"""
Fused Router GEMM + TopK + Softmax TileLang Kernels

Fuses the MoE router linear projection (GEMM) with TopK+Softmax into a single
kernel, eliminating the intermediate logits (T,E) global memory round-trip.

Pipeline:
  Separate: X(T,H) @ W(H,E) -> [global] logits(T,E) -> [global] (scores,indices)(T,K)
  Fused:    X(T,H) @ W(H,E) -> [shared] logits(BM,E) -> [global] (scores,indices)(T,K)

All four fused variants share an 8-warp-parallel register-cached epilogue:
each of 8 warps processes one row per round; each lane loads its slice of the
row into registers, runs the top-K algorithm with warp_reduce_max, then lane 0
of each warp computes softmax in registers and writes to global.

Variants:
  1. cpasync_selection : cp.async GEMM + selection-sort top-K
  2. cpasync_bitonic   : cp.async GEMM + bitonic-sort top-K
  3. tma_selection     : TMA+WS GEMM  + selection-sort top-K  (Hopper+)
  4. tma_bitonic       : TMA+WS GEMM  + bitonic-sort top-K    (Hopper+)

All internal GEMM buffers are padded to E_padded (next power of 2) because
TileLang's T.gemm and make_swizzled_layout require power-of-2 N.

TODO (warp-group split): currently TMA kernels use 128/128 producer/consumer
split per the Hopper reference example. A finer split (e.g. 32 producer + 96
compute consumer + 128 epilogue consumer in a 3-role pipeline) may overlap
epilogue with GEMM tail. Revisit after base variants are verified.
"""

import math
from typing import Dict

import torch
import tilelang
import tilelang.language as T


def _next_pow2(n: int) -> int:
    return 1 << math.ceil(math.log2(max(n, 2)))


# cp.async variants: pure cp.async path, disable both TMA and WS lowering.
# Keep TL_DISABLE_THREAD_STORAGE_SYNC OFF so auto-sync inserts barriers between
# shared writes (C_local->C_shared) and subsequent reads in the epilogue.
_cpasync_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

# TMA+WS variants: leave TMA and WS enabled. Only disable the auto-insertion
# of __syncthreads in divergent branches (we manage sync via mbarriers and
# explicit T.sync_threads across ws boundaries).
_tma_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


# ============================================================================
# Standalone Router GEMM (TL-sep baseline GEMM half)
# ============================================================================

@tilelang.jit(out_idx=[1])
def router_gemm_kernel(
    num_tokens, H, E_padded,
    block_M=64, block_K=64, num_stages=2, threads=256,
    dtype=T.bfloat16,
):
    """Standalone router GEMM: X(T,H) @ W(H,E_padded) -> logits(T,E_padded).

    Same tile structure as fused kernels (1D grid over M, BN=E_padded) so the
    TL-sep baseline differs from fused only in the epilogue.
    """
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        logits: T.Tensor([num_tokens, E_padded], accum_dtype),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, E_padded], dtype)
            C_local = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared = T.alloc_shared([block_M, E_padded], accum_dtype)

            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })

            m_start = bx * block_M
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=num_stages):
                T.copy(X[m_start, k * block_K], A_shared)
                T.copy(W[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, logits[m_start, 0])

    return kernel


_gemm_cache: Dict[str, any] = {}


def get_router_gemm_kernel(num_tokens, H, E, dtype=T.bfloat16):
    """Standalone GEMM with transparent E→E_padded padding."""
    E_padded = _next_pow2(E)
    cache_key = f"gemm_{num_tokens}_{H}_{E}"
    if cache_key in _gemm_cache:
        return _gemm_cache[cache_key]
    kernel = router_gemm_kernel(num_tokens, H, E_padded, dtype=dtype)
    if E == E_padded:
        _gemm_cache[cache_key] = kernel
    else:
        def padded_wrapper(X, W):
            W_p = torch.zeros(H, E_padded, device=W.device, dtype=W.dtype)
            W_p[:, :E] = W
            logits_p = kernel(X, W_p)
            return logits_p[:, :E].contiguous()
        _gemm_cache[cache_key] = padded_wrapper
    return _gemm_cache[cache_key]


# ============================================================================
# 1. cp.async + Selection Top-K Fused Kernel
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_cpasync_pass_configs)
def fused_router_cpasync_selection_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=64, block_K=64, num_stages=2, threads=256,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (cp.async) + Selection-Sort TopK + Softmax.

    Epilogue: 8 warps × 8 rounds, register-cached selection with warp_reduce_max.
    """
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    WARP_SIZE = 32
    num_warps = threads // WARP_SIZE  # 8
    rows_per_round = num_warps
    num_rounds = (block_M + rows_per_round - 1) // rows_per_round
    elems_per_lane = (E + WARP_SIZE - 1) // WARP_SIZE
    NEG_INF = -T.infinity(T.float32)

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, E_padded], dtype)
            C_local = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared = T.alloc_shared([block_M, E_padded], accum_dtype)

            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE
            lane_id = tx % WARP_SIZE

            # ---- GEMM ----
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=num_stages):
                T.copy(X[m_start, k * block_K], A_shared)
                T.copy(W[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.sync_threads()

            # ---- Epilogue: 8-warp-parallel selection ----
            for rnd in T.serial(num_rounds):
                row = rnd * rows_per_round + warp_id
                token_idx = m_start + row
                if token_idx < num_tokens:
                    # Per-lane register cache
                    my_vals = T.alloc_local([elems_per_lane], T.float32)
                    for i in T.serial(elems_per_lane):
                        col = lane_id * elems_per_lane + i
                        my_vals[i] = T.if_then_else(col < E, C_shared[row, col], NEG_INF)

                    local_vals = T.alloc_local([top_k], T.float32)
                    local_idx = T.alloc_local([top_k], T.int32)

                    for k_iter in T.serial(top_k):
                        my_max_val = T.alloc_var(T.float32)
                        my_max_idx = T.alloc_var(T.int32)
                        my_max_val = NEG_INF
                        my_max_idx = T.int32(-1)

                        for i in T.serial(elems_per_lane):
                            if my_vals[i] > my_max_val:
                                my_max_val = my_vals[i]
                                my_max_idx = T.int32(lane_id * elems_per_lane + i)

                        warp_max = T.warp_reduce_max(my_max_val)
                        cand = T.alloc_var(T.float32)
                        cand = T.float32(-1.0)
                        if my_max_val == warp_max:
                            cand = T.cast(my_max_idx, T.float32)
                        winner_f = T.warp_reduce_max(cand)
                        winner_idx = T.cast(winner_f, T.int32)

                        if lane_id == 0:
                            local_vals[k_iter] = warp_max
                            local_idx[k_iter] = winner_idx

                        for i in T.serial(elems_per_lane):
                            if lane_id * elems_per_lane + i == winner_idx:
                                my_vals[i] = NEG_INF

                    if lane_id == 0:
                        max_val = T.alloc_var(T.float32)
                        max_val = NEG_INF
                        for i in T.serial(top_k):
                            if local_vals[i] > max_val:
                                max_val = local_vals[i]

                        exp_sum = T.alloc_var(T.float32)
                        exp_sum = T.float32(0.0)
                        for i in T.serial(top_k):
                            local_vals[i] = T.exp(local_vals[i] - max_val)
                            exp_sum = exp_sum + local_vals[i]

                        for i in T.serial(top_k):
                            scores[token_idx, i] = local_vals[i] / exp_sum
                            indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# 2. cp.async + Bitonic Top-K Fused Kernel
# ============================================================================
# Warp-parallel bitonic: each warp sorts one row.  32 lanes each hold
# elems_per_lane values in register fragment. Local descending sort within
# each lane's slice, then bitonic-merge across 32 lanes via shfl_xor.
# After merge, lane 0 holds the top-K values in a local buffer.

@tilelang.jit(out_idx=[1, 2], pass_configs=_cpasync_pass_configs)
def fused_router_cpasync_bitonic_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=64, block_K=64, num_stages=2, threads=256,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (cp.async) + Bitonic-Sort TopK + Softmax.

    Epilogue: 8 warps × 8 rounds. Per-warp bitonic: each lane holds
    elems_per_lane uint32 bit-packed (value|col_idx) values, local-sort
    descending, then K-padded merge across 32 lanes via shfl_xor tree.
    """
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    WARP_SIZE = 32
    num_warps = threads // WARP_SIZE
    rows_per_round = num_warps
    num_rounds = (block_M + rows_per_round - 1) // rows_per_round

    # Bit-packing: encode column index into low log2(E_padded) bits of float32.
    elems_per_lane = E_padded // WARP_SIZE if E_padded >= WARP_SIZE else 1
    if elems_per_lane < 1:
        elems_per_lane = 1
    K_padded = _next_pow2(top_k)
    # frag must accommodate both the per-lane source elements AND the K_padded
    # working set used by the bitonic merge; otherwise frag[ki] for ki<K_padded
    # is OOB when elems_per_lane < K_padded.
    frag_size = max(elems_per_lane, K_padded)
    log_ept = int(math.log2(elems_per_lane)) if elems_per_lane > 1 else 0
    log_kp = int(math.log2(K_padded)) if K_padded > 1 else 0
    idx_bits = int(math.ceil(math.log2(max(E_padded, 2))))
    idx_mask_val = (1 << idx_bits) - 1
    num_merge_stages = int(math.log2(WARP_SIZE))  # 5

    NEG_INF = -T.infinity(T.float32)
    # Pre-compute bit pattern for a "padding" entry: NEG_INF value with col=0
    # encoded as negative (so enc bits = 0 & idx_mask = 0). After idx stripping
    # fval stays NEG_INF, losing every compare. frag_size - elems_per_lane such
    # slots occupy the tail when elems_per_lane < K_padded.

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, E_padded], dtype)
            C_local = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared = T.alloc_shared([block_M, E_padded], accum_dtype)

            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE
            lane_id = tx % WARP_SIZE
            idx_mask = T.uint32(idx_mask_val)

            # ---- GEMM ----
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=num_stages):
                T.copy(X[m_start, k * block_K], A_shared)
                T.copy(W[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.sync_threads()

            # ---- Epilogue: 8-warp-parallel bitonic ----
            for rnd in T.serial(num_rounds):
                row = rnd * rows_per_round + warp_id
                token_idx = m_start + row
                if token_idx < num_tokens:
                    # Bit-pack E elements into register fragment.
                    # Size = max(elems_per_lane, K_padded); pad tail with NEG_INF.
                    # NEG_INF_BITS_CLEAN = 0xFF800000 & ~idx_mask = 0xFF800000 (low
                    # 23 bits of -inf are already 0, so masking is a no-op).
                    NEG_INF_BITS_CLEAN = T.uint32(0xFF800000 & ~idx_mask_val)
                    frag = T.alloc_local([frag_size], T.uint32)
                    for e in T.serial(elems_per_lane):
                        col = lane_id * elems_per_lane + e
                        fval = T.if_then_else(col < E, C_shared[row, col], NEG_INF)
                        bits = T.reinterpret(T.uint32, fval)
                        col_u32 = T.Cast(T.uint32, col)
                        encoded_idx = T.if_then_else(
                            fval >= T.float32(0.0),
                            (~col_u32) & idx_mask,
                            col_u32 & idx_mask)
                        frag[e] = (bits & (~idx_mask)) | encoded_idx
                    # Pad tail slots if frag_size > elems_per_lane
                    for e in T.serial(frag_size - elems_per_lane):
                        frag[elems_per_lane + e] = NEG_INF_BITS_CLEAN

                    # Local partial top-K_padded via selection over all frag_size slots.
                    for ki in T.serial(K_padded):
                        for e in T.serial(frag_size):
                            if e > ki:
                                fi = T.reinterpret(T.float32, frag[ki])
                                fe = T.reinterpret(T.float32, frag[e])
                                if fe > fi:
                                    tmp = frag[ki]
                                    frag[ki] = frag[e]
                                    frag[e] = tmp

                    # Local bitonic sort on K_padded front
                    if log_kp > 0:
                        for i1 in T.serial(log_kp):
                            for i2 in T.serial(i1 + 1):
                                step = T.int32(1) << (i1 - i2)
                                dmask = T.int32(1) << (i1 + 1)
                                for i in T.serial(K_padded):
                                    j = i ^ step
                                    if i < j:
                                        ascending = (i & dmask) != 0
                                        fi = T.reinterpret(T.float32, frag[i])
                                        fj = T.reinterpret(T.float32, frag[j])
                                        do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                        if do_swap:
                                            tmp = frag[i]
                                            frag[i] = frag[j]
                                            frag[j] = tmp

                    # Cross-lane merge via shfl_xor tree (log2(32)=5 stages)
                    partner = T.alloc_local([K_padded], T.uint32)
                    for ms in T.serial(num_merge_stages):
                        stride = T.int32(WARP_SIZE) >> (ms + 1)
                        am_low = (lane_id & stride) == 0

                        for k in T.serial(K_padded):
                            val_i32 = T.reinterpret(T.int32, frag[k])
                            p_i32 = T.shfl_xor(val_i32, stride)
                            partner[k] = T.reinterpret(T.uint32, p_i32)

                        for k in T.serial(K_padded):
                            my_f = T.reinterpret(T.float32, frag[k])
                            p_f = T.reinterpret(T.float32, partner[K_padded - 1 - k])
                            should_take = T.if_then_else(am_low, p_f > my_f, my_f > p_f)
                            if should_take:
                                frag[k] = partner[K_padded - 1 - k]

                        # Re-sort K_padded front
                        if log_kp > 0:
                            for i1 in T.serial(log_kp):
                                for i2 in T.serial(i1 + 1):
                                    step = T.int32(1) << (i1 - i2)
                                    dmask = T.int32(1) << (i1 + 1)
                                    for i in T.serial(K_padded):
                                        j = i ^ step
                                        if i < j:
                                            ascending = (i & dmask) != 0
                                            fi = T.reinterpret(T.float32, frag[i])
                                            fj = T.reinterpret(T.float32, frag[j])
                                            do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                            if do_swap:
                                                tmp = frag[i]
                                                frag[i] = frag[j]
                                                frag[j] = tmp

                    # Decode + softmax (lane 0)
                    if lane_id == 0:
                        local_vals = T.alloc_local([top_k], T.float32)
                        local_idx = T.alloc_local([top_k], T.int32)
                        for ki in T.serial(top_k):
                            packed = frag[ki]
                            cleaned = packed & (~idx_mask)
                            fval = T.reinterpret(T.float32, cleaned)
                            enc = packed & idx_mask
                            col_idx = T.if_then_else(
                                fval >= T.float32(0.0),
                                (~enc) & idx_mask,
                                enc & idx_mask)
                            local_vals[ki] = fval
                            local_idx[ki] = T.Cast(T.int32, col_idx)

                        max_val = T.alloc_var(T.float32)
                        max_val = NEG_INF
                        for i in T.serial(top_k):
                            if local_vals[i] > max_val:
                                max_val = local_vals[i]

                        exp_sum = T.alloc_var(T.float32)
                        exp_sum = T.float32(0.0)
                        for i in T.serial(top_k):
                            local_vals[i] = T.exp(local_vals[i] - max_val)
                            exp_sum = exp_sum + local_vals[i]

                        for i in T.serial(top_k):
                            scores[token_idx, i] = local_vals[i] / exp_sum
                            indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# 3. TMA+WS + Selection Top-K Fused Kernel  (Hopper+)
# ============================================================================
# GEMM: standard Hopper-style warp specialization — ws(0)=consumer (WGMMA),
# ws(1)=producer (TMA). After GEMM, consumer dumps C_local→C_shared and
# signals epi_ready; both warp groups then converge and do 8-warp epilogue.
#
# TODO: a finer warp-group split (e.g. 32t producer + 96t WGMMA + 128t epilogue
# pre-worker) may overlap epilogue with GEMM tail.  Revisit after baseline.

@tilelang.jit(out_idx=[1, 2], pass_configs=_tma_pass_configs)
def fused_router_tma_selection_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=64, block_K=64, num_stages=2, threads=256,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (TMA+WS, Hopper) + Selection-Sort TopK + Softmax."""
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    WARP_SIZE = 32
    num_warps = threads // WARP_SIZE
    rows_per_round = num_warps
    num_rounds = (block_M + rows_per_round - 1) // rows_per_round
    elems_per_lane = (E + WARP_SIZE - 1) // WARP_SIZE
    NEG_INF = -T.infinity(T.float32)

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            # Multi-staged shared buffers for producer/consumer pipeline
            A_shared = T.alloc_shared([num_stages, block_M, block_K], dtype)
            B_shared = T.alloc_shared([num_stages, block_K, E_padded], dtype)
            C_local = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared = T.alloc_shared([block_M, E_padded], accum_dtype)

            # mbarrier IDs: [0..num_stages-1]=loaded, [num_stages..2*num_stages-1]=consumed,
            # [2*num_stages]=epi_ready (consumer→all: C_shared is written).
            T.create_list_of_mbarrier([128, 128] * num_stages + [128])
            EPI_READY = 2 * num_stages

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE
            lane_id = tx % WARP_SIZE

            # ---- GEMM: warp-specialized with TMA producer ----
            with T.ws(0):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local)
            with T.ws(1):
                T.annotate_producer_reg_dealloc(24)

            for ko in range(T.ceildiv(H, block_K)):
                with T.ws(1):  # producer: TMA copy
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages + num_stages,
                        parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(
                        X[m_start:m_start + block_M, ko * block_K:(ko + 1) * block_K],
                        A_shared[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, 0:E_padded],
                        B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                with T.ws(0):  # consumer: WGMMA compute
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            # Consumer stages output and signals readiness
            with T.ws(0):
                T.copy(C_local, C_shared)
                T.mbarrier_arrive(mbarrier=EPI_READY)

            # Both warp groups wait for C_shared
            T.mbarrier_wait_parity(mbarrier=EPI_READY, parity=0)
            T.sync_threads()  # explicit because TL_DISABLE_THREAD_STORAGE_SYNC

            # ---- Epilogue: 8-warp-parallel selection ----
            for rnd in T.serial(num_rounds):
                row = rnd * rows_per_round + warp_id
                token_idx = m_start + row
                if token_idx < num_tokens:
                    my_vals = T.alloc_local([elems_per_lane], T.float32)
                    for i in T.serial(elems_per_lane):
                        col = lane_id * elems_per_lane + i
                        my_vals[i] = T.if_then_else(col < E, C_shared[row, col], NEG_INF)

                    local_vals = T.alloc_local([top_k], T.float32)
                    local_idx = T.alloc_local([top_k], T.int32)

                    for k_iter in T.serial(top_k):
                        my_max_val = T.alloc_var(T.float32)
                        my_max_idx = T.alloc_var(T.int32)
                        my_max_val = NEG_INF
                        my_max_idx = T.int32(-1)

                        for i in T.serial(elems_per_lane):
                            if my_vals[i] > my_max_val:
                                my_max_val = my_vals[i]
                                my_max_idx = T.int32(lane_id * elems_per_lane + i)

                        warp_max = T.warp_reduce_max(my_max_val)
                        cand = T.alloc_var(T.float32)
                        cand = T.float32(-1.0)
                        if my_max_val == warp_max:
                            cand = T.cast(my_max_idx, T.float32)
                        winner_f = T.warp_reduce_max(cand)
                        winner_idx = T.cast(winner_f, T.int32)

                        if lane_id == 0:
                            local_vals[k_iter] = warp_max
                            local_idx[k_iter] = winner_idx

                        for i in T.serial(elems_per_lane):
                            if lane_id * elems_per_lane + i == winner_idx:
                                my_vals[i] = NEG_INF

                    if lane_id == 0:
                        max_val = T.alloc_var(T.float32)
                        max_val = NEG_INF
                        for i in T.serial(top_k):
                            if local_vals[i] > max_val:
                                max_val = local_vals[i]

                        exp_sum = T.alloc_var(T.float32)
                        exp_sum = T.float32(0.0)
                        for i in T.serial(top_k):
                            local_vals[i] = T.exp(local_vals[i] - max_val)
                            exp_sum = exp_sum + local_vals[i]

                        for i in T.serial(top_k):
                            scores[token_idx, i] = local_vals[i] / exp_sum
                            indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# 4. TMA+WS + Bitonic Top-K Fused Kernel  (Hopper+)
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_tma_pass_configs)
def fused_router_tma_bitonic_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=64, block_K=64, num_stages=2, threads=256,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (TMA+WS, Hopper) + Bitonic-Sort TopK + Softmax."""
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    WARP_SIZE = 32
    num_warps = threads // WARP_SIZE
    rows_per_round = num_warps
    num_rounds = (block_M + rows_per_round - 1) // rows_per_round

    elems_per_lane = E_padded // WARP_SIZE if E_padded >= WARP_SIZE else 1
    if elems_per_lane < 1:
        elems_per_lane = 1
    K_padded = _next_pow2(top_k)
    frag_size = max(elems_per_lane, K_padded)
    log_kp = int(math.log2(K_padded)) if K_padded > 1 else 0
    idx_bits = int(math.ceil(math.log2(max(E_padded, 2))))
    idx_mask_val = (1 << idx_bits) - 1
    num_merge_stages = int(math.log2(WARP_SIZE))

    NEG_INF = -T.infinity(T.float32)

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([num_stages, block_M, block_K], dtype)
            B_shared = T.alloc_shared([num_stages, block_K, E_padded], dtype)
            C_local = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared = T.alloc_shared([block_M, E_padded], accum_dtype)

            T.create_list_of_mbarrier([128, 128] * num_stages + [128])
            EPI_READY = 2 * num_stages

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE
            lane_id = tx % WARP_SIZE
            idx_mask = T.uint32(idx_mask_val)
            NEG_INF_BITS_CLEAN = T.uint32(0xFF800000 & ~idx_mask_val)

            with T.ws(0):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local)
            with T.ws(1):
                T.annotate_producer_reg_dealloc(24)

            for ko in range(T.ceildiv(H, block_K)):
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages + num_stages,
                        parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(
                        X[m_start:m_start + block_M, ko * block_K:(ko + 1) * block_K],
                        A_shared[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, 0:E_padded],
                        B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            with T.ws(0):
                T.copy(C_local, C_shared)
                T.mbarrier_arrive(mbarrier=EPI_READY)

            T.mbarrier_wait_parity(mbarrier=EPI_READY, parity=0)
            T.sync_threads()

            # ---- Epilogue: 8-warp-parallel bitonic ----
            for rnd in T.serial(num_rounds):
                row = rnd * rows_per_round + warp_id
                token_idx = m_start + row
                if token_idx < num_tokens:
                    frag = T.alloc_local([frag_size], T.uint32)
                    for e in T.serial(elems_per_lane):
                        col = lane_id * elems_per_lane + e
                        fval = T.if_then_else(col < E, C_shared[row, col], NEG_INF)
                        bits = T.reinterpret(T.uint32, fval)
                        col_u32 = T.Cast(T.uint32, col)
                        encoded_idx = T.if_then_else(
                            fval >= T.float32(0.0),
                            (~col_u32) & idx_mask,
                            col_u32 & idx_mask)
                        frag[e] = (bits & (~idx_mask)) | encoded_idx
                    for e in T.serial(frag_size - elems_per_lane):
                        frag[elems_per_lane + e] = NEG_INF_BITS_CLEAN

                    for ki in T.serial(K_padded):
                        for e in T.serial(frag_size):
                            if e > ki:
                                fi = T.reinterpret(T.float32, frag[ki])
                                fe = T.reinterpret(T.float32, frag[e])
                                if fe > fi:
                                    tmp = frag[ki]
                                    frag[ki] = frag[e]
                                    frag[e] = tmp

                    if log_kp > 0:
                        for i1 in T.serial(log_kp):
                            for i2 in T.serial(i1 + 1):
                                step = T.int32(1) << (i1 - i2)
                                dmask = T.int32(1) << (i1 + 1)
                                for i in T.serial(K_padded):
                                    j = i ^ step
                                    if i < j:
                                        ascending = (i & dmask) != 0
                                        fi = T.reinterpret(T.float32, frag[i])
                                        fj = T.reinterpret(T.float32, frag[j])
                                        do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                        if do_swap:
                                            tmp = frag[i]
                                            frag[i] = frag[j]
                                            frag[j] = tmp

                    partner = T.alloc_local([K_padded], T.uint32)
                    for ms in T.serial(num_merge_stages):
                        stride = T.int32(WARP_SIZE) >> (ms + 1)
                        am_low = (lane_id & stride) == 0

                        for k in T.serial(K_padded):
                            val_i32 = T.reinterpret(T.int32, frag[k])
                            p_i32 = T.shfl_xor(val_i32, stride)
                            partner[k] = T.reinterpret(T.uint32, p_i32)

                        for k in T.serial(K_padded):
                            my_f = T.reinterpret(T.float32, frag[k])
                            p_f = T.reinterpret(T.float32, partner[K_padded - 1 - k])
                            should_take = T.if_then_else(am_low, p_f > my_f, my_f > p_f)
                            if should_take:
                                frag[k] = partner[K_padded - 1 - k]

                        if log_kp > 0:
                            for i1 in T.serial(log_kp):
                                for i2 in T.serial(i1 + 1):
                                    step = T.int32(1) << (i1 - i2)
                                    dmask = T.int32(1) << (i1 + 1)
                                    for i in T.serial(K_padded):
                                        j = i ^ step
                                        if i < j:
                                            ascending = (i & dmask) != 0
                                            fi = T.reinterpret(T.float32, frag[i])
                                            fj = T.reinterpret(T.float32, frag[j])
                                            do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                            if do_swap:
                                                tmp = frag[i]
                                                frag[i] = frag[j]
                                                frag[j] = tmp

                    if lane_id == 0:
                        local_vals = T.alloc_local([top_k], T.float32)
                        local_idx = T.alloc_local([top_k], T.int32)
                        for ki in T.serial(top_k):
                            packed = frag[ki]
                            cleaned = packed & (~idx_mask)
                            fval = T.reinterpret(T.float32, cleaned)
                            enc = packed & idx_mask
                            col_idx = T.if_then_else(
                                fval >= T.float32(0.0),
                                (~enc) & idx_mask,
                                enc & idx_mask)
                            local_vals[ki] = fval
                            local_idx[ki] = T.Cast(T.int32, col_idx)

                        max_val = T.alloc_var(T.float32)
                        max_val = NEG_INF
                        for i in T.serial(top_k):
                            if local_vals[i] > max_val:
                                max_val = local_vals[i]

                        exp_sum = T.alloc_var(T.float32)
                        exp_sum = T.float32(0.0)
                        for i in T.serial(top_k):
                            local_vals[i] = T.exp(local_vals[i] - max_val)
                            exp_sum = exp_sum + local_vals[i]

                        for i in T.serial(top_k):
                            scores[token_idx, i] = local_vals[i] / exp_sum
                            indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# 5. TMA+WS 2-Consumer Variants (Hopper, threads=384)
#
# 1 producer WG (WG2) + 2 consumer WGs (WG0, WG1). Each consumer processes a
# full block_M=64 tile (same WGMMA shape as single-consumer TMA variants), so
# per-block total output rows = 2*block_M. Grid halves; total compute threads
# = 256 (same as cp.async) but now driven by TMA with warp specialization.
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_tma_pass_configs)
def fused_router_tma2_selection_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=64, block_K=64, num_stages=2, threads=384,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (TMA+WS, 1prod+2cons, Hopper) + Selection + Softmax."""
    block_M_total = 2 * block_M
    num_m_blocks = (num_tokens + block_M_total - 1) // block_M_total
    accum_dtype = T.float32
    WARP_SIZE = 32
    num_warps_per_wg = 128 // WARP_SIZE  # 4 warps/WG → rows_per_round=4
    rows_per_round = num_warps_per_wg
    num_rounds = (block_M + rows_per_round - 1) // rows_per_round  # 16
    elems_per_lane = (E + WARP_SIZE - 1) // WARP_SIZE
    NEG_INF = -T.infinity(T.float32)

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared_0 = T.alloc_shared([num_stages, block_M, block_K], dtype)
            A_shared_1 = T.alloc_shared([num_stages, block_M, block_K], dtype)
            B_shared   = T.alloc_shared([num_stages, block_K, E_padded], dtype)
            C_local_0  = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_local_1  = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared_0 = T.alloc_shared([block_M, E_padded], accum_dtype)
            C_shared_1 = T.alloc_shared([block_M, E_padded], accum_dtype)

            # Barriers: [0..ns-1]=loaded (producer→both cons, 128 arrives),
            #           [ns..2*ns-1]=consumed (both cons→producer, 256 arrives),
            #           [2*ns]=epi_ready_0, [2*ns+1]=epi_ready_1 (128 each)
            T.create_list_of_mbarrier(
                [128] * num_stages + [256] * num_stages + [128, 128])
            EPI0 = 2 * num_stages
            EPI1 = 2 * num_stages + 1

            m_start_0 = bx * block_M_total
            m_start_1 = bx * block_M_total + block_M
            tx = T.get_thread_binding()
            warp_id_wg = (tx // WARP_SIZE) % num_warps_per_wg
            lane_id = tx % WARP_SIZE

            with T.ws(0):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local_0)
            with T.ws(1):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local_1)
            with T.ws(2):
                T.annotate_producer_reg_dealloc(24)

            for ko in range(T.ceildiv(H, block_K)):
                with T.ws(2):  # producer: TMA copy A_0, A_1, B
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages + num_stages,
                        parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(
                        X[m_start_0:m_start_0 + block_M,
                          ko * block_K:(ko + 1) * block_K],
                        A_shared_0[ko % num_stages, :, :])
                    T.copy(
                        X[m_start_1:m_start_1 + block_M,
                          ko * block_K:(ko + 1) * block_K],
                        A_shared_1[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, 0:E_padded],
                        B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                with T.ws(0):  # consumer 0: WGMMA on A_0 × B
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared_0[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local_0)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)
                with T.ws(1):  # consumer 1: WGMMA on A_1 × B
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared_1[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local_1)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            # Stage per-WG C_local → C_shared; signal epi_ready per WG
            with T.ws(0):
                T.copy(C_local_0, C_shared_0)
                T.mbarrier_arrive(mbarrier=EPI0)
            with T.ws(1):
                T.copy(C_local_1, C_shared_1)
                T.mbarrier_arrive(mbarrier=EPI1)

            # ---- Epilogue: each consumer handles its own M slice ----
            with T.ws(0):
                T.mbarrier_wait_parity(mbarrier=EPI0, parity=0)
                for rnd in T.serial(num_rounds):
                    row = rnd * rows_per_round + warp_id_wg
                    token_idx = m_start_0 + row
                    if token_idx < num_tokens:
                        my_vals = T.alloc_local([elems_per_lane], T.float32)
                        for i in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + i
                            my_vals[i] = T.if_then_else(
                                col < E, C_shared_0[row, col], NEG_INF)

                        local_vals = T.alloc_local([top_k], T.float32)
                        local_idx = T.alloc_local([top_k], T.int32)

                        for k_iter in T.serial(top_k):
                            my_max_val = T.alloc_var(T.float32)
                            my_max_idx = T.alloc_var(T.int32)
                            my_max_val = NEG_INF
                            my_max_idx = T.int32(-1)

                            for i in T.serial(elems_per_lane):
                                if my_vals[i] > my_max_val:
                                    my_max_val = my_vals[i]
                                    my_max_idx = T.int32(lane_id * elems_per_lane + i)

                            warp_max = T.warp_reduce_max(my_max_val)
                            cand = T.alloc_var(T.float32)
                            cand = T.float32(-1.0)
                            if my_max_val == warp_max:
                                cand = T.cast(my_max_idx, T.float32)
                            winner_f = T.warp_reduce_max(cand)
                            winner_idx = T.cast(winner_f, T.int32)

                            if lane_id == 0:
                                local_vals[k_iter] = warp_max
                                local_idx[k_iter] = winner_idx

                            for i in T.serial(elems_per_lane):
                                if lane_id * elems_per_lane + i == winner_idx:
                                    my_vals[i] = NEG_INF

                        if lane_id == 0:
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]

                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]

                            for i in T.serial(top_k):
                                scores[token_idx, i] = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

            with T.ws(1):
                T.mbarrier_wait_parity(mbarrier=EPI1, parity=0)
                for rnd in T.serial(num_rounds):
                    row = rnd * rows_per_round + warp_id_wg
                    token_idx = m_start_1 + row
                    if token_idx < num_tokens:
                        my_vals = T.alloc_local([elems_per_lane], T.float32)
                        for i in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + i
                            my_vals[i] = T.if_then_else(
                                col < E, C_shared_1[row, col], NEG_INF)

                        local_vals = T.alloc_local([top_k], T.float32)
                        local_idx = T.alloc_local([top_k], T.int32)

                        for k_iter in T.serial(top_k):
                            my_max_val = T.alloc_var(T.float32)
                            my_max_idx = T.alloc_var(T.int32)
                            my_max_val = NEG_INF
                            my_max_idx = T.int32(-1)

                            for i in T.serial(elems_per_lane):
                                if my_vals[i] > my_max_val:
                                    my_max_val = my_vals[i]
                                    my_max_idx = T.int32(lane_id * elems_per_lane + i)

                            warp_max = T.warp_reduce_max(my_max_val)
                            cand = T.alloc_var(T.float32)
                            cand = T.float32(-1.0)
                            if my_max_val == warp_max:
                                cand = T.cast(my_max_idx, T.float32)
                            winner_f = T.warp_reduce_max(cand)
                            winner_idx = T.cast(winner_f, T.int32)

                            if lane_id == 0:
                                local_vals[k_iter] = warp_max
                                local_idx[k_iter] = winner_idx

                            for i in T.serial(elems_per_lane):
                                if lane_id * elems_per_lane + i == winner_idx:
                                    my_vals[i] = NEG_INF

                        if lane_id == 0:
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]

                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]

                            for i in T.serial(top_k):
                                scores[token_idx, i] = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# TMA+WS 2-Consumer Bitonic (Hopper, threads=384)
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_tma_pass_configs)
def fused_router_tma2_bitonic_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=64, block_K=64, num_stages=2, threads=384,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (TMA+WS, 1prod+2cons, Hopper) + Bitonic + Softmax."""
    block_M_total = 2 * block_M
    num_m_blocks = (num_tokens + block_M_total - 1) // block_M_total
    accum_dtype = T.float32
    WARP_SIZE = 32
    num_warps_per_wg = 128 // WARP_SIZE  # 4 warps/WG
    rows_per_round = num_warps_per_wg
    num_rounds = (block_M + rows_per_round - 1) // rows_per_round

    elems_per_lane = E_padded // WARP_SIZE if E_padded >= WARP_SIZE else 1
    if elems_per_lane < 1:
        elems_per_lane = 1
    K_padded = _next_pow2(top_k)
    frag_size = max(elems_per_lane, K_padded)
    log_kp = int(math.log2(K_padded)) if K_padded > 1 else 0
    idx_bits = int(math.ceil(math.log2(max(E_padded, 2))))
    idx_mask_val = (1 << idx_bits) - 1
    num_merge_stages = int(math.log2(WARP_SIZE))

    NEG_INF = -T.infinity(T.float32)
    NEG_INF_BITS_CLEAN_val = 0xFF800000 & ~idx_mask_val

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared_0 = T.alloc_shared([num_stages, block_M, block_K], dtype)
            A_shared_1 = T.alloc_shared([num_stages, block_M, block_K], dtype)
            B_shared   = T.alloc_shared([num_stages, block_K, E_padded], dtype)
            C_local_0  = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_local_1  = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared_0 = T.alloc_shared([block_M, E_padded], accum_dtype)
            C_shared_1 = T.alloc_shared([block_M, E_padded], accum_dtype)

            T.create_list_of_mbarrier(
                [128] * num_stages + [256] * num_stages + [128, 128])
            EPI0 = 2 * num_stages
            EPI1 = 2 * num_stages + 1

            m_start_0 = bx * block_M_total
            m_start_1 = bx * block_M_total + block_M
            tx = T.get_thread_binding()
            warp_id_wg = (tx // WARP_SIZE) % num_warps_per_wg
            lane_id = tx % WARP_SIZE
            idx_mask = T.uint32(idx_mask_val)
            NEG_INF_BITS_CLEAN = T.uint32(NEG_INF_BITS_CLEAN_val)

            with T.ws(0):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local_0)
            with T.ws(1):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local_1)
            with T.ws(2):
                T.annotate_producer_reg_dealloc(24)

            for ko in range(T.ceildiv(H, block_K)):
                with T.ws(2):  # producer
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages + num_stages,
                        parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(
                        X[m_start_0:m_start_0 + block_M,
                          ko * block_K:(ko + 1) * block_K],
                        A_shared_0[ko % num_stages, :, :])
                    T.copy(
                        X[m_start_1:m_start_1 + block_M,
                          ko * block_K:(ko + 1) * block_K],
                        A_shared_1[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, 0:E_padded],
                        B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared_0[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local_0)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared_1[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local_1)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            with T.ws(0):
                T.copy(C_local_0, C_shared_0)
                T.mbarrier_arrive(mbarrier=EPI0)
            with T.ws(1):
                T.copy(C_local_1, C_shared_1)
                T.mbarrier_arrive(mbarrier=EPI1)

            # ---- WG0 bitonic epilogue ----
            with T.ws(0):
                T.mbarrier_wait_parity(mbarrier=EPI0, parity=0)
                for rnd in T.serial(num_rounds):
                    row = rnd * rows_per_round + warp_id_wg
                    token_idx = m_start_0 + row
                    if token_idx < num_tokens:
                        frag = T.alloc_local([frag_size], T.uint32)
                        for e in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + e
                            fval = T.if_then_else(col < E, C_shared_0[row, col], NEG_INF)
                            bits = T.reinterpret(T.uint32, fval)
                            col_u32 = T.Cast(T.uint32, col)
                            encoded_idx = T.if_then_else(
                                fval >= T.float32(0.0),
                                (~col_u32) & idx_mask,
                                col_u32 & idx_mask)
                            frag[e] = (bits & (~idx_mask)) | encoded_idx
                        for e in T.serial(frag_size - elems_per_lane):
                            frag[elems_per_lane + e] = NEG_INF_BITS_CLEAN

                        for ki in T.serial(K_padded):
                            for e in T.serial(frag_size):
                                if e > ki:
                                    fi = T.reinterpret(T.float32, frag[ki])
                                    fe = T.reinterpret(T.float32, frag[e])
                                    if fe > fi:
                                        tmp = frag[ki]
                                        frag[ki] = frag[e]
                                        frag[e] = tmp

                        if log_kp > 0:
                            for i1 in T.serial(log_kp):
                                for i2 in T.serial(i1 + 1):
                                    step = T.int32(1) << (i1 - i2)
                                    dmask = T.int32(1) << (i1 + 1)
                                    for i in T.serial(K_padded):
                                        j = i ^ step
                                        if i < j:
                                            ascending = (i & dmask) != 0
                                            fi = T.reinterpret(T.float32, frag[i])
                                            fj = T.reinterpret(T.float32, frag[j])
                                            do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                            if do_swap:
                                                tmp = frag[i]
                                                frag[i] = frag[j]
                                                frag[j] = tmp

                        partner = T.alloc_local([K_padded], T.uint32)
                        for ms in T.serial(num_merge_stages):
                            stride = T.int32(WARP_SIZE) >> (ms + 1)
                            am_low = (lane_id & stride) == 0
                            for k in T.serial(K_padded):
                                val_i32 = T.reinterpret(T.int32, frag[k])
                                p_i32 = T.shfl_xor(val_i32, stride)
                                partner[k] = T.reinterpret(T.uint32, p_i32)
                            for k in T.serial(K_padded):
                                my_f = T.reinterpret(T.float32, frag[k])
                                p_f = T.reinterpret(T.float32, partner[K_padded - 1 - k])
                                should_take = T.if_then_else(am_low, p_f > my_f, my_f > p_f)
                                if should_take:
                                    frag[k] = partner[K_padded - 1 - k]
                            if log_kp > 0:
                                for i1 in T.serial(log_kp):
                                    for i2 in T.serial(i1 + 1):
                                        step = T.int32(1) << (i1 - i2)
                                        dmask = T.int32(1) << (i1 + 1)
                                        for i in T.serial(K_padded):
                                            j = i ^ step
                                            if i < j:
                                                ascending = (i & dmask) != 0
                                                fi = T.reinterpret(T.float32, frag[i])
                                                fj = T.reinterpret(T.float32, frag[j])
                                                do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                                if do_swap:
                                                    tmp = frag[i]
                                                    frag[i] = frag[j]
                                                    frag[j] = tmp

                        if lane_id == 0:
                            local_vals = T.alloc_local([top_k], T.float32)
                            local_idx = T.alloc_local([top_k], T.int32)
                            for ki in T.serial(top_k):
                                packed = frag[ki]
                                cleaned = packed & (~idx_mask)
                                fval = T.reinterpret(T.float32, cleaned)
                                enc = packed & idx_mask
                                col_idx = T.if_then_else(
                                    fval >= T.float32(0.0),
                                    (~enc) & idx_mask,
                                    enc & idx_mask)
                                local_vals[ki] = fval
                                local_idx[ki] = T.Cast(T.int32, col_idx)
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]
                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]
                            for i in T.serial(top_k):
                                scores[token_idx, i] = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

            # ---- WG1 bitonic epilogue ----
            with T.ws(1):
                T.mbarrier_wait_parity(mbarrier=EPI1, parity=0)
                for rnd in T.serial(num_rounds):
                    row = rnd * rows_per_round + warp_id_wg
                    token_idx = m_start_1 + row
                    if token_idx < num_tokens:
                        frag = T.alloc_local([frag_size], T.uint32)
                        for e in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + e
                            fval = T.if_then_else(col < E, C_shared_1[row, col], NEG_INF)
                            bits = T.reinterpret(T.uint32, fval)
                            col_u32 = T.Cast(T.uint32, col)
                            encoded_idx = T.if_then_else(
                                fval >= T.float32(0.0),
                                (~col_u32) & idx_mask,
                                col_u32 & idx_mask)
                            frag[e] = (bits & (~idx_mask)) | encoded_idx
                        for e in T.serial(frag_size - elems_per_lane):
                            frag[elems_per_lane + e] = NEG_INF_BITS_CLEAN

                        for ki in T.serial(K_padded):
                            for e in T.serial(frag_size):
                                if e > ki:
                                    fi = T.reinterpret(T.float32, frag[ki])
                                    fe = T.reinterpret(T.float32, frag[e])
                                    if fe > fi:
                                        tmp = frag[ki]
                                        frag[ki] = frag[e]
                                        frag[e] = tmp

                        if log_kp > 0:
                            for i1 in T.serial(log_kp):
                                for i2 in T.serial(i1 + 1):
                                    step = T.int32(1) << (i1 - i2)
                                    dmask = T.int32(1) << (i1 + 1)
                                    for i in T.serial(K_padded):
                                        j = i ^ step
                                        if i < j:
                                            ascending = (i & dmask) != 0
                                            fi = T.reinterpret(T.float32, frag[i])
                                            fj = T.reinterpret(T.float32, frag[j])
                                            do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                            if do_swap:
                                                tmp = frag[i]
                                                frag[i] = frag[j]
                                                frag[j] = tmp

                        partner = T.alloc_local([K_padded], T.uint32)
                        for ms in T.serial(num_merge_stages):
                            stride = T.int32(WARP_SIZE) >> (ms + 1)
                            am_low = (lane_id & stride) == 0
                            for k in T.serial(K_padded):
                                val_i32 = T.reinterpret(T.int32, frag[k])
                                p_i32 = T.shfl_xor(val_i32, stride)
                                partner[k] = T.reinterpret(T.uint32, p_i32)
                            for k in T.serial(K_padded):
                                my_f = T.reinterpret(T.float32, frag[k])
                                p_f = T.reinterpret(T.float32, partner[K_padded - 1 - k])
                                should_take = T.if_then_else(am_low, p_f > my_f, my_f > p_f)
                                if should_take:
                                    frag[k] = partner[K_padded - 1 - k]
                            if log_kp > 0:
                                for i1 in T.serial(log_kp):
                                    for i2 in T.serial(i1 + 1):
                                        step = T.int32(1) << (i1 - i2)
                                        dmask = T.int32(1) << (i1 + 1)
                                        for i in T.serial(K_padded):
                                            j = i ^ step
                                            if i < j:
                                                ascending = (i & dmask) != 0
                                                fi = T.reinterpret(T.float32, frag[i])
                                                fj = T.reinterpret(T.float32, frag[j])
                                                do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                                if do_swap:
                                                    tmp = frag[i]
                                                    frag[i] = frag[j]
                                                    frag[j] = tmp

                        if lane_id == 0:
                            local_vals = T.alloc_local([top_k], T.float32)
                            local_idx = T.alloc_local([top_k], T.int32)
                            for ki in T.serial(top_k):
                                packed = frag[ki]
                                cleaned = packed & (~idx_mask)
                                fval = T.reinterpret(T.float32, cleaned)
                                enc = packed & idx_mask
                                col_idx = T.if_then_else(
                                    fval >= T.float32(0.0),
                                    (~enc) & idx_mask,
                                    enc & idx_mask)
                                local_vals[ki] = fval
                                local_idx[ki] = T.Cast(T.int32, col_idx)
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]
                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]
                            for i in T.serial(top_k):
                                scores[token_idx, i] = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# 6. TMA+WS N-Split Variants (Hopper, threads=384)
#
# N-split: keep block_M=64, split E_padded between 2 consumer WGs.
# WG0: GEMM(A, B[:,0:E_half])   → C_shared_0[block_M, E_half]
# WG1: GEMM(A, B[:,E_half:Ep])  → C_shared_1[block_M, E_half]
# Grid = ceil(T/block_M) — same as standard TMA (not halved).
# Epilogue: WG0 rows 0..half_M-1, WG1 rows half_M..block_M-1,
#           each WG reads both C_shared_0 and C_shared_1 for full E coverage.
# Shared mem (DS-V2, E_padded=256): 16+32+32+32+32 = 144KB  (tma2: 224KB).
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_tma_pass_configs)
def fused_router_tma2n_selection_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=64, block_K=64, num_stages=2, threads=384,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (TMA+WS, N-split 1prod+2cons, Hopper) + Selection + Softmax."""
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    WARP_SIZE = 32
    num_warps_per_wg = 128 // WARP_SIZE  # 4
    E_half = E_padded // 2
    half_M = block_M // 2
    rows_per_round = num_warps_per_wg
    num_rounds_half = half_M // rows_per_round  # 8
    elems_per_lane = (E + WARP_SIZE - 1) // WARP_SIZE
    NEG_INF = -T.infinity(T.float32)

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared   = T.alloc_shared([num_stages, block_M, block_K], dtype)
            B_shared_0 = T.alloc_shared([num_stages, block_K, E_half], dtype)
            B_shared_1 = T.alloc_shared([num_stages, block_K, E_half], dtype)
            C_local_0  = T.alloc_fragment([block_M, E_half], accum_dtype)
            C_local_1  = T.alloc_fragment([block_M, E_half], accum_dtype)
            C_shared_0 = T.alloc_shared([block_M, E_half], accum_dtype)
            C_shared_1 = T.alloc_shared([block_M, E_half], accum_dtype)

            # [0..ns-1]: loaded (prod→both, count=128)
            # [ns..2ns-1]: consumed (both cons→prod, count=256)
            # [2ns]: EPI_READY (both cons after C_shared write, count=256)
            T.create_list_of_mbarrier([128] * num_stages + [256] * num_stages + [256])
            EPI_READY = 2 * num_stages

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id_wg = (tx // WARP_SIZE) % num_warps_per_wg
            lane_id = tx % WARP_SIZE

            with T.ws(0):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local_0)
            with T.ws(1):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local_1)
            with T.ws(2):
                T.annotate_producer_reg_dealloc(24)

            for ko in range(T.ceildiv(H, block_K)):
                with T.ws(2):  # producer: load A + both B halves
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages + num_stages,
                        parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(
                        X[m_start:m_start + block_M, ko * block_K:(ko + 1) * block_K],
                        A_shared[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, 0:E_half],
                        B_shared_0[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, E_half:E_padded],
                        B_shared_1[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                with T.ws(0):  # consumer 0: GEMM for left E half
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared_0[ko % num_stages, :, :],
                        C_local_0)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)
                with T.ws(1):  # consumer 1: GEMM for right E half
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared_1[ko % num_stages, :, :],
                        C_local_1)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            with T.ws(0):
                T.copy(C_local_0, C_shared_0)
                T.mbarrier_arrive(mbarrier=EPI_READY)
            with T.ws(1):
                T.copy(C_local_1, C_shared_1)
                T.mbarrier_arrive(mbarrier=EPI_READY)

            # ---- WG0 epilogue: rows 0..half_M-1 ----
            with T.ws(0):
                T.mbarrier_wait_parity(mbarrier=EPI_READY, parity=0)
                for rnd in T.serial(num_rounds_half):
                    row = rnd * rows_per_round + warp_id_wg
                    token_idx = m_start + row
                    if token_idx < num_tokens:
                        my_vals = T.alloc_local([elems_per_lane], T.float32)
                        for i in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + i
                            # Read from correct half; adj_col keeps index non-negative
                            adj_col = T.if_then_else(col >= E_half, col - E_half, T.int32(0))
                            val = T.if_then_else(
                                col < E_half, C_shared_0[row, col], C_shared_1[row, adj_col])
                            my_vals[i] = T.if_then_else(col < E, val, NEG_INF)

                        local_vals = T.alloc_local([top_k], T.float32)
                        local_idx = T.alloc_local([top_k], T.int32)

                        for k_iter in T.serial(top_k):
                            my_max_val = T.alloc_var(T.float32)
                            my_max_idx = T.alloc_var(T.int32)
                            my_max_val = NEG_INF
                            my_max_idx = T.int32(-1)
                            for i in T.serial(elems_per_lane):
                                if my_vals[i] > my_max_val:
                                    my_max_val = my_vals[i]
                                    my_max_idx = T.int32(lane_id * elems_per_lane + i)
                            warp_max = T.warp_reduce_max(my_max_val)
                            cand = T.alloc_var(T.float32)
                            cand = T.float32(-1.0)
                            if my_max_val == warp_max:
                                cand = T.cast(my_max_idx, T.float32)
                            winner_f = T.warp_reduce_max(cand)
                            winner_idx = T.cast(winner_f, T.int32)
                            if lane_id == 0:
                                local_vals[k_iter] = warp_max
                                local_idx[k_iter] = winner_idx
                            for i in T.serial(elems_per_lane):
                                if lane_id * elems_per_lane + i == winner_idx:
                                    my_vals[i] = NEG_INF

                        if lane_id == 0:
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]
                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]
                            for i in T.serial(top_k):
                                scores[token_idx, i] = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

            # ---- WG1 epilogue: rows half_M..block_M-1 ----
            with T.ws(1):
                T.mbarrier_wait_parity(mbarrier=EPI_READY, parity=0)
                for rnd in T.serial(num_rounds_half):
                    row = half_M + rnd * rows_per_round + warp_id_wg
                    token_idx = m_start + row
                    if token_idx < num_tokens:
                        my_vals = T.alloc_local([elems_per_lane], T.float32)
                        for i in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + i
                            adj_col = T.if_then_else(col >= E_half, col - E_half, T.int32(0))
                            val = T.if_then_else(
                                col < E_half, C_shared_0[row, col], C_shared_1[row, adj_col])
                            my_vals[i] = T.if_then_else(col < E, val, NEG_INF)

                        local_vals = T.alloc_local([top_k], T.float32)
                        local_idx = T.alloc_local([top_k], T.int32)

                        for k_iter in T.serial(top_k):
                            my_max_val = T.alloc_var(T.float32)
                            my_max_idx = T.alloc_var(T.int32)
                            my_max_val = NEG_INF
                            my_max_idx = T.int32(-1)
                            for i in T.serial(elems_per_lane):
                                if my_vals[i] > my_max_val:
                                    my_max_val = my_vals[i]
                                    my_max_idx = T.int32(lane_id * elems_per_lane + i)
                            warp_max = T.warp_reduce_max(my_max_val)
                            cand = T.alloc_var(T.float32)
                            cand = T.float32(-1.0)
                            if my_max_val == warp_max:
                                cand = T.cast(my_max_idx, T.float32)
                            winner_f = T.warp_reduce_max(cand)
                            winner_idx = T.cast(winner_f, T.int32)
                            if lane_id == 0:
                                local_vals[k_iter] = warp_max
                                local_idx[k_iter] = winner_idx
                            for i in T.serial(elems_per_lane):
                                if lane_id * elems_per_lane + i == winner_idx:
                                    my_vals[i] = NEG_INF

                        if lane_id == 0:
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]
                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]
                            for i in T.serial(top_k):
                                scores[token_idx, i] = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

    return kernel


@tilelang.jit(out_idx=[1, 2], pass_configs=_tma_pass_configs)
def fused_router_tma2n_bitonic_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=64, block_K=64, num_stages=2, threads=384,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (TMA+WS, N-split 1prod+2cons, Hopper) + Bitonic + Softmax."""
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    WARP_SIZE = 32
    num_warps_per_wg = 128 // WARP_SIZE  # 4
    E_half = E_padded // 2
    half_M = block_M // 2
    rows_per_round = num_warps_per_wg
    num_rounds_half = half_M // rows_per_round  # 8

    elems_per_lane = E_padded // WARP_SIZE if E_padded >= WARP_SIZE else 1
    elems_per_half_lane = E_half // WARP_SIZE if E_half >= WARP_SIZE else 1
    K_padded = _next_pow2(top_k)
    frag_size = max(elems_per_lane, K_padded)
    log_kp = int(math.log2(K_padded)) if K_padded > 1 else 0
    idx_bits = int(math.ceil(math.log2(max(E_padded, 2))))
    idx_mask_val = (1 << idx_bits) - 1
    num_merge_stages = int(math.log2(WARP_SIZE))

    NEG_INF = -T.infinity(T.float32)
    NEG_INF_BITS_CLEAN_val = 0xFF800000 & ~idx_mask_val

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared   = T.alloc_shared([num_stages, block_M, block_K], dtype)
            B_shared_0 = T.alloc_shared([num_stages, block_K, E_half], dtype)
            B_shared_1 = T.alloc_shared([num_stages, block_K, E_half], dtype)
            C_local_0  = T.alloc_fragment([block_M, E_half], accum_dtype)
            C_local_1  = T.alloc_fragment([block_M, E_half], accum_dtype)
            C_shared_0 = T.alloc_shared([block_M, E_half], accum_dtype)
            C_shared_1 = T.alloc_shared([block_M, E_half], accum_dtype)

            T.create_list_of_mbarrier([128] * num_stages + [256] * num_stages + [256])
            EPI_READY = 2 * num_stages

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id_wg = (tx // WARP_SIZE) % num_warps_per_wg
            lane_id = tx % WARP_SIZE
            idx_mask = T.uint32(idx_mask_val)
            NEG_INF_BITS_CLEAN = T.uint32(NEG_INF_BITS_CLEAN_val)

            with T.ws(0):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local_0)
            with T.ws(1):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local_1)
            with T.ws(2):
                T.annotate_producer_reg_dealloc(24)

            for ko in range(T.ceildiv(H, block_K)):
                with T.ws(2):
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages + num_stages,
                        parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(
                        X[m_start:m_start + block_M, ko * block_K:(ko + 1) * block_K],
                        A_shared[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, 0:E_half],
                        B_shared_0[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, E_half:E_padded],
                        B_shared_1[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared_0[ko % num_stages, :, :],
                        C_local_0)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared_1[ko % num_stages, :, :],
                        C_local_1)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            with T.ws(0):
                T.copy(C_local_0, C_shared_0)
                T.mbarrier_arrive(mbarrier=EPI_READY)
            with T.ws(1):
                T.copy(C_local_1, C_shared_1)
                T.mbarrier_arrive(mbarrier=EPI_READY)

            # Bitonic epilogue helper: load full-row bit-packed frag from two half-buffers.
            # Each lane holds elems_per_lane elements spanning [lane*ept, (lane+1)*ept).
            # For col in [0, E_half): read C_shared_0[row, col].
            # For col in [E_half, E_padded): read C_shared_1[row, col-E_half].
            # adj_col = col - E_half clamped to >=0 so TVM bounds-check passes.

            # ---- WG0 bitonic epilogue: rows 0..half_M-1 ----
            with T.ws(0):
                T.mbarrier_wait_parity(mbarrier=EPI_READY, parity=0)
                for rnd in T.serial(num_rounds_half):
                    row = rnd * rows_per_round + warp_id_wg
                    token_idx = m_start + row
                    if token_idx < num_tokens:
                        frag = T.alloc_local([frag_size], T.uint32)
                        for e in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + e
                            adj_col = T.if_then_else(col >= E_half, col - E_half, T.int32(0))
                            raw = T.if_then_else(
                                col < E_half, C_shared_0[row, col], C_shared_1[row, adj_col])
                            fval = T.if_then_else(col < E, raw, NEG_INF)
                            bits = T.reinterpret(T.uint32, fval)
                            col_u32 = T.Cast(T.uint32, col)
                            encoded_idx = T.if_then_else(
                                fval >= T.float32(0.0),
                                (~col_u32) & idx_mask,
                                col_u32 & idx_mask)
                            frag[e] = (bits & (~idx_mask)) | encoded_idx
                        for e in T.serial(frag_size - elems_per_lane):
                            frag[elems_per_lane + e] = NEG_INF_BITS_CLEAN

                        for ki in T.serial(K_padded):
                            for e in T.serial(frag_size):
                                if e > ki:
                                    fi = T.reinterpret(T.float32, frag[ki])
                                    fe = T.reinterpret(T.float32, frag[e])
                                    if fe > fi:
                                        tmp = frag[ki]
                                        frag[ki] = frag[e]
                                        frag[e] = tmp

                        if log_kp > 0:
                            for i1 in T.serial(log_kp):
                                for i2 in T.serial(i1 + 1):
                                    step = T.int32(1) << (i1 - i2)
                                    dmask = T.int32(1) << (i1 + 1)
                                    for i in T.serial(K_padded):
                                        j = i ^ step
                                        if i < j:
                                            ascending = (i & dmask) != 0
                                            fi = T.reinterpret(T.float32, frag[i])
                                            fj = T.reinterpret(T.float32, frag[j])
                                            do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                            if do_swap:
                                                tmp = frag[i]
                                                frag[i] = frag[j]
                                                frag[j] = tmp

                        partner = T.alloc_local([K_padded], T.uint32)
                        for ms in T.serial(num_merge_stages):
                            stride = T.int32(WARP_SIZE) >> (ms + 1)
                            am_low = (lane_id & stride) == 0
                            for k in T.serial(K_padded):
                                val_i32 = T.reinterpret(T.int32, frag[k])
                                p_i32 = T.shfl_xor(val_i32, stride)
                                partner[k] = T.reinterpret(T.uint32, p_i32)
                            for k in T.serial(K_padded):
                                my_f = T.reinterpret(T.float32, frag[k])
                                p_f = T.reinterpret(T.float32, partner[K_padded - 1 - k])
                                should_take = T.if_then_else(am_low, p_f > my_f, my_f > p_f)
                                if should_take:
                                    frag[k] = partner[K_padded - 1 - k]
                            if log_kp > 0:
                                for i1 in T.serial(log_kp):
                                    for i2 in T.serial(i1 + 1):
                                        step = T.int32(1) << (i1 - i2)
                                        dmask = T.int32(1) << (i1 + 1)
                                        for i in T.serial(K_padded):
                                            j = i ^ step
                                            if i < j:
                                                ascending = (i & dmask) != 0
                                                fi = T.reinterpret(T.float32, frag[i])
                                                fj = T.reinterpret(T.float32, frag[j])
                                                do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                                if do_swap:
                                                    tmp = frag[i]
                                                    frag[i] = frag[j]
                                                    frag[j] = tmp

                        if lane_id == 0:
                            local_vals = T.alloc_local([top_k], T.float32)
                            local_idx = T.alloc_local([top_k], T.int32)
                            for ki in T.serial(top_k):
                                packed = frag[ki]
                                cleaned = packed & (~idx_mask)
                                fval = T.reinterpret(T.float32, cleaned)
                                enc = packed & idx_mask
                                col_idx = T.if_then_else(
                                    fval >= T.float32(0.0),
                                    (~enc) & idx_mask,
                                    enc & idx_mask)
                                local_vals[ki] = fval
                                local_idx[ki] = T.Cast(T.int32, col_idx)
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]
                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]
                            for i in T.serial(top_k):
                                scores[token_idx, i] = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

            # ---- WG1 bitonic epilogue: rows half_M..block_M-1 ----
            with T.ws(1):
                T.mbarrier_wait_parity(mbarrier=EPI_READY, parity=0)
                for rnd in T.serial(num_rounds_half):
                    row = half_M + rnd * rows_per_round + warp_id_wg
                    token_idx = m_start + row
                    if token_idx < num_tokens:
                        frag = T.alloc_local([frag_size], T.uint32)
                        for e in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + e
                            adj_col = T.if_then_else(col >= E_half, col - E_half, T.int32(0))
                            raw = T.if_then_else(
                                col < E_half, C_shared_0[row, col], C_shared_1[row, adj_col])
                            fval = T.if_then_else(col < E, raw, NEG_INF)
                            bits = T.reinterpret(T.uint32, fval)
                            col_u32 = T.Cast(T.uint32, col)
                            encoded_idx = T.if_then_else(
                                fval >= T.float32(0.0),
                                (~col_u32) & idx_mask,
                                col_u32 & idx_mask)
                            frag[e] = (bits & (~idx_mask)) | encoded_idx
                        for e in T.serial(frag_size - elems_per_lane):
                            frag[elems_per_lane + e] = NEG_INF_BITS_CLEAN

                        for ki in T.serial(K_padded):
                            for e in T.serial(frag_size):
                                if e > ki:
                                    fi = T.reinterpret(T.float32, frag[ki])
                                    fe = T.reinterpret(T.float32, frag[e])
                                    if fe > fi:
                                        tmp = frag[ki]
                                        frag[ki] = frag[e]
                                        frag[e] = tmp

                        if log_kp > 0:
                            for i1 in T.serial(log_kp):
                                for i2 in T.serial(i1 + 1):
                                    step = T.int32(1) << (i1 - i2)
                                    dmask = T.int32(1) << (i1 + 1)
                                    for i in T.serial(K_padded):
                                        j = i ^ step
                                        if i < j:
                                            ascending = (i & dmask) != 0
                                            fi = T.reinterpret(T.float32, frag[i])
                                            fj = T.reinterpret(T.float32, frag[j])
                                            do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                            if do_swap:
                                                tmp = frag[i]
                                                frag[i] = frag[j]
                                                frag[j] = tmp

                        partner = T.alloc_local([K_padded], T.uint32)
                        for ms in T.serial(num_merge_stages):
                            stride = T.int32(WARP_SIZE) >> (ms + 1)
                            am_low = (lane_id & stride) == 0
                            for k in T.serial(K_padded):
                                val_i32 = T.reinterpret(T.int32, frag[k])
                                p_i32 = T.shfl_xor(val_i32, stride)
                                partner[k] = T.reinterpret(T.uint32, p_i32)
                            for k in T.serial(K_padded):
                                my_f = T.reinterpret(T.float32, frag[k])
                                p_f = T.reinterpret(T.float32, partner[K_padded - 1 - k])
                                should_take = T.if_then_else(am_low, p_f > my_f, my_f > p_f)
                                if should_take:
                                    frag[k] = partner[K_padded - 1 - k]
                            if log_kp > 0:
                                for i1 in T.serial(log_kp):
                                    for i2 in T.serial(i1 + 1):
                                        step = T.int32(1) << (i1 - i2)
                                        dmask = T.int32(1) << (i1 + 1)
                                        for i in T.serial(K_padded):
                                            j = i ^ step
                                            if i < j:
                                                ascending = (i & dmask) != 0
                                                fi = T.reinterpret(T.float32, frag[i])
                                                fj = T.reinterpret(T.float32, frag[j])
                                                do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                                if do_swap:
                                                    tmp = frag[i]
                                                    frag[i] = frag[j]
                                                    frag[j] = tmp

                        if lane_id == 0:
                            local_vals = T.alloc_local([top_k], T.float32)
                            local_idx = T.alloc_local([top_k], T.int32)
                            for ki in T.serial(top_k):
                                packed = frag[ki]
                                cleaned = packed & (~idx_mask)
                                fval = T.reinterpret(T.float32, cleaned)
                                enc = packed & idx_mask
                                col_idx = T.if_then_else(
                                    fval >= T.float32(0.0),
                                    (~enc) & idx_mask,
                                    enc & idx_mask)
                                local_vals[ki] = fval
                                local_idx[ki] = T.Cast(T.int32, col_idx)
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]
                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]
                            for i in T.serial(top_k):
                                scores[token_idx, i] = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# 9. TMA 3-WG (1 producer + 2 consumer) + Selection Top-K  (Hopper+)
#
# 1:2 warp specialization with M-split cooperative consumers, expressed as
# plain `if tx < 256` / `if tx >= 256` branches:
#   tx <  256 → consumer 0+1: T.gemm with FullRow policy → 2 × wgmma<64,
#               E_padded, 16> M-split across the 2 consumer WGs.
#   tx >= 256 → producer (1 WG): TMA loads of A[block_M=128] + W per K-step.
#
# Note: W TMA is split into 4 loads per K-step when E_padded > 64 (128B-swizzle
# × bf16 caps inner stripe at 64 elements). Tried W-transpose to fold these
# into 1 TMA, but the resulting strided global access pattern (stride = H*2
# bytes vs E_padded*2 bytes original) hurt TMA bandwidth more than the issue-
# count reduction helped — net regression on H100. So we keep the natural
# W[H, E_padded] layout. Optimization B (drop redundant sync) is preserved.
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_tma_pass_configs)
def fused_router_tma3_selection_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=128, block_K=64, num_stages=2, threads=384,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (TMA, 1 producer + 2 M-split cooperative consumer WGs)
    + Selection-Sort TopK + Softmax.

    Single A_shared/B_shared/C_local; FullRow policy → wgmma<64, E_padded, 16>
    per consumer WG (M-split). 8 consumer warps × 16 rounds covers 128 rows.
    """
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    WARP_SIZE = 32
    consumer_threads = 256
    num_warps = consumer_threads // WARP_SIZE  # 8
    rows_per_round = num_warps                 # 8
    num_rounds = block_M // rows_per_round     # 16
    elems_per_lane = (E + WARP_SIZE - 1) // WARP_SIZE
    NEG_INF = -T.infinity(T.float32)

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([num_stages, block_M, block_K], dtype)
            B_shared = T.alloc_shared([num_stages, block_K, E_padded], dtype)
            C_local  = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared = T.alloc_shared([block_M, E_padded], accum_dtype)

            # Barriers:
            # [0..ns-1]    load_done    : producer(128t) → consumers
            # [ns..2ns-1]  consume_done : consumers(256t) → producer
            # [2ns]        epi_ready    : consumers(256t) signal C_shared written
            T.create_list_of_mbarrier(
                [128] * num_stages + [256] * num_stages + [256])
            EPI_READY = 2 * num_stages

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE   # 0..7 across both consumer WGs
            lane_id = tx % WARP_SIZE

            # Register reallocation
            if tx < 256:
                T.set_max_nreg(240, 1)
                T.clear(C_local)
            elif tx >= 256:
                T.set_max_nreg(24, 0)

            for ko in range(T.ceildiv(H, block_K)):
                if tx >= 256:           # producer: TMA async load
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages + num_stages,
                        parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(
                        X[m_start:m_start + block_M,
                          ko * block_K:(ko + 1) * block_K],
                        A_shared[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, 0:E_padded],
                        B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                if tx < 256:            # consumers (cooperative M-split WGMMA)
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            if tx < 256:
                T.copy(C_local, C_shared)
                T.mbarrier_arrive(mbarrier=EPI_READY)

            T.mbarrier_wait_parity(mbarrier=EPI_READY, parity=0)
            # Optimization B: dropped redundant T.sync_threads() — mbarrier with
            # arrive_count=256 already releases all 384 waiters in lockstep once
            # all consumer arrives are observed.

            # ---- Epilogue: 8-warp-parallel selection (consumer threads only) ----
            if tx < 256:
                for rnd in T.serial(num_rounds):
                    row = rnd * rows_per_round + warp_id
                    token_idx = m_start + row
                    if token_idx < num_tokens:
                        my_vals = T.alloc_local([elems_per_lane], T.float32)
                        for i in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + i
                            my_vals[i] = T.if_then_else(col < E, C_shared[row, col], NEG_INF)

                        local_vals = T.alloc_local([top_k], T.float32)
                        local_idx  = T.alloc_local([top_k], T.int32)

                        for k_iter in T.serial(top_k):
                            my_max_val = T.alloc_var(T.float32)
                            my_max_idx = T.alloc_var(T.int32)
                            my_max_val = NEG_INF
                            my_max_idx = T.int32(-1)

                            for i in T.serial(elems_per_lane):
                                if my_vals[i] > my_max_val:
                                    my_max_val = my_vals[i]
                                    my_max_idx = T.int32(lane_id * elems_per_lane + i)

                            warp_max = T.warp_reduce_max(my_max_val)
                            cand = T.alloc_var(T.float32)
                            cand = T.float32(-1.0)
                            if my_max_val == warp_max:
                                cand = T.cast(my_max_idx, T.float32)
                            winner_f   = T.warp_reduce_max(cand)
                            winner_idx = T.cast(winner_f, T.int32)

                            if lane_id == 0:
                                local_vals[k_iter] = warp_max
                                local_idx[k_iter]  = winner_idx

                            for i in T.serial(elems_per_lane):
                                if lane_id * elems_per_lane + i == winner_idx:
                                    my_vals[i] = NEG_INF

                        if lane_id == 0:
                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]

                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]

                            for i in T.serial(top_k):
                                scores[token_idx, i]  = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# 10. TMA 3-WG (1 producer + 2 consumer) + Bitonic Top-K  (Hopper+)
#
# Same M-split cooperative GEMM design as tma3_selection above; epilogue
# replaced with bit-packed bitonic top-K + softmax.
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_tma_pass_configs)
def fused_router_tma3_bitonic_kernel(
    num_tokens, H, E, E_padded, top_k,
    block_M=128, block_K=64, num_stages=2, threads=384,
    dtype=T.bfloat16,
):
    """Fused Router GEMM (TMA, 1 producer + 2 M-split cooperative consumer WGs)
    + Bitonic-Sort TopK + Softmax."""
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    WARP_SIZE = 32
    consumer_threads = 256
    num_warps = consumer_threads // WARP_SIZE  # 8
    rows_per_round = num_warps                 # 8
    num_rounds = block_M // rows_per_round     # 16

    elems_per_lane = E_padded // WARP_SIZE if E_padded >= WARP_SIZE else 1
    if elems_per_lane < 1:
        elems_per_lane = 1
    K_padded = _next_pow2(top_k)
    frag_size = max(elems_per_lane, K_padded)
    log_kp = int(math.log2(K_padded)) if K_padded > 1 else 0
    idx_bits = int(math.ceil(math.log2(max(E_padded, 2))))
    idx_mask_val = (1 << idx_bits) - 1
    num_merge_stages = int(math.log2(WARP_SIZE))
    NEG_INF = -T.infinity(T.float32)
    NEG_INF_BITS_CLEAN_val = 0xFF800000 & ~idx_mask_val

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E_padded], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([num_stages, block_M, block_K], dtype)
            B_shared = T.alloc_shared([num_stages, block_K, E_padded], dtype)
            C_local  = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared = T.alloc_shared([block_M, E_padded], accum_dtype)

            T.create_list_of_mbarrier(
                [128] * num_stages + [256] * num_stages + [256])
            EPI_READY = 2 * num_stages

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE
            lane_id = tx % WARP_SIZE
            idx_mask = T.uint32(idx_mask_val)
            NEG_INF_BITS_CLEAN = T.uint32(NEG_INF_BITS_CLEAN_val)

            if tx < 256:
                T.set_max_nreg(240, 1)
                T.clear(C_local)
            elif tx >= 256:
                T.set_max_nreg(24, 0)

            for ko in range(T.ceildiv(H, block_K)):
                if tx >= 256:
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages + num_stages,
                        parity=((ko // num_stages) % num_stages) ^ 1)
                    T.copy(
                        X[m_start:m_start + block_M,
                          ko * block_K:(ko + 1) * block_K],
                        A_shared[ko % num_stages, :, :])
                    T.copy(
                        W[ko * block_K:(ko + 1) * block_K, 0:E_padded],
                        B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbarrier=ko % num_stages)
                if tx < 256:
                    T.mbarrier_wait_parity(
                        mbarrier=ko % num_stages,
                        parity=(ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.mbarrier_arrive(mbarrier=ko % num_stages + num_stages)

            if tx < 256:
                T.copy(C_local, C_shared)
                T.mbarrier_arrive(mbarrier=EPI_READY)

            T.mbarrier_wait_parity(mbarrier=EPI_READY, parity=0)
            # Optimization B: dropped redundant T.sync_threads() — mbarrier
            # already provides full 384-thread sync.

            # ---- Epilogue: 8-warp-parallel bitonic (consumer threads only) ----
            if tx < 256:
                for rnd in T.serial(num_rounds):
                    row = rnd * rows_per_round + warp_id
                    token_idx = m_start + row
                    if token_idx < num_tokens:
                        frag = T.alloc_local([frag_size], T.uint32)
                        for e in T.serial(elems_per_lane):
                            col = lane_id * elems_per_lane + e
                            fval = T.if_then_else(col < E, C_shared[row, col], NEG_INF)
                            bits = T.reinterpret(T.uint32, fval)
                            col_u32 = T.Cast(T.uint32, col)
                            encoded_idx = T.if_then_else(
                                fval >= T.float32(0.0),
                                (~col_u32) & idx_mask,
                                col_u32 & idx_mask)
                            frag[e] = (bits & (~idx_mask)) | encoded_idx
                        for e in T.serial(frag_size - elems_per_lane):
                            frag[elems_per_lane + e] = NEG_INF_BITS_CLEAN

                        for ki in T.serial(K_padded):
                            for e in T.serial(frag_size):
                                if e > ki:
                                    fi = T.reinterpret(T.float32, frag[ki])
                                    fe = T.reinterpret(T.float32, frag[e])
                                    if fe > fi:
                                        tmp = frag[ki]
                                        frag[ki] = frag[e]
                                        frag[e] = tmp

                        if log_kp > 0:
                            for i1 in T.serial(log_kp):
                                for i2 in T.serial(i1 + 1):
                                    step = T.int32(1) << (i1 - i2)
                                    dmask = T.int32(1) << (i1 + 1)
                                    for i in T.serial(K_padded):
                                        j = i ^ step
                                        if i < j:
                                            ascending = (i & dmask) != 0
                                            fi = T.reinterpret(T.float32, frag[i])
                                            fj = T.reinterpret(T.float32, frag[j])
                                            do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                            if do_swap:
                                                tmp = frag[i]
                                                frag[i] = frag[j]
                                                frag[j] = tmp

                        partner = T.alloc_local([K_padded], T.uint32)
                        for ms in T.serial(num_merge_stages):
                            stride = T.int32(WARP_SIZE) >> (ms + 1)
                            am_low = (lane_id & stride) == 0

                            for k in T.serial(K_padded):
                                val_i32 = T.reinterpret(T.int32, frag[k])
                                p_i32 = T.shfl_xor(val_i32, stride)
                                partner[k] = T.reinterpret(T.uint32, p_i32)

                            for k in T.serial(K_padded):
                                my_f = T.reinterpret(T.float32, frag[k])
                                p_f = T.reinterpret(T.float32, partner[K_padded - 1 - k])
                                should_take = T.if_then_else(am_low, p_f > my_f, my_f > p_f)
                                if should_take:
                                    frag[k] = partner[K_padded - 1 - k]

                            if log_kp > 0:
                                for i1 in T.serial(log_kp):
                                    for i2 in T.serial(i1 + 1):
                                        step = T.int32(1) << (i1 - i2)
                                        dmask = T.int32(1) << (i1 + 1)
                                        for i in T.serial(K_padded):
                                            j = i ^ step
                                            if i < j:
                                                ascending = (i & dmask) != 0
                                                fi = T.reinterpret(T.float32, frag[i])
                                                fj = T.reinterpret(T.float32, frag[j])
                                                do_swap = T.if_then_else(ascending, fi > fj, fi < fj)
                                                if do_swap:
                                                    tmp = frag[i]
                                                    frag[i] = frag[j]
                                                    frag[j] = tmp

                        if lane_id == 0:
                            local_vals = T.alloc_local([top_k], T.float32)
                            local_idx  = T.alloc_local([top_k], T.int32)
                            for ki in T.serial(top_k):
                                packed = frag[ki]
                                cleaned = packed & (~idx_mask)
                                fval = T.reinterpret(T.float32, cleaned)
                                enc = packed & idx_mask
                                col_idx = T.if_then_else(
                                    fval >= T.float32(0.0),
                                    (~enc) & idx_mask,
                                    enc & idx_mask)
                                local_vals[ki] = fval
                                local_idx[ki]  = T.Cast(T.int32, col_idx)

                            max_val = T.alloc_var(T.float32)
                            max_val = NEG_INF
                            for i in T.serial(top_k):
                                if local_vals[i] > max_val:
                                    max_val = local_vals[i]

                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for i in T.serial(top_k):
                                local_vals[i] = T.exp(local_vals[i] - max_val)
                                exp_sum = exp_sum + local_vals[i]

                            for i in T.serial(top_k):
                                scores[token_idx, i]  = local_vals[i] / exp_sum
                                indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# Kernel Cache + Wrapper (handles W padding for non-power-of-2 E)
# ============================================================================

_fused_cache: Dict[str, any] = {}

_ALGO_FACTORIES = {
    "cpasync_selection":  fused_router_cpasync_selection_kernel,
    "cpasync_bitonic":    fused_router_cpasync_bitonic_kernel,
    "tma_selection":      fused_router_tma_selection_kernel,
    "tma_bitonic":        fused_router_tma_bitonic_kernel,
    "tma2_selection":     fused_router_tma2_selection_kernel,
    "tma2_bitonic":       fused_router_tma2_bitonic_kernel,
    "tma2n_selection":    fused_router_tma2n_selection_kernel,
    "tma2n_bitonic":      fused_router_tma2n_bitonic_kernel,
    "tma3_selection":     fused_router_tma3_selection_kernel,
    "tma3_bitonic":       fused_router_tma3_bitonic_kernel,
}


def _pad_W(W, E, E_padded):
    if E == E_padded:
        return W
    H = W.shape[0]
    W_p = torch.zeros(H, E_padded, device=W.device, dtype=W.dtype)
    W_p[:, :E] = W
    return W_p


def get_fused_router_kernel(algo, num_tokens, H, E, top_k, dtype=T.bfloat16,
                             block_M=64, block_K=64, num_stages=2, threads=256):
    """Get or compile a fused router kernel.

    Args:
        algo: one of "cpasync_selection", "cpasync_bitonic",
              "tma_selection", "tma_bitonic"
    Returns:
        Callable(X, W) where W is [H, E] (original, unpadded).  Padding to
        E_padded is handled internally.
    """
    if algo not in _ALGO_FACTORIES:
        raise ValueError(f"Unknown algorithm: {algo}. "
                         f"Valid: {list(_ALGO_FACTORIES.keys())}")

    E_padded = _next_pow2(E)
    cache_key = f"{algo}_{num_tokens}_{H}_{E}_{top_k}"
    if cache_key in _fused_cache:
        return _fused_cache[cache_key]

    factory = _ALGO_FACTORIES[algo]
    kernel = factory(
        num_tokens, H, E, E_padded, top_k,
        block_M=block_M, block_K=block_K,
        num_stages=num_stages, threads=threads, dtype=dtype)

    if E == E_padded:
        _fused_cache[cache_key] = kernel
    else:
        def padded_wrapper(X, W, _k=kernel, _E=E, _Ep=E_padded):
            return _k(X, _pad_W(W, _E, _Ep))
        _fused_cache[cache_key] = padded_wrapper

    return _fused_cache[cache_key]
