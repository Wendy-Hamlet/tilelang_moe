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
_cpasync_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
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
                        my_max_val = T.alloc_var(T.float32, init=NEG_INF)
                        my_max_idx = T.alloc_var(T.int32, init=-1)

                        for i in T.serial(elems_per_lane):
                            if my_vals[i] > my_max_val:
                                my_max_val = my_vals[i]
                                my_max_idx = T.int32(lane_id * elems_per_lane + i)

                        warp_max = T.warp_reduce_max(my_max_val)
                        cand = T.alloc_var(T.float32, init=T.float32(-1.0))
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
                        max_val = T.alloc_var(T.float32, init=NEG_INF)
                        for i in T.serial(top_k):
                            if local_vals[i] > max_val:
                                max_val = local_vals[i]

                        exp_sum = T.alloc_var(T.float32, init=T.float32(0.0))
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
    log_ept = int(math.log2(elems_per_lane)) if elems_per_lane > 1 else 0
    log_kp = int(math.log2(K_padded)) if K_padded > 1 else 0
    idx_bits = int(math.ceil(math.log2(max(E_padded, 2))))
    idx_mask_val = (1 << idx_bits) - 1
    num_merge_stages = int(math.log2(WARP_SIZE))  # 5

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
                    # Bit-pack E elements into register fragment
                    frag = T.alloc_local([elems_per_lane], T.uint32)
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

                    # Local partial top-K_padded via selection (elems_per_lane iterations)
                    for ki in T.serial(K_padded):
                        for e in T.serial(elems_per_lane):
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

                        max_val = T.alloc_var(T.float32, init=NEG_INF)
                        for i in T.serial(top_k):
                            if local_vals[i] > max_val:
                                max_val = local_vals[i]

                        exp_sum = T.alloc_var(T.float32, init=T.float32(0.0))
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

            # mbars[0..num_stages-1]           : loaded   (producer arrives, consumer waits)
            # mbars[num_stages..2*num_stages-1]: consumed (consumer arrives, producer waits)
            mbars = T.alloc_barrier([128, 128] * num_stages)
            epi_ready = T.alloc_barrier([128])  # consumer→everyone: C_shared is written

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE
            lane_id = tx % WARP_SIZE

            # ---- GEMM: warp-specialized with TMA producer ----
            with T.ws(0):
                T.clear(C_local)

            for ko in range(T.ceildiv(H, block_K)):
                with T.ws(1):  # producer: TMA copy
                    T.mbarrier_wait_parity(
                        mbars[ko % num_stages + num_stages],
                        ((ko // num_stages) % num_stages) ^ 1)
                    T.copy(X[m_start, ko * block_K], A_shared[ko % num_stages, :, :])
                    T.copy(W[ko * block_K, 0], B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbars[ko % num_stages])
                with T.ws(0):  # consumer: WGMMA compute
                    T.mbarrier_wait_parity(
                        mbars[ko % num_stages],
                        (ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local)
                    T.mbarrier_arrive(mbars[ko % num_stages + num_stages])

            # Consumer stages output and signals readiness
            with T.ws(0):
                T.copy(C_local, C_shared)
                T.mbarrier_arrive(epi_ready)

            # Both warp groups wait for C_shared
            T.mbarrier_wait_parity(epi_ready, 0)
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
                        my_max_val = T.alloc_var(T.float32, init=NEG_INF)
                        my_max_idx = T.alloc_var(T.int32, init=-1)

                        for i in T.serial(elems_per_lane):
                            if my_vals[i] > my_max_val:
                                my_max_val = my_vals[i]
                                my_max_idx = T.int32(lane_id * elems_per_lane + i)

                        warp_max = T.warp_reduce_max(my_max_val)
                        cand = T.alloc_var(T.float32, init=T.float32(-1.0))
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
                        max_val = T.alloc_var(T.float32, init=NEG_INF)
                        for i in T.serial(top_k):
                            if local_vals[i] > max_val:
                                max_val = local_vals[i]

                        exp_sum = T.alloc_var(T.float32, init=T.float32(0.0))
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

            mbars = T.alloc_barrier([128, 128] * num_stages)
            epi_ready = T.alloc_barrier([128])

            m_start = bx * block_M
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE
            lane_id = tx % WARP_SIZE
            idx_mask = T.uint32(idx_mask_val)

            with T.ws(0):
                T.clear(C_local)

            for ko in range(T.ceildiv(H, block_K)):
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbars[ko % num_stages + num_stages],
                        ((ko // num_stages) % num_stages) ^ 1)
                    T.copy(X[m_start, ko * block_K], A_shared[ko % num_stages, :, :])
                    T.copy(W[ko * block_K, 0], B_shared[ko % num_stages, :, :])
                    T.mbarrier_arrive(mbars[ko % num_stages])
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbars[ko % num_stages],
                        (ko // num_stages) % num_stages)
                    T.gemm(
                        A_shared[ko % num_stages, :, :],
                        B_shared[ko % num_stages, :, :],
                        C_local)
                    T.mbarrier_arrive(mbars[ko % num_stages + num_stages])

            with T.ws(0):
                T.copy(C_local, C_shared)
                T.mbarrier_arrive(epi_ready)

            T.mbarrier_wait_parity(epi_ready, 0)
            T.sync_threads()

            # ---- Epilogue: 8-warp-parallel bitonic ----
            for rnd in T.serial(num_rounds):
                row = rnd * rows_per_round + warp_id
                token_idx = m_start + row
                if token_idx < num_tokens:
                    frag = T.alloc_local([elems_per_lane], T.uint32)
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

                    for ki in T.serial(K_padded):
                        for e in T.serial(elems_per_lane):
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

                        max_val = T.alloc_var(T.float32, init=NEG_INF)
                        for i in T.serial(top_k):
                            if local_vals[i] > max_val:
                                max_val = local_vals[i]

                        exp_sum = T.alloc_var(T.float32, init=T.float32(0.0))
                        for i in T.serial(top_k):
                            local_vals[i] = T.exp(local_vals[i] - max_val)
                            exp_sum = exp_sum + local_vals[i]

                        for i in T.serial(top_k):
                            scores[token_idx, i] = local_vals[i] / exp_sum
                            indices[token_idx, i] = local_idx[i]

    return kernel


# ============================================================================
# Kernel Cache + Wrapper (handles W padding for non-power-of-2 E)
# ============================================================================

_fused_cache: Dict[str, any] = {}

_ALGO_FACTORIES = {
    "cpasync_selection": fused_router_cpasync_selection_kernel,
    "cpasync_bitonic":   fused_router_cpasync_bitonic_kernel,
    "tma_selection":     fused_router_tma_selection_kernel,
    "tma_bitonic":       fused_router_tma_bitonic_kernel,
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
