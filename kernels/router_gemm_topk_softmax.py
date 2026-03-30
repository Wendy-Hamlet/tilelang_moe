"""
Fused Router GEMM + TopK + Softmax TileLang Kernels

Fuses the MoE router linear projection (GEMM) with TopK+Softmax into a single
kernel, eliminating the intermediate logits (T,E) global memory round-trip.

Pipeline:
  Separate: X(T,H) @ W(H,E) → [global] logits(T,E) → [global] (scores,indices)(T,K)
  Fused:    X(T,H) @ W(H,E) → [shared] logits(BM,E) → [global] (scores,indices)(T,K)

Three TopK epilogue variants:
  1. Selection sort: O(K*E) parallel max-reduction, simplest
  2. Bitonic v1: O(E*log²E) parallel bitonic sort, full sort then extract
  3. Bitonic v3: register-based partial sort + warp shuffle merge
"""

import math
from typing import Dict, Tuple

import tilelang
import tilelang.language as T


# TileLang pass configs to disable features that conflict with manual thread control
_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

_bitonic_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


# ============================================================================
# Selection Sort Fused Kernel
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_pass_configs)
def fused_router_selection_kernel(
    num_tokens, H, E, top_k,
    block_M=64, block_K=64, num_stages=2, threads=256,
    dtype=T.bfloat16,
):
    """
    Fused Router GEMM + Selection-Sort TopK + Softmax.

    GEMM: X(num_tokens, H) @ W(H, E) → logits(BM, E) in shared memory
    Epilogue: selection sort topk + softmax on shared memory logits

    Grid: 1D over M-blocks. Each CTA computes BM full logits rows,
    then does topk+softmax epilogue before writing (scores, indices).
    """
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    elems_per_thread = (E + threads - 1) // threads

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            # --- Shared memory allocations ---
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, E], dtype)
            C_local = T.alloc_fragment([block_M, E], accum_dtype)
            C_shared = T.alloc_shared([block_M, E], accum_dtype)

            # TopK workspace (reuses shared memory after GEMM pipeline is done)
            s_topk_vals = T.alloc_shared([top_k], T.float32)
            s_topk_idx = T.alloc_shared([top_k], T.int32)
            s_thread_vals = T.alloc_shared([threads], T.float32)
            s_thread_idx = T.alloc_shared([threads], T.int32)

            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })

            m_start = bx * block_M
            tx = T.get_thread_binding()

            # ======== GEMM Phase ========
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=num_stages):
                T.copy(X[m_start, k * block_K], A_shared)
                T.copy(W[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # Copy logits from fragment to shared memory
            T.copy(C_local, C_shared)

            # ======== TopK + Softmax Epilogue ========
            # Process each row (token) sequentially; all threads cooperate per row
            for row in T.serial(block_M):
                token_idx = m_start + row
                if token_idx < num_tokens:
                    # Initialize topk buffers
                    if tx == 0:
                        for i in T.serial(top_k):
                            s_topk_vals[i] = T.float32(-1e38)
                            s_topk_idx[i] = T.int32(-1)
                    T.sync_threads()

                    # K iterations of selection (find max, exclude, repeat)
                    for k_iter in T.serial(top_k):
                        thread_max_val = T.alloc_var(T.float32)
                        thread_max_idx = T.alloc_var(T.int32)
                        thread_max_val = T.float32(-1e38)
                        thread_max_idx = T.int32(-1)

                        for i in T.serial(elems_per_thread):
                            idx = tx * elems_per_thread + i
                            if idx < E:
                                val = C_shared[row, idx]
                                already_selected = T.alloc_var(T.bool)
                                already_selected = T.bool(False)
                                for j in T.serial(k_iter):
                                    if s_topk_idx[j] == idx:
                                        already_selected = T.bool(True)
                                if (not already_selected) and (val > thread_max_val):
                                    thread_max_val = val
                                    thread_max_idx = T.int32(idx)

                        s_thread_vals[tx] = thread_max_val
                        s_thread_idx[tx] = thread_max_idx
                        T.sync_threads()

                        # Tree reduction to find global max
                        if tx < 128:
                            if s_thread_vals[tx + 128] > s_thread_vals[tx]:
                                s_thread_vals[tx] = s_thread_vals[tx + 128]
                                s_thread_idx[tx] = s_thread_idx[tx + 128]
                        T.sync_threads()
                        if tx < 64:
                            if s_thread_vals[tx + 64] > s_thread_vals[tx]:
                                s_thread_vals[tx] = s_thread_vals[tx + 64]
                                s_thread_idx[tx] = s_thread_idx[tx + 64]
                        T.sync_threads()
                        if tx < 32:
                            if s_thread_vals[tx + 32] > s_thread_vals[tx]:
                                s_thread_vals[tx] = s_thread_vals[tx + 32]
                                s_thread_idx[tx] = s_thread_idx[tx + 32]
                        T.sync_threads()
                        if tx < 16:
                            if s_thread_vals[tx + 16] > s_thread_vals[tx]:
                                s_thread_vals[tx] = s_thread_vals[tx + 16]
                                s_thread_idx[tx] = s_thread_idx[tx + 16]
                        T.sync_threads()
                        if tx < 8:
                            if s_thread_vals[tx + 8] > s_thread_vals[tx]:
                                s_thread_vals[tx] = s_thread_vals[tx + 8]
                                s_thread_idx[tx] = s_thread_idx[tx + 8]
                        T.sync_threads()
                        if tx < 4:
                            if s_thread_vals[tx + 4] > s_thread_vals[tx]:
                                s_thread_vals[tx] = s_thread_vals[tx + 4]
                                s_thread_idx[tx] = s_thread_idx[tx + 4]
                        T.sync_threads()
                        if tx < 2:
                            if s_thread_vals[tx + 2] > s_thread_vals[tx]:
                                s_thread_vals[tx] = s_thread_vals[tx + 2]
                                s_thread_idx[tx] = s_thread_idx[tx + 2]
                        T.sync_threads()
                        if tx < 1:
                            if s_thread_vals[tx + 1] > s_thread_vals[tx]:
                                s_thread_vals[tx] = s_thread_vals[tx + 1]
                                s_thread_idx[tx] = s_thread_idx[tx + 1]
                        T.sync_threads()

                        if tx == 0:
                            s_topk_vals[k_iter] = s_thread_vals[0]
                            s_topk_idx[k_iter] = s_thread_idx[0]
                        T.sync_threads()

                    # Softmax over top-K values
                    if tx == 0:
                        max_val = T.alloc_var(T.float32)
                        max_val = T.float32(-1e38)
                        for i in T.serial(top_k):
                            if s_topk_vals[i] > max_val:
                                max_val = s_topk_vals[i]

                        exp_sum = T.alloc_var(T.float32)
                        exp_sum = T.float32(0.0)
                        for i in T.serial(top_k):
                            s_topk_vals[i] = T.exp(s_topk_vals[i] - max_val)
                            exp_sum = exp_sum + s_topk_vals[i]

                        for i in T.serial(top_k):
                            scores[token_idx, i] = s_topk_vals[i] / exp_sum
                            indices[token_idx, i] = s_topk_idx[i]
                    T.sync_threads()

    return kernel


# ============================================================================
# Bitonic Sort v1 Fused Kernel
# ============================================================================

@tilelang.jit(out_idx=[1, 2], pass_configs=_pass_configs)
def fused_router_bitonic_kernel(
    num_tokens, H, E, top_k,
    block_M=64, block_K=64, num_stages=2, threads=256,
    dtype=T.bfloat16,
):
    """
    Fused Router GEMM + Bitonic-Sort TopK + Softmax.

    After GEMM, uses parallel bitonic sort on shared memory logits.
    Requires threads >= E_padded for the sort phase.
    """
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    N_padded = 1 << math.ceil(math.log2(max(E, 2)))
    num_iters = int(math.log2(N_padded))

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, E], dtype)
            C_local = T.alloc_fragment([block_M, E], accum_dtype)
            C_shared = T.alloc_shared([block_M, E], accum_dtype)

            # Bitonic sort workspace
            s_vals = T.alloc_shared([N_padded], T.float32)
            s_idx = T.alloc_shared([N_padded], T.int32)
            s_softmax = T.alloc_shared([top_k], T.float32)
            s_exp_sum = T.alloc_shared([1], T.float32)

            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })

            m_start = bx * block_M
            tx = T.get_thread_binding()

            # ======== GEMM Phase ========
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=num_stages):
                T.copy(X[m_start, k * block_K], A_shared)
                T.copy(W[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)

            # ======== Bitonic Sort TopK + Softmax Epilogue ========
            for row in T.serial(block_M):
                token_idx = m_start + row
                if token_idx < num_tokens:
                    # Load logits into sort buffer
                    if tx < N_padded:
                        if tx < E:
                            s_vals[tx] = C_shared[row, tx]
                            s_idx[tx] = T.int32(tx)
                        else:
                            s_vals[tx] = T.float32(-1e38)
                            s_idx[tx] = T.int32(-1)
                    T.sync_threads()

                    # Bitonic sort (descending)
                    for i1 in T.serial(num_iters):
                        for i2 in T.serial(i1 + 1):
                            if tx < N_padded:
                                block_size_sort = 1 << (i1 - i2 + 1)
                                half_block = block_size_sort >> 1
                                group = tx // block_size_sort
                                pos_in_block = tx % block_size_sort

                                if pos_in_block < half_block:
                                    j = group * block_size_sort + (block_size_sort - 1 - pos_in_block)
                                    # Descending: swap if s_vals[tx] < s_vals[j]
                                    ascending = (tx >> (i1 + 1)) & 1
                                    if ascending == 1:
                                        if s_vals[tx] > s_vals[j]:
                                            tmp_v = s_vals[tx]
                                            s_vals[tx] = s_vals[j]
                                            s_vals[j] = tmp_v
                                            tmp_i = s_idx[tx]
                                            s_idx[tx] = s_idx[j]
                                            s_idx[j] = tmp_i
                                    else:
                                        if s_vals[tx] < s_vals[j]:
                                            tmp_v = s_vals[tx]
                                            s_vals[tx] = s_vals[j]
                                            s_vals[j] = tmp_v
                                            tmp_i = s_idx[tx]
                                            s_idx[tx] = s_idx[j]
                                            s_idx[j] = tmp_i
                            T.sync_threads()

                    # Softmax over top-K (sorted descending, top-K are s_vals[0..K-1])
                    if tx < top_k:
                        s_softmax[tx] = T.exp(s_vals[tx] - s_vals[0])
                    T.sync_threads()

                    if tx == 0:
                        s_exp_sum[0] = T.float32(0.0)
                        for i in T.serial(top_k):
                            s_exp_sum[0] = s_exp_sum[0] + s_softmax[i]
                    T.sync_threads()

                    if tx < top_k:
                        scores[token_idx, tx] = s_softmax[tx] / s_exp_sum[0]
                        indices[token_idx, tx] = s_idx[tx]
                    T.sync_threads()

    return kernel


# ============================================================================
# Bitonic v3 Fused Kernel (Register-based partial sort + warp shuffle)
# ============================================================================

def _bitonic_v3_thread_config(E, top_k):
    """Compute thread layout for bitonic v3 epilogue."""
    N_padded = 1 << math.ceil(math.log2(max(E, 2)))
    K_padded = 1 << math.ceil(math.log2(max(top_k, 2)))
    threads_per_row = max(min(N_padded // top_k, 32, N_padded // 64), 1)
    elems_per_thread = N_padded // threads_per_row
    return N_padded, K_padded, threads_per_row, elems_per_thread


@tilelang.jit(out_idx=[1, 2], pass_configs=_pass_configs)
def fused_router_bitonic_v3_kernel(
    num_tokens, H, E, top_k,
    block_M=64, block_K=64, num_stages=2, threads=256,
    dtype=T.bfloat16,
):
    """
    Fused Router GEMM + Bitonic-v3 TopK + Softmax.

    After GEMM, uses register-based partial sort with warp shuffle merge.
    Each thread loads elems_per_thread elements from C_shared, does local
    partial sort, then merges across threads via shfl_xor.
    """
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32
    N_padded, K_padded, threads_per_row, elems_per_thread = _bitonic_v3_thread_config(E, top_k)
    idx_bits = int(math.ceil(math.log2(max(N_padded, 2))))
    idx_mask = (1 << idx_bits) - 1
    num_merge_stages = int(math.log2(threads_per_row)) if threads_per_row > 1 else 0

    @T.prim_func
    def kernel(
        X: T.Tensor([num_tokens, H], dtype),
        scores: T.Tensor([num_tokens, top_k], T.float32),
        indices: T.Tensor([num_tokens, top_k], T.int32),
        W: T.Tensor([H, E], dtype),
    ):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, E], dtype)
            C_local = T.alloc_fragment([block_M, E], accum_dtype)
            C_shared = T.alloc_shared([block_M, E], accum_dtype)

            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })

            m_start = bx * block_M
            tx = T.get_thread_binding()

            # ======== GEMM Phase ========
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=num_stages):
                T.copy(X[m_start, k * block_K], A_shared)
                T.copy(W[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)

            # ======== Bitonic v3 TopK + Softmax Epilogue ========
            for row in T.serial(block_M):
                token_idx = m_start + row
                if token_idx < num_tokens:
                    # Determine which lane this thread is for the current row
                    lane_in_row = tx % threads_per_row
                    row_assignment = tx // threads_per_row

                    # Only threads assigned to row 0 (within the row group) do work
                    # For simplicity with block_M rows processed sequentially,
                    # all threads with lane_in_row < threads_per_row participate
                    if row_assignment == 0:
                        # Load + bit-pack into local array
                        frag = T.alloc_local([elems_per_thread], T.uint32)
                        for e in T.serial(elems_per_thread):
                            col = lane_in_row * elems_per_thread + e
                            if col < E:
                                fval = C_shared[row, col]
                                fval_uint = T.reinterpret(fval, T.uint32)
                                encoded_idx = (~T.uint32(col)) & T.uint32(idx_mask)
                                if fval < T.float32(0.0):
                                    encoded_idx = T.uint32(col) & T.uint32(idx_mask)
                                frag[e] = (fval_uint & (~T.uint32(idx_mask))) | encoded_idx
                            else:
                                neg_inf = T.reinterpret(T.float32(-1e38), T.uint32)
                                frag[e] = (neg_inf & (~T.uint32(idx_mask))) | (T.uint32(col) & T.uint32(idx_mask))

                        # Local partial sort: selection-based top-K extraction
                        for ki in T.serial(K_padded):
                            for e in T.serial(elems_per_thread):
                                if e > ki:
                                    frag_ki_f = T.reinterpret(frag[ki], T.float32)
                                    frag_e_f = T.reinterpret(frag[e], T.float32)
                                    if frag_e_f > frag_ki_f:
                                        tmp = frag[ki]
                                        frag[ki] = frag[e]
                                        frag[e] = tmp

                        # Bitonic sort on K_padded candidates
                        log_kp = int(math.log2(K_padded)) if K_padded > 1 else 0
                        for i1 in T.serial(log_kp):
                            for i2 in T.serial(i1 + 1):
                                block_s = 1 << (i1 - i2 + 1)
                                half_b = block_s >> 1
                                for ki in T.serial(K_padded):
                                    grp = ki // block_s
                                    pos = ki % block_s
                                    if pos < half_b:
                                        j = grp * block_s + (block_s - 1 - pos)
                                        fi = T.reinterpret(frag[ki], T.float32)
                                        fj = T.reinterpret(frag[j], T.float32)
                                        ascending = (ki >> (i1 + 1)) & 1
                                        if ascending == 1:
                                            if fi > fj:
                                                tmp = frag[ki]
                                                frag[ki] = frag[j]
                                                frag[j] = tmp
                                        else:
                                            if fi < fj:
                                                tmp = frag[ki]
                                                frag[ki] = frag[j]
                                                frag[j] = tmp

                        # Cross-thread merge via shfl_xor (if threads_per_row > 1)
                        if threads_per_row > 1:
                            partner = T.alloc_local([K_padded], T.uint32)
                            for stage in T.serial(num_merge_stages):
                                stride = 1 << stage
                                for ki in T.serial(K_padded):
                                    partner[ki] = T.shfl_xor(frag[ki], stride)
                                # Merge: keep top-K from frag + partner
                                for ki in T.serial(K_padded):
                                    p_f = T.reinterpret(partner[ki], T.float32)
                                    f_f = T.reinterpret(frag[ki], T.float32)
                                    if p_f > f_f:
                                        frag[ki] = partner[ki]
                                # Re-sort K_padded
                                for i1_m in T.serial(log_kp):
                                    for i2_m in T.serial(i1_m + 1):
                                        block_s2 = 1 << (i1_m - i2_m + 1)
                                        half_b2 = block_s2 >> 1
                                        for ki in T.serial(K_padded):
                                            grp2 = ki // block_s2
                                            pos2 = ki % block_s2
                                            if pos2 < half_b2:
                                                j2 = grp2 * block_s2 + (block_s2 - 1 - pos2)
                                                fi2 = T.reinterpret(frag[ki], T.float32)
                                                fj2 = T.reinterpret(frag[j2], T.float32)
                                                asc2 = (ki >> (i1_m + 1)) & 1
                                                if asc2 == 1:
                                                    if fi2 > fj2:
                                                        tmp = frag[ki]
                                                        frag[ki] = frag[j2]
                                                        frag[j2] = tmp
                                                else:
                                                    if fi2 < fj2:
                                                        tmp = frag[ki]
                                                        frag[ki] = frag[j2]
                                                        frag[j2] = tmp

                        # Decode + softmax (lane 0 only)
                        if lane_in_row == 0:
                            # Decode top-K values and indices
                            max_val = T.alloc_var(T.float32)
                            max_val = T.float32(-1e38)
                            for ki in T.serial(top_k):
                                fval = T.reinterpret(frag[ki], T.float32)
                                if fval > max_val:
                                    max_val = fval

                            exp_sum = T.alloc_var(T.float32)
                            exp_sum = T.float32(0.0)
                            for ki in T.serial(top_k):
                                raw_uint = frag[ki]
                                fval = T.reinterpret(raw_uint, T.float32)
                                encoded_idx = raw_uint & T.uint32(idx_mask)
                                if fval >= T.float32(0.0):
                                    decoded_col = T.int32((~encoded_idx) & T.uint32(idx_mask))
                                else:
                                    decoded_col = T.int32(encoded_idx & T.uint32(idx_mask))

                                exp_val = T.exp(fval - max_val)
                                exp_sum = exp_sum + exp_val
                                # Temporarily store in scores (will normalize next)
                                scores[token_idx, ki] = exp_val
                                indices[token_idx, ki] = decoded_col

                            for ki in T.serial(top_k):
                                scores[token_idx, ki] = scores[token_idx, ki] / exp_sum
                    T.sync_threads()

    return kernel


# ============================================================================
# Kernel Cache
# ============================================================================
_fused_cache: Dict[str, any] = {}


def get_fused_router_kernel(algo, num_tokens, H, E, top_k, dtype=T.bfloat16):
    """Get or compile a fused router kernel."""
    # Adaptive block config
    block_M = 64
    block_K = 64
    num_stages = 2
    threads = 256

    cache_key = f"{algo}_{num_tokens}_{H}_{E}_{top_k}"
    if cache_key in _fused_cache:
        return _fused_cache[cache_key]

    if algo == "selection":
        kernel = fused_router_selection_kernel(
            num_tokens, H, E, top_k,
            block_M=block_M, block_K=block_K,
            num_stages=num_stages, threads=threads, dtype=dtype)
    elif algo == "bitonic":
        kernel = fused_router_bitonic_kernel(
            num_tokens, H, E, top_k,
            block_M=block_M, block_K=block_K,
            num_stages=num_stages, threads=threads, dtype=dtype)
    elif algo == "bitonic_v3":
        kernel = fused_router_bitonic_v3_kernel(
            num_tokens, H, E, top_k,
            block_M=block_M, block_K=block_K,
            num_stages=num_stages, threads=threads, dtype=dtype)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    _fused_cache[cache_key] = kernel
    return kernel
