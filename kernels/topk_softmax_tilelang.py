"""
TopK+Softmax TileLang Implementations

This file contains all TileLang implementations for TopK+Softmax:
1. Radix TopK: Official TileLang radix selection algorithm
2. Selection Sort: O(K*N) selection sort based fusion kernel
3. Bitonic Sort: O(N*log²N) parallel bitonic sort fusion (Sonic-style)
4. Separate: TileLang TopK + TileLang Softmax (non-fused, all TileLang)
5. Best: Auto-selection based on config

Note: The real Sonic library uses Bitonic Sort for TopK.
      Our "selection" kernel is a simpler O(K*N) approach.

Recommended usage:
    from kernels.topk_softmax_tilelang import topk_softmax  # auto-select best
"""

import math
import torch
import tilelang
import tilelang as tl
import tilelang.language as T
from typing import Tuple, Dict, Callable


# ============================================================================
# Part 1: Radix TopK (Official TileLang Implementation)
# ============================================================================

_radix_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


def _convert_to_uint16(x):
    """Convert float to sortable uint16 by manipulating sign bit"""
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def _convert_to_uint32(x):
    """Convert float32 to sortable uint32 by manipulating sign bit"""
    bits_uint = T.reinterpret(T.uint32, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


@tilelang.jit(pass_configs=_radix_pass_configs)
def _tl_topk_radix_impl(topk, in_dtype=T.float32, out_dtype=T.int32):
    num_tokens = T.dynamic("num_tokens")
    num_experts = T.dynamic("num_experts")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = 4096

    @T.prim_func
    def tl_topk_radix_kernel(
        input: T.Tensor[(num_tokens, num_experts), in_dtype],
        index: T.Tensor[(num_tokens, topk), out_dtype],
        starts: T.Tensor[(num_tokens), out_dtype],
        ends: T.Tensor[(num_tokens), out_dtype],
    ):
        with T.Kernel(num_tokens, threads=BLOCK_SIZE) as (bx):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_histogram = T.alloc_shared([RADIX + 1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_num_input = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_start_pos = T.alloc_var(T.int32)
            l_start_idx = T.alloc_var(T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)
            pos = T.alloc_var(T.int32)

            l_new_topk = topk
            l_start_idx = starts[bx]
            l_end_idx = ends[bx]

            # Stage 1: Quick 8-bit TopK
            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)
            T.sync_threads()

            for s in T.serial(T.ceildiv(num_experts, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < num_experts:
                    inval_int16 = _convert_to_uint16(input[bx, input_idx])
                    T.atomic_add(s_histogram[inval_int16], 1)
            T.sync_threads()

            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx

            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            for s in T.serial(T.ceildiv(num_experts, BLOCK_SIZE)):
                T.sync_threads()
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < num_experts:
                    bin_id = _convert_to_uint16(input[bx, input_idx])
                    l_bin_id32 = T.Cast(T.int32, bin_id)
                    
                    if l_bin_id32 > l_threshold_bin_id:
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, pos] = input_idx
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        s_input_idx[0, pos] = input_idx

            # Stage 2: Iterative Refinement
            for round in T.serial(4):
                if l_new_topk <= 0:
                    break

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]

                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast(
                            T.int32, 
                            ((_convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) 
                              >> (24 - round * 8)) & 0xFF)
                        )
                        T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()

                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx

                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast(
                            T.int32,
                            ((_convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]])
                              >> (24 - round * 8)) & 0xFF)
                        )
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            if round == 3:
                                l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

    return tl_topk_radix_kernel


def topk_radix_tilelang(logits: torch.Tensor, topk: int = 6) -> torch.Tensor:
    """
    TileLang Radix TopK implementation
    
    Args:
        logits: [num_tokens, num_experts] tensor
        topk: number of experts to select
    
    Returns:
        indices: [num_tokens, topk] indices of top-k experts
    """
    if logits.dtype != torch.float32:
        logits = logits.to(torch.float32)
    
    num_tokens, num_experts = logits.shape
    num_tokens = int(num_tokens)
    num_experts = int(num_experts)
    
    if isinstance(topk, torch.Tensor):
        topk = int(topk.item())
    else:
        topk = int(topk)
    
    indices = torch.zeros((num_tokens, topk), dtype=torch.int32, device=logits.device)
    starts = torch.zeros((num_tokens,), dtype=torch.int32, device=logits.device)
    ends = torch.full((num_tokens,), num_experts, dtype=torch.int32, device=logits.device)
    
    kernel = _tl_topk_radix_impl(topk)
    kernel(logits, indices, starts, ends)
    
    return indices


# ============================================================================
# Part 1.5: TileLang Online Softmax (Official Implementation)
# ============================================================================

# Softmax kernel cache: (M, N) -> kernel
_softmax_kernel_cache: Dict[Tuple[int, int], Callable] = {}


@tl.jit(out_idx=[1])
def _tl_softmax_kernel(M: int, N: int, dtype: T.dtype = T.float32):
    """
    TileLang Online Softmax implementation
    Based on: tilelang/examples/online_softmax/online_softmax.py
    """
    BN = min(tl.next_power_of_2(N), 8192)
    NN = tl.cdiv(N, BN)
    accum_dtype = T.float32
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor([M, N], dtype),
        Y: T.Tensor([M, N], dtype),
    ):
        with T.Kernel(M, threads=128) as (i_m):
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], accum_dtype)
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], accum_dtype)
            sum_exp_x = T.alloc_fragment([1], accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                T.reduce_max(x, max_x, dim=0, clear=True)
                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)
                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)
                lse[0] = max_x[0] * scale + T.log2(T.exp2(lse[0] - max_x[0] * scale) + sum_exp_x[0])

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])
                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main


def softmax_tilelang(x: torch.Tensor) -> torch.Tensor:
    """
    TileLang Softmax wrapper
    
    Args:
        x: [M, N] input tensor
    Returns:
        y: [M, N] softmax output
    """
    M, N = x.shape
    cache_key = (M, N)
    
    if cache_key not in _softmax_kernel_cache:
        dtype = T.float32 if x.dtype == torch.float32 else T.float16
        _softmax_kernel_cache[cache_key] = _tl_softmax_kernel(M, N, dtype)
    
    kernel = _softmax_kernel_cache[cache_key]
    return kernel(x)


# ============================================================================
# Part 2: Selection Sort TopK+Softmax
# ============================================================================

@tilelang.jit(
    out_idx=[1, 2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def topk_softmax_selection_kernel(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_size: int = 256,
    dtype = T.float32,
):
    """
    Selection Sort based TopK+Softmax fusion kernel
    
    Algorithm: Selection sort O(K*N) + single-thread softmax
    Best for: Small K with high padding ratio (fewer iterations needed)
    
    Note: This is NOT the actual Sonic algorithm. Real Sonic uses Bitonic Sort.
    """
    @T.prim_func
    def main(
        logits: T.Tensor((num_tokens, num_experts), dtype),
        scores: T.Tensor((num_tokens, top_k), T.float32),
        indices: T.Tensor((num_tokens, top_k), T.int32),
    ):
        with T.Kernel(num_tokens, threads=block_size) as bx:
            s_topk_vals = T.alloc_shared((top_k,), T.float32)
            s_topk_idx = T.alloc_shared((top_k,), T.int32)
            s_thread_vals = T.alloc_shared((block_size,), T.float32)
            s_thread_idx = T.alloc_shared((block_size,), T.int32)
            
            tx = T.get_thread_binding()
            elems_per_thread = T.ceildiv(num_experts, block_size)
            
            for i in T.serial(top_k):
                if tx == 0:
                    s_topk_vals[i] = T.float32(-1e38)
                    s_topk_idx[i] = T.int32(-1)
            T.sync_threads()
            
            for k_iter in T.serial(top_k):
                thread_max_val = T.alloc_var(T.float32)
                thread_max_idx = T.alloc_var(T.int32)
                thread_max_val = T.float32(-1e38)
                thread_max_idx = T.int32(-1)
                
                for i in T.serial(elems_per_thread):
                    idx = tx * elems_per_thread + i
                    if idx < num_experts:
                        val = logits[bx, idx]
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
                
                # Tree reduction (unrolled for block_size=256)
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
            
            # Softmax
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
                    s_topk_vals[i] = s_topk_vals[i] / exp_sum
            
            T.sync_threads()
            
            if tx == 0:
                for i in T.serial(top_k):
                    scores[bx, i] = s_topk_vals[i]
                    indices[bx, i] = s_topk_idx[i]
    
    return main


# ============================================================================
# Part 3: Bitonic Sort TopK+Softmax (Sonic-style, Parallel Sort)
# ============================================================================

_bitonic_pass_configs = {
    tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def _make_topk_softmax_bitonic_kernel(num_tokens: int, num_experts: int, top_k: int):
    """
    Create fused TopK+Softmax kernel with Bitonic Sort (Sonic-style)
    
    This is the same algorithm used by Sonic library (from QuACK's bitonic_topk).
    
    Algorithm: O(N * log²N) parallel sort
    Best for: N is power of 2 (no padding waste), high parallelism
    """
    N_padded = 1 << int(math.ceil(math.log2(max(num_experts, 2))))
    num_iters = int(math.log2(N_padded))
    
    @tl.jit(pass_configs=_bitonic_pass_configs)
    def kernel_factory():
        
        @T.macro
        def bitonic_sort_descending(s_vals: T.Buffer, s_idx: T.Buffer):
            T.sync_threads()
            for i1 in T.serial(num_iters):
                for i2 in T.serial(i1 + 1):
                    for i in T.Parallel(N_padded):
                        ascending = (i & (1 << (i1 + 1))) != 0
                        j = i ^ (1 << (i1 - i2))
                        if i < j and (
                            (ascending and s_vals[i] > s_vals[j])
                            or (not ascending and s_vals[i] < s_vals[j])
                        ):
                            val = s_vals[i]
                            s_vals[i] = s_vals[j]
                            s_vals[j] = val
                            idx = s_idx[i]
                            s_idx[i] = s_idx[j]
                            s_idx[j] = idx
                    T.sync_threads()
        
        @T.prim_func
        def main(
            logits: T.Tensor((num_tokens, num_experts), T.float32),
            scores: T.Tensor((num_tokens, top_k), T.float32),
            indices: T.Tensor((num_tokens, top_k), T.int32),
        ):
            with T.Kernel(num_tokens, threads=N_padded) as bx:
                s_vals = T.alloc_shared((N_padded,), T.float32)
                s_idx = T.alloc_shared((N_padded,), T.int32)
                
                for i in T.Parallel(N_padded):
                    if i < num_experts:
                        s_vals[i] = logits[bx, i]
                        s_idx[i] = T.int32(i)
                    else:
                        s_vals[i] = T.float32(-1e38)
                        s_idx[i] = T.int32(-1)
                T.sync_threads()
                
                bitonic_sort_descending(s_vals, s_idx)
                
                s_softmax = T.alloc_shared((top_k,), T.float32)
                s_exp_sum = T.alloc_shared((1,), T.float32)
                
                for i in T.Parallel(top_k):
                    s_softmax[i] = T.exp(s_vals[i] - s_vals[0])
                T.sync_threads()
                
                T.fill(s_exp_sum, T.float32(0.0))
                T.sync_threads()
                for i in T.serial(top_k):
                    s_exp_sum[0] = s_exp_sum[0] + s_softmax[i]
                T.sync_threads()
                
                for i in T.Parallel(top_k):
                    scores[bx, i] = s_softmax[i] / s_exp_sum[0]
                    indices[bx, i] = s_idx[i]
        
        return main
    
    return kernel_factory()


# ============================================================================
# Part 3b: Bitonic Sort v2 -- Multi-Token + Bit-Packing (Sonic-aligned)
#
# Key improvements over v1:
#   - Multiple tokens per block (cols_per_block) instead of 1
#   - Bit-packing: column index encoded into float32 low bits → no separate
#     index array, halves shared memory and swap cost
#   - 128 threads per block (fixed), each handling elems_per_thread elements
#   - Same bitonic sorting network, but much better SM utilization
# ============================================================================

_BITONIC_V2_BLOCK_THREADS = 128


def _bitonic_v2_thread_config(num_experts: int, top_k: int):
    """Compute thread layout parameters matching Sonic's strategy."""
    N_padded = 1 << int(math.ceil(math.log2(max(num_experts, 2))))
    threads_per_row = max(min(N_padded // top_k, 32, N_padded // 64), 1)
    cols_per_block = _BITONIC_V2_BLOCK_THREADS // threads_per_row
    elems_per_thread = N_padded // threads_per_row
    return N_padded, threads_per_row, cols_per_block, elems_per_thread


def _make_topk_softmax_bitonic_v2_kernel(num_tokens: int, num_experts: int, top_k: int):
    """
    Bitonic Sort v2: multi-token per block + bit-packing.

    Thread layout (all compile-time constants):
      BLOCK_THREADS  = 128
      threads_per_row = max(min(N_padded//K, 32, N_padded//64), 1)
      cols_per_block  = 128 // threads_per_row   (tokens per block)
      elems_per_thread = N_padded // threads_per_row

    Bit-packing: column index encoded into low log2(N_padded) bits of the
    float32 value (reinterpreted as uint32).  Positive values invert the
    index bits for stable tie-breaking.  After sort, low bits are cleared
    to recover the value, and index is decoded.
    """
    N_padded, threads_per_row, cols_per_block, elems_per_thread = \
        _bitonic_v2_thread_config(num_experts, top_k)
    num_iters = int(math.log2(N_padded))
    log_N = int(math.log2(N_padded))
    idx_mask_val = (1 << log_N) - 1
    num_blocks = (num_tokens + cols_per_block - 1) // cols_per_block
    total_smem = cols_per_block * N_padded  # uint32 elements

    @tl.jit(pass_configs=_bitonic_pass_configs)
    def kernel_factory():

        @T.prim_func
        def main(
            logits: T.Tensor((num_tokens, num_experts), T.float32),
            scores: T.Tensor((num_tokens, top_k), T.float32),
            indices: T.Tensor((num_tokens, top_k), T.int32),
        ):
            with T.Kernel(num_blocks, threads=_BITONIC_V2_BLOCK_THREADS) as bx:
                # Shared memory: bit-packed values (value + encoded index)
                s_data = T.alloc_shared((total_smem,), T.uint32)

                tx = T.get_thread_binding()
                row_in_block = tx // threads_per_row
                lane_in_row = tx % threads_per_row
                token_idx = bx * cols_per_block + row_in_block
                row_off = row_in_block * N_padded
                idx_mask = T.uint32(idx_mask_val)

                # ----------------------------------------------------------
                # Step 1: Load + bit-pack encode
                # ----------------------------------------------------------
                for e in T.serial(elems_per_thread):
                    col = lane_in_row * elems_per_thread + e
                    smem_pos = row_off + col

                    # Use if_then_else to avoid scoping issues
                    fval = T.if_then_else(
                        T.boolean(token_idx < num_tokens) & T.boolean(col < num_experts),
                        logits[token_idx, T.min(col, num_experts - 1)],
                        T.float32(-1e38),
                    )

                    bits = T.reinterpret(T.uint32, fval)
                    col_u32 = T.Cast(T.uint32, col)
                    # Positive values: invert index bits for stable tie-break
                    encoded_idx = T.if_then_else(
                        fval >= T.float32(0.0),
                        (~col_u32) & idx_mask,
                        col_u32 & idx_mask,
                    )
                    s_data[smem_pos] = (bits & (~idx_mask)) | encoded_idx

                T.sync_threads()

                # ----------------------------------------------------------
                # Step 2: Bitonic sort (descending) per row in shared memory
                #   Each thread handles elems_per_thread elements via serial
                #   loop.  The i<j convention ensures no write conflicts.
                # ----------------------------------------------------------
                for i1 in T.serial(num_iters):
                    for i2 in T.serial(i1 + 1):
                        step_bit = T.int32(1) << (i1 - i2)
                        dir_bit = T.int32(1) << (i1 + 1)

                        for e in T.serial(elems_per_thread):
                            p = lane_in_row * elems_per_thread + e
                            q = p ^ step_bit
                            if p < q and q < N_padded:
                                ascending = (p & dir_bit) != 0
                                sp = s_data[row_off + p]
                                sq = s_data[row_off + q]
                                # Reinterpret as float for comparison
                                fp = T.reinterpret(T.float32, sp)
                                fq = T.reinterpret(T.float32, sq)
                                do_swap = T.if_then_else(
                                    ascending,
                                    fp > fq,
                                    fp < fq,
                                )
                                if do_swap:
                                    s_data[row_off + p] = sq
                                    s_data[row_off + q] = sp

                        T.sync_threads()

                # ----------------------------------------------------------
                # Step 3: Decode top-K + softmax (first thread of each row)
                # ----------------------------------------------------------
                if lane_in_row == 0 and token_idx < num_tokens:
                    # Decode top-K values and indices
                    # After descending sort, top-K are at positions 0..K-1
                    s_topk_vals = T.alloc_shared((top_k,), T.float32)
                    s_topk_idx = T.alloc_shared((top_k,), T.int32)

                    for ki in T.serial(top_k):
                        packed = s_data[row_off + ki]
                        cleaned = packed & (~idx_mask)
                        fval = T.reinterpret(T.float32, cleaned)

                        # Decode index
                        encoded_idx_val = packed & idx_mask
                        col_idx = T.if_then_else(
                            fval >= T.float32(0.0),
                            (~encoded_idx_val) & idx_mask,
                            encoded_idx_val & idx_mask,
                        )
                        s_topk_vals[ki] = fval
                        s_topk_idx[ki] = T.Cast(T.int32, col_idx)

                    # Softmax: find max
                    max_val = T.alloc_var(T.float32)
                    max_val = s_topk_vals[0]
                    for ki in T.serial(top_k):
                        if s_topk_vals[ki] > max_val:
                            max_val = s_topk_vals[ki]

                    # exp and sum
                    exp_sum = T.alloc_var(T.float32)
                    exp_sum = T.float32(0.0)
                    for ki in T.serial(top_k):
                        s_topk_vals[ki] = T.exp(s_topk_vals[ki] - max_val)
                        exp_sum = exp_sum + s_topk_vals[ki]

                    # normalize and write
                    for ki in T.serial(top_k):
                        scores[token_idx, ki] = s_topk_vals[ki] / exp_sum
                        indices[token_idx, ki] = s_topk_idx[ki]

        return main

    return kernel_factory()


_bitonic_kernel_cache = {}
_bitonic_v2_kernel_cache = {}


def topk_softmax_bitonic(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused TopK+Softmax using Bitonic Sort

    Args:
        logits: (num_tokens, num_experts) tensor
        k: number of top elements

    Returns:
(scores, indices): both shape (num_tokens, k)
    """
    num_tokens, num_experts = logits.shape

    if num_experts > 1024:
        raise ValueError(f"Bitonic sort kernel supports num_experts <= 1024, got {num_experts}")

    cache_key = (num_tokens, num_experts, k)
    if cache_key not in _bitonic_kernel_cache:
        _bitonic_kernel_cache[cache_key] = _make_topk_softmax_bitonic_kernel(num_tokens, num_experts, k)

    kernel = _bitonic_kernel_cache[cache_key]

    scores = torch.empty(num_tokens, k, device=logits.device, dtype=torch.float32)
    indices = torch.empty(num_tokens, k, device=logits.device, dtype=torch.int32)

    kernel(logits, scores, indices)
    return scores, indices


def topk_softmax_bitonic_v2(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused TopK+Softmax using Bitonic Sort v2 (multi-token + bit-packing).

    Same interface as topk_softmax_bitonic but processes multiple tokens per
    block (like Sonic) for much better SM utilization at large T.
    """
    num_tokens, num_experts = logits.shape

    if num_experts > 1024:
        raise ValueError(f"Bitonic v2 kernel supports num_experts <= 1024, got {num_experts}")

    cache_key = (num_tokens, num_experts, k)
    if cache_key not in _bitonic_v2_kernel_cache:
        _bitonic_v2_kernel_cache[cache_key] = _make_topk_softmax_bitonic_v2_kernel(
            num_tokens, num_experts, k
        )

    kernel = _bitonic_v2_kernel_cache[cache_key]

    scores = torch.empty(num_tokens, k, device=logits.device, dtype=torch.float32)
    indices = torch.empty(num_tokens, k, device=logits.device, dtype=torch.int32)

    kernel(logits, scores, indices)
    return scores, indices


# ============================================================================
# Part 4: Selection Sort Wrapper
# ============================================================================

_selection_kernel_cache = {}


def topk_softmax_selection(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Selection Sort based TopK+Softmax
    
    Args:
        logits: (num_tokens, num_experts) tensor
        k: number of top elements
    
    Returns:
(scores, indices): both shape (num_tokens, k)
    """
    num_tokens, num_experts = logits.shape
    
    cache_key = (num_tokens, num_experts, k)
    if cache_key not in _selection_kernel_cache:
        _selection_kernel_cache[cache_key] = topk_softmax_selection_kernel(num_tokens, num_experts, k)
    
    kernel = _selection_kernel_cache[cache_key]
    return kernel(logits)


# Alias for backward compatibility
topk_softmax_sonic = topk_softmax_selection
topk_softmax_sonic_kernel = topk_softmax_selection_kernel


# ============================================================================
# Part 5: Separate Implementation (Non-fused, All TileLang)
# ============================================================================

def topk_softmax_tilelang_separate(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang non-fused version: TileLang Softmax + TileLang Radix TopK
    
    Both operators use official TileLang implementations:
    - Softmax: online_softmax from tilelang/examples/
    - TopK: radix_topk from tilelang/examples/deepseek_v32/
    
    Args:
        logits: (num_tokens, num_experts) float32
        k: top-k count
    Returns:
(scores, indices): both shape (num_tokens, k)
    """
    # Ensure float32
    if logits.dtype != torch.float32:
        logits = logits.to(torch.float32)
    
    # Step 1: TileLang Softmax (global)
    scores = softmax_tilelang(logits)
    
    # Step 2: TileLang Radix TopK
    topk_indices = topk_radix_tilelang(logits, k)
    
    # Step 3: Gather top-k scores
    topk_scores = torch.gather(scores, dim=-1, index=topk_indices.to(torch.int64))
    
    return topk_scores, topk_indices.to(torch.int32)


# ============================================================================
# Part 6: Smart Auto-Selection (Best Implementation)
# ============================================================================

def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _get_padding_ratio(n: int) -> float:
    n_padded = 1 << int(math.ceil(math.log2(max(n, 2))))
    return (n_padded - n) / n


def _select_strategy(num_experts: int, k: int) -> str:
    """
    Smart strategy selection based on algorithm cost estimation
    
    Returns: 'bitonic', 'selection', or 'radix'
    
    Note:
    - 'bitonic' = Bitonic Sort O(N*log²N), same as real Sonic library
    - 'selection' = Selection Sort O(K*N), simpler but fewer iterations for small K
    """
    if num_experts > 1024:
        return 'radix'
    
    padding_ratio = _get_padding_ratio(num_experts)
    
    if num_experts <= 64:
        return 'bitonic'
    
    if k <= 8 and padding_ratio > 0.4:
        return 'selection'
    
    if 8 <= k <= 16 and 0.2 <= padding_ratio <= 0.5:
        return 'bitonic' if num_experts >= 384 else 'selection'
    
    if padding_ratio > 0.5:
        return 'selection'
    
    if _is_power_of_2(num_experts) or padding_ratio < 0.2:
        if num_experts >= 128 and k <= 6:
            return 'selection'
        if num_experts >= 256 and k <= 8:
            return 'selection'
        if num_experts >= 1024 and k >= 16:
            return 'selection'
        return 'bitonic'
    
    return 'bitonic'


def topk_softmax_best(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Auto-select best TopK+Softmax implementation
    
    Selection is based on:
    - N value and padding overhead for Bitonic (Sonic-style)
    - K value for Selection Sort iteration count
    - N > 1024 falls back to Radix
    
    Args:
        logits: (num_tokens, num_experts) float32
        k: top-k count
    Returns:
(scores, indices): both shape (num_tokens, k)
    """
    num_tokens, num_experts = logits.shape
    strategy = _select_strategy(num_experts, k)
    
    if strategy == 'bitonic':
        return topk_softmax_bitonic(logits, k)
    elif strategy == 'selection':
        return topk_softmax_selection(logits, k)
    else:  # radix
        topk_indices = topk_radix_tilelang(logits, k)
        topk_values = torch.gather(logits, dim=-1, index=topk_indices.to(torch.int64))
        scores = torch.softmax(topk_values, dim=-1, dtype=torch.float32)
        return scores, topk_indices.to(torch.int32)


# Default export: auto-select best
topk_softmax = topk_softmax_best



def topk_softmax_conv_b_radix_unfused(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    BL-5: Convention B + TileLang radix topk (unfused) + PyTorch softmax(K)

    Used in Experiment 1 (Q1/Q2 chain) as the TileLang unfused baseline.
    Radix topk is the only TileLang algorithm naturally separable from softmax:
    it outputs indices only, so softmax must be applied in a separate step.

    Contrast with bitonic/selection: those algorithms produce scores within a
    single kernel not because of classical fusion but because the algorithm
    itself never creates an intermediate index tensor in global memory.
    """
    if logits.dtype != torch.float32:
        logits = logits.to(torch.float32)
    indices = topk_radix_tilelang(logits, k)
    topk_values = torch.gather(logits, dim=-1, index=indices.to(torch.int64))
    scores = torch.softmax(topk_values, dim=-1)
    return scores, indices.to(torch.int32)
