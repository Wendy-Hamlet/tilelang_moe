"""
TopK+Softmax TileLang Implementations

This file contains all TileLang implementations for TopK+Softmax:
1. Radix TopK: Official TileLang radix selection algorithm
2. Selection Sort: O(K*N) selection sort based fusion kernel
3. Bitonic Sort: O(N*log²N) parallel bitonic sort fusion (Sonic-style)
4. Separate: TileLang TopK + PyTorch Softmax (non-fused)
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
from typing import Tuple


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
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = 4096

    @T.prim_func
    def tl_topk_radix_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
        starts: T.Tensor[(batch), out_dtype],
        ends: T.Tensor[(batch), out_dtype],
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
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

            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
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

            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                T.sync_threads()
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
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
        logits: [batch, num_experts] tensor
        topk: number of experts to select
    
    Returns:
        indices: [batch, topk] indices of top-k experts
    """
    if logits.dtype != torch.float32:
        logits = logits.to(torch.float32)
    
    batch, num_experts = logits.shape
    batch = int(batch)
    num_experts = int(num_experts)
    
    if isinstance(topk, torch.Tensor):
        topk = int(topk.item())
    else:
        topk = int(topk)
    
    indices = torch.zeros((batch, topk), dtype=torch.int32, device=logits.device)
    starts = torch.zeros((batch,), dtype=torch.int32, device=logits.device)
    ends = torch.full((batch,), num_experts, dtype=torch.int32, device=logits.device)
    
    kernel = _tl_topk_radix_impl(topk)
    kernel(logits, indices, starts, ends)
    
    return indices


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
    batch_size: int,
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
        logits: T.Tensor((batch_size, num_experts), dtype),
        scores: T.Tensor((batch_size, top_k), T.float32),
        indices: T.Tensor((batch_size, top_k), T.int32),
    ):
        with T.Kernel(batch_size, threads=block_size) as bx:
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


def _make_topk_softmax_bitonic_kernel(batch_size: int, num_experts: int, top_k: int):
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
            logits: T.Tensor((batch_size, num_experts), T.float32),
            scores: T.Tensor((batch_size, top_k), T.float32),
            indices: T.Tensor((batch_size, top_k), T.int32),
        ):
            with T.Kernel(batch_size, threads=N_padded) as bx:
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


_bitonic_kernel_cache = {}


def topk_softmax_bitonic(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused TopK+Softmax using Bitonic Sort
    
    Args:
        logits: (batch, num_experts) tensor
        k: number of top elements
    
    Returns:
        (scores, indices): both shape (batch, k)
    """
    batch_size, num_experts = logits.shape
    
    if num_experts > 1024:
        raise ValueError(f"Bitonic sort kernel supports num_experts <= 1024, got {num_experts}")
    
    cache_key = (batch_size, num_experts, k)
    if cache_key not in _bitonic_kernel_cache:
        _bitonic_kernel_cache[cache_key] = _make_topk_softmax_bitonic_kernel(batch_size, num_experts, k)
    
    kernel = _bitonic_kernel_cache[cache_key]
    
    scores = torch.empty(batch_size, k, device=logits.device, dtype=torch.float32)
    indices = torch.empty(batch_size, k, device=logits.device, dtype=torch.int32)
    
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
        logits: (batch, num_experts) tensor
        k: number of top elements
    
    Returns:
        (scores, indices): both shape (batch, k)
    """
    batch_size, num_experts = logits.shape
    
    cache_key = (batch_size, num_experts, k)
    if cache_key not in _selection_kernel_cache:
        _selection_kernel_cache[cache_key] = topk_softmax_selection_kernel(batch_size, num_experts, k)
    
    kernel = _selection_kernel_cache[cache_key]
    return kernel(logits)


# Alias for backward compatibility
topk_softmax_sonic = topk_softmax_selection
topk_softmax_sonic_kernel = topk_softmax_selection_kernel


# ============================================================================
# Part 5: Separate Implementation (Non-fused)
# ============================================================================

def topk_softmax_tilelang_separate(logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang non-fused version: Global softmax + Radix TopK
    
    Args:
        logits: (batch, num_experts) float32
        k: top-k count
    Returns:
        (scores, indices): both shape (batch, k)
    """
    import torch.nn.functional as F
    scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    topk_indices = topk_radix_tilelang(logits, k)
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
        logits: (batch, num_experts) float32
        k: top-k count
    Returns:
        (scores, indices): both shape (batch, k)
    """
    batch_size, num_experts = logits.shape
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
