"""
TopK implementation inspired by SonicMoE's approach
Uses index encoding and optimized sorting for small k values
"""

import torch
import tilelang
import tilelang.language as T
import math


def next_power_of_2(n):
    """Get next power of 2 >= n"""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@tilelang.jit
def tl_topk_sonic_style_impl(topk, num_experts=160):
    """
    SonicMoE-inspired TopK implementation with index encoding
    
    Key optimizations:
    1. Index encoding: Store index in the mantissa bits of float
    2. Thread-level sorting: Each thread handles one row
    3. Register-based computation: Minimize shared memory use
    """
    batch = T.dynamic("batch")
    N = num_experts  # Fixed at compile time for better optimization
    
    # Use power of 2 for better performance
    N_padded = next_power_of_2(N)
    
    @T.prim_func
    def tl_topk_sonic_kernel(
        logits: T.Tensor[(batch, N), T.float32],
        indices: T.Tensor[(batch, topk), T.int32],
        values: T.Tensor[(batch, topk), T.float32],
    ):
        # One thread per row for simplicity and to avoid synchronization
        with T.Kernel(batch, threads=128) as bx:
            tx = T.get_thread_binding()
            
            if tx < batch:
                # Each thread processes one row independently
                # Allocate local arrays for the entire row
                local_values = T.alloc_local([N_padded], T.float32)
                local_indices = T.alloc_local([N_padded], T.int32)
                
                # Load data and initialize indices
                for i in T.serial(N):
                    local_values[i] = logits[tx, i]
                    local_indices[i] = i
                
                # Pad with -inf for power-of-2 size
                for i in T.serial(N, N_padded):
                    local_values[i] = T.float32(-1e9)
                    local_indices[i] = -1
                
                # Selection sort to find top-k
                # For small k (k=6), this is competitive
                for k_iter in T.serial(topk):
                    # Find max in remaining elements
                    max_idx = T.alloc_var(T.int32)
                    max_val = T.alloc_var(T.float32)
                    max_idx = k_iter
                    max_val = local_values[k_iter]
                    
                    for j in T.serial(k_iter + 1, N_padded):
                        if local_values[j] > max_val:
                            max_val = local_values[j]
                            max_idx = j
                    
                    # Swap
                    if max_idx != k_iter:
                        # Swap values
                        temp_val = local_values[k_iter]
                        local_values[k_iter] = local_values[max_idx]
                        local_values[max_idx] = temp_val
                        
                        # Swap indices
                        temp_idx = local_indices[k_iter]
                        local_indices[k_iter] = local_indices[max_idx]
                        local_indices[max_idx] = temp_idx
                
                # Write results
                for k_iter in T.serial(topk):
                    indices[tx, k_iter] = local_indices[k_iter]
                    values[tx, k_iter] = local_values[k_iter]
    
    return tl_topk_sonic_kernel


@tilelang.jit
def tl_topk_warp_impl(topk, num_experts=160):
    """
    Warp-level parallel TopK using reduction
    Each warp cooperatively processes one row
    """
    batch = T.dynamic("batch")
    N = num_experts
    warp_size = 32
    
    @T.prim_func
    def tl_topk_warp_kernel(
        logits: T.Tensor[(batch, N), T.float32],
        indices: T.Tensor[(batch, topk), T.int32],
        values: T.Tensor[(batch, topk), T.float32],
    ):
        # Use 32 threads per warp for parallel reduction
        with T.Kernel(batch, threads=warp_size) as bx:
            tx = T.get_thread_binding()
            
            # Shared memory for the row
            s_values = T.alloc_shared([N], T.float32)
            s_indices = T.alloc_shared([N], T.int32)
            s_topk_values = T.alloc_shared([topk], T.float32)
            s_topk_indices = T.alloc_shared([topk], T.int32)
            
            # Load data cooperatively
            for i in T.serial(T.ceildiv(N, warp_size)):
                idx = i * warp_size + tx
                if idx < N:
                    s_values[idx] = logits[bx, idx]
                    s_indices[idx] = idx
            
            T.sync_threads()
            
            # Find top-k using parallel selection
            for k_iter in T.serial(topk):
                # Find local max
                local_max_val = T.alloc_var(T.float32)
                local_max_idx = T.alloc_var(T.int32)
                local_max_val = T.float32(-1e9)
                local_max_idx = -1
                
                # Each thread finds max in its chunk
                for i in T.serial(T.ceildiv(N - k_iter, warp_size)):
                    idx = k_iter + i * warp_size + tx
                    if idx < N:
                        if s_values[idx] > local_max_val:
                            local_max_val = s_values[idx]
                            local_max_idx = idx
                
                # Store local max in shared memory for reduction
                s_temp_vals = T.alloc_shared([warp_size], T.float32)
                s_temp_idxs = T.alloc_shared([warp_size], T.int32)
                s_temp_vals[tx] = local_max_val
                s_temp_idxs[tx] = local_max_idx
                T.sync_threads()
                
                # Tree reduction to find global max
                stride = warp_size // 2
                while stride > 0:
                    if tx < stride:
                        if s_temp_vals[tx + stride] > s_temp_vals[tx]:
                            s_temp_vals[tx] = s_temp_vals[tx + stride]
                            s_temp_idxs[tx] = s_temp_idxs[tx + stride]
                    T.sync_threads()
                    stride = stride // 2
                
                # Thread 0 swaps the max to position k_iter
                if tx == 0:
                    max_idx = s_temp_idxs[0]
                    s_topk_values[k_iter] = s_values[max_idx]
                    s_topk_indices[k_iter] = s_indices[max_idx]
                    
                    # Swap with position k_iter
                    if max_idx != k_iter:
                        temp_val = s_values[k_iter]
                        s_values[k_iter] = s_values[max_idx]
                        s_values[max_idx] = temp_val
                        
                        temp_idx = s_indices[k_iter]
                        s_indices[k_iter] = s_indices[max_idx]
                        s_indices[max_idx] = temp_idx
                
                T.sync_threads()
            
            # Write results
            if tx < topk:
                indices[bx, tx] = s_topk_indices[tx]
                values[bx, tx] = s_topk_values[tx]
    
    return tl_topk_warp_kernel


def topk_sonic_style(logits, topk=6, use_warp=False):
    """
    SonicMoE-inspired TopK implementation
    
    Args:
        logits: [batch, num_experts] tensor
        topk: number of experts to select
        use_warp: if True, use warp-level parallel version
    
    Returns:
        tuple of (values, indices)
    """
    batch, num_experts = logits.shape
    
    indices = torch.zeros(batch, topk, dtype=torch.int32, device=logits.device)
    values = torch.zeros(batch, topk, dtype=torch.float32, device=logits.device)
    
    if use_warp:
        kernel = tl_topk_warp_impl(topk, num_experts)
    else:
        kernel = tl_topk_sonic_style_impl(topk, num_experts)
    
    kernel(logits, indices, values)
    
    return values, indices
