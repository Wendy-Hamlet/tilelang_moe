"""
TileLang implementation of Top-K Gating for MoE
- Baseline: Iterative selection (from official example_topk.py)
- Optimized: Radix Select algorithm (from DeepSeek-V3.2)
"""
import torch
import tilelang
import tilelang.language as T
import itertools


def topk_gating_pytorch(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch baseline implementation of Top-K gating"""
    topk_weights, topk_indices = torch.topk(scores, k=k, dim=-1, sorted=False)
    return topk_indices, topk_weights


# ============================================================================
# Baseline Implementation: Iterative Selection (Official Example)
# ============================================================================


def get_topk_configs():
    """
    Auto-tune configurations for TopK kernel (Iterative version)
    Note: threads must be divisible by N (num_experts) for reduce operations
    For DeepSeek-V2 (N=160), valid thread counts: 160, 320, 480, 640, 800, 960
    """
    iter_params = dict(
        blk_m=[64, 128, 256],
        threads=[160, 320, 480],  # Must divide 160 (num_experts for DeepSeek-V2)
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_topk_configs())
@tilelang.jit(out_idx=[1, 2], target="cuda")
def tl_topk_iterative(M, N, topk, blk_m, threads=128):
    """
    Iterative Top-K selection (baseline from official example)
    Algorithm: Selection sort - iteratively find max and mask it
    Time Complexity: O(k × M × N) = O(6 × M × 160)
    """
    dtype = T.float32

    @T.prim_func
    def topk_kernel(
        logits: T.Tensor([M, N], dtype),
        topk_gates: T.Tensor([M, topk], dtype),
        topk_indices: T.Tensor([M, topk], T.int32),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            # Allocate fragments for computation
            logits_frag = T.alloc_fragment([blk_m, N], dtype=dtype)
            max_val = T.alloc_fragment([blk_m], dtype=dtype)
            expand_max_idx = T.alloc_fragment([blk_m, N], T.int32)
            max_idx = T.alloc_fragment([blk_m], T.int32)

            # Load input logits
            T.copy(logits[bx * blk_m, 0], logits_frag)

            # Iteratively select top-k elements
            for k in T.serial(topk):
                T.fill(expand_max_idx, -1)
                
                # Find maximum value
                T.reduce_max(logits_frag, max_val, dim=1, clear=True)

                # Find index of maximum value
                for i, j in T.Parallel(blk_m, N):
                    expand_max_idx[i, j] = T.if_then_else(
                        max_val[i] == logits_frag[i, j], j, expand_max_idx[i, j]
                    )

                T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)

                # Mask out selected value for next iteration
                for i, j in T.Parallel(blk_m, N):
                    logits_frag[i, j] = T.if_then_else(
                        max_val[i] == logits_frag[i, j], -10000.0, logits_frag[i, j]
                    )

                # Store results
                for i in T.Parallel(blk_m):
                    topk_gates[bx * blk_m + i, k] = max_val[i]
                    topk_indices[bx * blk_m + i, k] = max_idx[i]

    return topk_kernel


# ============================================================================
# Optimized Implementation: Bitonic Sort (TileLang-friendly)
# ============================================================================

def get_optimized_topk_configs():
    """
    Auto-tune configurations for optimized TopK kernel
    """
    iter_params = dict(
        blk_m=[32, 64, 128],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_optimized_topk_configs())
@tilelang.jit(out_idx=[1, 2], target="cuda")
def tl_topk_optimized(M, N, topk, blk_m, threads=128):
    """
    Optimized Top-K using selection sort with proper synchronization
    
    Key fixes:
    - Remove T.Parallel for selection sort to avoid race conditions
    - Process each row sequentially within each thread
    - Thread-level parallelism for loading/storing only
    """
    dtype = T.float32

    @T.prim_func
    def topk_kernel_opt(
        logits: T.Tensor([M, N], dtype),
        topk_gates: T.Tensor([M, topk], dtype),
        topk_indices: T.Tensor([M, topk], T.int32),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            # Shared memory for sorting
            s_values = T.alloc_shared([blk_m, N], dtype)
            s_indices = T.alloc_shared([blk_m, N], T.int32)
            
            tid = T.get_thread_binding()
            
            # Load data to shared memory and initialize indices (parallel)
            for i in T.Parallel(blk_m):
                row = bx * blk_m + i
                if row < M:
                    for j in range(tid, N, threads):
                        s_values[i, j] = logits[row, j]
                        s_indices[i, j] = j
            
            T.sync_threads()
            
            # Selection sort - process rows sequentially to avoid races
            # Each thread handles different rows based on thread ID
            for row_offset in T.serial(T.ceildiv(blk_m, threads)):
                local_row = tid + row_offset * threads
                if local_row < blk_m:
                    global_row = bx * blk_m + local_row
                    if global_row < M:
                        # For each position in top-k
                        for k_pos in T.serial(topk):
                            # Find max in remaining elements [k_pos:N]
                            max_val = s_values[local_row, k_pos]
                            max_idx_local = k_pos
                            
                            for j in T.serial(k_pos + 1, N):
                                if s_values[local_row, j] > max_val:
                                    max_val = s_values[local_row, j]
                                    max_idx_local = j
                            
                            # Swap to k_pos position
                            if max_idx_local != k_pos:
                                # Swap values
                                temp_val = s_values[local_row, k_pos]
                                s_values[local_row, k_pos] = s_values[local_row, max_idx_local]
                                s_values[local_row, max_idx_local] = temp_val
                                
                                # Swap indices
                                temp_idx = s_indices[local_row, k_pos]
                                s_indices[local_row, k_pos] = s_indices[local_row, max_idx_local]
                                s_indices[local_row, max_idx_local] = temp_idx
            
            T.sync_threads()
            
            # Write top-k results to output (parallel)
            for i in T.Parallel(blk_m):
                row = bx * blk_m + i
                if row < M:
                    for k_idx in T.serial(topk):
                        topk_gates[row, k_idx] = s_values[i, k_idx]
                        topk_indices[row, k_idx] = s_indices[i, k_idx]

    return topk_kernel_opt


# ============================================================================
# User-facing API Functions
# ============================================================================

def topk_gating_tilelang_baseline(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang Top-K using iterative selection (baseline from official example)
    
    Args:
        scores: [batch, num_experts] - Gate scores
        k: Number of experts to select per token
        
    Returns:
        indices: [batch, k] - Selected expert indices
        weights: [batch, k] - Corresponding weights (scores)
    """
    M, N = scores.shape
    kernel = tl_topk_iterative(M=M, N=N, topk=k)
    topk_weights, topk_indices = kernel(scores)
    return topk_indices, topk_weights


def topk_gating_tilelang(scores: torch.Tensor, k: int, use_optimized: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang Top-K gating (auto-selects best implementation)
    
    Args:
        scores: [batch, num_experts] - Gate scores (after softmax)
        k: Number of experts to select per token
        use_optimized: If True, use optimized warp-level impl; else use baseline
                      (Default: False - optimized version has compilation issues)
        
    Returns:
        indices: [batch, k] - Selected expert indices
        weights: [batch, k] - Corresponding weights (scores)
        
    Note:
        Block size (blk_m) and thread count are auto-tuned automatically.
    """
    M, N = scores.shape
    
    if use_optimized:
        # Use optimized warp-level implementation (currently has issues)
        kernel = tl_topk_optimized(M=M, N=N, topk=k)
    else:
        # Use baseline iterative implementation (stable)
        kernel = tl_topk_iterative(M=M, N=N, topk=k)
    
    # Execute kernel
    topk_weights, topk_indices = kernel(scores)
    
    return topk_indices, topk_weights


if __name__ == "__main__":
    # Quick test (for local verification)
    batch, num_experts, k = 320, 160, 6
    scores = torch.randn(batch, num_experts, device="cuda", dtype=torch.float32)
    
    print("Testing Top-K implementations...")
    print(f"Config: batch={batch}, num_experts={num_experts}, k={k}")
    
    # PyTorch baseline
    indices_pt, weights_pt = topk_gating_pytorch(scores, k)
    print(f"✓ PyTorch: indices={indices_pt.shape}, weights={weights_pt.shape}")
    
    # TileLang baseline version
    indices_tl_base, weights_tl_base = topk_gating_tilelang(scores, k, use_optimized=False)
    print(f"✓ TileLang (baseline): indices={indices_tl_base.shape}, weights={weights_tl_base.shape}")
    
    # TileLang optimized version
    indices_tl_opt, weights_tl_opt = topk_gating_tilelang(scores, k, use_optimized=True)
    print(f"✓ TileLang (optimized): indices={indices_tl_opt.shape}, weights={weights_tl_opt.shape}")
    
    # Verify correctness
    print("\nVerifying correctness...")
    torch.testing.assert_close(weights_tl_base, weights_pt, rtol=1e-5, atol=1e-5)
    print("✓ Baseline implementation correct!")
    torch.testing.assert_close(weights_tl_opt, weights_pt, rtol=1e-5, atol=1e-5)
    print("✓ Optimized implementation correct!")

