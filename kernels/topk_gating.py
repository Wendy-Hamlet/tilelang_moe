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
# Optimized Implementation: Warp-level Top-K with Shared Memory
# ============================================================================

def get_optimized_topk_configs():
    """
    Auto-tune configurations for optimized TopK kernel
    Using warp-level primitives for better performance
    """
    iter_params = dict(
        blk_m=[64, 128, 256],
        threads=[128, 256, 512],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_optimized_topk_configs())
@tilelang.jit(out_idx=[1, 2], target="cuda")
def tl_topk_optimized(M, N, topk, blk_m, threads=256):
    """
    Optimized Top-K using warp-level reduction and shared memory heap
    Algorithm: Parallel heap-based selection within warps
    Time Complexity: O(M × N + M × k × log(k))
    
    Strategy:
    1. Each warp handles blk_m/num_warps rows
    2. Use warp shuffle for parallel max reduction
    3. Maintain top-k heap in registers per thread
    4. Final merge in shared memory
    """
    dtype = T.float32
    warp_size = 32
    num_warps = threads // warp_size

    @T.prim_func
    def topk_kernel_opt(
        logits: T.Tensor([M, N], dtype),
        topk_gates: T.Tensor([M, topk], dtype),
        topk_indices: T.Tensor([M, topk], T.int32),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            # Shared memory for reduction and temporary storage
            s_values = T.alloc_shared([blk_m, topk], dtype)
            s_indices = T.alloc_shared([blk_m, topk], T.int32)
            s_temp_max = T.alloc_shared([blk_m, warp_size], dtype)
            s_temp_idx = T.alloc_shared([blk_m, warp_size], T.int32)
            
            # Register storage for local top-k per thread
            r_topk_vals = T.alloc_fragment([topk], dtype)
            r_topk_idx = T.alloc_fragment([topk], T.int32)
            
            tid = T.get_thread_binding()
            warp_id = tid // warp_size
            lane_id = tid % warp_size
            
            # Each thread handles multiple rows
            for row_offset in T.serial(T.ceildiv(blk_m, num_warps)):
                row = warp_id * T.ceildiv(blk_m, num_warps) + row_offset
                if row < blk_m:
                    global_row = bx * blk_m + row
                    if global_row < M:
                        # Initialize local top-k with negative infinity
                        for k_idx in T.Parallel(topk):
                            r_topk_vals[k_idx] = -10000.0
                            r_topk_idx[k_idx] = -1
                        
                        # Scan through N dimensions with warp-level parallelism
                        for n_base in T.serial(T.ceildiv(N, warp_size)):
                            n = n_base * warp_size + lane_id
                            val = T.if_then_else(n < N, logits[global_row, n], -10000.0)
                            
                            # Insert into local heap if value is larger than smallest in top-k
                            if val > r_topk_vals[topk - 1]:
                                # Find insertion position (simple linear scan for small k=6)
                                for k_idx in T.serial(topk):
                                    if val > r_topk_vals[k_idx]:
                                        # Shift elements down
                                        for shift_idx in T.serial(topk - 1 - k_idx):
                                            dest = topk - 1 - shift_idx
                                            src = dest - 1
                                            r_topk_vals[dest] = r_topk_vals[src]
                                            r_topk_idx[dest] = r_topk_idx[src]
                                        # Insert new value
                                        r_topk_vals[k_idx] = val
                                        r_topk_idx[k_idx] = n
                                        break
                        
                        # Write to shared memory for final merge
                        for k_idx in T.Parallel(topk):
                            s_temp_max[row, lane_id] = r_topk_vals[k_idx]
                            s_temp_idx[row, lane_id] = r_topk_idx[k_idx]
                            
                            # Warp-level reduction to find max
                            for offset in T.serial(5):  # log2(32) = 5
                                stride = 1 << (4 - offset)
                                if lane_id < stride:
                                    other_val = s_temp_max[row, lane_id + stride]
                                    other_idx = s_temp_idx[row, lane_id + stride]
                                    if other_val > s_temp_max[row, lane_id]:
                                        s_temp_max[row, lane_id] = other_val
                                        s_temp_idx[row, lane_id] = other_idx
                            
                            # Lane 0 writes final result
                            if lane_id == 0:
                                s_values[row, k_idx] = s_temp_max[row, 0]
                                s_indices[row, k_idx] = s_temp_idx[row, 0]
            
            T.sync_threads()
            
            # Copy results to global memory
            for row in T.Parallel(blk_m):
                global_row = bx * blk_m + row
                if global_row < M:
                    for k_idx in T.serial(topk):
                        topk_gates[global_row, k_idx] = s_values[row, k_idx]
                        topk_indices[global_row, k_idx] = s_indices[row, k_idx]

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


def topk_gating_tilelang(scores: torch.Tensor, k: int, use_optimized: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang Top-K gating (auto-selects best implementation)
    
    Args:
        scores: [batch, num_experts] - Gate scores (after softmax)
        k: Number of experts to select per token
        use_optimized: If True, use optimized warp-level impl; else use baseline
        
    Returns:
        indices: [batch, k] - Selected expert indices
        weights: [batch, k] - Corresponding weights (scores)
        
    Note:
        Block size (blk_m) and thread count are auto-tuned automatically.
    """
    M, N = scores.shape
    
    if use_optimized:
        # Use optimized warp-level implementation
        kernel = tl_topk_optimized(M=M, N=N, topk=k)
    else:
        # Use baseline iterative implementation
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

