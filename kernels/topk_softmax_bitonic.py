"""
TopK+Softmax with Bitonic Sort Fusion

Key optimizations:
1. Bitonic Sort: O(N * log²N) parallel sort, highly efficient on GPU
2. Kernel Fusion: Single kernel = ONE read + ONE write to global memory
3. Top-K extraction: After sort, top-k are at known positions

Reference: tilelang/examples/dsa_sparse_finetune/indexer_topk_reducesum.py
"""

import math
import tilelang as tl
import tilelang.language as T


# Pass configs that work well for this kernel
pass_configs = {
    tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def _make_topk_softmax_bitonic_kernel(batch_size: int, num_experts: int, top_k: int):
    """
    Create fused TopK+Softmax kernel with Bitonic Sort
    
    Memory pattern:
    - Read: logits[batch, num_experts] - ONE read
    - Write: scores[batch, k], indices[batch, k] - ONE write
    """
    # Pad to power of 2 for bitonic sort
    N_padded = 1 << int(math.ceil(math.log2(max(num_experts, 2))))
    num_iters = int(math.log2(N_padded))
    
    @tl.jit(pass_configs=pass_configs)
    def kernel_factory():
        
        @T.macro
        def bitonic_sort_descending(
            s_vals: T.Buffer,
            s_idx: T.Buffer,
        ):
            """Bitonic sort - descending order (largest first)"""
            T.sync_threads()
            for i1 in T.serial(num_iters):
                for i2 in T.serial(i1 + 1):
                    for i in T.Parallel(N_padded):
                        # Original condition produces descending order
                        ascending = (i & (1 << (i1 + 1))) != 0
                        j = i ^ (1 << (i1 - i2))
                        if i < j and (
                            (ascending and s_vals[i] > s_vals[j])
                            or (not ascending and s_vals[i] < s_vals[j])
                        ):
                            # Swap values
                            val = s_vals[i]
                            s_vals[i] = s_vals[j]
                            s_vals[j] = val
                            # Swap indices
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
                # Shared memory for sorting
                s_vals = T.alloc_shared((N_padded,), T.float32)
                s_idx = T.alloc_shared((N_padded,), T.int32)
                
                # ========================================
                # Stage 1: Load data (parallel, coalesced)
                # ========================================
                for i in T.Parallel(N_padded):
                    if i < num_experts:
                        s_vals[i] = logits[bx, i]
                        s_idx[i] = T.int32(i)
                    else:
                        s_vals[i] = T.float32(-1e38)  # -inf for padding
                        s_idx[i] = T.int32(-1)
                T.sync_threads()
                
                # ========================================
                # Stage 2: Bitonic Sort (descending)
                # ========================================
                bitonic_sort_descending(s_vals, s_idx)
                
                # After descending sort: largest at s_vals[0], smallest at s_vals[N_padded-1]
                # Top-k largest are at positions [0, top_k-1]
                
                # ========================================
                # Stage 3: Softmax on top-k elements
                # ========================================
                # Use shared memory for softmax computation
                s_softmax = T.alloc_shared((top_k,), T.float32)
                s_exp_sum = T.alloc_shared((1,), T.float32)
                
                # Compute exp(x - max) where max = s_vals[0] (already sorted)
                for i in T.Parallel(top_k):
                    s_softmax[i] = T.exp(s_vals[i] - s_vals[0])
                T.sync_threads()
                
                # Sum reduction (simple serial for small k)
                T.fill(s_exp_sum, T.float32(0.0))
                T.sync_threads()
                for i in T.serial(top_k):
                    s_exp_sum[0] = s_exp_sum[0] + s_softmax[i]
                T.sync_threads()
                
                # ========================================
                # Stage 4: Write results (parallel)
                # ========================================
                for i in T.Parallel(top_k):
                    scores[bx, i] = s_softmax[i] / s_exp_sum[0]
                    indices[bx, i] = s_idx[i]
        
        return main
    
    return kernel_factory()


# Kernel cache
_bitonic_kernel_cache = {}


def topk_softmax_bitonic(logits, k):
    """
    Fused TopK+Softmax using Bitonic Sort
    
    Args:
        logits: (batch, num_experts) tensor
        k: number of top elements
    
    Returns:
        (scores, indices): both shape (batch, k)
    """
    import torch
    
    batch_size, num_experts = logits.shape
    
    # Bitonic sort has practical limits on N
    if num_experts > 1024:
        raise ValueError(f"Bitonic sort kernel supports num_experts <= 1024, got {num_experts}")
    
    cache_key = (batch_size, num_experts, k)
    if cache_key not in _bitonic_kernel_cache:
        _bitonic_kernel_cache[cache_key] = _make_topk_softmax_bitonic_kernel(
            batch_size, num_experts, k
        )
    
    kernel = _bitonic_kernel_cache[cache_key]
    
    # Allocate output tensors
    scores = torch.empty(batch_size, k, device=logits.device, dtype=torch.float32)
    indices = torch.empty(batch_size, k, device=logits.device, dtype=torch.int32)
    
    kernel(logits, scores, indices)
    return scores, indices


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    import torch
    from tilelang.profiler import do_bench
    
    print("=" * 80)
    print("TopK+Softmax Bitonic Sort Fusion Benchmark")
    print("Benefit: Single kernel = ONE read + ONE write to global memory")
    print("=" * 80)
    
    configs = [
        # (batch, num_experts, top_k)
        (64, 64, 8),
        (64, 128, 8),
        (64, 256, 8),
        (64, 512, 16),
        (128, 256, 8),
        (256, 256, 8),
        (64, 1024, 16),
    ]
    
    print(f"\n{'Config':<25} | {'PyTorch':>10} | {'Bitonic':>10} | {'Speedup':>8} | {'Status'}")
    print("-" * 80)
    
    for batch_size, num_experts, top_k in configs:
        config_str = f"B{batch_size}_E{num_experts}_K{top_k}"
        
        torch.manual_seed(42)
        logits = torch.randn(batch_size, num_experts, device='cuda', dtype=torch.float32)
        
        # PyTorch baseline
        def pytorch_fn():
            topk = torch.topk(logits, top_k, dim=-1)
            return torch.softmax(topk.values, dim=-1), topk.indices
        
        pytorch_ms = do_bench(pytorch_fn, warmup=100, rep=1000, return_mode='median')
        ref_scores, ref_indices = pytorch_fn()
        
        try:
            # Bitonic fusion kernel
            def bitonic_fn():
                return topk_softmax_bitonic(logits, top_k)
            
            # Warmup and benchmark
            bitonic_ms = do_bench(bitonic_fn, warmup=100, rep=1000, return_mode='median')
            test_scores, test_indices = bitonic_fn()
            
            # Correctness check
            overlap = 0.0
            for i in range(batch_size):
                ref_set = set(ref_indices[i].cpu().tolist())
                test_set = set(test_indices[i].cpu().tolist())
                overlap += len(ref_set & test_set) / top_k
            overlap /= batch_size
            
            speedup = pytorch_ms / bitonic_ms
            status = "✓" if overlap > 0.99 else f"⚠{overlap:.0%}"
            
            print(f"{config_str:<25} | {pytorch_ms:>8.4f}ms | {bitonic_ms:>8.4f}ms | {speedup:>6.2f}x | {status}")
            
        except Exception as e:
            print(f"{config_str:<25} | {pytorch_ms:>8.4f}ms | {'ERROR':>10} | {'N/A':>8} | {str(e)[:30]}")
    
    print("\n" + "=" * 80)
    print("Note: Bitonic sort is O(N log²N) parallel vs PyTorch topk O(N log k)")
    print("Best for: small to medium num_experts (≤512), fusion reduces memory traffic")
    print("=" * 80)
