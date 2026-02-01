"""
TopK+Softmax Best Implementation: Auto-selection Strategy

Automatically selects the best implementation based on num_experts:
- num_experts <= 1024: Bitonic Sort fusion kernel (best parallelism, single kernel)
- num_experts > 1024: Radix TopK + PyTorch Softmax (better for large N)

Bitonic Sort advantages:
- Fully parallel O(N * log²N) sort
- Single fused kernel (ONE read + ONE write to global memory)
- Better occupancy with many parallel threads
"""

import torch
from typing import Tuple


def _get_bitonic_kernel():
    from .topk_softmax_bitonic import topk_softmax_bitonic
    return topk_softmax_bitonic


def _get_radix_topk():
    from .topk_radix import topk_radix_tilelang
    return topk_radix_tilelang


def topk_softmax_best(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Auto-select best TopK+Softmax implementation
    
    Args:
        logits: (batch, num_experts) float32
        k: top-k count
    Returns:
        (scores, indices): both shape (batch, k)
    """
    batch_size, num_experts = logits.shape
    
    # Strategy selection based on benchmark results:
    # - num_experts <= 1024: Bitonic Sort fusion (1.0x-2.0x speedup vs PyTorch)
    # - num_experts > 1024: Radix TopK + Softmax (handles large N efficiently)
    if num_experts <= 1024:
        # Use Bitonic Sort fusion kernel
        topk_softmax_bitonic = _get_bitonic_kernel()
        return topk_softmax_bitonic(logits, k)
    else:
        # Use Radix TopK + PyTorch Softmax
        topk_radix_tilelang = _get_radix_topk()
        
        # 1. Radix TopK
        topk_indices = topk_radix_tilelang(logits, k)
        
        # 2. Gather top-k values
        topk_values = torch.gather(logits, dim=-1, index=topk_indices.to(torch.int64))
        
        # 3. Softmax on top-k values
        scores = torch.softmax(topk_values, dim=-1, dtype=torch.float32)
        
        return scores, topk_indices.to(torch.int32)


# Export as default implementation
topk_softmax = topk_softmax_best


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Redefine imports for direct execution
    def _get_bitonic_kernel():
        from kernels.topk_softmax_bitonic import topk_softmax_bitonic
        return topk_softmax_bitonic
    
    def _get_radix_topk():
        from kernels.topk_radix import topk_radix_tilelang
        return topk_radix_tilelang
    
    from tilelang.profiler import do_bench
    
    torch.set_float32_matmul_precision('high')
    
    print("TopK+Softmax Best Implementation Test")
    print("=" * 80)
    
    configs = [
        (64, 64, 8),
        (64, 160, 6),
        (64, 256, 8),
        (128, 512, 16),
        (64, 1024, 8),
        (64, 2048, 16),
        (64, 4096, 8),
    ]
    
    print(f"{'Config':<20} | {'PyTorch':>10} | {'Best':>10} | {'Speedup':>10} | Strategy")
    print("-" * 75)
    
    for batch_size, num_experts, top_k in configs:
        torch.manual_seed(42)
        logits = torch.randn(batch_size, num_experts, device='cuda', dtype=torch.float32)
        
        # PyTorch baseline
        def pytorch_fn():
            topk = torch.topk(logits, top_k, dim=-1)
            return torch.softmax(topk.values, dim=-1), topk.indices
        
        # Best
        def best_fn():
            return topk_softmax_best(logits, top_k)
        
        pytorch_ms = do_bench(pytorch_fn, warmup=100, rep=1000, return_mode='median')
        best_ms = do_bench(best_fn, warmup=100, rep=1000, return_mode='median')
        
        # Correctness check
        ref_scores, ref_indices = pytorch_fn()
        test_scores, test_indices = best_fn()
        
        # Check index overlap
        overlap = 0
        for i in range(batch_size):
            ref_set = set(ref_indices[i].cpu().tolist())
            test_set = set(test_indices[i].cpu().tolist())
            overlap += len(ref_set & test_set) / top_k
        overlap /= batch_size
        
        strategy = "Bitonic" if num_experts <= 1024 else "Radix"
        speedup = pytorch_ms / best_ms
        status = "✓" if overlap > 0.95 else "✗"
        
        print(f"B{batch_size}_E{num_experts}_K{top_k:<3} | {pytorch_ms:>8.4f}ms | {best_ms:>8.4f}ms | {speedup:>8.2f}x | {strategy} {status}")
