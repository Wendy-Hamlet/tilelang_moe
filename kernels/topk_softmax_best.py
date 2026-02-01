"""
TopK+Softmax最佳实现：自动选择策略

根据num_experts大小自动选择最佳实现：
- 小num_experts (<=1024): 使用Sonic融合kernel
- 大num_experts (>1024): 使用Radix TopK + PyTorch Softmax

原因：
- Sonic使用选择排序O(k*N/T)，在小N时高效
- Radix使用基数选择O(N*log(N))，在大N时更高效
"""

import torch
from typing import Tuple

# 延迟导入以支持直接运行
def _get_sonic_kernel():
    from .topk_softmax_sonic import topk_softmax_sonic
    return topk_softmax_sonic

def _get_radix_topk():
    from .topk_radix import topk_radix_tilelang
    return topk_radix_tilelang


# 缓存编译好的kernel
_sonic_kernel_cache = {}


def topk_softmax_best(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    自动选择最佳TopK+Softmax实现
    
    Args:
        logits: (batch, num_experts) float32
        k: top-k count
    Returns:
        (scores, indices): both shape (batch, k)
    """
    batch_size, num_experts = logits.shape
    
    # 根据num_experts选择策略
    # 阈值基于benchmark结果：
    # - num_experts <= 1024: Sonic更快 (1.0x-2.1x vs PyTorch)
    # - num_experts > 1024 且 <= 2048: Sonic仍略好于Radix，但两者都比PyTorch慢
    # - num_experts > 2048: Radix更快
    # 综合考虑，使用2048作为阈值（Sonic在<=2048时总是优于或等于Radix）
    if num_experts <= 2048:
        # 使用Sonic融合kernel
        topk_softmax_sonic = _get_sonic_kernel()
        cache_key = (batch_size, num_experts, k)
        if cache_key not in _sonic_kernel_cache:
            _sonic_kernel_cache[cache_key] = topk_softmax_sonic(batch_size, num_experts, k)
        
        kernel = _sonic_kernel_cache[cache_key]
        scores, indices = kernel(logits)
        return scores, indices
    else:
        # 使用Radix TopK + PyTorch Softmax
        topk_radix_tilelang = _get_radix_topk()
        
        # 1. Radix TopK
        topk_indices = topk_radix_tilelang(logits, k)
        
        # 2. Gather top-k values
        topk_values = torch.gather(logits, dim=-1, index=topk_indices.to(torch.int64))
        
        # 3. Softmax on top-k values
        scores = torch.softmax(topk_values, dim=-1, dtype=torch.float32)
        
        return scores, topk_indices.to(torch.int32)


# 导出为默认实现
topk_softmax = topk_softmax_best


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # 重新定义导入函数以支持直接运行
    def _get_sonic_kernel():
        from kernels.topk_softmax_sonic import topk_softmax_sonic
        return topk_softmax_sonic
    
    def _get_radix_topk():
        from kernels.topk_radix import topk_radix_tilelang
        return topk_radix_tilelang
    
    from tilelang.profiler import do_bench
    
    print("TopK+Softmax最佳实现测试")
    print("=" * 80)
    
    configs = [
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
        
        # 验证正确性
        ref_scores, ref_indices = pytorch_fn()
        test_scores, test_indices = best_fn()
        
        # 检查索引重叠
        overlap = 0
        for i in range(batch_size):
            ref_set = set(ref_indices[i].cpu().tolist())
            test_set = set(test_indices[i].cpu().tolist())
            overlap += len(ref_set & test_set) / top_k
        overlap /= batch_size
        
        strategy = "Sonic" if num_experts <= 2048 else "Radix"
        speedup = pytorch_ms / best_ms
        status = "✓" if overlap > 0.95 else "✗"
        
        print(f"B{batch_size}_E{num_experts}_K{top_k:<3} | {pytorch_ms:>8.4f}ms | {best_ms:>8.4f}ms | {speedup:>8.2f}x | {strategy} {status}")
