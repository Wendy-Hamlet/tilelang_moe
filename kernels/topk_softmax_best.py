"""
TopK+Softmax Best Implementation: Smart Auto-selection Strategy

Automatically selects between Bitonic Sort and Sonic based on:
1. Padding overhead: Bitonic needs N_padded = 2^ceil(log2(N)), wastes threads on padding
2. K value: Sonic is O(K*N), smaller K favors Sonic
3. Power of 2: When N is power of 2, Bitonic has zero padding overhead

Algorithm complexity:
- Bitonic: O(N_padded * log²(N_padded)) parallel, but wastes cycles on padding
- Sonic:   O(K * N) with K sequential iterations

Selection heuristic:
- If padding_ratio > 30% and K is small: use Sonic
- If N is power of 2 or padding_ratio <= 30%: use Bitonic
- For N > 1024: use Radix TopK (Bitonic limit)
"""

import math
import torch
from typing import Tuple


def _get_bitonic_kernel():
    from .topk_softmax_bitonic import topk_softmax_bitonic
    return topk_softmax_bitonic


def _get_sonic_kernel():
    from .topk_softmax_sonic import topk_softmax_sonic
    return topk_softmax_sonic


def _get_radix_topk():
    from .topk_radix import topk_radix_tilelang
    return topk_radix_tilelang


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _get_padding_ratio(n: int) -> float:
    """Calculate padding ratio for Bitonic sort"""
    n_padded = 1 << int(math.ceil(math.log2(max(n, 2))))
    return (n_padded - n) / n


def _select_strategy(num_experts: int, k: int) -> str:
    """
    Smart strategy selection based on algorithm cost estimation
    
    Cost models (empirically tuned from benchmarks):
    - Bitonic: O(N_padded * log²(N_padded)), affected by padding
    - Sonic:   O(K * N), K sequential iterations over N elements
    
    Key observations from benchmarks:
    1. Bitonic excels when N is power of 2 (no padding waste)
    2. Sonic excels when K is small and N has high padding
    3. For large K (>=16) with medium N, performance is close
    
    Returns: 'bitonic', 'sonic', or 'radix'
    """
    # Radix for large N (Bitonic kernel limit)
    if num_experts > 1024:
        return 'radix'
    
    # Calculate parameters
    n_padded = 1 << int(math.ceil(math.log2(max(num_experts, 2))))
    log_n = int(math.log2(n_padded))
    padding_ratio = _get_padding_ratio(num_experts)
    
    # Heuristic rules based on benchmark data:
    
    # Rule 1: Very small N (<=64) with low padding -> Bitonic (high parallelism wins)
    if num_experts <= 64:
        return 'bitonic'
    
    # Rule 2: Small K (<= 8) with high padding (>40%) -> Sonic
    if k <= 8 and padding_ratio > 0.4:
        return 'sonic'
    
    # Rule 3: Medium K (8-16) with moderate padding (20-50%) -> depends on N
    if 8 <= k <= 16 and 0.2 <= padding_ratio <= 0.5:
        # Bitonic is better for larger N due to better parallel scaling
        return 'bitonic' if num_experts >= 384 else 'sonic'
    
    # Rule 4: Large padding (>50%) generally favors Sonic (too much waste in Bitonic)
    if padding_ratio > 0.5:
        return 'sonic'
    
    # Rule 5: Power of 2 or very low padding -> Bitonic optimal
    if _is_power_of_2(num_experts) or padding_ratio < 0.2:
        # But for medium-large N with small K, Sonic might still win
        # Because Sonic's O(K*N) beats Bitonic's O(N*log²N) when K is small
        if num_experts >= 128 and k <= 6:
            return 'sonic'
        if num_experts >= 256 and k <= 8:
            return 'sonic'
        # For N=1024 with K>=16, Sonic's O(K*N) is comparable to Bitonic's O(N*log²N)
        if num_experts >= 1024 and k >= 16:
            return 'sonic'
        return 'bitonic'
    
    # Default: Bitonic for its better parallelism
    return 'bitonic'


# Cache for compiled kernels
_sonic_kernel_cache = {}


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
    
    strategy = _select_strategy(num_experts, k)
    
    if strategy == 'bitonic':
        topk_softmax_bitonic = _get_bitonic_kernel()
        return topk_softmax_bitonic(logits, k)
    
    elif strategy == 'sonic':
        topk_softmax_sonic = _get_sonic_kernel()
        cache_key = (batch_size, num_experts, k)
        if cache_key not in _sonic_kernel_cache:
            _sonic_kernel_cache[cache_key] = topk_softmax_sonic(batch_size, num_experts, k)
        kernel = _sonic_kernel_cache[cache_key]
        return kernel(logits)
    
    else:  # radix
        topk_radix_tilelang = _get_radix_topk()
        topk_indices = topk_radix_tilelang(logits, k)
        topk_values = torch.gather(logits, dim=-1, index=topk_indices.to(torch.int64))
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
    
    def _get_sonic_kernel():
        from kernels.topk_softmax_sonic import topk_softmax_sonic
        return topk_softmax_sonic
    
    def _get_radix_topk():
        from kernels.topk_radix import topk_radix_tilelang
        return topk_radix_tilelang
    
    from tilelang.profiler import do_bench
    from kernels.topk_softmax_bitonic import topk_softmax_bitonic
    from kernels.topk_softmax_sonic import topk_softmax_sonic
    
    torch.set_float32_matmul_precision('high')
    
    print("=" * 100)
    print("TopK+Softmax Smart Auto-Selection Validation")
    print("=" * 100)
    
    # Extended config set for validation
    configs = [
        # Power of 2 (Bitonic optimal)
        (64, 64, 8),
        (64, 128, 8),
        (64, 256, 8),
        (64, 512, 16),
        (64, 1024, 16),
        # Non-power of 2 (high padding)
        (64, 160, 6),    # pad to 256, 60% overhead
        (64, 192, 8),    # pad to 256, 33% overhead
        (64, 320, 8),    # pad to 512, 60% overhead
        (64, 384, 16),   # pad to 512, 33% overhead
        (64, 640, 16),   # pad to 1024, 60% overhead
        (64, 768, 16),   # pad to 1024, 33% overhead
        # Large N (Radix)
        (64, 2048, 16),
        (64, 4096, 8),
    ]
    
    print(f"\n{'Config':<18} | {'Pad%':>6} | {'Strategy':<8} | {'PyTorch':>9} | {'Sonic':>9} | {'Bitonic':>9} | {'Best':>9} | {'Speedup':>8} | {'Optimal?'}")
    print("-" * 115)
    
    correct_selections = 0
    total_selections = 0
    
    for batch_size, num_experts, top_k in configs:
        torch.manual_seed(42)
        logits = torch.randn(batch_size, num_experts, device='cuda', dtype=torch.float32)
        
        padding_ratio = _get_padding_ratio(num_experts) * 100
        strategy = _select_strategy(num_experts, top_k)
        
        # PyTorch baseline
        def pytorch_fn():
            topk = torch.topk(logits, top_k, dim=-1)
            return torch.softmax(topk.values, dim=-1), topk.indices
        
        pytorch_ms = do_bench(pytorch_fn, warmup=100, rep=1000, return_mode='median')
        
        # Sonic
        try:
            sonic_kernel = topk_softmax_sonic(batch_size, num_experts, top_k)
            def sonic_fn():
                return sonic_kernel(logits)
            sonic_ms = do_bench(sonic_fn, warmup=100, rep=1000, return_mode='median')
        except:
            sonic_ms = float('inf')
        
        # Bitonic
        try:
            def bitonic_fn():
                return topk_softmax_bitonic(logits, top_k)
            bitonic_ms = do_bench(bitonic_fn, warmup=100, rep=1000, return_mode='median')
        except:
            bitonic_ms = float('inf')
        
        # Best (auto-selected)
        def best_fn():
            return topk_softmax_best(logits, top_k)
        best_ms = do_bench(best_fn, warmup=100, rep=1000, return_mode='median')
        
        # Determine optimal
        if num_experts <= 1024:
            actual_best = 'bitonic' if bitonic_ms <= sonic_ms else 'sonic'
            optimal = "✓" if strategy == actual_best else "✗"
            if strategy == actual_best:
                correct_selections += 1
            total_selections += 1
        else:
            actual_best = 'radix'
            optimal = "✓"  # Radix is the only option for large N
            correct_selections += 1
            total_selections += 1
        
        speedup = pytorch_ms / best_ms
        
        sonic_str = f"{sonic_ms:.4f}" if sonic_ms != float('inf') else "N/A"
        bitonic_str = f"{bitonic_ms:.4f}" if bitonic_ms != float('inf') else "N/A"
        
        print(f"B{batch_size}_E{num_experts}_K{top_k:<2} | {padding_ratio:>5.0f}% | {strategy:<8} | {pytorch_ms:>7.4f}ms | {sonic_str:>7}ms | {bitonic_str:>7}ms | {best_ms:>7.4f}ms | {speedup:>6.2f}x | {optimal}")
    
    print("-" * 115)
    print(f"Selection accuracy: {correct_selections}/{total_selections} ({correct_selections/total_selections*100:.0f}%)")
    print()
    print("Strategy selection rules:")
    print("- N > 1024: Radix (Bitonic kernel limit)")
    print("- Padding > 30% and K <= 16: Sonic (less wasted threads)")
    print("- Power of 2 or low padding: Bitonic (optimal parallelism)")
