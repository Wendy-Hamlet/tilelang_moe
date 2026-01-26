"""
TopK+Softmax性能对比测试

对比三个实现：
1. PyTorch Baseline (分离版本)
2. PyTorch Fused (融合版本)
3. TileLang Sonic (Sonic优化)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from kernels.topk_softmax_baseline import (
    topk_softmax_separate,
    topk_softmax_fused,
    topk_softmax_fused_compiled,
)
from kernels.topk_softmax_sonic import topk_softmax_sonic


def benchmark_version(
    name: str,
    func,
    logits: torch.Tensor,
    k: int,
    num_warmup: int = 50,
    num_runs: int = 1000,
    **kwargs
) -> dict:
    """
    Benchmark单个版本
    
    Returns:
        dict with 'latency_ms', 'scores', 'indices'
    """
    device = logits.device
    batch_size, num_experts = logits.shape
    
    # Warmup
    for _ in range(num_warmup):
        if 'tilelang' in name.lower():
            # TileLang kernel with out_idx returns outputs automatically
            _ = func(logits)
        else:
            _ = func(logits, k, **kwargs)
    torch.cuda.synchronize()
    
    # Benchmark using CUDA events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        if 'tilelang' in name.lower():
            # TileLang kernel with out_idx returns outputs automatically
            scores_out, indices_out = func(logits)
        else:
            scores_out, indices_out = func(logits, k, **kwargs)
    end_event.record()
    
    torch.cuda.synchronize()
    latency_ms = start_event.elapsed_time(end_event) / num_runs
    
    return {
        'latency_ms': latency_ms,
        'scores': scores_out,
        'indices': indices_out,
    }


def verify_correctness(
    scores: torch.Tensor,
    indices: torch.Tensor,
    ref_scores: torch.Tensor,
    ref_indices: torch.Tensor,
) -> dict:
    """验证正确性"""
    
    # 索引匹配率（考虑相同值可能有不同索引）
    batch_size, k = indices.shape
    
    # 对每个batch检查overlap
    overlaps = []
    for i in range(batch_size):
        ref_set = set(ref_indices[i].cpu().tolist())
        test_set = set(indices[i].cpu().tolist())
        overlap = len(ref_set & test_set) / k
        overlaps.append(overlap)
    
    avg_overlap = sum(overlaps) / len(overlaps)
    
    # 分数差异
    score_diff = (scores.float() - ref_scores.float()).abs()
    max_diff = score_diff.max().item()
    mean_diff = score_diff.mean().item()
    
    # 归一化检查
    sum_scores = scores.sum(dim=-1)
    sum_valid = ((sum_scores - 1.0).abs() < 1e-4).float().mean().item()
    
    return {
        'overlap': avg_overlap,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'sum_valid': sum_valid,
    }


def benchmark_config(
    batch_size: int,
    num_experts: int,
    top_k: int,
    config_name: str,
):
    """测试单个配置"""
    print(f"\n{'=' * 100}")
    print(f"{config_name}: batch={batch_size}, experts={num_experts}, k={top_k}")
    print(f"{'=' * 100}")
    
    # 准备数据
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    logits = torch.randn(batch_size, num_experts, device='cuda', dtype=torch.float32)
    
    results = {}
    
    # 1. PyTorch Baseline (分离)
    print("\n[1/4] PyTorch Baseline (Separate)...")
    try:
        results['baseline'] = benchmark_version(
            "PyTorch Baseline",
            topk_softmax_separate,
            logits,
            top_k,
        )
        print(f"  ✓ Latency: {results['baseline']['latency_ms']:.4f} ms")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['baseline'] = None
    
    # 2. PyTorch Fused
    print("\n[2/4] PyTorch Fused...")
    try:
        results['fused'] = benchmark_version(
            "PyTorch Fused",
            topk_softmax_fused,
            logits,
            top_k,
        )
        print(f"  ✓ Latency: {results['fused']['latency_ms']:.4f} ms")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['fused'] = None
    
    # 3. PyTorch Compiled
    print("\n[3/4] PyTorch Compiled...")
    try:
        results['compiled'] = benchmark_version(
            "PyTorch Compiled",
            topk_softmax_fused_compiled,
            logits,
            top_k,
        )
        print(f"  ✓ Latency: {results['compiled']['latency_ms']:.4f} ms")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results['compiled'] = None
    
    # 4. TileLang Sonic
    print("\n[4/4] TileLang Sonic...")
    try:
        kernel = topk_softmax_sonic(batch_size, num_experts, top_k)
        results['sonic'] = benchmark_version(
            "TileLang Sonic",
            kernel,
            logits,
            top_k,
        )
        print(f"  ✓ Latency: {results['sonic']['latency_ms']:.4f} ms")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results['sonic'] = None
    
    # 验证正确性
    if results['baseline'] is not None:
        ref_scores = results['baseline']['scores']
        ref_indices = results['baseline']['indices']
        
        print(f"\n{'─' * 100}")
        print("正确性验证 (vs Baseline)")
        print(f"{'─' * 100}")
        print(f"{'版本':<20} {'Overlap':<12} {'MaxDiff':<12} {'MeanDiff':<12} {'SumValid':<12}")
        print(f"{'─' * 100}")
        
        for name, result in results.items():
            if result is not None and name != 'baseline':
                correctness = verify_correctness(
                    result['scores'],
                    result['indices'],
                    ref_scores,
                    ref_indices,
                )
                print(f"{name:<20} {correctness['overlap']*100:>10.2f}% "
                      f"{correctness['max_diff']:>10.2e} "
                      f"{correctness['mean_diff']:>10.2e} "
                      f"{correctness['sum_valid']*100:>10.2f}%")
    
    # 性能总结
    print(f"\n{'─' * 100}")
    print("性能总结")
    print(f"{'─' * 100}")
    print(f"{'版本':<20} {'Latency (ms)':<15} {'加速比':<12} {'状态':<10}")
    print(f"{'─' * 100}")
    
    baseline_time = results['baseline']['latency_ms'] if results['baseline'] else None
    
    for name, result in results.items():
        if result is not None:
            speedup = baseline_time / result['latency_ms'] if baseline_time else 1.0
            status = "✓"
            print(f"{name:<20} {result['latency_ms']:<15.4f} {speedup:<12.2f}x {status:<10}")
        else:
            print(f"{name:<20} {'N/A':<15} {'N/A':<12} {'✗':<10}")
    
    print(f"{'=' * 100}\n")
    
    return results


def main():
    print("=" * 100)
    print("TopK + Softmax 性能对比测试")
    print("=" * 100)
    print("\n对比版本:")
    print("  1. PyTorch Baseline    - 先全局softmax再topk")
    print("  2. PyTorch Fused       - 先topk再局部softmax（Sonic策略）")
    print("  3. PyTorch Compiled    - torch.compile优化的融合版本")
    print("  4. TileLang Sonic      - TileLang手写融合kernel")
    print("\n测试配置: 50次预热 + 1000次测试取平均")
    print("=" * 100)
    
    # 测试场景
    test_configs = [
        (64, 160, 6, "DeepSeek-V2 (小规模)"),
        (64, 256, 8, "DeepSeek-V3 (标准)"),
        (128, 512, 16, "大规模"),
    ]
    
    all_results = {}
    
    for batch_size, num_experts, top_k, config_name in test_configs:
        try:
            results = benchmark_config(batch_size, num_experts, top_k, config_name)
            all_results[config_name] = results
        except Exception as e:
            print(f"\n✗ {config_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 最终总结
    print("\n" + "=" * 100)
    print("跨配置性能对比")
    print("=" * 100)
    print(f"{'配置':<25} {'Baseline':<12} {'Fused':<12} {'Compiled':<12} {'Sonic':<12}")
    print("=" * 100)
    
    for config_name, results in all_results.items():
        row = f"{config_name:<25}"
        for key in ['baseline', 'fused', 'compiled', 'sonic']:
            if results.get(key):
                row += f"{results[key]['latency_ms']:<12.4f}"
            else:
                row += f"{'N/A':<12}"
        print(row)
    
    print("=" * 100)
    
    # 最佳实现总结
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    print("预期性能排名（基于Sonic论文）:")
    print("  1. TileLang Sonic    - 3.0x vs Baseline (融合+优化算法)")
    print("  2. PyTorch Compiled  - 1.5x vs Baseline (编译器优化)")
    print("  3. PyTorch Fused     - 1.2x vs Baseline (减少内存访问)")
    print("  4. PyTorch Baseline  - 1.0x (基准)")
    print("=" * 100)


if __name__ == "__main__":
    main()
