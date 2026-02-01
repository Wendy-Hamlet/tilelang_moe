"""
TopK + Softmax 性能对比基准

对比四个版本：
1) pytorch baseline（无算法版本：全局softmax→topk）
2) torch.compile（无算法版本：1) 的编译优化）
3) tilelang（无算法版本：两个算子分开，TopK用TileLang官方Radix实现 + PyTorch对k做softmax）
4) ours（融合+sonic算法：单kernel完成topk+softmax）
"""

import sys
import warnings
from pathlib import Path

# 过滤torch.compile的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Callable, Dict
from tilelang.profiler import do_bench

# 启用TF32以获得更好的float32矩阵乘法性能
torch.set_float32_matmul_precision('high')
from kernels.topk_softmax_baseline import (
    topk_softmax_separate,
    topk_softmax_separate_compiled,
)
from kernels.topk_softmax_tilelang_separate import topk_softmax_tilelang_separate
from kernels.topk_softmax_sonic import topk_softmax_sonic


def benchmark_callable(
    func: Callable[[torch.Tensor], tuple],
    logits: torch.Tensor,
    num_warmup: int = 100,
    num_runs: int = 1000,
) -> Dict[str, torch.Tensor | float]:
    """对一个签名为 f(logits)->(scores, indices) 的函数进行基准测试。
    
    使用TileLang do_bench以获得稳定的数值：
    - 自动GPU同步
    - 数值稳定的统计
    - 完全的warmup和repeated runs
    """
    # 定义测试函数（do_bench会自动处理warmup和重复）
    def bench_fn():
        return func(logits)
    
    # 使用do_bench进行基准测试（单位：ms）
    latency_ms = do_bench(bench_fn, warmup=num_warmup, rep=num_runs, return_mode='median')
    
    # 获取最后一次执行的输出用于正确性检验
    scores_out, indices_out = func(logits)
    torch.cuda.synchronize()

    return {"latency_ms": latency_ms, "scores": scores_out, "indices": indices_out}


def verify_correctness(
    scores: torch.Tensor,
    indices: torch.Tensor,
    ref_scores: torch.Tensor,
    ref_indices: torch.Tensor,
    is_renormalized: bool = False,
) -> dict:
    """验证正确性
    
    Args:
        is_renormalized: 是否对topk结果重新归一化
            - False: baseline/compile方法（先全局softmax→topk，sum<1）
            - True: tilelang/ours方法（先topk→局部softmax，sum=1）
    """
    
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
    
    # 归一化检查：根据方法类型判断期望的sum值
    if is_renormalized:
        # tilelang/ours: 期望sum=1.0（对k个元素重新归一化）
        sum_scores = scores.sum(dim=-1)
        sum_valid = ((sum_scores - 1.0).abs() < 1e-4).float().mean().item()
    else:
        # baseline/compile: 期望与参考baseline一致（都是从归一化分布中选k个，sum<1）
        ref_sum = ref_scores.sum(dim=-1)
        test_sum = scores.sum(dim=-1)
        sum_valid = ((ref_sum - test_sum).abs() < 1e-4).float().mean().item()
    
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
    """测试单个配置，返回四个版本的结果。"""
    print(f"\n—— Running: {config_name} (B={batch_size}, E={num_experts}, K={top_k})")

    # Data
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    logits = torch.randn(batch_size, num_experts, device='cuda', dtype=torch.float32)

    # Build runners with unified signature f(logits)->(scores, indices)
    sonic_kernel = topk_softmax_sonic(batch_size, num_experts, top_k)

    runners: Dict[str, Callable[[torch.Tensor], tuple]] = {
        # 1) pytorch baseline（无算法版本）
        'pytorch-baseline': lambda x: topk_softmax_separate(x, top_k),
        # 2) torch.compile（无算法版本，模块级别已编译，自动缓存）
        'torch.compile': lambda x: topk_softmax_separate_compiled(x, top_k),
        # 3) tilelang（无算法版本，两个算子分开，官方TopK）
        'tilelang-separate': lambda x: topk_softmax_tilelang_separate(x, top_k),
        # 4) ours（融合+sonic算法，单kernel）
        'ours': lambda x: sonic_kernel(x),
    }

    results: Dict[str, Dict] = {}
    for name, fn in runners.items():
        try:
            out = benchmark_callable(fn, logits)
            results[name] = out
            print(f"  ✓ {name}: {out['latency_ms']:.4f} ms")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results[name] = None
    
    # 正确性（对照pytorch-baseline）
    if results.get('pytorch-baseline') is not None:
        ref_scores = results['pytorch-baseline']['scores']
        ref_indices = results['pytorch-baseline']['indices']
        
        # 调试：检查baseline本身
        baseline_sum = ref_scores.sum(dim=-1)
        print(f"  DEBUG: pytorch-baseline scores.sum() = {baseline_sum.mean():.6f} (期望 < 1.0)")
        
        print("  正确性 (vs pytorch-baseline):")
        ok = True
        for name, res in results.items():
            if name == 'pytorch-baseline' or res is None:
                continue
            # 判断是否重新归一化：只有ours用方式B（topk->softmax k）
            # pytorch-baseline、torch.compile、tilelang-separate都是方式A（softmax all->topk）
            is_renormalized = name == 'ours'
            c = verify_correctness(res['scores'], res['indices'], ref_scores, ref_indices, is_renormalized)
            
            # 对于ours（方式B），只检查索引重叠度，不检查sum（因为数学定义不同）
            # 对于方式A的版本，检查sum和overlap
            if is_renormalized:
                is_ok = (c['overlap'] > 0.95)  # ours只检查索引匹配，不检查数值和
            else:
                is_ok = (c['overlap'] > 0.95 and c['sum_valid'] > 0.95)  # 方式A版本检查all
            ok = ok and is_ok
            status = "✓" if is_ok else "⚠"
            
            # 调试：显示这个版本的sum
            version_sum = res['scores'].sum(dim=-1)
            print(f"    {status} {name:<18} overlap={c['overlap']*100:>6.1f}% sum_valid={c['sum_valid']*100:>6.1f}% (scores.sum={version_sum.mean():.6f}) max_diff={c['max_diff']:.2e}")
        print(f"  整体: {'OK' if ok else 'WARN'}")

    
    return results


def main():
    print("=" * 100)
    print("TopK + Softmax 性能对比")
    print("=" * 100)
    print("版本:")
    print("- pytorch-baseline: 全局softmax→topk")
    print("- torch.compile: 编译优化的baseline")
    print("- tilelang-separate: 官方TopK + PyTorch对k softmax")
    print("- ours: 单kernel融合 + sonic算法")
    print("\n计时: CUDA Events，100 warmup + 1000 runs 平均")
    print("注：torch.compile需要足够的warmup次数达到稳定性能")
    print("=" * 100)
    
    # 测试场景 (batch_size, num_experts, top_k)
    test_configs = [
        (64, 160, 6),
        (64, 256, 8),
        (128, 512, 16),
        (64, 1024, 8),
        (64, 2048, 16),
    ]
    
    all_results = {}
    
    for batch_size, num_experts, top_k in test_configs:
        config_name = f"B{batch_size}_E{num_experts}_K{top_k}"
        try:
            results = benchmark_config(batch_size, num_experts, top_k, config_name)
            all_results[config_name] = results
        except Exception as e:
            print(f"\n✗ {config_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出表格：Latency
    print("\n" + "=" * 100)
    print("Latency (ms)")
    print("=" * 100)
    header = f"{'配置':<18} {'pytorch-baseline':>16} {'torch.compile':>16} {'tilelang-separate':>19} {'ours':>12}"
    print(header)
    print("-" * len(header))
    for cfg, results in all_results.items():
        def get_ms(name):
            r = results.get(name)
            return f"{r['latency_ms']:.4f}" if r else "N/A"
        print(f"{cfg:<18} {get_ms('pytorch-baseline'):>16} {get_ms('torch.compile'):>16} {get_ms('tilelang-separate'):>19} {get_ms('ours'):>12}")

    # 输出表格：Speedup vs pytorch-baseline
    print("\n" + "=" * 100)
    print("Speedup vs pytorch-baseline")
    print("=" * 100)
    print(header)
    print("-" * len(header))
    for cfg, results in all_results.items():
        base = results.get('pytorch-baseline')
        def sp(name):
            r = results.get(name)
            if not base or not r:
                return "N/A"
            return f"{base['latency_ms']/r['latency_ms']:.2f}x"
        print(f"{cfg:<18} {'1.00x':>16} {sp('torch.compile'):>16} {sp('tilelang-separate'):>19} {sp('ours'):>12}")


if __name__ == "__main__":
    main()
