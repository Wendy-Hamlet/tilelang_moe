"""
TopK + Softmax 性能对比基准 - 官方配置

只测试TileLang官方示例中的两个配置：
1. DeepSeek V2 (example_topk.py): M=320 tokens, N=128 experts, K=6
2. DeepSeek V3 (example_fusedmoe.py): M=8192 tokens, N=8 experts, K=4

对比四个版本：
1) PyTorch: 标准实现（全局softmax→topk）
2) torch.compile: JIT编译优化
3) TileLang-Official: 官方TopK + PyTorch对K做softmax（分离版本）
4) Ours: 融合kernel + Sonic算法（topk→softmax）
"""

import sys
import warnings
from pathlib import Path

# 过滤torch.compile的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Callable, Dict
from tabulate import tabulate
from tilelang.profiler import do_bench

# 启用TF32以获得更好的float32矩阵乘法性能
torch.set_float32_matmul_precision('high')

from kernels.topk_softmax_baseline import (
    topk_softmax_separate,
    topk_softmax_separate_compiled,
)
from kernels.topk_softmax_tilelang import (
    topk_softmax_tilelang_separate,
    topk_softmax_best,
)


# ============ 官方配置 ============
# 来源: tilelang/examples/
OFFICIAL_CONFIGS = [
    {
        'name': 'DeepSeek-V2',
        'source': 'example_topk.py',
        'M': 320,      # num_tokens
        'N': 128,      # num_experts  
        'K': 6,        # topk
        'desc': '320 tokens × 128 experts → top-6',
    },
    {
        'name': 'DeepSeek-V3',
        'source': 'example_fusedmoe.py',
        'M': 8192,     # batch_size * seq_len = 1 * 8192
        'N': 8,        # n_routed_experts
        'K': 4,        # n_experts_per_token
        'desc': '8192 tokens × 8 experts → top-4',
    },
]


def benchmark_config(M: int, N: int, K: int, config_name: str, warmup: int = 100, rep: int = 1000):
    """测试单个配置，返回四个版本的结果。"""
    
    # Data
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    logits = torch.randn(M, N, device='cuda', dtype=torch.float32)

    # 四个版本
    versions = {
        'PyTorch': lambda x: topk_softmax_separate(x, K),
        'torch.compile': lambda x: topk_softmax_separate_compiled(x, K),
        'TL-Official': lambda x: topk_softmax_tilelang_separate(x, K),
        'Ours': lambda x: topk_softmax_best(x, K),
    }

    results = {}
    for name, fn in versions.items():
        try:
            # Warmup
            _ = fn(logits)
            torch.cuda.synchronize()
            
            # Benchmark
            def bench_fn():
                return fn(logits)
            
            latency_ms = do_bench(bench_fn, warmup=warmup, rep=rep, return_mode='median')
            results[name] = latency_ms
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results[name] = None
    
    return results


def main():
    print("=" * 100)
    print("TopK + Softmax 官方配置性能对比")
    print("=" * 100)
    print()
    print("配置来源:")
    for cfg in OFFICIAL_CONFIGS:
        print(f"  - {cfg['name']}: {cfg['source']} ({cfg['desc']})")
    print()
    print("版本说明:")
    print("  - PyTorch:      标准实现 (softmax→topk)")
    print("  - torch.compile: JIT编译优化")
    print("  - TL-Official:  TileLang官方TopK + PyTorch softmax (分离)")
    print("  - Ours:         融合kernel + Sonic算法 (topk→softmax)")
    print()
    print("=" * 100)
    
    all_results = {}
    
    for cfg in OFFICIAL_CONFIGS:
        print(f"\n▶ Testing {cfg['name']} ({cfg['desc']})...")
        results = benchmark_config(cfg['M'], cfg['N'], cfg['K'], cfg['name'])
        all_results[cfg['name']] = {
            'config': cfg,
            'results': results,
        }
        
        # 打印单个配置结果
        for name, latency in results.items():
            if latency is not None:
                print(f"    {name:<14}: {latency:.4f} ms")
            else:
                print(f"    {name:<14}: FAILED")
    
    # ============ 表格1: 延迟对比 ============
    print("\n" + "=" * 100)
    print("表格1: 延迟对比 (ms)")
    print("=" * 100)
    
    latency_table = []
    for name, data in all_results.items():
        cfg = data['config']
        res = data['results']
        row = [
            f"{cfg['name']} (M={cfg['M']}, N={cfg['N']}, K={cfg['K']})",
            f"{res['PyTorch']:.4f}" if res['PyTorch'] else "N/A",
            f"{res['torch.compile']:.4f}" if res['torch.compile'] else "N/A",
            f"{res['TL-Official']:.4f}" if res['TL-Official'] else "N/A",
            f"{res['Ours']:.4f}" if res['Ours'] else "N/A",
        ]
        latency_table.append(row)
    
    headers = ["配置", "PyTorch", "torch.compile", "TL-Official", "Ours"]
    print(tabulate(latency_table, headers=headers, tablefmt="grid"))
    
    # ============ 表格2: Ours vs 最强其他 ============
    print("\n" + "=" * 100)
    print("表格2: Ours vs 最强其他方案")
    print("=" * 100)
    
    speedup_table = []
    for name, data in all_results.items():
        cfg = data['config']
        res = data['results']
        
        # 找最强其他（PyTorch, torch.compile, TL-Official中最快的）
        others = {k: v for k, v in res.items() if k != 'Ours' and v is not None}
        if not others or res['Ours'] is None:
            continue
        
        best_other_name = min(others, key=lambda k: others[k])
        best_other_latency = others[best_other_name]
        ours_latency = res['Ours']
        
        speedup = best_other_latency / ours_latency
        
        row = [
            cfg['name'],
            f"{best_other_name} ({best_other_latency:.4f} ms)",
            f"{ours_latency:.4f} ms",
            f"{speedup:.2f}x",
            "✅" if speedup > 1.0 else "❌",
        ]
        speedup_table.append(row)
    
    headers = ["配置", "最强其他", "Ours", "加速比", "状态"]
    print(tabulate(speedup_table, headers=headers, tablefmt="grid"))
    
    # ============ 总结 ============
    print("\n" + "=" * 100)
    print("总结")
    print("=" * 100)
    
    speedups = []
    for name, data in all_results.items():
        res = data['results']
        others = {k: v for k, v in res.items() if k != 'Ours' and v is not None}
        if others and res['Ours'] is not None:
            best_other = min(others.values())
            speedups.append(best_other / res['Ours'])
    
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"  平均加速比 (vs 最强其他): {avg_speedup:.2f}x")
        print(f"  最大加速比: {max(speedups):.2f}x")
        print(f"  最小加速比: {min(speedups):.2f}x")


if __name__ == "__main__":
    main()
