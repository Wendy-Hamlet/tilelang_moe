"""
TopK + Softmax 性能对比基准

官方配置:
1. DeepSeek-V2 (example_topk.py): M=320, N=128, K=6
2. DeepSeek-V3 (deepseek_v32/topk_selector.py): batch=64, seq_len=32768, K=2048

对比四个版本：
1) PyTorch: 全局softmax→topk
2) torch.compile: PyTorch编译优化
3) TL-Official: TileLang官方TopK + PyTorch softmax (separate)
4) Ours: 融合kernel + 自动选择最优算法
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Callable, Dict
from tabulate import tabulate
from tilelang.profiler import do_bench

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
OFFICIAL_CONFIGS = [
    # (name, batch/M, num_experts/N, topk, source)
    ("DeepSeek-V2", 320, 128, 6, "example_topk.py"),
    ("DeepSeek-V3", 64, 32768, 2048, "deepseek_v32/topk_selector.py"),
]


def benchmark_config(batch_size: int, num_experts: int, top_k: int, config_name: str):
    """测试单个配置，返回四个版本的延迟结果。"""
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    logits = torch.randn(batch_size, num_experts, device='cuda', dtype=torch.float32)

    versions = {
        'PyTorch': lambda x: topk_softmax_separate(x, top_k),
        'torch.compile': lambda x: topk_softmax_separate_compiled(x, top_k),
        'TL-Official': lambda x: topk_softmax_tilelang_separate(x, top_k),
        'Ours': lambda x: topk_softmax_best(x, top_k),
    }

    results = {}
    for name, fn in versions.items():
        try:
            # Warmup
            for _ in range(10):
                _ = fn(logits)
            torch.cuda.synchronize()
            
            # Benchmark
            latency_ms = do_bench(lambda: fn(logits), warmup=100, rep=1000, return_mode='median')
            results[name] = latency_ms
            
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            results[name] = None
    
    return results


def main():
    print("=" * 100)
    print("TopK + Softmax Performance Benchmark")
    print("=" * 100)
    print()
    print("Official Configurations:")
    for name, M, N, K, source in OFFICIAL_CONFIGS:
        print(f"  - {name}: M={M}, N={N}, K={K} (from {source})")
    print()
    print("Versions:")
    print("  - PyTorch: softmax(all) → topk")
    print("  - torch.compile: compiled PyTorch")
    print("  - TL-Official: TileLang TopK + PyTorch softmax (separate)")
    print("  - Ours: fused kernel + auto-select algorithm")
    print()
    
    all_results = []
    
    for name, M, N, K, source in OFFICIAL_CONFIGS:
        print(f"Testing {name} (M={M}, N={N}, K={K})...")
        results = benchmark_config(M, N, K, name)
        results['config'] = name
        results['params'] = f"M={M}, N={N}, K={K}"
        all_results.append(results)
        print(f"  Done: PyTorch={results.get('PyTorch', 'N/A'):.4f}ms, Ours={results.get('Ours', 'N/A'):.4f}ms")
        print()
    
    # ============ 表格1: 延迟对比 (ms) ============
    print("=" * 100)
    print("Table 1: Latency Comparison (ms)")
    print("=" * 100)
    
    latency_table = []
    for r in all_results:
        row = [
            r['config'],
            r['params'],
            f"{r['PyTorch']:.4f}" if r.get('PyTorch') else "N/A",
            f"{r['torch.compile']:.4f}" if r.get('torch.compile') else "N/A",
            f"{r['TL-Official']:.4f}" if r.get('TL-Official') else "N/A",
            f"{r['Ours']:.4f}" if r.get('Ours') else "N/A",
        ]
        latency_table.append(row)
    
    headers = ["Config", "Params", "PyTorch", "torch.compile", "TL-Official", "Ours"]
    print(tabulate(latency_table, headers=headers, tablefmt="grid"))
    
    # ============ 表格2: Ours vs 最强其他方案 ============
    print()
    print("=" * 100)
    print("Table 2: Ours vs Best Other")
    print("=" * 100)
    
    compare_table = []
    for r in all_results:
        # 找出最强其他方案
        others = {
            'PyTorch': r.get('PyTorch'),
            'torch.compile': r.get('torch.compile'),
            'TL-Official': r.get('TL-Official'),
        }
        
        valid_others = {k: v for k, v in others.items() if v is not None}
        if not valid_others:
            continue
            
        best_other_name = min(valid_others, key=valid_others.get)
        best_other_latency = valid_others[best_other_name]
        ours_latency = r.get('Ours')
        
        if ours_latency is None:
            continue
        
        speedup = best_other_latency / ours_latency
        status = "✅" if speedup > 1.05 else ("≈" if speedup > 0.95 else "❌")
        
        row = [
            r['config'],
            f"{best_other_name} ({best_other_latency:.4f} ms)",
            f"{ours_latency:.4f} ms",
            f"{speedup:.2f}x",
            status,
        ]
        compare_table.append(row)
    
    headers = ["Config", "Best Other", "Ours", "Speedup", "Status"]
    print(tabulate(compare_table, headers=headers, tablefmt="grid"))
    
    # ============ 平均加速比 ============
    print()
    speedups = []
    for r in all_results:
        others = [r.get('PyTorch'), r.get('torch.compile'), r.get('TL-Official')]
        valid_others = [v for v in others if v is not None]
        if valid_others and r.get('Ours'):
            speedups.append(min(valid_others) / r['Ours'])
    
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"Average Speedup (Ours vs Best Other): {avg_speedup:.2f}x")
    
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
