"""
稳定性优化的基准测试 - 解决加速比波动大的问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import statistics
from kernels.topk_radix import topk_radix_tilelang
from kernels.topk_radix_optimized import topk_radix_optimized

def run_benchmark_multiple_times(N, k, batch_size, num_runs=1000, num_trials=5):
    """
    运行多次基准测试以获得稳定结果
    返回：(平均时间, 中位数时间, 最小值, 最大值, 标准差)
    """
    device = torch.device('cuda')
    
    # 固定随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # 准备数据
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32)
    
    results = []
    
    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials}...", end=" ", flush=True)
        
        # ========== PyTorch baseline ==========
        for _ in range(50):  # 更多预热
            _ = torch.topk(x, k, dim=-1)
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            indices_pt = torch.topk(x, k, dim=-1).indices
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) / num_runs * 1000
        
        # ========== TileLang Official ==========
        for _ in range(50):
            _ = topk_radix_tilelang(x, k)
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            indices_orig = topk_radix_tilelang(x, k)
        torch.cuda.synchronize()
        official_time = (time.perf_counter() - start) / num_runs * 1000
        
        # ========== TileLang Optimized ==========
        for _ in range(50):
            _ = topk_radix_optimized(x, k)
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            indices_opt = topk_radix_optimized(x, k)
        torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start) / num_runs * 1000
        
        speedup_opt = pytorch_time / optimized_time
        speedup_orig = pytorch_time / official_time
        opt_vs_orig = official_time / optimized_time  # >1 表示Optimized快
        
        results.append((pytorch_time, official_time, optimized_time, speedup_opt, speedup_orig, opt_vs_orig))
        print(f"Opt/Orig={opt_vs_orig:.4f}x")
    
    # 计算统计数据
    pytorch_times = [r[0] for r in results]
    official_times = [r[1] for r in results]
    optimized_times = [r[2] for r in results]
    speedup_opts = [r[3] for r in results]
    speedup_origs = [r[4] for r in results]
    opt_vs_origs = [r[5] for r in results]
    
    return {
        'pytorch': {
            'avg': statistics.mean(pytorch_times),
            'median': statistics.median(pytorch_times),
            'min': min(pytorch_times),
            'max': max(pytorch_times),
            'stdev': statistics.stdev(pytorch_times) if len(pytorch_times) > 1 else 0,
        },
        'official': {
            'avg': statistics.mean(official_times),
            'median': statistics.median(official_times),
            'min': min(official_times),
            'max': max(official_times),
            'stdev': statistics.stdev(official_times) if len(official_times) > 1 else 0,
        },
        'optimized': {
            'avg': statistics.mean(optimized_times),
            'median': statistics.median(optimized_times),
            'min': min(optimized_times),
            'max': max(optimized_times),
            'stdev': statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0,
        },
        'speedup_opt': {
            'avg': statistics.mean(speedup_opts),
            'median': statistics.median(speedup_opts),
            'min': min(speedup_opts),
            'max': max(speedup_opts),
            'stdev': statistics.stdev(speedup_opts) if len(speedup_opts) > 1 else 0,
        },
        'speedup_orig': {
            'avg': statistics.mean(speedup_origs),
            'median': statistics.median(speedup_origs),
            'min': min(speedup_origs),
            'max': max(speedup_origs),
            'stdev': statistics.stdev(speedup_origs) if len(speedup_origs) > 1 else 0,
        },
        'opt_vs_orig': {
            'avg': statistics.mean(opt_vs_origs),
            'median': statistics.median(opt_vs_origs),
            'min': min(opt_vs_origs),
            'max': max(opt_vs_origs),
            'stdev': statistics.stdev(opt_vs_origs) if len(opt_vs_origs) > 1 else 0,
        },
    }

def main():
    print("="*120)
    print("稳定性优化的基准测试 - 运行多次以消除波动")
    print("="*120)
    
    # 关键场景：N=32768, k=2048
    N, k, batch = 32768, 2048, 64
    
    print(f"\n场景: N={N}, k={k}, batch={batch}")
    print(f"配置: 50次预热 + 1000次测试 × 5次Trial")
    print("-"*120)
    
    results = run_benchmark_multiple_times(N, k, batch, num_runs=1000, num_trials=5)
    
    print("\n" + "="*120)
    print("详细结果分析")
    print("="*120)
    
    print(f"\n【PyTorch基准】")
    print(f"  平均: {results['pytorch']['avg']:.4f} ms")
    print(f"  中位: {results['pytorch']['median']:.4f} ms")
    print(f"  范围: {results['pytorch']['min']:.4f} - {results['pytorch']['max']:.4f} ms")
    print(f"  波动: ±{results['pytorch']['stdev']:.4f} ms ({results['pytorch']['stdev']/results['pytorch']['avg']*100:.2f}%)")
    
    print(f"\n【TileLang Official】")
    print(f"  平均: {results['official']['avg']:.4f} ms")
    print(f"  中位: {results['official']['median']:.4f} ms")
    print(f"  范围: {results['official']['min']:.4f} - {results['official']['max']:.4f} ms")
    print(f"  波动: ±{results['official']['stdev']:.4f} ms ({results['official']['stdev']/results['official']['avg']*100:.2f}%)")
    
    print(f"\n【TileLang Optimized】")
    print(f"  平均: {results['optimized']['avg']:.4f} ms")
    print(f"  中位: {results['optimized']['median']:.4f} ms")
    print(f"  范围: {results['optimized']['min']:.4f} - {results['optimized']['max']:.4f} ms")
    print(f"  波动: ±{results['optimized']['stdev']:.4f} ms ({results['optimized']['stdev']/results['optimized']['avg']*100:.2f}%)")
    
    print(f"\n【加速比 - Optimized vs PyTorch】")
    print(f"  平均: {results['speedup_opt']['avg']:.2f}x")
    print(f"  中位: {results['speedup_opt']['median']:.2f}x")
    print(f"  范围: {results['speedup_opt']['min']:.2f}x - {results['speedup_opt']['max']:.2f}x")
    print(f"  波动: ±{results['speedup_opt']['stdev']:.2f}x ({results['speedup_opt']['stdev']/results['speedup_opt']['avg']*100:.2f}%)")
    
    print(f"\n【加速比 - Official vs PyTorch】")
    print(f"  平均: {results['speedup_orig']['avg']:.2f}x")
    print(f"  中位: {results['speedup_orig']['median']:.2f}x")
    print(f"  范围: {results['speedup_orig']['min']:.2f}x - {results['speedup_orig']['max']:.2f}x")
    print(f"  波动: ±{results['speedup_orig']['stdev']:.2f}x ({results['speedup_orig']['stdev']/results['speedup_orig']['avg']*100:.2f}%)")
    
    print(f"\n【对比 - Optimized vs Official (>1表示Optimized更快)】")
    print(f"  平均: {results['opt_vs_orig']['avg']:.4f}x")
    print(f"  中位: {results['opt_vs_orig']['median']:.4f}x")
    print(f"  范围: {results['opt_vs_orig']['min']:.4f}x - {results['opt_vs_orig']['max']:.4f}x")
    print(f"  波动: ±{results['opt_vs_orig']['stdev']:.4f}x ({results['opt_vs_orig']['stdev']/results['opt_vs_orig']['avg']*100:.2f}%)")
    
    winner = "Optimized更快" if results['opt_vs_orig']['median'] > 1 else "Official更快"
    improvement = abs(results['opt_vs_orig']['median'] - 1) * 100
    
    print(f"\n" + "="*120)
    print(f"结论: {winner} ({improvement:.2f}%)")
    print(f"加速比波动控制在 ±{results['opt_vs_orig']['stdev']/results['opt_vs_orig']['avg']*100:.2f}%")
    print("="*120)

if __name__ == "__main__":
    main()
