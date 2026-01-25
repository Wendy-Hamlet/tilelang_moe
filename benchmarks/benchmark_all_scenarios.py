"""
全场景稳定性测试 - 验证Optimized版本在各种k值下的表现
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import statistics
from kernels.topk_radix import topk_radix_tilelang
from kernels.topk_radix_optimized import topk_radix_optimized

def run_benchmark_multiple_times(N, k, batch_size, num_runs=100, num_trials=3):
    """运行多次基准测试以获得稳定结果"""
    device = torch.device('cuda')
    
    # 固定随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # 准备数据
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32)
    
    official_times = []
    optimized_times = []
    
    for trial in range(num_trials):
        # ========== TileLang Official ==========
        for _ in range(50):
            _ = topk_radix_tilelang(x, k)
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = topk_radix_tilelang(x, k)
        torch.cuda.synchronize()
        official_time = (time.perf_counter() - start) / num_runs * 1000
        
        # ========== TileLang Optimized ==========
        for _ in range(50):
            _ = topk_radix_optimized(x, k)
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = topk_radix_optimized(x, k)
        torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start) / num_runs * 1000
        
        official_times.append(official_time)
        optimized_times.append(optimized_time)
    
    # 使用中位数作为最可信的值
    official_median = statistics.median(official_times)
    optimized_median = statistics.median(optimized_times)
    speedup = official_median / optimized_median
    
    official_stdev = statistics.stdev(official_times) if len(official_times) > 1 else 0
    optimized_stdev = statistics.stdev(optimized_times) if len(optimized_times) > 1 else 0
    
    return {
        'official_median': official_median,
        'official_stdev': official_stdev,
        'optimized_median': optimized_median,
        'optimized_stdev': optimized_stdev,
        'speedup': speedup,
        'official_var': official_stdev / official_median * 100 if official_median > 0 else 0,
        'optimized_var': optimized_stdev / optimized_median * 100 if optimized_median > 0 else 0,
    }

def main():
    print("="*120)
    print("全场景稳定性测试 - Optimized vs Official")
    print("="*120)
    
    test_cases = [
        # (N, k, batch, k/N%, name)
        (160, 6, 64, 3.75, "小规模 (DeepSeek-V2)"),
        (1024, 32, 64, 3.13, "Small"),
        (4096, 128, 32, 3.13, "Medium"),
        (8192, 256, 32, 3.13, "Large"),
        (16384, 512, 32, 3.13, "XLarge"),
        (32768, 2048, 64, 6.25, "大规模 (DeepSeek-V3) ⚠️ 大k"),
    ]
    
    results = []
    
    for N, k, batch, k_ratio, name in test_cases:
        print(f"\n{name}: N={N}, k={k}, k/N={k_ratio:.2f}%")
        print("-"*120)
        
        try:
            result = run_benchmark_multiple_times(N, k, batch, num_runs=100 if k < 2048 else 1000, num_trials=3)
            results.append((name, N, k, k_ratio, result))
            
            speedup_pct = (result['speedup'] - 1) * 100
            symbol = "✅" if speedup_pct > 1 else ("❌" if speedup_pct < -1 else "⚖️")
            
            print(f"  Official:  {result['official_median']:8.4f} ± {result['official_stdev']:6.4f} ms (波动{result['official_var']:5.2f}%)")
            print(f"  Optimized: {result['optimized_median']:8.4f} ± {result['optimized_stdev']:6.4f} ms (波动{result['optimized_var']:5.2f}%)")
            print(f"  {symbol} Speedup: {result['speedup']:.4f}x ({speedup_pct:+.2f}%)")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "="*120)
    print("全场景总结")
    print("="*120)
    print(f"{'场景':<25} {'N':<8} {'k':<6} {'k/N%':<8} {'Official(ms)':<15} {'Optimized(ms)':<15} {'Speedup':<12} {'评价':<15}")
    print("-"*120)
    
    small_k_wins = 0
    large_k_result = None
    
    for name, N, k, k_ratio, result in results:
        speedup_pct = (result['speedup'] - 1) * 100
        
        if speedup_pct > 1:
            eval_text = f"✅ 快{speedup_pct:.2f}%"
            if k < 1000:
                small_k_wins += 1
        elif speedup_pct < -1:
            eval_text = f"❌ 慢{abs(speedup_pct):.2f}%"
        else:
            eval_text = f"⚖️ 持平±{abs(speedup_pct):.2f}%"
        
        if k >= 1000:
            large_k_result = (name, speedup_pct)
        
        print(f"{name:<25} {N:<8} {k:<6} {k_ratio:<8.2f} {result['official_median']:<15.4f} {result['optimized_median']:<15.4f} {result['speedup']:<12.4f} {eval_text:<15}")
    
    print("\n" + "="*120)
    print("关键结论")
    print("="*120)
    print(f"✅ 小k场景 (k<1000): Optimized在{small_k_wins}个场景中更快")
    if large_k_result:
        name, speedup_pct = large_k_result
        if speedup_pct > 1:
            print(f"✅ 大k场景 ({name}): Optimized快{speedup_pct:.2f}%")
        else:
            print(f"❌ 大k场景 ({name}): Optimized慢{abs(speedup_pct):.2f}%")
    
    print("\n建议:")
    if small_k_wins >= 4:
        print("✅ Optimized版本应该保留，在小k场景下有显著优势")
    if large_k_result and large_k_result[1] > 1:
        print("✅ 考虑添加自适应策略，根据k值选择使用Optimized还是Official")
    else:
        print("⚠️ 考虑为大k场景开发专门的优化版本（如large-k版本）")
    
    print("="*120)

if __name__ == "__main__":
    main()
