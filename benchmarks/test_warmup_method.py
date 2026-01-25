"""
验证预热方式对性能测试的影响
对比：联合预热 vs 独立预热
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import statistics
from kernels.topk_radix import topk_radix_tilelang
from kernels.topk_radix_optimized import topk_radix_optimized

def test_joint_warmup(N, k, batch_size, warmup, num_runs):
    """联合预热：所有版本一起预热（benchmark_large_k.py的方式）"""
    device = torch.device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32)
    
    # 联合预热 - 所有版本一起跑
    print(f"  联合预热 {warmup}次...", end=" ", flush=True)
    for _ in range(warmup):
        _ = topk_radix_tilelang(x, k)
        _ = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    print("完成")
    
    # Benchmark Official
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    official_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark Optimized
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / num_runs * 1000
    
    return official_time, optimized_time

def test_isolated_warmup(N, k, batch_size, warmup, num_runs):
    """独立预热：每个版本单独预热（benchmark_stability.py的方式）"""
    device = torch.device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32)
    
    # Official - 独立预热
    print(f"  Official独立预热 {warmup}次...", end=" ", flush=True)
    for _ in range(warmup):
        _ = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    official_time = (time.perf_counter() - start) / num_runs * 1000
    print(f"{official_time:.4f}ms")
    
    # Optimized - 独立预热
    print(f"  Optimized独立预热 {warmup}次...", end=" ", flush=True)
    for _ in range(warmup):
        _ = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / num_runs * 1000
    print(f"{optimized_time:.4f}ms")
    
    return official_time, optimized_time

def main():
    N, k, batch = 32768, 2048, 64
    
    print("="*120)
    print("预热方式对性能测试的影响分析")
    print(f"场景: N={N}, k={k}, batch={batch}")
    print("="*120)
    
    # 测试1: 联合预热（100次）
    print("\n【测试1】联合预热 (100次预热，所有版本一起跑)")
    print("-"*120)
    joint_trials = []
    for trial in range(3):
        print(f"Trial {trial+1}:")
        official_time, optimized_time = test_joint_warmup(N, k, batch, warmup=100, num_runs=1000)
        speedup = official_time / optimized_time
        print(f"  Official: {official_time:.4f}ms, Optimized: {optimized_time:.4f}ms, Speedup: {speedup:.4f}x")
        joint_trials.append((official_time, optimized_time, speedup))
    
    # 测试2: 独立预热（50次）
    print("\n【测试2】独立预热 (50次预热，每个版本单独跑)")
    print("-"*120)
    isolated_trials = []
    for trial in range(3):
        print(f"Trial {trial+1}:")
        official_time, optimized_time = test_isolated_warmup(N, k, batch, warmup=50, num_runs=1000)
        speedup = official_time / optimized_time
        print(f"  Official: {official_time:.4f}ms, Optimized: {optimized_time:.4f}ms, Speedup: {speedup:.4f}x")
        isolated_trials.append((official_time, optimized_time, speedup))
    
    # 统计结果
    print("\n" + "="*120)
    print("统计结果对比 (取中位数)")
    print("="*120)
    
    joint_official_median = statistics.median([t[0] for t in joint_trials])
    joint_optimized_median = statistics.median([t[1] for t in joint_trials])
    joint_speedup = joint_official_median / joint_optimized_median
    
    isolated_official_median = statistics.median([t[0] for t in isolated_trials])
    isolated_optimized_median = statistics.median([t[1] for t in isolated_trials])
    isolated_speedup = isolated_official_median / isolated_optimized_median
    
    print(f"\n联合预热 (benchmark_large_k.py方式):")
    print(f"  Official:  {joint_official_median:.4f} ms")
    print(f"  Optimized: {joint_optimized_median:.4f} ms")
    print(f"  Speedup:   {joint_speedup:.4f}x ({(joint_speedup-1)*100:+.2f}%)")
    
    print(f"\n独立预热 (benchmark_stability.py方式):")
    print(f"  Official:  {isolated_official_median:.4f} ms")
    print(f"  Optimized: {isolated_optimized_median:.4f} ms")
    print(f"  Speedup:   {isolated_speedup:.4f}x ({(isolated_speedup-1)*100:+.2f}%)")
    
    # 关键发现
    print("\n" + "="*120)
    print("🔍 关键发现")
    print("="*120)
    
    official_diff = (joint_official_median - isolated_official_median) / isolated_official_median * 100
    optimized_diff = (joint_optimized_median - isolated_optimized_median) / isolated_optimized_median * 100
    
    print(f"\nOfficial版本性能差异: {official_diff:+.2f}%")
    print(f"  联合预热: {joint_official_median:.4f}ms")
    print(f"  独立预热: {isolated_official_median:.4f}ms")
    if abs(official_diff) > 5:
        print(f"  ⚠️  Official受预热方式影响显著（{abs(official_diff):.1f}%）")
    
    print(f"\nOptimized版本性能差异: {optimized_diff:+.2f}%")
    print(f"  联合预热: {joint_optimized_median:.4f}ms")
    print(f"  独立预热: {isolated_optimized_median:.4f}ms")
    if abs(optimized_diff) > 5:
        print(f"  ⚠️  Optimized受预热方式影响显著（{abs(optimized_diff):.1f}%）")
    
    print(f"\n相对性能变化:")
    print(f"  联合预热时Optimized优势: {(joint_speedup-1)*100:+.2f}%")
    print(f"  独立预热时Optimized优势: {(isolated_speedup-1)*100:+.2f}%")
    print(f"  优势变化: {(joint_speedup-isolated_speedup)*100:+.2f} 百分点")
    
    # 结论
    print("\n" + "="*120)
    print("💡 结论")
    print("="*120)
    
    if abs(official_diff) > abs(optimized_diff) and official_diff > 10:
        print("✅ 确认：联合预热对Official版本影响更大")
        print(f"   - Official在联合预热时慢了 {official_diff:.1f}%")
        print(f"   - Optimized相对稳定（只差 {abs(optimized_diff):.1f}%）")
        print(f"   - 这导致联合预热时Optimized显得更快")
        print(f"\n⚠️  之前benchmark_large_k.py中看到的46%提升是测试方法的人为产物！")
        print(f"   真实性能：Optimized只比Official快 {(isolated_speedup-1)*100:.2f}%")
    elif abs(isolated_speedup - 1) < 0.02:
        print("✅ 确认：独立预热时两个版本基本持平")
        print(f"   - Optimized快 {(isolated_speedup-1)*100:.2f}%，在波动范围内")
        print(f"   - 联合预热引入了偏差，让Optimized看起来快很多")
    else:
        print("🤔 预热方式有影响，但不是唯一因素")
    
    print("\n建议：使用独立预热方式进行性能测试，更准确反映真实性能")
    print("="*120)

if __name__ == "__main__":
    main()
