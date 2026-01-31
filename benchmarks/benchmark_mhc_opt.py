"""
DeepSeek mHC Optimization Benchmark

对比测试mHC Post的优化效果:
1. PyTorch baseline
2. torch.compile
3. TileLang official
4. TileLang optimized (our implementation)

运行方法:
python benchmarks/benchmark_mhc_opt.py
"""

import torch
import torch.utils.benchmark as benchmark
from typing import List, Dict
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.mhc_baseline import (
    mhc_post_pytorch,
    mhc_post_pytorch_compiled,
)

from kernels.mhc_optimized import (
    mhc_post_official,
    mhc_post_optimized,
)


# ============ Benchmark Configuration ============

BENCHMARK_CONFIGS = [
    # (num_tokens, hc_mult, hidden_size)
    (256, 4, 2560),
    (512, 4, 2560),
    (1024, 4, 2560),
    (2048, 4, 2560),
    (4096, 4, 2560),
    (8192, 4, 2560),
    # 更大的hidden_size
    (1024, 4, 5120),
    (2048, 4, 5120),
]

WARMUP_ITERS = 10
BENCHMARK_ITERS = 100


# ============ Test Correctness ============

def test_mhc_post_correctness(
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
) -> Dict[str, float]:
    """测试mHC Post各实现的正确性"""
    device = "cuda"
    
    # 生成测试数据
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    
    # Reference
    output_ref = mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)
    
    results = {}
    
    # torch.compile
    output_c = mhc_post_pytorch_compiled(x, residual, post_layer_mix, comb_res_mix)
    results["torch.compile vs pytorch"] = (output_c.float() - output_ref.float()).abs().max().item()
    
    # TileLang official
    output_official = mhc_post_official(x, residual, post_layer_mix, comb_res_mix)
    results["official vs pytorch"] = (output_official.float() - output_ref.float()).abs().max().item()
    
    # TileLang optimized
    output_opt = mhc_post_optimized(x, residual, post_layer_mix, comb_res_mix)
    results["optimized vs pytorch"] = (output_opt.float() - output_ref.float()).abs().max().item()
    results["optimized vs official"] = (output_opt.float() - output_official.float()).abs().max().item()
    
    return results


# ============ Benchmark Functions ============

def benchmark_mhc_post(
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
) -> Dict[str, float]:
    """Benchmark mHC Post各实现"""
    device = "cuda"
    
    # 生成数据
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    
    results = {}
    
    # ========== PyTorch baseline ==========
    for _ in range(WARMUP_ITERS):
        _ = mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    
    timer = benchmark.Timer(
        stmt="mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)",
        globals={"mhc_post_pytorch": mhc_post_pytorch,
                 "x": x, "residual": residual, "post_layer_mix": post_layer_mix, "comb_res_mix": comb_res_mix},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["pytorch"] = result.mean * 1000  # ms
    
    # ========== torch.compile ==========
    for _ in range(WARMUP_ITERS):
        _ = mhc_post_pytorch_compiled(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    
    timer = benchmark.Timer(
        stmt="mhc_post_pytorch_compiled(x, residual, post_layer_mix, comb_res_mix)",
        globals={"mhc_post_pytorch_compiled": mhc_post_pytorch_compiled,
                 "x": x, "residual": residual, "post_layer_mix": post_layer_mix, "comb_res_mix": comb_res_mix},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["torch.compile"] = result.mean * 1000
    
    # ========== TileLang official ==========
    for _ in range(WARMUP_ITERS):
        _ = mhc_post_official(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    
    timer = benchmark.Timer(
        stmt="mhc_post_official(x, residual, post_layer_mix, comb_res_mix)",
        globals={"mhc_post_official": mhc_post_official,
                 "x": x, "residual": residual, "post_layer_mix": post_layer_mix, "comb_res_mix": comb_res_mix},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["official"] = result.mean * 1000
    
    # ========== TileLang optimized ==========
    for _ in range(WARMUP_ITERS):
        _ = mhc_post_optimized(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    
    timer = benchmark.Timer(
        stmt="mhc_post_optimized(x, residual, post_layer_mix, comb_res_mix)",
        globals={"mhc_post_optimized": mhc_post_optimized,
                 "x": x, "residual": residual, "post_layer_mix": post_layer_mix, "comb_res_mix": comb_res_mix},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["optimized"] = result.mean * 1000
    
    return results


# ============ Main ============

def run_correctness_tests():
    """运行正确性测试"""
    print("=" * 80)
    print("Running mHC Post Correctness Tests")
    print("=" * 80)
    
    num_tokens, hc_mult, hidden_size = 1024, 4, 2560
    print(f"\nConfig: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
    
    results = test_mhc_post_correctness(num_tokens, hc_mult, hidden_size)
    
    print("\n--- Correctness Results ---")
    all_pass = True
    for name, diff in results.items():
        status = "✅" if diff < 0.1 else "❌"
        if diff >= 0.1:
            all_pass = False
        print(f"  {name}: max_diff={diff:.6f} {status}")
    
    return all_pass


def run_benchmarks():
    """运行性能benchmark"""
    print("\n" + "=" * 80)
    print("Running mHC Post Performance Benchmarks")
    print("=" * 80)
    
    results = []
    
    for num_tokens, hc_mult, hidden_size in BENCHMARK_CONFIGS:
        config_str = f"({num_tokens}, {hc_mult}, {hidden_size})"
        print(f"\nBenchmarking config: {config_str}")
        
        try:
            perf = benchmark_mhc_post(num_tokens, hc_mult, hidden_size)
            perf["config"] = config_str
            perf["num_tokens"] = num_tokens
            perf["hc_mult"] = hc_mult
            perf["hidden_size"] = hidden_size
            results.append(perf)
            
            # 打印简要结果
            print(f"  pytorch={perf['pytorch']:.3f}ms, torch.compile={perf['torch.compile']:.3f}ms, "
                  f"official={perf['official']:.3f}ms, optimized={perf['optimized']:.3f}ms")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def print_summary(results: List[Dict]):
    """打印汇总结果"""
    print("\n" + "=" * 100)
    print("mHC Post Benchmark Summary (Latency in ms)")
    print("=" * 100)
    
    # Table 1: All implementations vs PyTorch
    print("\n--- All Implementations vs PyTorch ---")
    print(f"{'Config':20} | {'PyTorch':>10} | {'Compile':>10} | {'Official':>10} | {'Optimized':>10} | "
          f"{'Speedup':>8} | {'Opt/Off':>8}")
    print("-" * 100)
    
    for r in results:
        speedup_vs_pytorch = r['pytorch'] / r['optimized']
        opt_vs_official = r['official'] / r['optimized']
        
        line = (f"{r['config']:20} | {r['pytorch']:>10.3f} | {r['torch.compile']:>10.3f} | "
                f"{r['official']:>10.3f} | {r['optimized']:>10.3f} | "
                f"{speedup_vs_pytorch:>7.2f}x | {opt_vs_official:>7.2f}x")
        print(line)
    
    # 计算平均值
    avg_speedup_pytorch = sum(r['pytorch'] / r['optimized'] for r in results) / len(results)
    avg_opt_vs_official = sum(r['official'] / r['optimized'] for r in results) / len(results)
    
    print("-" * 100)
    print(f"{'Average':20} | {'-':>10} | {'-':>10} | {'-':>10} | {'-':>10} | "
          f"{avg_speedup_pytorch:>7.2f}x | {avg_opt_vs_official:>7.2f}x")
    
    # Table 2: Optimized vs Official detailed comparison
    print("\n--- Optimized vs Official Detailed Comparison ---")
    print(f"{'Config':20} | {'Official':>10} | {'Optimized':>10} | {'Diff (ms)':>10} | {'Ratio':>10} | {'Winner':>10}")
    print("-" * 85)
    
    opt_wins = 0
    official_wins = 0
    
    for r in results:
        diff = r['official'] - r['optimized']
        ratio = r['official'] / r['optimized']
        
        if ratio > 1.05:
            winner = "Optimized"
            opt_wins += 1
        elif ratio < 0.95:
            winner = "Official"
            official_wins += 1
        else:
            winner = "Tie"
        
        line = (f"{r['config']:20} | {r['official']:>10.3f} | {r['optimized']:>10.3f} | "
                f"{diff:>+10.3f} | {ratio:>9.2f}x | {winner:>10}")
        print(line)
    
    print("-" * 85)
    print(f"\nSummary: Optimized wins {opt_wins}, Official wins {official_wins}, Ties {len(results) - opt_wins - official_wins}")
    
    # Decision
    print("\n" + "=" * 80)
    if avg_opt_vs_official > 1.1:
        print(f"✅ OPTIMIZATION SUCCESS: Optimized version is {avg_opt_vs_official:.2f}x faster than official!")
        print("   Recommendation: Keep the optimized version.")
    elif avg_opt_vs_official < 0.9:
        print(f"❌ OPTIMIZATION FAILED: Optimized version is {1/avg_opt_vs_official:.2f}x slower than official.")
        print("   Recommendation: Revert to official and try next optimization.")
    else:
        print(f"⚠️ MARGINAL DIFFERENCE: Optimized version is {avg_opt_vs_official:.2f}x vs official.")
        print("   Recommendation: Consider other optimizations.")
    print("=" * 80)


if __name__ == "__main__":
    # 检查CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
    
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # 运行正确性测试
    correctness_pass = run_correctness_tests()
    
    if not correctness_pass:
        print("\n⚠️ Warning: Some correctness tests failed!")
    
    # 运行性能benchmark
    results = run_benchmarks()
    
    # 打印汇总
    if results:
        print_summary(results)
