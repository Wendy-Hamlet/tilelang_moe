"""
DeepSeek mHC Optimization Benchmark

对比测试mHC Pre和Post的优化效果:
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
    mhc_pre_pytorch,
    mhc_pre_pytorch_compiled,
    mhc_post_pytorch,
    mhc_post_pytorch_compiled,
)

from kernels.mhc_optimized import (
    mhc_pre_official,
    mhc_pre_optimized,
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

def test_mhc_pre_correctness(
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
) -> Dict[str, float]:
    """测试mHC Pre各实现的正确性"""
    device = "cuda"
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    
    # 生成测试数据
    torch.manual_seed(42)
    residual = (torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.float32)
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).bfloat16()
    fn = (torch.randn(hc_mult3, hc_mult, hidden_size, device=device, dtype=torch.float32) * 1e-4
          * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).flatten(1, 2)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    
    params = dict(
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=10,
    )
    
    # Reference
    post_ref, comb_ref, layer_ref = mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params)
    
    results = {}
    
    # torch.compile
    post_c, comb_c, layer_c = mhc_pre_pytorch_compiled(residual, fn, hc_scale, hc_base, **params)
    results["torch.compile vs pytorch (layer)"] = (layer_c.float() - layer_ref.float()).abs().max().item()
    
    # TileLang official
    post_off, comb_off, layer_off = mhc_pre_official(residual, fn, hc_scale, hc_base, **params)
    results["official vs pytorch (layer)"] = (layer_off.float() - layer_ref.float()).abs().max().item()
    
    # TileLang optimized
    post_opt, comb_opt, layer_opt = mhc_pre_optimized(residual, fn, hc_scale, hc_base, **params)
    results["optimized vs pytorch (layer)"] = (layer_opt.float() - layer_ref.float()).abs().max().item()
    results["optimized vs official (layer)"] = (layer_opt.float() - layer_off.float()).abs().max().item()
    
    return results


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


def benchmark_mhc_pre(
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
) -> Dict[str, float]:
    """Benchmark mHC Pre各实现"""
    device = "cuda"
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    
    # 生成数据
    torch.manual_seed(42)
    residual = (torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.float32)
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).bfloat16()
    fn = (torch.randn(hc_mult3, hc_mult, hidden_size, device=device, dtype=torch.float32) * 1e-4
          * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).flatten(1, 2)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    
    params = dict(
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=10,
    )
    
    results = {}
    
    # ========== PyTorch baseline ==========
    for _ in range(WARMUP_ITERS):
        _ = mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params)
    torch.cuda.synchronize()
    
    timer = benchmark.Timer(
        stmt="mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params)",
        globals={"mhc_pre_pytorch": mhc_pre_pytorch,
                 "residual": residual, "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base, "params": params},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["pytorch"] = result.mean * 1000  # ms
    
    # ========== torch.compile ==========
    for _ in range(WARMUP_ITERS):
        _ = mhc_pre_pytorch_compiled(residual, fn, hc_scale, hc_base, **params)
    torch.cuda.synchronize()
    
    timer = benchmark.Timer(
        stmt="mhc_pre_pytorch_compiled(residual, fn, hc_scale, hc_base, **params)",
        globals={"mhc_pre_pytorch_compiled": mhc_pre_pytorch_compiled,
                 "residual": residual, "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base, "params": params},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["torch.compile"] = result.mean * 1000
    
    # ========== TileLang official ==========
    for _ in range(WARMUP_ITERS):
        _ = mhc_pre_official(residual, fn, hc_scale, hc_base, **params)
    torch.cuda.synchronize()
    
    timer = benchmark.Timer(
        stmt="mhc_pre_official(residual, fn, hc_scale, hc_base, **params)",
        globals={"mhc_pre_official": mhc_pre_official,
                 "residual": residual, "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base, "params": params},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["official"] = result.mean * 1000
    
    # ========== TileLang optimized ==========
    for _ in range(WARMUP_ITERS):
        _ = mhc_pre_optimized(residual, fn, hc_scale, hc_base, **params)
    torch.cuda.synchronize()
    
    timer = benchmark.Timer(
        stmt="mhc_pre_optimized(residual, fn, hc_scale, hc_base, **params)",
        globals={"mhc_pre_optimized": mhc_pre_optimized,
                 "residual": residual, "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base, "params": params},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["optimized"] = result.mean * 1000
    
    return results


def print_summary(results: List[Dict], title: str):
    """打印汇总结果"""
    print("\n" + "=" * 100)
    print(f"{title} Benchmark Summary (Latency in ms)")
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
    
    return avg_opt_vs_official


# ============ Main ============

def run_correctness_tests():
    """运行正确性测试"""
    print("=" * 80)
    print("Running mHC Correctness Tests")
    print("=" * 80)
    
    num_tokens, hc_mult, hidden_size = 1024, 4, 2560
    print(f"\nConfig: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
    
    # mHC Pre
    print("\n--- mHC Pre Correctness ---")
    pre_results = test_mhc_pre_correctness(num_tokens, hc_mult, hidden_size)
    all_pass = True
    for name, diff in pre_results.items():
        status = "✅" if diff < 0.1 else "❌"
        if diff >= 0.1:
            all_pass = False
        print(f"  {name}: max_diff={diff:.6f} {status}")
    
    # mHC Post
    print("\n--- mHC Post Correctness ---")
    post_results = test_mhc_post_correctness(num_tokens, hc_mult, hidden_size)
    for name, diff in post_results.items():
        status = "✅" if diff < 0.1 else "❌"
        if diff >= 0.1:
            all_pass = False
        print(f"  {name}: max_diff={diff:.6f} {status}")
    
    return all_pass


def run_benchmarks():
    """运行性能benchmark"""
    pre_results = []
    post_results = []
    
    print("\n" + "=" * 80)
    print("Running mHC Performance Benchmarks")
    print("=" * 80)
    
    for num_tokens, hc_mult, hidden_size in BENCHMARK_CONFIGS:
        config_str = f"({num_tokens}, {hc_mult}, {hidden_size})"
        print(f"\nBenchmarking config: {config_str}")
        
        try:
            # mHC Pre
            pre_perf = benchmark_mhc_pre(num_tokens, hc_mult, hidden_size)
            pre_perf["config"] = config_str
            pre_perf["num_tokens"] = num_tokens
            pre_perf["hc_mult"] = hc_mult
            pre_perf["hidden_size"] = hidden_size
            pre_results.append(pre_perf)
            
            # mHC Post
            post_perf = benchmark_mhc_post(num_tokens, hc_mult, hidden_size)
            post_perf["config"] = config_str
            post_perf["num_tokens"] = num_tokens
            post_perf["hc_mult"] = hc_mult
            post_perf["hidden_size"] = hidden_size
            post_results.append(post_perf)
            
            # 打印简要结果
            print(f"  Pre:  pytorch={pre_perf['pytorch']:.3f}ms, compile={pre_perf['torch.compile']:.3f}ms, "
                  f"official={pre_perf['official']:.3f}ms, optimized={pre_perf['optimized']:.3f}ms")
            print(f"  Post: pytorch={post_perf['pytorch']:.3f}ms, compile={post_perf['torch.compile']:.3f}ms, "
                  f"official={post_perf['official']:.3f}ms, optimized={post_perf['optimized']:.3f}ms")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return pre_results, post_results


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
    pre_results, post_results = run_benchmarks()
    
    # 打印汇总
    if pre_results:
        pre_ratio = print_summary(pre_results, "mHC Pre")
    if post_results:
        post_ratio = print_summary(post_results, "mHC Post")
    
    # 总结
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    if pre_results:
        print(f"mHC Pre Optimization: {pre_ratio:.2f}x vs official")
    if post_results:
        avg_post = sum(r['official'] / r['optimized'] for r in post_results) / len(post_results)
        print(f"mHC Post Optimization: {avg_post:.2f}x vs official")
