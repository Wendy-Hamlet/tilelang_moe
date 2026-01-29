"""
DeepSeek mHC Benchmark

对比测试mHC Pre和mHC Post的三个版本:
1. PyTorch baseline
2. torch.compile
3. TileLang official

运行方法:
python benchmarks/benchmark_mhc.py
"""

import torch
import torch.utils.benchmark as benchmark
from typing import List, Dict, Tuple
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

try:
    from kernels.mhc_tilelang_official import (
        mhc_pre_tilelang_official,
        mhc_post_tilelang_official,
    )
    HAS_TILELANG = True
except ImportError as e:
    print(f"Warning: TileLang not available: {e}")
    HAS_TILELANG = False


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
    # 更多的hc_mult
    (1024, 8, 2560),
    (2048, 8, 2560),
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
    
    # 生成测试数据 (与官方测试相同的初始化方式)
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
    results["torch.compile_post"] = (post_c - post_ref).abs().max().item()
    results["torch.compile_comb"] = (comb_c - comb_ref).abs().max().item()
    results["torch.compile_layer"] = (layer_c.float() - layer_ref.float()).abs().max().item()
    
    # TileLang
    if HAS_TILELANG:
        post_tl, comb_tl, layer_tl = mhc_pre_tilelang_official(residual, fn, hc_scale, hc_base, **params)
        results["tilelang_post"] = (post_tl - post_ref).abs().max().item()
        results["tilelang_comb"] = (comb_tl - comb_ref).abs().max().item()
        results["tilelang_layer"] = (layer_tl.float() - layer_ref.float()).abs().max().item()
    
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
    results["torch.compile"] = (output_c.float() - output_ref.float()).abs().max().item()
    
    # TileLang
    if HAS_TILELANG:
        output_tl = mhc_post_tilelang_official(x, residual, post_layer_mix, comb_res_mix)
        results["tilelang"] = (output_tl.float() - output_ref.float()).abs().max().item()
    
    return results


# ============ Benchmark Functions ============

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
    
    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params)
    torch.cuda.synchronize()
    
    # PyTorch baseline
    timer = benchmark.Timer(
        stmt="mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params)",
        globals={"mhc_pre_pytorch": mhc_pre_pytorch, 
                 "residual": residual, "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base, "params": params},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["pytorch"] = result.mean * 1000  # ms
    
    # torch.compile
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
    
    # TileLang
    if HAS_TILELANG:
        for _ in range(WARMUP_ITERS):
            _ = mhc_pre_tilelang_official(residual, fn, hc_scale, hc_base, **params)
        torch.cuda.synchronize()
        
        timer = benchmark.Timer(
            stmt="mhc_pre_tilelang_official(residual, fn, hc_scale, hc_base, **params)",
            globals={"mhc_pre_tilelang_official": mhc_pre_tilelang_official,
                     "residual": residual, "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base, "params": params},
        )
        result = timer.timeit(BENCHMARK_ITERS)
        results["tilelang"] = result.mean * 1000
    
    return results


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
    
    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)
    torch.cuda.synchronize()
    
    # PyTorch baseline
    timer = benchmark.Timer(
        stmt="mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)",
        globals={"mhc_post_pytorch": mhc_post_pytorch,
                 "x": x, "residual": residual, "post_layer_mix": post_layer_mix, "comb_res_mix": comb_res_mix},
    )
    result = timer.timeit(BENCHMARK_ITERS)
    results["pytorch"] = result.mean * 1000
    
    # torch.compile
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
    
    # TileLang
    if HAS_TILELANG:
        for _ in range(WARMUP_ITERS):
            _ = mhc_post_tilelang_official(x, residual, post_layer_mix, comb_res_mix)
        torch.cuda.synchronize()
        
        timer = benchmark.Timer(
            stmt="mhc_post_tilelang_official(x, residual, post_layer_mix, comb_res_mix)",
            globals={"mhc_post_tilelang_official": mhc_post_tilelang_official,
                     "x": x, "residual": residual, "post_layer_mix": post_layer_mix, "comb_res_mix": comb_res_mix},
        )
        result = timer.timeit(BENCHMARK_ITERS)
        results["tilelang"] = result.mean * 1000
    
    return results


# ============ Main ============

def run_correctness_tests():
    """运行所有正确性测试"""
    print("=" * 80)
    print("Running mHC Correctness Tests")
    print("=" * 80)
    
    # 选择一个配置进行详细测试
    num_tokens, hc_mult, hidden_size = 1024, 4, 2560
    
    print(f"\nConfig: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
    
    # mHC Pre
    print("\n--- mHC Pre Correctness ---")
    pre_results = test_mhc_pre_correctness(num_tokens, hc_mult, hidden_size)
    for name, diff in pre_results.items():
        status = "✅" if diff < 0.1 else "❌"
        print(f"  {name}: max_diff={diff:.6f} {status}")
    
    # mHC Post
    print("\n--- mHC Post Correctness ---")
    post_results = test_mhc_post_correctness(num_tokens, hc_mult, hidden_size)
    for name, diff in post_results.items():
        status = "✅" if diff < 0.1 else "❌"
        print(f"  {name}: max_diff={diff:.6f} {status}")
    
    return pre_results, post_results


def run_benchmarks():
    """运行所有benchmark"""
    print("\n" + "=" * 80)
    print("Running mHC Performance Benchmarks")
    print("=" * 80)
    
    pre_results = []
    post_results = []
    
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
            print(f"  Pre:  pytorch={pre_perf['pytorch']:.3f}ms, torch.compile={pre_perf['torch.compile']:.3f}ms", end="")
            if HAS_TILELANG:
                print(f", tilelang={pre_perf['tilelang']:.3f}ms")
            else:
                print()
            
            print(f"  Post: pytorch={post_perf['pytorch']:.3f}ms, torch.compile={post_perf['torch.compile']:.3f}ms", end="")
            if HAS_TILELANG:
                print(f", tilelang={post_perf['tilelang']:.3f}ms")
            else:
                print()
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return pre_results, post_results


def print_summary(pre_results: List[Dict], post_results: List[Dict]):
    """打印汇总结果"""
    print("\n" + "=" * 80)
    print("mHC Pre Benchmark Summary (Latency in ms)")
    print("=" * 80)
    
    if HAS_TILELANG:
        headers = ["Config", "PyTorch", "torch.compile", "TileLang", "Speedup vs PyTorch"]
    else:
        headers = ["Config", "PyTorch", "torch.compile", "Speedup vs PyTorch"]
    
    print(f"{'Config':25} | {'PyTorch':>10} | {'torch.compile':>14}", end="")
    if HAS_TILELANG:
        print(f" | {'TileLang':>10} | {'Speedup':>10}")
    else:
        print(f" | {'Speedup':>10}")
    print("-" * 80)
    
    for r in pre_results:
        line = f"{r['config']:25} | {r['pytorch']:>10.3f} | {r['torch.compile']:>14.3f}"
        if HAS_TILELANG:
            speedup = r['pytorch'] / r['tilelang']
            line += f" | {r['tilelang']:>10.3f} | {speedup:>9.2f}x"
        else:
            speedup = r['pytorch'] / r['torch.compile']
            line += f" | {speedup:>9.2f}x"
        print(line)
    
    print("\n" + "=" * 80)
    print("mHC Post Benchmark Summary (Latency in ms)")
    print("=" * 80)
    
    print(f"{'Config':25} | {'PyTorch':>10} | {'torch.compile':>14}", end="")
    if HAS_TILELANG:
        print(f" | {'TileLang':>10} | {'Speedup':>10}")
    else:
        print(f" | {'Speedup':>10}")
    print("-" * 80)
    
    for r in post_results:
        line = f"{r['config']:25} | {r['pytorch']:>10.3f} | {r['torch.compile']:>14.3f}"
        if HAS_TILELANG:
            speedup = r['pytorch'] / r['tilelang']
            line += f" | {r['tilelang']:>10.3f} | {speedup:>9.2f}x"
        else:
            speedup = r['pytorch'] / r['torch.compile']
            line += f" | {speedup:>9.2f}x"
        print(line)
    
    # 计算平均加速比
    if HAS_TILELANG:
        pre_speedups = [r['pytorch'] / r['tilelang'] for r in pre_results]
        post_speedups = [r['pytorch'] / r['tilelang'] for r in post_results]
        print(f"\nmHC Pre Average Speedup (TileLang vs PyTorch): {sum(pre_speedups)/len(pre_speedups):.2f}x")
        print(f"mHC Post Average Speedup (TileLang vs PyTorch): {sum(post_speedups)/len(post_speedups):.2f}x")


if __name__ == "__main__":
    # 检查CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"TileLang available: {HAS_TILELANG}")
    
    # 运行正确性测试
    run_correctness_tests()
    
    # 运行性能benchmark
    pre_results, post_results = run_benchmarks()
    
    # 打印汇总
    print_summary(pre_results, post_results)
