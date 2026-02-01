"""
Benchmark mHC Post optimized versions vs official implementation
"""

import torch
import time
import sys
sys.path.insert(0, '.')

from kernels.mhc_post_optimized import (
    mhc_post_optimized_v5,
    mhc_post_optimized_v6,
    mhc_post_optimized_v7,
)
from kernels.mhc_optimized import mhc_post_official


def benchmark_fn(fn, *args, warmup=10, iters=100):
    """Benchmark a function with warmup and iterations"""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    
    torch.cuda.synchronize()
    
    # Timing
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iters * 1000  # ms


def verify_correctness(output, reference, name, rtol=1e-2, atol=1e-2):
    """Verify output correctness"""
    if torch.allclose(output.float(), reference.float(), rtol=rtol, atol=atol):
        max_diff = (output.float() - reference.float()).abs().max().item()
        print(f"  ✅ {name}: max_diff = {max_diff:.6f}")
        return True
    else:
        max_diff = (output.float() - reference.float()).abs().max().item()
        print(f"  ❌ {name}: max_diff = {max_diff:.6f} (FAILED)")
        return False


def run_benchmark(num_tokens, hc_mult=4, hidden_size=5120):
    """Run benchmark for given configuration"""
    print(f"\n{'='*60}")
    print(f"Config: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    
    # Create inputs
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device)
    
    # Reference (official)
    print("\n[Correctness Check]")
    ref_output = mhc_post_official(x, residual, post_layer_mix, comb_res_mix)
    
    results = {}
    
    # Test each version
    versions = [
        ("Official", mhc_post_official),
        ("v5 (2D Grid)", mhc_post_optimized_v5),
        ("v6 (h=1024, split)", mhc_post_optimized_v6),
        ("v7 (row write)", mhc_post_optimized_v7),
    ]
    
    for name, fn in versions:
        try:
            output = fn(x, residual, post_layer_mix, comb_res_mix)
            if name != "Official":
                verify_correctness(output, ref_output, name)
            results[name] = fn
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    # Benchmark
    print("\n[Performance Benchmark]")
    times = {}
    
    for name, fn in results.items():
        try:
            t = benchmark_fn(fn, x, residual, post_layer_mix, comb_res_mix)
            times[name] = t
            print(f"  {name}: {t:.4f} ms")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    # Speedup comparison
    if "Official" in times:
        official_time = times["Official"]
        print("\n[Speedup vs Official]")
        for name, t in times.items():
            if name != "Official":
                speedup = official_time / t
                print(f"  {name}: {speedup:.2f}x")
    
    return times


def main():
    print("=" * 60)
    print("mHC Post Kernel Benchmark")
    print("=" * 60)
    
    # Test different token counts
    token_counts = [64, 128, 256, 512, 1024]
    
    all_results = {}
    for n in token_counts:
        try:
            times = run_benchmark(n)
            all_results[n] = times
        except Exception as e:
            print(f"Error for n={n}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (time in ms)")
    print("=" * 80)
    
    # Header
    versions = ["Official", "v5 (2D Grid)", "v6 (h=1024, split)", "v7 (row write)"]
    header = f"{'Tokens':<10}"
    for v in versions:
        header += f"{v[:15]:<18}"
    print(header)
    print("-" * 80)
    
    # Data rows
    for n, times in all_results.items():
        row = f"{n:<10}"
        for v in versions:
            if v in times:
                row += f"{times[v]:.4f} ms        "
            else:
                row += f"{'N/A':<18}"
        print(row)
    
    # Speedup summary
    print("\n" + "=" * 80)
    print("SPEEDUP vs OFFICIAL")
    print("=" * 80)
    
    header = f"{'Tokens':<10}"
    for v in versions[1:]:  # Skip official
        header += f"{v[:15]:<18}"
    print(header)
    print("-" * 80)
    
    for n, times in all_results.items():
        if "Official" not in times:
            continue
        official_t = times["Official"]
        row = f"{n:<10}"
        for v in versions[1:]:
            if v in times:
                speedup = official_t / times[v]
                row += f"{speedup:.2f}x             "
            else:
                row += f"{'N/A':<18}"
        print(row)
    
    # Find best version for each token count
    print("\n" + "=" * 80)
    print("BEST VERSION PER TOKEN COUNT")
    print("=" * 80)
    
    for n, times in all_results.items():
        if len(times) > 1:
            best_name = min(times.items(), key=lambda x: x[1])[0]
            best_time = times[best_name]
            if "Official" in times:
                speedup = times["Official"] / best_time
                print(f"  n={n}: {best_name} ({speedup:.2f}x vs Official)")
            else:
                print(f"  n={n}: {best_name}")


if __name__ == "__main__":
    main()
