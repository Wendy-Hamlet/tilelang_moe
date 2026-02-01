"""
Benchmark mHC Pre GEMM+sqrsum optimized versions
"""

import torch
import time
import sys
sys.path.insert(0, '.')

from kernels.mhc_pre_gemm_optimized import (
    mhc_pre_gemm_sqrsum_v2_kernel,
    mhc_pre_gemm_sqrsum_v3_kernel,
    mhc_pre_gemm_sqrsum_v4_kernel,
    mhc_pre_gemm_sqrsum_v5_kernel,
)
from kernels.mhc_optimized import (
    mhc_pre_gemm_sqrsum_optimized_kernel,
    mhc_pre_gemm_sqrsum_official_kernel,
)


def benchmark_fn(fn, *args, warmup=10, iters=100):
    """Benchmark a function"""
    for _ in range(warmup):
        fn(*args)
    
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iters * 1000  # ms


def verify_correctness(out, sqrsum, ref_out, ref_sqrsum, name, rtol=1e-2, atol=1e-2):
    """Verify output correctness"""
    out_match = torch.allclose(out, ref_out, rtol=rtol, atol=atol)
    sqrsum_match = torch.allclose(sqrsum, ref_sqrsum, rtol=rtol, atol=atol)
    
    if out_match and sqrsum_match:
        out_diff = (out - ref_out).abs().max().item()
        sqrsum_diff = (sqrsum - ref_sqrsum).abs().max().item()
        print(f"  ✅ {name}: out_diff={out_diff:.6f}, sqrsum_diff={sqrsum_diff:.6f}")
        return True
    else:
        out_diff = (out - ref_out).abs().max().item()
        sqrsum_diff = (sqrsum - ref_sqrsum).abs().max().item()
        print(f"  ❌ {name}: out_diff={out_diff:.6f}, sqrsum_diff={sqrsum_diff:.6f}")
        return False


def run_benchmark(num_tokens, hc_mult=4, hidden_size=5120):
    print(f"\n{'='*60}")
    print(f"Config: tokens={num_tokens}, hc={hc_mult}, hidden={hidden_size}")
    print(f"{'='*60}")
    
    device = "cuda"
    torch.manual_seed(42)
    
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    
    x = torch.randn(num_tokens, hc_hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device=device)
    
    # Reference outputs (official)
    ref_out = torch.empty(num_tokens, hc_mult3, dtype=torch.float32, device=device)
    ref_sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=device)
    mhc_pre_gemm_sqrsum_official_kernel(x, fn, ref_out, ref_sqrsum, hc_mult3, hc_hidden_size)
    
    versions = [
        ("Official (h=256, 2-stage)", mhc_pre_gemm_sqrsum_official_kernel),
        ("Current (h=128, 3-stage)", mhc_pre_gemm_sqrsum_optimized_kernel),
        ("v2 (separate sqrsum)", mhc_pre_gemm_sqrsum_v2_kernel),
        ("v3 (h=256, 2-stage)", mhc_pre_gemm_sqrsum_v3_kernel),
        ("v4 (h=128, 4-stage)", mhc_pre_gemm_sqrsum_v4_kernel),
        ("v5 (sqrsum_8)", mhc_pre_gemm_sqrsum_v5_kernel),
    ]
    
    print("\n[Correctness Check]")
    valid_versions = []
    
    for name, kernel in versions:
        out = torch.empty(num_tokens, hc_mult3, dtype=torch.float32, device=device)
        sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=device)
        
        try:
            kernel(x, fn, out, sqrsum, hc_mult3, hc_hidden_size)
            
            if name != "Official (h=256, 2-stage)":
                if verify_correctness(out, sqrsum, ref_out, ref_sqrsum, name):
                    valid_versions.append((name, kernel))
            else:
                valid_versions.append((name, kernel))
                print(f"  ✅ {name}: Reference")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    print("\n[Performance Benchmark]")
    times = {}
    
    for name, kernel in valid_versions:
        out = torch.empty(num_tokens, hc_mult3, dtype=torch.float32, device=device)
        sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=device)
        
        try:
            t = benchmark_fn(kernel, x, fn, out, sqrsum, hc_mult3, hc_hidden_size)
            times[name] = t
            print(f"  {name}: {t:.4f} ms")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    # Speedup
    if "Official (h=256, 2-stage)" in times:
        official_time = times["Official (h=256, 2-stage)"]
        print("\n[Speedup vs Official]")
        for name, t in times.items():
            if name != "Official (h=256, 2-stage)":
                speedup = official_time / t
                print(f"  {name}: {speedup:.2f}x")
    
    return times


def main():
    print("=" * 60)
    print("mHC Pre GEMM+sqrsum Benchmark")
    print("=" * 60)
    
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
    
    # Summary
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY vs Official")
    print("=" * 80)
    
    version_names = ["Current (h=128, 3-stage)", "v2 (separate sqrsum)", 
                     "v3 (h=256, 2-stage)", "v4 (h=128, 4-stage)", "v5 (sqrsum_8)"]
    
    header = f"{'Tokens':<10}"
    for v in version_names:
        header += f"{v[:18]:<20}"
    print(header)
    print("-" * 110)
    
    for n, times in all_results.items():
        if "Official (h=256, 2-stage)" not in times:
            continue
        official_t = times["Official (h=256, 2-stage)"]
        row = f"{n:<10}"
        for v in version_names:
            if v in times:
                speedup = official_t / times[v]
                row += f"{speedup:.2f}x               "
            else:
                row += f"{'N/A':<20}"
        print(row)
    
    # Best version
    print("\n" + "=" * 80)
    print("BEST VERSION PER TOKEN COUNT")
    print("=" * 80)
    
    for n, times in all_results.items():
        if len(times) > 1:
            best_name = min(times.items(), key=lambda x: x[1])[0]
            best_time = times[best_name]
            if "Official (h=256, 2-stage)" in times:
                speedup = times["Official (h=256, 2-stage)"] / best_time
                print(f"  n={n}: {best_name} ({speedup:.2f}x vs Official)")


if __name__ == "__main__":
    main()
