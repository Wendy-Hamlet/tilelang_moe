"""
mHC Profiling Script

分析 mHC Pre 和 Post 各部分的时间占比
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernels.mhc_optimized import (
    mhc_pre_gemm_sqrsum_optimized,
    mhc_pre_big_fuse_optimized,
    mhc_pre_gemm_sqrsum_official,
    mhc_pre_big_fuse_official,
    mhc_post_optimized,
)


def profile_mhc():
    """Profile each component of mHC"""
    device = "cuda"
    warmup_iterations = 20
    benchmark_iterations = 100
    
    # Test configurations
    configs = [
        {"num_tokens": 256, "hc_mult": 4, "hidden_size": 5120},
        {"num_tokens": 512, "hc_mult": 4, "hidden_size": 5120},
        {"num_tokens": 1024, "hc_mult": 4, "hidden_size": 5120},
    ]
    
    print("=" * 90)
    print("mHC Component Profiling")
    print("=" * 90)
    
    for config in configs:
        num_tokens = config["num_tokens"]
        hc_mult = config["hc_mult"]
        hidden_size = config["hidden_size"]
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # 24
        hc_hidden_size = hc_mult * hidden_size
        
        print(f"\nConfig: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
        print("-" * 70)
        
        # Create test inputs
        torch.manual_seed(42)
        residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
        fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device=device) * 0.01
        hc_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
        hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device=device)
        
        rms_eps = 1e-6
        hc_pre_eps = 1e-6
        hc_sinkhorn_eps = 1e-6
        hc_post_mult_value = 1.0
        sinkhorn_repeat = 10
        
        # ============ Profile Pre GEMM+sqrsum (Optimized) ============
        for _ in range(warmup_iterations):
            mixes, sqrsum = mhc_pre_gemm_sqrsum_optimized(residual, fn)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            mixes, sqrsum = mhc_pre_gemm_sqrsum_optimized(residual, fn)
        torch.cuda.synchronize()
        gemm_opt_time = (time.perf_counter() - start) / benchmark_iterations * 1000
        
        # ============ Profile Pre GEMM+sqrsum (Official) ============
        for _ in range(warmup_iterations):
            mixes, sqrsum = mhc_pre_gemm_sqrsum_official(residual, fn)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            mixes, sqrsum = mhc_pre_gemm_sqrsum_official(residual, fn)
        torch.cuda.synchronize()
        gemm_off_time = (time.perf_counter() - start) / benchmark_iterations * 1000
        
        # Get mixes and sqrsum for big_fuse
        mixes, sqrsum = mhc_pre_gemm_sqrsum_optimized(residual, fn)
        
        # ============ Profile Pre big_fuse (Optimized) ============
        for _ in range(warmup_iterations):
            post_mix, comb_mix, layer_input = mhc_pre_big_fuse_optimized(
                residual, mixes, sqrsum, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat
            )
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            post_mix, comb_mix, layer_input = mhc_pre_big_fuse_optimized(
                residual, mixes, sqrsum, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat
            )
        torch.cuda.synchronize()
        fuse_opt_time = (time.perf_counter() - start) / benchmark_iterations * 1000
        
        # ============ Profile Pre big_fuse (Official) ============
        for _ in range(warmup_iterations):
            post_mix, comb_mix, layer_input = mhc_pre_big_fuse_official(
                residual, mixes, sqrsum, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat
            )
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            post_mix, comb_mix, layer_input = mhc_pre_big_fuse_official(
                residual, mixes, sqrsum, hc_scale, hc_base,
                rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat
            )
        torch.cuda.synchronize()
        fuse_off_time = (time.perf_counter() - start) / benchmark_iterations * 1000
        
        # Create inputs for Post
        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        
        # ============ Profile Post ============
        for _ in range(warmup_iterations):
            output = mhc_post_optimized(x, residual, post_mix, comb_mix)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(benchmark_iterations):
            output = mhc_post_optimized(x, residual, post_mix, comb_mix)
        torch.cuda.synchronize()
        post_time = (time.perf_counter() - start) / benchmark_iterations * 1000
        
        # ============ Summary ============
        total_opt = gemm_opt_time + fuse_opt_time + post_time
        total_off = gemm_off_time + fuse_off_time + post_time
        
        print(f"{'Component':<25} {'Official':<12} {'Optimized':<12} {'Speedup':<10} {'% of Total':<12}")
        print("-" * 70)
        print(f"{'Pre GEMM+sqrsum':<25} {gemm_off_time:<12.4f} {gemm_opt_time:<12.4f} {gemm_off_time/gemm_opt_time:<10.2f}x {gemm_opt_time/total_opt*100:<12.1f}%")
        print(f"{'Pre big_fuse':<25} {fuse_off_time:<12.4f} {fuse_opt_time:<12.4f} {fuse_off_time/fuse_opt_time:<10.2f}x {fuse_opt_time/total_opt*100:<12.1f}%")
        print(f"{'Post':<25} {post_time:<12.4f} {post_time:<12.4f} {'1.00':<10}x {post_time/total_opt*100:<12.1f}%")
        print("-" * 70)
        print(f"{'Total':<25} {total_off:<12.4f} {total_opt:<12.4f} {total_off/total_opt:<10.2f}x {'100.0':<12}%")
        
        print(f"\nBreakdown (Optimized):")
        print(f"  Pre GEMM+sqrsum: {gemm_opt_time:.4f} ms ({gemm_opt_time/total_opt*100:.1f}%)")
        print(f"  Pre big_fuse:    {fuse_opt_time:.4f} ms ({fuse_opt_time/total_opt*100:.1f}%)")
        print(f"  Post:            {post_time:.4f} ms ({post_time/total_opt*100:.1f}%)")


if __name__ == "__main__":
    profile_mhc()
