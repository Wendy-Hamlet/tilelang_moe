"""
Benchmark for mHC Full Fusion Implementation

比较:
1. Official: 官方实现 (2 kernels: GEMM+sqrsum, big_fuse)
2. Optimized v1: 当前最优参数实现 (2 kernels, 1.10x)
3. Full Fusion: 全融合实现 (1 kernel)
"""

import torch
import time
import argparse
import sys
import os

# Add kernels directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def benchmark_mhc_pre_full_fusion():
    """Benchmark mHC Pre implementations"""
    from kernels.mhc_optimized import (
        mhc_pre_optimized_v1,
        mhc_pre_official,
    )
    from kernels.mhc_full_fusion_simple import (
        mhc_pre_full_fusion_simple,
    )
    
    # Test configurations
    configs = [
        {"num_tokens": 128, "hc_mult": 4, "hidden_size": 5120},
        {"num_tokens": 256, "hc_mult": 4, "hidden_size": 5120},
        {"num_tokens": 512, "hc_mult": 4, "hidden_size": 5120},
        {"num_tokens": 1024, "hc_mult": 4, "hidden_size": 5120},
    ]
    
    warmup_iterations = 10
    benchmark_iterations = 100
    device = "cuda"
    
    print("=" * 80)
    print("mHC Pre Benchmark: Official vs Optimized v1 vs Full Fusion")
    print("=" * 80)
    
    results = []
    
    for config in configs:
        num_tokens = config["num_tokens"]
        hc_mult = config["hc_mult"]
        hidden_size = config["hidden_size"]
        hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # 24
        hc_hidden_size = hc_mult * hidden_size
        
        print(f"\nConfig: num_tokens={num_tokens}, hc_mult={hc_mult}, hidden_size={hidden_size}")
        print("-" * 60)
        
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
        
        # ============ Benchmark Official ============
        try:
            # Warmup
            for _ in range(warmup_iterations):
                _ = mhc_pre_official(
                    residual, fn, hc_scale, hc_base,
                    rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat
                )
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(benchmark_iterations):
                _ = mhc_pre_official(
                    residual, fn, hc_scale, hc_base,
                    rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat
                )
            torch.cuda.synchronize()
            official_time = (time.perf_counter() - start) / benchmark_iterations * 1000
            print(f"Official:     {official_time:.4f} ms")
        except Exception as e:
            print(f"Official:     FAILED - {e}")
            official_time = float('inf')
        
        # ============ Benchmark Optimized v1 ============
        try:
            # Warmup
            for _ in range(warmup_iterations):
                _ = mhc_pre_optimized_v1(
                    residual, fn, hc_scale, hc_base,
                    rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat
                )
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(benchmark_iterations):
                _ = mhc_pre_optimized_v1(
                    residual, fn, hc_scale, hc_base,
                    rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat
                )
            torch.cuda.synchronize()
            v1_time = (time.perf_counter() - start) / benchmark_iterations * 1000
            v1_speedup = official_time / v1_time
            print(f"Optimized v1: {v1_time:.4f} ms ({v1_speedup:.2f}x vs official)")
        except Exception as e:
            print(f"Optimized v1: FAILED - {e}")
            v1_time = float('inf')
            v1_speedup = 0
        
        # ============ Benchmark Full Fusion ============
        try:
            # Warmup
            for _ in range(warmup_iterations):
                _ = mhc_pre_full_fusion_simple(
                    residual, fn, hc_scale, hc_base,
                    rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat
                )
            torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(benchmark_iterations):
                _ = mhc_pre_full_fusion_simple(
                    residual, fn, hc_scale, hc_base,
                    rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                    hc_post_mult_value, sinkhorn_repeat
                )
            torch.cuda.synchronize()
            fusion_time = (time.perf_counter() - start) / benchmark_iterations * 1000
            fusion_vs_official = official_time / fusion_time
            fusion_vs_v1 = v1_time / fusion_time
            print(f"Full Fusion:  {fusion_time:.4f} ms ({fusion_vs_official:.2f}x vs official, {fusion_vs_v1:.2f}x vs v1)")
        except Exception as e:
            print(f"Full Fusion:  FAILED - {e}")
            import traceback
            traceback.print_exc()
            fusion_time = float('inf')
            fusion_vs_official = 0
            fusion_vs_v1 = 0
        
        results.append({
            "num_tokens": num_tokens,
            "official_ms": official_time,
            "v1_ms": v1_time,
            "v1_speedup": v1_speedup,
            "fusion_ms": fusion_time,
            "fusion_vs_official": fusion_vs_official,
            "fusion_vs_v1": fusion_vs_v1,
        })
    
    # ============ Summary ============
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Tokens':<10} {'Official':<12} {'Opt v1':<12} {'v1 Speedup':<12} {'Fusion':<12} {'Fusion/Off':<12} {'Fusion/v1':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['num_tokens']:<10} {r['official_ms']:<12.4f} {r['v1_ms']:<12.4f} {r['v1_speedup']:<12.2f}x {r['fusion_ms']:<12.4f} {r['fusion_vs_official']:<12.2f}x {r['fusion_vs_v1']:<12.2f}x")
    
    # Average speedups
    avg_v1 = sum(r['v1_speedup'] for r in results if r['v1_speedup'] > 0) / len([r for r in results if r['v1_speedup'] > 0])
    avg_fusion_off = sum(r['fusion_vs_official'] for r in results if r['fusion_vs_official'] > 0) / max(1, len([r for r in results if r['fusion_vs_official'] > 0]))
    avg_fusion_v1 = sum(r['fusion_vs_v1'] for r in results if r['fusion_vs_v1'] > 0) / max(1, len([r for r in results if r['fusion_vs_v1'] > 0]))
    
    print("-" * 80)
    print(f"{'Average':<10} {'':<12} {'':<12} {avg_v1:<12.2f}x {'':<12} {avg_fusion_off:<12.2f}x {avg_fusion_v1:<12.2f}x")
    
    return results


def verify_correctness():
    """Verify Full Fusion produces correct results"""
    from kernels.mhc_optimized import mhc_pre_official
    from kernels.mhc_full_fusion_simple import mhc_pre_full_fusion_simple
    
    print("=" * 80)
    print("Correctness Verification: Full Fusion vs Official")
    print("=" * 80)
    
    device = "cuda"
    num_tokens = 64
    hc_mult = 4
    hidden_size = 2560
    hc_mult3 = 24
    hc_hidden_size = hc_mult * hidden_size
    
    torch.manual_seed(42)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device=device) * 0.01
    hc_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device=device)
    
    # Official
    post_off, comb_off, layer_off = mhc_pre_official(
        residual, fn, hc_scale, hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0, sinkhorn_repeat=10
    )
    
    # Full Fusion
    post_fuse, comb_fuse, layer_fuse = mhc_pre_full_fusion_simple(
        residual, fn, hc_scale, hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0, sinkhorn_repeat=10
    )
    
    # Compare
    post_diff = (post_off - post_fuse).abs().max().item()
    comb_diff = (comb_off - comb_fuse).abs().max().item()
    layer_diff = (layer_off.float() - layer_fuse.float()).abs().max().item()
    
    print(f"post_mix max diff:   {post_diff:.6f}")
    print(f"comb_mix max diff:   {comb_diff:.6f}")
    print(f"layer_input max diff: {layer_diff:.6f}")
    
    threshold = 0.01
    if post_diff < threshold and comb_diff < threshold and layer_diff < threshold:
        print("✅ Correctness PASSED")
        return True
    else:
        print("❌ Correctness FAILED")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Verify correctness only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    args = parser.parse_args()
    
    if args.verify:
        verify_correctness()
    elif args.benchmark:
        benchmark_mhc_pre_full_fusion()
    else:
        # Default: both
        correct = verify_correctness()
        if correct:
            print("\n")
            benchmark_mhc_pre_full_fusion()
        else:
            print("\nSkipping benchmark due to correctness failure")
