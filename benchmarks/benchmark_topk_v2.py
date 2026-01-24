"""
Benchmark: Top-K Gating Performance Comparison
PyTorch vs TileLang (Baseline Iterative vs Optimized Warp-level)

Run on server with CUDA-capable GPU

IMPORTANT: Clear TileLang cache after pulling code updates:
  rm -rf ~/.tilelang_cache
"""
import torch
import time
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.topk_gating import topk_gating_pytorch, topk_gating_tilelang


def benchmark_topk(batch_sizes, num_experts_list, k_values, num_warmup=10, num_runs=100):
    """
    Benchmark Top-K implementations: PyTorch, TileLang Baseline, TileLang Optimized
    """
    results = []
    
    print("\n" + "=" * 80)
    print("Top-K Gating Benchmark: PyTorch vs TileLang (Baseline vs Optimized)")
    print("=" * 80)
    
    for batch in batch_sizes:
        for num_experts in num_experts_list:
            for k in k_values:
                print("\n" + "─" * 80)
                print(f"Config: batch={batch}, num_experts={num_experts}, k={k}")
                print("─" * 80)
                
                scores = torch.randn(batch, num_experts, device="cuda", dtype=torch.float32)
                
                # Benchmark PyTorch
                print("  [1/3] Benchmarking PyTorch...")
                torch.cuda.synchronize()
                for _ in range(num_warmup):
                    _ = topk_gating_pytorch(scores, k)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(num_runs):
                    indices_pt, weights_pt = topk_gating_pytorch(scores, k)
                torch.cuda.synchronize()
                pytorch_time = (time.perf_counter() - start) / num_runs * 1000  # ms
                
                # Benchmark TileLang Baseline
                print("  [2/3] Benchmarking TileLang (Baseline - Iterative)...")
                torch.cuda.synchronize()
                for _ in range(num_warmup):
                    _ = topk_gating_tilelang(scores, k, use_optimized=False)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(num_runs):
                    indices_tl_base, weights_tl_base = topk_gating_tilelang(scores, k, use_optimized=False)
                torch.cuda.synchronize()
                tilelang_baseline_time = (time.perf_counter() - start) / num_runs * 1000  # ms
                
                # Benchmark TileLang Optimized
                print("  [3/3] Benchmarking TileLang (Optimized - Warp-level)...")
                torch.cuda.synchronize()
                for _ in range(num_warmup):
                    _ = topk_gating_tilelang(scores, k, use_optimized=True)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(num_runs):
                    indices_tl_opt, weights_tl_opt = topk_gating_tilelang(scores, k, use_optimized=True)
                torch.cuda.synchronize()
                tilelang_opt_time = (time.perf_counter() - start) / num_runs * 1000  # ms
                
                # Verify correctness
                correct_baseline = torch.allclose(weights_tl_base.sort(descending=True)[0], 
                                                  weights_pt.sort(descending=True)[0], rtol=1e-4, atol=1e-4)
                correct_optimized = torch.allclose(weights_tl_opt.sort(descending=True)[0], 
                                                   weights_pt.sort(descending=True)[0], rtol=1e-4, atol=1e-4)
                
                speedup_baseline = pytorch_time / tilelang_baseline_time
                speedup_optimized = pytorch_time / tilelang_opt_time
                
                print(f"\n  Results:")
                print(f"    PyTorch:             {pytorch_time:.6f} ms")
                print(f"    TileLang (Baseline): {tilelang_baseline_time:.6f} ms (speedup: {speedup_baseline:.2f}x)")
                print(f"    TileLang (Optimized):{tilelang_opt_time:.6f} ms (speedup: {speedup_optimized:.2f}x)")
                print(f"    Correct (Baseline):  {'✓' if correct_baseline else '✗'}")
                print(f"    Correct (Optimized): {'✓' if correct_optimized else '✗'}")
                
                results.append({
                    'batch': batch,
                    'num_experts': num_experts,
                    'k': k,
                    'pytorch_ms': pytorch_time,
                    'tilelang_baseline_ms': tilelang_baseline_time,
                    'tilelang_opt_ms': tilelang_opt_time,
                    'speedup_baseline': speedup_baseline,
                    'speedup_optimized': speedup_optimized,
                    'correct_baseline': correct_baseline,
                    'correct_optimized': correct_optimized,
                })
    
    return pd.DataFrame(results)


def main():
    print("\n" + "=" * 80)
    print("Top-K Gating Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # DeepSeek-V2 configurations + scaling tests
    batch_sizes = [256, 512, 1024, 2048, 4096, 8192]
    num_experts_list = [160]  # DeepSeek-V2
    k_values = [6]  # DeepSeek-V2
    
    print(f"\nTest configurations:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Num experts: {num_experts_list}")
    print(f"  K values: {k_values}")
    
    # Run benchmark
    df = benchmark_topk(batch_sizes, num_experts_list, k_values, num_warmup=20, num_runs=100)
    
    # Save results
    save_dir = Path("benchmarks/results")
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "topk_benchmark_results_v2.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(df.to_string(index=False))
    
    print(f"\nBaseline (Iterative):")
    print(f"  Average speedup: {df['speedup_baseline'].mean():.2f}x")
    print(f"  Max speedup: {df['speedup_baseline'].max():.2f}x")
    print(f"  Min speedup: {df['speedup_baseline'].min():.2f}x")
    print(f"  All tests correct: {'✓' if df['correct_baseline'].all() else '✗'}")
    
    print(f"\nOptimized (Warp-level):")
    print(f"  Average speedup: {df['speedup_optimized'].mean():.2f}x")
    print(f"  Max speedup: {df['speedup_optimized'].max():.2f}x")
    print(f"  Min speedup: {df['speedup_optimized'].min():.2f}x")
    print(f"  All tests correct: {'✓' if df['correct_optimized'].all() else '✗'}")
    
    print(f"\nResults saved to: {save_dir / 'topk_benchmark_results_v2.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
