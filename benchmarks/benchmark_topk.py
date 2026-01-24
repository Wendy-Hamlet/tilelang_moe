"""
Benchmark: Top-K Gating Performance Comparison
PyTorch vs TileLang (with Hopper features on H100)

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
    Benchmark Top-K performance across different configurations
    
    Args:
        batch_sizes: List of batch sizes to test
        num_experts_list: List of expert counts to test
        k_values: List of k values to test
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("\n" + "=" * 80)
    print("Top-K Gating Benchmark: PyTorch vs TileLang (Hopper Features Enabled)")
    print("=" * 80)
    
    for batch in batch_sizes:
        for num_experts in num_experts_list:
            for k in k_values:
                print(f"\n{'─' * 80}")
                print(f"Config: batch={batch}, num_experts={num_experts}, k={k}")
                print(f"{'─' * 80}")
                
                # Generate test data
                scores = torch.randn(batch, num_experts, device="cuda", dtype=torch.float32)
                
                # Benchmark PyTorch
                print("  [1/2] Benchmarking PyTorch...")
                torch.cuda.synchronize()
                for _ in range(num_warmup):
                    _ = topk_gating_pytorch(scores, k)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(num_runs):
                    indices_pt, weights_pt = topk_gating_pytorch(scores, k)
                torch.cuda.synchronize()
                pytorch_time = (time.perf_counter() - start) / num_runs * 1000  # ms
                
                # Benchmark TileLang
                print("  [2/2] Benchmarking TileLang (Hopper)...")
                torch.cuda.synchronize()
                for _ in range(num_warmup):
                    _ = topk_gating_tilelang(scores, k)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(num_runs):
                    indices_tl, weights_tl = topk_gating_tilelang(scores, k)
                torch.cuda.synchronize()
                tilelang_time = (time.perf_counter() - start) / num_runs * 1000  # ms
                
                # Verify correctness
                weights_pt_sorted = torch.sort(weights_pt, dim=-1, descending=True)[0]
                weights_tl_sorted = torch.sort(weights_tl, dim=-1, descending=True)[0]
                is_correct = torch.allclose(weights_pt_sorted, weights_tl_sorted, rtol=1e-3, atol=1e-5)
                
                speedup = pytorch_time / tilelang_time if tilelang_time > 0 else 0
                
                results.append({
                    'batch': batch,
                    'num_experts': num_experts,
                    'k': k,
                    'pytorch_ms': pytorch_time,
                    'tilelang_ms': tilelang_time,
                    'speedup': speedup,
                    'correct': is_correct
                })
                
                # Print results
                print(f"\n  Results:")
                print(f"    PyTorch:  {pytorch_time:.4f} ms")
                print(f"    TileLang: {tilelang_time:.4f} ms")
                print(f"    Speedup:  {speedup:.2f}x")
                print(f"    Correct:  {'✓' if is_correct else '✗'}")
    
    return pd.DataFrame(results)


def main():
    print("\n" + "=" * 80)
    print("Top-K Gating Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # DeepSeek-V2 configurations + scaling tests (extended for trend analysis)
    batch_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    num_experts_list = [160]  # DeepSeek-V2
    k_values = [6]  # DeepSeek-V2
    
    print("\n⚠️  Testing large batch sizes to analyze scaling behavior")
    print("   Hypothesis: TileLang may outperform PyTorch at large scale")
    
    print(f"\nTest configurations:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Num experts: {num_experts_list}")
    print(f"  K values: {k_values}")
    
    # Run benchmark
    df = benchmark_topk(batch_sizes, num_experts_list, k_values, num_warmup=20, num_runs=100)
    
    # Save results
    save_dir = Path("benchmarks/results")
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "topk_benchmark_results.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    print(f"Average speedup: {df['speedup'].mean():.2f}x")
    print(f"Max speedup: {df['speedup'].max():.2f}x")
    print(f"Min speedup: {df['speedup'].min():.2f}x")
    print(f"All tests correct: {'✓' if df['correct'].all() else '✗'}")
    print(f"\nResults saved to: {save_dir}/topk_benchmark_results.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()

