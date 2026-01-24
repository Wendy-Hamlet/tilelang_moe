"""
Benchmark TopK Radix Selection across batch sizes 1-256
Test correctness and performance compared to PyTorch
"""

import torch
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.topk_radix import topk_radix_tilelang


def benchmark_topk_radix(batch_sizes=None, num_experts=160, topk=6, warmup=10, runs=50):
    """
    Benchmark TopK Radix implementation across different batch sizes
    
    Args:
        batch_sizes: list of batch sizes to test (default: 1 to 256)
        num_experts: number of experts (DeepSeek-V2 default: 160)
        topk: number of experts to select (DeepSeek-V2 default: 6)
        warmup: warmup iterations
        runs: benchmark iterations
    """
    if batch_sizes is None:
        # Test batch sizes from 1 to 256
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Configuration: num_experts={num_experts}, topk={topk}")
    print(f"Warmup: {warmup}, Runs: {runs}")
    print("=" * 80)
    
    results = []
    
    for batch in batch_sizes:
        print(f"\nBatch size: {batch}")
        print("-" * 40)
        
        # Generate random logits
        torch.manual_seed(42)
        logits = torch.randn(batch, num_experts, dtype=torch.float32, device=device)
        
        # ========== PyTorch Baseline ==========
        # Warmup
        for _ in range(warmup):
            _ = torch.topk(logits, topk, dim=-1, largest=True, sorted=False)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(runs):
            indices_torch = torch.topk(logits, topk, dim=-1, largest=True, sorted=False)[1]
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start_time) / runs * 1000  # ms
        
        # ========== TileLang Radix ==========
        # Warmup
        for _ in range(warmup):
            _ = topk_radix_tilelang(logits, topk)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(runs):
            indices_tl = topk_radix_tilelang(logits, topk)
        torch.cuda.synchronize()
        tilelang_time = (time.perf_counter() - start_time) / runs * 1000  # ms
        
        # ========== Correctness Check ==========
        # Get values for both outputs
        values_torch = logits.gather(1, indices_torch.long())
        values_tl = logits.gather(1, indices_tl.long())
        
        # Sort both by value (descending)
        values_torch_sorted, _ = torch.sort(values_torch, dim=-1, descending=True)
        values_tl_sorted, _ = torch.sort(values_tl, dim=-1, descending=True)
        
        # Check if sorted values match
        max_diff = (values_torch_sorted - values_tl_sorted).abs().max().item()
        is_correct = max_diff < 1e-5
        
        # Calculate set intersection (alternative correctness metric)
        intersection_ratios = []
        for i in range(batch):
            ref_set = set(indices_torch[i].cpu().numpy())
            tl_set = set(indices_tl[i].cpu().numpy())
            intersection = len(ref_set & tl_set)
            ratio = intersection / topk
            intersection_ratios.append(ratio)
        avg_intersection = sum(intersection_ratios) / len(intersection_ratios)
        
        # ========== Results ==========
        speedup = pytorch_time / tilelang_time
        
        print(f"PyTorch:      {pytorch_time:.4f} ms")
        print(f"TileLang:     {tilelang_time:.4f} ms")
        print(f"Speedup:      {speedup:.2f}x")
        print(f"Correct:      {is_correct} (max_diff={max_diff:.2e})")
        print(f"Intersection: {avg_intersection*100:.1f}%")
        
        results.append({
            'batch': batch,
            'pytorch_ms': pytorch_time,
            'tilelang_ms': tilelang_time,
            'speedup': speedup,
            'correct': is_correct,
            'max_diff': max_diff,
            'intersection': avg_intersection
        })
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Batch':<10} {'PyTorch(ms)':<15} {'TileLang(ms)':<15} {'Speedup':<10} {'Correct':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['batch']:<10} {r['pytorch_ms']:<15.4f} {r['tilelang_ms']:<15.4f} "
              f"{r['speedup']:<10.2f} {str(r['correct']):<10}")
    
    # Save results
    os.makedirs('benchmarks/results', exist_ok=True)
    output_file = 'benchmarks/results/topk_radix_benchmark.csv'
    with open(output_file, 'w') as f:
        f.write('batch,pytorch_ms,tilelang_ms,speedup,correct,max_diff,intersection\n')
        for r in results:
            f.write(f"{r['batch']},{r['pytorch_ms']:.6f},{r['tilelang_ms']:.6f},"
                   f"{r['speedup']:.4f},{int(r['correct'])},{r['max_diff']:.6e},{r['intersection']:.6f}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    # Run benchmark for batch sizes 1 to 256
    results = benchmark_topk_radix()
