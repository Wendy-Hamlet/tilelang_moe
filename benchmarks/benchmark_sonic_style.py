"""
Benchmark SonicMoE-style TopK implementations
Compare thread-level and warp-level versions
"""

import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.topk_sonic_style import topk_sonic_style


def benchmark_sonic_style(batch_sizes=None, num_experts=160, topk=6, warmup=10, runs=50):
    """
    Benchmark SonicMoE-style TopK implementations
    
    Compare:
    1. PyTorch baseline
    2. Thread-level version (one thread per row)
    3. Warp-level version (32 threads per row)
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Configuration: num_experts={num_experts}, topk={topk}")
    print(f"Warmup: {warmup}, Runs: {runs}")
    print("=" * 100)
    
    results = []
    
    for batch in batch_sizes:
        print(f"\nBatch size: {batch}")
        print("-" * 100)
        
        # Generate random logits
        torch.manual_seed(42)
        logits = torch.randn(batch, num_experts, dtype=torch.float32, device=device)
        
        # ========== PyTorch Baseline ==========
        for _ in range(warmup):
            _ = torch.topk(logits, topk, dim=-1, largest=True, sorted=False)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(runs):
            values_torch, indices_torch = torch.topk(logits, topk, dim=-1, largest=True, sorted=False)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start_time) / runs * 1000
        
        # ========== Thread-Level Version ==========
        for _ in range(warmup):
            _ = topk_sonic_style(logits, topk, use_warp=False)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(runs):
            values_thread, indices_thread = topk_sonic_style(logits, topk, use_warp=False)
        torch.cuda.synchronize()
        thread_time = (time.perf_counter() - start_time) / runs * 1000
        
        # ========== Warp-Level Version ==========
        for _ in range(warmup):
            _ = topk_sonic_style(logits, topk, use_warp=True)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(runs):
            values_warp, indices_warp = topk_sonic_style(logits, topk, use_warp=True)
        torch.cuda.synchronize()
        warp_time = (time.perf_counter() - start_time) / runs * 1000
        
        # ========== Correctness Check ==========
        def check_correctness(values, indices, name):
            # Sort by value to compare
            values_sorted, sort_idx = torch.sort(values, dim=-1, descending=True)
            values_torch_sorted, _ = torch.sort(values_torch, dim=-1, descending=True)
            
            max_diff = (values_sorted - values_torch_sorted).abs().max().item()
            is_correct = max_diff < 1e-5
            
            # Calculate intersection
            intersection_ratios = []
            for i in range(batch):
                ref_set = set(indices_torch[i].cpu().numpy())
                test_set = set(indices[i].cpu().numpy())
                intersection = len(ref_set & test_set)
                ratio = intersection / topk
                intersection_ratios.append(ratio)
            avg_intersection = sum(intersection_ratios) / len(intersection_ratios)
            
            return is_correct, max_diff, avg_intersection
        
        thread_correct, thread_diff, thread_intersection = check_correctness(values_thread, indices_thread, "Thread")
        warp_correct, warp_diff, warp_intersection = check_correctness(values_warp, indices_warp, "Warp")
        
        # ========== Results ==========
        thread_speedup = pytorch_time / thread_time
        warp_speedup = pytorch_time / warp_time
        
        print(f"{'Method':<20} {'Time(ms)':<12} {'Speedup':<10} {'Correct':<10} {'Intersection':<12}")
        print("-" * 100)
        print(f"{'PyTorch':<20} {pytorch_time:<12.4f} {'1.00x':<10} {'True':<10} {'100.0%':<12}")
        print(f"{'Thread-Level':<20} {thread_time:<12.4f} {thread_speedup:<10.2f} {str(thread_correct):<10} {thread_intersection*100:<12.1f}%")
        print(f"{'Warp-Level':<20} {warp_time:<12.4f} {warp_speedup:<10.2f} {str(warp_correct):<10} {warp_intersection*100:<12.1f}%")
        
        results.append({
            'batch': batch,
            'pytorch_ms': pytorch_time,
            'thread_ms': thread_time,
            'warp_ms': warp_time,
            'thread_speedup': thread_speedup,
            'warp_speedup': warp_speedup,
            'thread_correct': thread_correct,
            'warp_correct': warp_correct,
            'thread_intersection': thread_intersection,
            'warp_intersection': warp_intersection,
        })
    
    # ========== Summary ==========
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Batch':<10} {'PyTorch':<12} {'Thread':<12} {'Warp':<12} {'Thread-Speedup':<15} {'Warp-Speedup':<15}")
    print("-" * 100)
    for r in results:
        print(f"{r['batch']:<10} {r['pytorch_ms']:<12.4f} {r['thread_ms']:<12.4f} {r['warp_ms']:<12.4f} "
              f"{r['thread_speedup']:<15.2f} {r['warp_speedup']:<15.2f}")
    
    # Save results
    os.makedirs('benchmarks/results', exist_ok=True)
    output_file = 'benchmarks/results/topk_sonic_style_benchmark.csv'
    with open(output_file, 'w') as f:
        f.write('batch,pytorch_ms,thread_ms,warp_ms,thread_speedup,warp_speedup,'
                'thread_correct,warp_correct,thread_intersection,warp_intersection\n')
        for r in results:
            f.write(f"{r['batch']},{r['pytorch_ms']:.6f},{r['thread_ms']:.6f},{r['warp_ms']:.6f},"
                   f"{r['thread_speedup']:.4f},{r['warp_speedup']:.4f},"
                   f"{int(r['thread_correct'])},{int(r['warp_correct'])},"
                   f"{r['thread_intersection']:.6f},{r['warp_intersection']:.6f}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Find best performer
    best_thread = max(results, key=lambda x: x['thread_speedup'])
    best_warp = max(results, key=lambda x: x['warp_speedup'])
    
    print("\n" + "=" * 100)
    print("BEST PERFORMANCE")
    print("=" * 100)
    print(f"Thread-Level: batch={best_thread['batch']}, speedup={best_thread['thread_speedup']:.2f}x")
    print(f"Warp-Level:   batch={best_warp['batch']}, speedup={best_warp['warp_speedup']:.2f}x")
    
    return results


if __name__ == '__main__':
    results = benchmark_sonic_style()
