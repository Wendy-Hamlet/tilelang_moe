"""
Extreme scale benchmark - test with truly large N similar to TileLang's scenario
Focus on finding if TileLang gains advantage at extreme scales
"""

import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.topk_sonic_style import topk_sonic_style


def benchmark_extreme_scales():
    """
    Test extreme problem sizes
    Focus on the TileLang scenario and beyond
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 120)
    print("EXTREME SCALE BENCHMARK")
    print("=" * 120)
    
    # Extreme configurations - more similar to TileLang's use case
    configs = [
        # Baseline: Small scale (our current scenario)
        ("Baseline (N=160)", 160, 6, 256, 50),
        
        # Growing scales  
        ("Small (N=1024)", 1024, 32, 128, 50),
        ("Medium (N=4096)", 4096, 128, 64, 50),
        ("Large (N=8192)", 8192, 256, 32, 30),
        ("XLarge (N=16384)", 16384, 512, 16, 20),
        
        # TileLang's scenario
        ("TileLang-like (N=32768)", 32768, 2048, 8, 10),
        
        # Even larger
        ("Extreme (N=65536)", 65536, 4096, 4, 5),
    ]
    
    results = []
    
    for name, num_features, topk, batch, runs in configs:
        print(f"\n{'='*120}")
        print(f"{name}: N={num_features}, k={topk}, batch={batch}")
        print("-" * 120)
        
        # Generate random data
        torch.manual_seed(42)
        data = torch.randn(batch, num_features, dtype=torch.float32, device=device)
        
        memory_gb = (batch * num_features * 4) / (1024**3)
        print(f"Memory: {memory_gb:.3f} GB")
        
        # PyTorch baseline
        print("PyTorch...", end=" ", flush=True)
        for _ in range(3):  # Fewer warmups
            _ = torch.topk(data, topk, dim=-1, largest=True, sorted=False)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(runs):
            values_torch, indices_torch = torch.topk(data, topk, dim=-1, largest=True, sorted=False)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start_time) / runs * 1000
        print(f"{pytorch_time:.4f} ms")
        
        # TileLang
        print("TileLang...", end=" ", flush=True)
        try:
            for _ in range(3):
                _ = topk_sonic_style(data, topk, use_warp=False)
            
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(runs):
                values_tl, indices_tl = topk_sonic_style(data, topk, use_warp=False)
            torch.cuda.synchronize()
            tilelang_time = (time.perf_counter() - start_time) / runs * 1000
            
            # Check correctness
            values_torch_sorted, _ = torch.sort(values_torch, dim=-1, descending=True)
            values_tl_sorted, _ = torch.sort(values_tl, dim=-1, descending=True)
            max_diff = (values_torch_sorted - values_tl_sorted).abs().max().item()
            is_correct = max_diff < 1e-5
            
            speedup = pytorch_time / tilelang_time
            
            status = "✅" if is_correct else "❌"
            direction = "🚀 FASTER!" if speedup > 1.0 else "🐌"
            
            print(f"{tilelang_time:.4f} ms")
            print(f"Speedup: {speedup:.2f}x {direction} {status}")
            
            results.append({
                'name': name,
                'N': num_features,
                'k': topk,
                'batch': batch,
                'pytorch_ms': pytorch_time,
                'tilelang_ms': tilelang_time,
                'speedup': speedup,
                'correct': is_correct,
            })
            
        except Exception as e:
            print(f"FAILED: {str(e)[:50]}")
            results.append({
                'name': name,
                'N': num_features,
                'k': topk,
                'batch': batch,
                'pytorch_ms': pytorch_time,
                'tilelang_ms': None,
                'speedup': None,
                'correct': False,
            })
    
    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"{'Scenario':<25} {'N':<10} {'k':<10} {'PyTorch':<12} {'TileLang':<12} {'Speedup':<12}")
    print("-" * 120)
    
    for r in results:
        if r['tilelang_ms'] is not None:
            speedup_str = f"{r['speedup']:.2f}x"
            marker = "→" if r['speedup'] > 1.0 else " "
            print(f"{marker} {r['name']:<23} {r['N']:<10} {r['k']:<10} "
                  f"{r['pytorch_ms']:<12.4f} {r['tilelang_ms']:<12.4f} {speedup_str:<12}")
        else:
            print(f"  {r['name']:<23} {r['N']:<10} {r['k']:<10} "
                  f"{r['pytorch_ms']:<12.4f} {'FAILED':<12} {'—':<12}")
    
    # Analysis
    print("\n" + "=" * 120)
    print("ANALYSIS: At what scale does TileLang become faster?")
    print("=" * 120)
    
    valid_results = [r for r in results if r['tilelang_ms'] is not None]
    
    if not valid_results:
        print("❌ TileLang failed at all scales")
    else:
        # Find crossover
        slower_results = [r for r in valid_results if r['speedup'] < 1.0]
        faster_results = [r for r in valid_results if r['speedup'] > 1.0]
        
        if slower_results:
            slowest = min(slower_results, key=lambda x: x['speedup'])
            print(f"\n📊 PyTorch faster at all achieved scales")
            print(f"   Worst case (smallest speedup): {slowest['name']}")
            print(f"   N={slowest['N']}, k={slowest['k']}, speedup={slowest['speedup']:.2f}x")
        
        if faster_results:
            fastest = max(faster_results, key=lambda x: x['speedup'])
            print(f"\n🎉 TileLang is faster at:")
            print(f"   {fastest['name']}")
            print(f"   N={fastest['N']}, k={fastest['k']}")
            print(f"   Speedup: {fastest['speedup']:.2f}x")
        
        # Show scaling trend
        print(f"\n📈 Scaling trend (speedup ratio):")
        for r in valid_results:
            ratio = r['tilelang_ms'] / r['pytorch_ms']
            trend = "↑" if ratio > 1 else "↓"
            print(f"   {trend} N={r['N']:<10}: {ratio:.3f}x")
    
    # Save results
    os.makedirs('benchmarks/results', exist_ok=True)
    output_file = 'benchmarks/results/topk_extreme_scale_benchmark.csv'
    with open(output_file, 'w') as f:
        f.write('scenario,N,k,batch,pytorch_ms,tilelang_ms,speedup,correct\n')
        for r in results:
            if r['tilelang_ms'] is not None:
                f.write(f"{r['name']},{r['N']},{r['k']},{r['batch']},"
                       f"{r['pytorch_ms']:.6f},{r['tilelang_ms']:.6f},"
                       f"{r['speedup']:.4f},{int(r['correct'])}\n")
            else:
                f.write(f"{r['name']},{r['N']},{r['k']},{r['batch']},"
                       f"{r['pytorch_ms']:.6f},FAILED,—,0\n")
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    results = benchmark_extreme_scales()
