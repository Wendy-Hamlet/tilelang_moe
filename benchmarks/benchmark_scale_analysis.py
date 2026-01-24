"""
Benchmark TopK with varying problem sizes to find the crossover point
Test if TileLang gains advantage with larger N and k values
"""

import torch
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.topk_sonic_style import topk_sonic_style


def benchmark_varying_scales(warmup=10, runs=50):
    """
    Benchmark with different problem scales to find crossover point
    
    Test configurations:
    1. Tiny (our current scenario): N=160, k=6
    2. Small: N=512, k=16  
    3. Medium: N=1024, k=32
    4. Large: N=4096, k=128
    5. XLarge: N=16384, k=512
    6. XXLarge: N=32768, k=2048 (TileLang's scenario)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Warmup: {warmup}, Runs: {runs}")
    print("=" * 120)
    
    # Test configurations
    configs = [
        ("Tiny (Our MoE)", 160, 6, 256),          # Our current scenario
        ("Small", 512, 16, 256),
        ("Medium", 1024, 32, 256),
        ("Large", 4096, 128, 128),
        ("XLarge", 16384, 512, 64),
        ("XXLarge", 32768, 2048, 32),             # TileLang's scenario
    ]
    
    results = []
    
    for name, num_features, topk, batch in configs:
        print(f"\n{name}: N={num_features}, k={topk}, batch={batch}")
        print("-" * 120)
        
        # Generate random data
        torch.manual_seed(42)
        data = torch.randn(batch, num_features, dtype=torch.float32, device=device)
        
        # ========== PyTorch Baseline ==========
        for _ in range(warmup):
            _ = torch.topk(data, topk, dim=-1, largest=True, sorted=False)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(runs):
            values_torch, indices_torch = torch.topk(data, topk, dim=-1, largest=True, sorted=False)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start_time) / runs * 1000
        
        # ========== TileLang Thread-Level ==========
        try:
            for _ in range(warmup):
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
            
        except Exception as e:
            print(f"  Error: {e}")
            tilelang_time = None
            speedup = None
            is_correct = False
        
        # ========== Results ==========
        if tilelang_time is not None:
            status = "✅" if is_correct else "❌"
            direction = "🚀" if speedup > 1.0 else "🐌"
            
            print(f"PyTorch:     {pytorch_time:.4f} ms")
            print(f"TileLang:    {tilelang_time:.4f} ms")
            print(f"Speedup:     {speedup:.2f}x {direction} {status}")
            
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
        else:
            print(f"TileLang:    FAILED")
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
    
    # ========== Summary ==========
    print("\n" + "=" * 120)
    print("SUMMARY - Finding Crossover Point")
    print("=" * 120)
    print(f"{'Scenario':<20} {'N':<10} {'k':<10} {'PyTorch':<12} {'TileLang':<12} {'Speedup':<12} {'Status':<10}")
    print("-" * 120)
    
    crossover_found = False
    for r in results:
        if r['tilelang_ms'] is not None:
            status = "✅" if r['correct'] else "❌"
            speedup_str = f"{r['speedup']:.2f}x"
            
            # Mark crossover point
            if r['speedup'] > 1.0 and not crossover_found:
                print(f"{'>>> CROSSOVER >>>':<20}")
                crossover_found = True
            
            print(f"{r['name']:<20} {r['N']:<10} {r['k']:<10} "
                  f"{r['pytorch_ms']:<12.4f} {r['tilelang_ms']:<12.4f} "
                  f"{speedup_str:<12} {status:<10}")
        else:
            print(f"{r['name']:<20} {r['N']:<10} {r['k']:<10} "
                  f"{r['pytorch_ms']:<12.4f} {'FAILED':<12} {'—':<12} {'❌':<10}")
    
    # Save results
    os.makedirs('benchmarks/results', exist_ok=True)
    output_file = 'benchmarks/results/topk_scale_analysis.csv'
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
    
    # Analysis
    print("\n" + "=" * 120)
    print("ANALYSIS")
    print("=" * 120)
    
    valid_results = [r for r in results if r['tilelang_ms'] is not None]
    if valid_results:
        slowest = min(valid_results, key=lambda x: x['speedup'])
        fastest = max(valid_results, key=lambda x: x['speedup'])
        
        print(f"\nPyTorch Advantage:")
        print(f"  Strongest: {slowest['name']} ({slowest['speedup']:.2f}x)")
        print(f"  Weakest: {fastest['name']} ({fastest['speedup']:.2f}x)")
        
        if fastest['speedup'] > 1.0:
            print(f"\n🎉 TileLang has advantage at: {fastest['name']}")
            print(f"   N={fastest['N']}, k={fastest['k']}")
            print(f"   Speedup: {fastest['speedup']:.2f}x")
        else:
            print(f"\n📊 PyTorch is faster at all tested scales")
            print(f"   Scaling from N=160 to N=32768 shows:")
            for r in valid_results:
                ratio = r['tilelang_ms'] / r['pytorch_ms']
                print(f"     N={r['N']:<10} ratio={ratio:.2f}x (k={r['k']})")
    
    return results


if __name__ == '__main__':
    results = benchmark_varying_scales()
