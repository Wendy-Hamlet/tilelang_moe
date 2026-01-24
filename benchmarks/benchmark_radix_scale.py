"""
最终测试：Radix Selection 在大规模下的表现
测试 topk_radix.py（真正的 O(N) 实现）vs PyTorch
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from kernels.topk_radix import tl_topk_radix

def benchmark_config(N, k, batch_size=8, num_runs=50, warmup=10):
    """测试单个配置"""
    device = torch.device('cuda')
    
    # 准备数据
    x = torch.randn(batch_size, N, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup):
        _ = torch.topk(x, k, dim=-1)
        _ = tl_topk_radix(x, k)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_pt = torch.topk(x, k, dim=-1).indices
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark TileLang Radix
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_tl = tl_topk_radix(x, k)
    torch.cuda.synchronize()
    tilelang_time = (time.perf_counter() - start) / num_runs * 1000
    
    # 验证正确性
    correct = torch.allclose(
        torch.sort(indices_pt, dim=-1)[0],
        torch.sort(indices_tl, dim=-1)[0]
    )
    
    speedup = pytorch_time / tilelang_time
    memory_gb = batch_size * N * 2 / (1024**3)  # float16
    
    return {
        'N': N,
        'k': k,
        'batch': batch_size,
        'pytorch_ms': pytorch_time,
        'tilelang_ms': tilelang_time,
        'speedup': speedup,
        'memory_gb': memory_gb,
        'correct': correct
    }

def main():
    print("="*100)
    print("RADIX SELECTION 规模测试")
    print("测试 topk_radix.py (真正的 O(N) 算法) vs PyTorch")
    print("="*100)
    
    # 测试配置：从小到大
    configs = [
        # (N, k, batch, runs, name)
        (160, 6, 64, 100, "Baseline (DeepSeek-V2)"),
        (512, 16, 32, 50, "Small"),
        (1024, 32, 16, 50, "Medium"),
        (2048, 64, 8, 30, "Large"),
        (4096, 128, 8, 20, "XLarge"),
        (8192, 256, 4, 10, "XXLarge"),
        (16384, 512, 2, 5, "Huge"),
        (32768, 2048, 1, 5, "TileLang-like (DeepSeek-V3)"),
    ]
    
    results = []
    
    for N, k, batch, runs, name in configs:
        print(f"\n{'='*100}")
        print(f"{name} (N={N:,}): N={N}, k={k}, batch={batch}")
        print("-"*100)
        
        try:
            result = benchmark_config(N, k, batch, num_runs=runs)
            results.append(result)
            
            print(f"Memory: {result['memory_gb']:.3f} GB")
            print(f"PyTorch... {result['pytorch_ms']:.4f} ms")
            print(f"TileLang Radix... {result['tilelang_ms']:.4f} ms")
            print(f"Speedup: {result['speedup']:.2f}x")
            print(f"Correct: {'✓' if result['correct'] else '✗ FAILED'}")
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'N': N, 'k': k, 'batch': batch,
                'pytorch_ms': None, 'tilelang_ms': None,
                'speedup': None, 'memory_gb': None,
                'correct': False
            })
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY - Radix Selection 性能")
    print("="*100)
    print(f"{'Scenario':<30} {'N':<10} {'k':<10} {'PyTorch':<12} {'Radix':<12} {'Speedup':<12}")
    print("-"*100)
    
    for i, (_, _, _, _, name) in enumerate(configs):
        r = results[i]
        if r['pytorch_ms'] is not None:
            print(f"{name:<30} {r['N']:<10} {r['k']:<10} {r['pytorch_ms']:<12.4f} "
                  f"{r['tilelang_ms']:<12.4f} {r['speedup']:<12.2f}x")
        else:
            print(f"{name:<30} {r['N']:<10} {r['k']:<10} {'—':<12} {'FAILED':<12} {'—':<12}")
    
    # 分析
    print("\n" + "="*100)
    print("ANALYSIS: Radix Selection 的规模效应")
    print("="*100)
    
    valid_results = [r for r in results if r['speedup'] is not None]
    
    if not valid_results:
        print("\n❌ 所有测试都失败了")
        return
    
    best = max(valid_results, key=lambda r: r['speedup'])
    worst = min(valid_results, key=lambda r: r['speedup'])
    
    print(f"\n📊 最佳性能: N={best['N']}, k={best['k']}, speedup={best['speedup']:.2f}x")
    print(f"📊 最差性能: N={worst['N']}, k={worst['k']}, speedup={worst['speedup']:.2f}x")
    
    # 检查是否有提升
    if best['speedup'] > 1.0:
        print(f"\n✅ 找到了！Radix Selection 在 N={best['N']} 时比 PyTorch 快 {best['speedup']:.2f}x!")
    else:
        print(f"\n❌ Radix Selection 在所有规模都比 PyTorch 慢")
        print(f"   最好情况：慢 {1/best['speedup']:.2f}x")
    
    # 分析趋势
    print(f"\n📈 Speedup 趋势:")
    for r in valid_results:
        bar_len = int(r['speedup'] * 20) if r['speedup'] < 5 else 100
        bar = '█' * bar_len
        print(f"   N={r['N']:<6}: {r['speedup']:6.2f}x {bar}")
    
    # 复杂度分析
    print(f"\n📐 理论复杂度对比:")
    print(f"   PyTorch (假设最小堆):    O(N log k)")
    print(f"   Radix Selection:          O(N × passes)")
    print(f"   ")
    print(f"   当 N 很大时:")
    print(f"   - Radix: O(N) → 线性增长")
    print(f"   - Heap:  O(N log k) → 对数增长开销")
    print(f"   ")
    print(f"   但实际中:")
    print(f"   - PyTorch 的常数项极小（硬件优化）")
    print(f"   - Radix 的 passes 开销可能很大（共享内存、原子操作）")

if __name__ == "__main__":
    main()
