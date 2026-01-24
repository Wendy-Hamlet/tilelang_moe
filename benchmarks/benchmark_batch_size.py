"""
测试 batch size 对 TileLang Radix TopK 性能的影响
固定 N, k，测试 batch 从 1 到 256
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from kernels.topk_radix import topk_radix_tilelang

def benchmark_batch_size(N, k, batch_size, num_runs=50, warmup=5):
    """测试单个 batch size 配置"""
    device = torch.device('cuda')
    
    # 准备数据
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = torch.topk(x, k, dim=-1)
        _ = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = torch.topk(x, k, dim=-1).indices
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark TileLang Radix
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    tilelang_time = (time.perf_counter() - start) / num_runs * 1000
    
    speedup = pytorch_time / tilelang_time
    
    return {
        'batch': batch_size,
        'pytorch_ms': pytorch_time,
        'tilelang_ms': tilelang_time,
        'speedup': speedup,
    }

def test_scenario(N, k, name):
    """测试一个 (N, k) 场景下不同 batch size 的性能"""
    print("="*100)
    print(f"{name}: N={N}, k={k}")
    print("="*100)
    
    # Batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    results = []
    
    for batch in batch_sizes:
        print(f"\nBatch={batch}...", end=" ", flush=True)
        try:
            result = benchmark_batch_size(N, k, batch, num_runs=50)
            results.append(result)
            print(f"PyTorch: {result['pytorch_ms']:.4f}ms, TileLang: {result['tilelang_ms']:.4f}ms, Speedup: {result['speedup']:.2f}x")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'batch': batch,
                'pytorch_ms': None,
                'tilelang_ms': None,
                'speedup': None,
            })
    
    # Summary
    print("\n" + "-"*100)
    print(f"{'Batch':<10} {'PyTorch (ms)':<15} {'TileLang (ms)':<15} {'Speedup':<10} {'Trend':<30}")
    print("-"*100)
    
    for r in results:
        if r['speedup'] is not None:
            bar_len = int(r['speedup'] * 10) if r['speedup'] < 10 else 100
            bar = '█' * bar_len if r['speedup'] > 1 else '░' * int(abs(r['speedup']) * 10)
            status = "✓" if r['speedup'] > 1 else "✗"
            print(f"{r['batch']:<10} {r['pytorch_ms']:<15.4f} {r['tilelang_ms']:<15.4f} {r['speedup']:<10.2f}x {status} {bar}")
        else:
            print(f"{r['batch']:<10} {'FAILED':<15} {'FAILED':<15} {'—':<10}")
    
    # Analysis
    valid = [r for r in results if r['speedup'] is not None]
    if valid:
        best = max(valid, key=lambda r: r['speedup'])
        worst = min(valid, key=lambda r: r['speedup'])
        
        print("\n" + "="*100)
        print("ANALYSIS")
        print("="*100)
        print(f"📊 最佳 batch: {best['batch']}, speedup={best['speedup']:.2f}x")
        print(f"📊 最差 batch: {worst['batch']}, speedup={worst['speedup']:.2f}x")
        print(f"📈 性能提升范围: {worst['speedup']:.2f}x → {best['speedup']:.2f}x")
        
        # Kernel overhead analysis
        if len(valid) > 1:
            small_batch = valid[0]  # batch=1
            large_batch = valid[-1]  # batch=256
            
            print(f"\n🔍 Overhead 分析:")
            print(f"   Small batch (batch={small_batch['batch']}):")
            print(f"     - TileLang: {small_batch['tilelang_ms']:.4f}ms")
            print(f"     - PyTorch:  {small_batch['pytorch_ms']:.4f}ms")
            print(f"     - Speedup:  {small_batch['speedup']:.2f}x")
            print(f"   Large batch (batch={large_batch['batch']}):")
            print(f"     - TileLang: {large_batch['tilelang_ms']:.4f}ms")
            print(f"     - PyTorch:  {large_batch['pytorch_ms']:.4f}ms")
            print(f"     - Speedup:  {large_batch['speedup']:.2f}x")
            
            if large_batch['speedup'] > small_batch['speedup'] * 1.2:
                print(f"\n✅ Batch size 增大显著提升 TileLang 性能")
                print(f"   推荐：使用 batch >= {large_batch['batch'] // 2} 以获得最佳性能")
            elif abs(large_batch['speedup'] - small_batch['speedup']) < 0.3:
                print(f"\n⚠️  Batch size 对性能影响不大")
                print(f"   说明：kernel 计算本身占主导，overhead 较小")
    
    return results

def main():
    print("="*100)
    print("BATCH SIZE 影响测试 - TileLang Radix TopK vs PyTorch")
    print("="*100)
    
    scenarios = [
        # (N, k, name)
        (160, 6, "小规模 (DeepSeek-V2)"),
        (8192, 256, "中等规模"),
        (32768, 2048, "大规模 (DeepSeek-V3)"),
    ]
    
    all_results = {}
    
    for N, k, name in scenarios:
        results = test_scenario(N, k, name)
        all_results[(N, k)] = results
        print("\n" * 2)
    
    # Final summary
    print("="*100)
    print("总结：Batch Size 对性能的影响")
    print("="*100)
    
    for (N, k), results in all_results.items():
        valid = [r for r in results if r['speedup'] is not None]
        if valid:
            best = max(valid, key=lambda r: r['speedup'])
            print(f"\nN={N}, k={k}:")
            print(f"  最佳 batch={best['batch']}, speedup={best['speedup']:.2f}x")
            print(f"  建议 batch >= {best['batch'] // 2} 以获得良好性能")

if __name__ == "__main__":
    main()
