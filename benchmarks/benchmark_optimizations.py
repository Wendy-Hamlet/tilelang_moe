"""
对比测试：原始版 vs 优化版 Radix TopK
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from kernels.topk_radix import topk_radix_tilelang
from kernels.topk_radix_optimized import topk_radix_optimized

def benchmark_comparison(N, k, batch_size, num_runs=50):
    """对比三个实现的性能"""
    device = torch.device('cuda')
    
    # 准备数据
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        _ = torch.topk(x, k, dim=-1)
        _ = topk_radix_tilelang(x, k)
        _ = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_pt = torch.topk(x, k, dim=-1).indices
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark 原始 TileLang
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_orig = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    original_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark 优化版 TileLang
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_opt = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / num_runs * 1000
    
    # 验证正确性
    orig_correct = True
    opt_correct = True
    for i in range(min(2, batch_size)):
        ref_set = set(indices_pt[i].cpu().to(torch.int32).numpy())
        orig_set = set(indices_orig[i].cpu().numpy())
        opt_set = set(indices_opt[i].cpu().numpy())
        
        if len(ref_set & orig_set) < len(ref_set) * 0.95:
            orig_correct = False
        if len(ref_set & opt_set) < len(ref_set) * 0.95:
            opt_correct = False
    
    return {
        'pytorch_ms': pytorch_time,
        'original_ms': original_time,
        'optimized_ms': optimized_time,
        'speedup_orig': pytorch_time / original_time,
        'speedup_opt': pytorch_time / optimized_time,
        'improvement': original_time / optimized_time,
        'orig_correct': orig_correct,
        'opt_correct': opt_correct,
    }

def main():
    print("="*100)
    print("优化效果对比测试 - Radix TopK")
    print("="*100)
    print("\n对比三个实现：")
    print("  1. PyTorch baseline")
    print("  2. 原始 TileLang (官方算法)")
    print("  3. 优化 TileLang (Warp归约 + AutoTune + 缓存)")
    print("="*100)
    
    test_cases = [
        # (N, k, batch, name)
        (160, 6, 64, "小规模 (DeepSeek-V2)"),
        (1024, 32, 64, "Small"),
        (4096, 128, 32, "Medium"),
        (8192, 256, 32, "Large"),
        (16384, 512, 32, "XLarge"),
        (32768, 2048, 64, "大规模 (DeepSeek-V3)"),
    ]
    
    results = []
    
    for N, k, batch, name in test_cases:
        print(f"\n{name}: N={N}, k={k}, batch={batch}")
        print("-"*100)
        
        try:
            result = benchmark_comparison(N, k, batch, num_runs=50)
            results.append((name, N, k, result))
            
            print(f"  PyTorch:        {result['pytorch_ms']:8.4f} ms")
            print(f"  原始 TileLang:  {result['original_ms']:8.4f} ms  (speedup: {result['speedup_orig']:5.2f}x vs PyTorch)")
            print(f"  优化 TileLang:  {result['optimized_ms']:8.4f} ms  (speedup: {result['speedup_opt']:5.2f}x vs PyTorch)")
            print(f"  📈 优化提升:     {result['improvement']:5.2f}x  (优化版 vs 原始版)")
            print(f"  ✓ 正确性: 原始={result['orig_correct']}, 优化={result['opt_correct']}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*100)
    print("总结")
    print("="*100)
    print(f"{'场景':<25} {'N':<8} {'k':<6} {'原始(ms)':<12} {'优化(ms)':<12} {'优化提升':<10}")
    print("-"*100)
    
    for name, N, k, r in results:
        print(f"{name:<25} {N:<8} {k:<6} {r['original_ms']:<12.4f} {r['optimized_ms']:<12.4f} {r['improvement']:<10.2f}x")
    
    if results:
        avg_improvement = sum(r['improvement'] for _, _, _, r in results) / len(results)
        max_improvement = max((r['improvement'], name) for name, _, _, r in results)
        
        print("\n" + "="*100)
        print("关键发现")
        print("="*100)
        print(f"📊 平均优化提升: {avg_improvement:.2f}x")
        print(f"📊 最大优化提升: {max_improvement[0]:.2f}x (在 {max_improvement[1]} 场景)")
        
        print("\n主要优化技术：")
        print("  ✓ Warp-level 原子操作归约 (减少冲突 32x)")
        print("  ✓ AutoTune 参数调优 (BLOCK_SIZE, SMEM_INPUT_SIZE)")
        print("  ✓ 缓存 uint32 值 (避免重复计算)")
        print("  ✓ Padding 共享内存 (避免 bank conflict)")

if __name__ == "__main__":
    main()
