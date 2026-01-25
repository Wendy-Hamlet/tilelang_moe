"""
四版本TopK对比测试: PyTorch vs PyTorch Compile vs TileLang Official vs TileLang Optimized
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from kernels.topk_radix import topk_radix_tilelang
from kernels.topk_radix_optimized import topk_radix_optimized

def benchmark_comparison(N, k, batch_size, num_runs=100):
    """对比四个实现的性能 - 每个版本独立预热和测试，避免相互干扰"""
    device = torch.device('cuda')
    
    # 固定随机种子确保结果可重复
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # 准备数据
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32)
    
    # ========== 1. PyTorch baseline - 独立测试 ==========
    print("  测试 PyTorch baseline...")
    for _ in range(10):
        _ = torch.topk(x, k, dim=-1)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_pt = torch.topk(x, k, dim=-1).indices
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    # ========== 2. PyTorch Compiled - 独立测试 ==========
    print("  测试 PyTorch Compiled...")
    def topk_wrapper(x, k):
        return torch.topk(x, k, dim=-1).indices
    topk_compiled = torch.compile(topk_wrapper, mode='max-autotune')
    
    for _ in range(10):
        _ = topk_compiled(x, k)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_compiled = topk_compiled(x, k)
    torch.cuda.synchronize()
    compiled_time = (time.perf_counter() - start) / num_runs * 1000
    
    # ========== 3. TileLang Official - 独立测试 ==========
    print("  测试 TileLang Official...")
    for _ in range(10):
        _ = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_orig = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    original_time = (time.perf_counter() - start) / num_runs * 1000
    
    # ========== 4. TileLang Optimized - 独立测试 ==========
    print("  测试 TileLang Optimized...")
    for _ in range(10):
        _ = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_opt = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / num_runs * 1000
    
    # 验证正确性
    compiled_correct = True
    orig_correct = True
    opt_correct = True
    for i in range(min(2, batch_size)):
        ref_set = set(indices_pt[i].cpu().to(torch.int32).numpy())
        compiled_set = set(indices_compiled[i].cpu().to(torch.int32).numpy())
        orig_set = set(indices_orig[i].cpu().numpy())
        opt_set = set(indices_opt[i].cpu().numpy())
        
        if len(ref_set & compiled_set) < len(ref_set) * 0.95:
            compiled_correct = False
        if len(ref_set & orig_set) < len(ref_set) * 0.95:
            orig_correct = False
        if len(ref_set & opt_set) < len(ref_set) * 0.95:
            opt_correct = False
    
    return {
        'pytorch_ms': pytorch_time,
        'compiled_ms': compiled_time,
        'original_ms': original_time,
        'optimized_ms': optimized_time,
        'speedup_compiled': pytorch_time / compiled_time,
        'speedup_orig': pytorch_time / original_time,
        'speedup_opt': pytorch_time / optimized_time,
        'compiled_correct': compiled_correct,
        'orig_correct': orig_correct,
        'opt_correct': opt_correct,
    }

def main():
    print("="*120)
    print("四版本TopK性能对比测试")
    print("="*120)
    print("\n对比四个实现：")
    print("  1. PyTorch baseline (torch.topk)")
    print("  2. PyTorch Compiled (torch.compile with max-autotune)")
    print("  3. TileLang Official (官方Radix Selection)")
    print("  4. TileLang Optimized (缓存优化版)")
    print("\n⚙️  测试配置: 每个版本独立预热10次 + 测试100次取平均")
    print("="*120)
    
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
        print("-"*120)
        
        try:
            result = benchmark_comparison(N, k, batch, num_runs=100)
            results.append((name, N, k, result))
            
            print(f"  PyTorch:          {result['pytorch_ms']:8.4f} ms  (baseline)")
            print(f"  PyTorch Compiled: {result['compiled_ms']:8.4f} ms  (speedup: {result['speedup_compiled']:5.2f}x)")
            print(f"  TileLang Official:{result['original_ms']:8.4f} ms  (speedup: {result['speedup_orig']:5.2f}x)")
            print(f"  TileLang Optimized:{result['optimized_ms']:8.4f} ms  (speedup: {result['speedup_opt']:5.2f}x)")
            print(f"  ✓ 正确性: Compiled={result['compiled_correct']}, Official={result['orig_correct']}, Optimized={result['opt_correct']}")
            
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*120)
    print("性能对比总结")
    print("="*120)
    print(f"{'场景':<25} {'N':<8} {'k':<6} {'PyTorch':<10} {'Compiled':<10} {'Official':<10} {'Optimized':<10}")
    print("-"*120)
    
    for name, N, k, r in results:
        print(f"{name:<25} {N:<8} {k:<6} {r['pytorch_ms']:<10.4f} {r['compiled_ms']:<10.4f} {r['original_ms']:<10.4f} {r['optimized_ms']:<10.4f}")
    
    if results:
        print("\n" + "="*120)
        print("加速比总结 (相对PyTorch baseline)")
        print("="*120)
        print(f"{'场景':<25} {'Compiled':<12} {'Official':<12} {'Optimized':<12}")
        print("-"*120)
        
        for name, N, k, r in results:
            print(f"{name:<25} {r['speedup_compiled']:<12.2f}x {r['speedup_orig']:<12.2f}x {r['speedup_opt']:<12.2f}x")

if __name__ == "__main__":
    main()

