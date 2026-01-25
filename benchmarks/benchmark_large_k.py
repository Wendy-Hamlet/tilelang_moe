"""
专门针对 N=32768, k=2048 场景的性能测试
对比官方版、通用优化版和大k优化版
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from kernels.topk_radix import topk_radix_tilelang
from kernels.topk_radix_optimized import topk_radix_optimized
from kernels.topk_radix_large_k import topk_radix_large_k

def benchmark_large_k(N=32768, k=2048, batch_size=64, num_runs=1000):
    """专门测试大k场景 - 与benchmark_optimizations.py保持一致的参数"""
    device = torch.device('cuda')
    
    # 准备数据
    x = torch.randn(batch_size, N, device=device, dtype=torch.float32)
    
    # 预热 - 100次预热确保完全消除编译开销
    print(f"预热中 (N={N}, k={k}, batch={batch_size})...")
    for _ in range(100):
        _ = torch.topk(x, k, dim=-1)
        _ = topk_radix_tilelang(x, k)
        _ = topk_radix_optimized(x, k)
        _ = topk_radix_large_k(x, k)
    torch.cuda.synchronize()
    
    print("开始性能测试...")
    
    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_pt = torch.topk(x, k, dim=-1).indices
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark Official TileLang
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_orig = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    original_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark Optimized (general)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_opt = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / num_runs * 1000
    
    # Benchmark Large-k Optimized
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        indices_large_k = topk_radix_large_k(x, k)
    torch.cuda.synchronize()
    large_k_time = (time.perf_counter() - start) / num_runs * 1000
    
    # 验证正确性 - 使用更多样本
    orig_correct = True
    opt_correct = True
    large_k_correct = True
    
    check_samples = min(5, batch_size)  # 检查5个样本
    for i in range(check_samples):
        ref_set = set(indices_pt[i].cpu().to(torch.int32).numpy())
        orig_set = set(indices_orig[i].cpu().numpy())
        opt_set = set(indices_opt[i].cpu().numpy())
        large_k_set = set(indices_large_k[i].cpu().numpy())
        
        overlap_orig = len(ref_set & orig_set)
        overlap_opt = len(ref_set & opt_set)
        overlap_large_k = len(ref_set & large_k_set)
        
        if overlap_orig < len(ref_set) * 0.95:
            orig_correct = False
            print(f"  ⚠️  Official样本{i}重叠度: {overlap_orig}/{len(ref_set)} = {overlap_orig/len(ref_set)*100:.1f}%")
        if overlap_opt < len(ref_set) * 0.95:
            opt_correct = False
            print(f"  ⚠️  Optimized样本{i}重叠度: {overlap_opt}/{len(ref_set)} = {overlap_opt/len(ref_set)*100:.1f}%")
        if overlap_large_k < len(ref_set) * 0.95:
            large_k_correct = False
            print(f"  ⚠️  Large-k样本{i}重叠度: {overlap_large_k}/{len(ref_set)} = {overlap_large_k/len(ref_set)*100:.1f}%")
        if len(ref_set & large_k_set) < len(ref_set) * 0.95:
            large_k_correct = False
    
    return {
        'pytorch_ms': pytorch_time,
        'original_ms': original_time,
        'optimized_ms': optimized_time,
        'large_k_ms': large_k_time,
        'orig_correct': orig_correct,
        'opt_correct': opt_correct,
        'large_k_correct': large_k_correct,
    }

def main():
    print("="*120)
    print("大k场景 (N=32768, k=2048) 专项优化测试")
    print("="*120)
    print("\n对比四个实现：")
    print("  1. PyTorch baseline")
    print("  2. TileLang Official (官方Radix Selection)")
    print("  3. TileLang Optimized (通用优化版 - 小k场景优化)")
    print("  4. TileLang Large-k (大k场景专门优化)")
    print("\n优化策略：")
    print("  - BLOCK_SIZE: 1024 (CUDA标准限制)")
    print("  - SMEM_INPUT_SIZE: 12288 (利用H100的232KB opt-in共享内存)")
    print("  - 缓存uint32值 (避免Stage 2重复float→uint32转换)")
    print("  - 共享内存使用: 约148KB (需要H100大共享内存支持)")
    print("\n⚙️  测试配置: 预热100次 + 测试1000次取平均 (与benchmark_optimizations.py一致)")
    print("="*120)
    
    result = benchmark_large_k(N=32768, k=2048, batch_size=64, num_runs=1000)
    
    print(f"\n{'版本':<30} {'时间(ms)':<12} {'加速比':<10} {'正确性'}")
    print("-"*120)
    print(f"{'PyTorch baseline':<30} {result['pytorch_ms']:<12.4f} {'1.00x':<10} ✓")
    print(f"{'TileLang Official':<30} {result['original_ms']:<12.4f} {result['pytorch_ms']/result['original_ms']:<10.2f}x {'✓' if result['orig_correct'] else '✗'}")
    print(f"{'TileLang Optimized (通用)':<30} {result['optimized_ms']:<12.4f} {result['pytorch_ms']/result['optimized_ms']:<10.2f}x {'✓' if result['opt_correct'] else '✗'}")
    print(f"{'TileLang Large-k (大k优化)':<30} {result['large_k_ms']:<12.4f} {result['pytorch_ms']/result['large_k_ms']:<10.2f}x {'✓' if result['large_k_correct'] else '✗'}")
    
    print("\n" + "="*120)
    print("性能对比")
    print("="*120)
    
    if result['large_k_ms'] < result['original_ms']:
        improvement = (result['original_ms'] / result['large_k_ms'] - 1) * 100
        print(f"✅ Large-k优化版相比官方版提升: {improvement:.2f}%")
    else:
        degradation = (result['large_k_ms'] / result['original_ms'] - 1) * 100
        print(f"❌ Large-k优化版相比官方版退化: {degradation:.2f}%")
    
    if result['large_k_ms'] < result['optimized_ms']:
        improvement = (result['optimized_ms'] / result['large_k_ms'] - 1) * 100
        print(f"✅ Large-k优化版相比通用优化版提升: {improvement:.2f}%")
    else:
        degradation = (result['large_k_ms'] / result['optimized_ms'] - 1) * 100
        print(f"❌ Large-k优化版相比通用优化版退化: {degradation:.2f}%")

if __name__ == '__main__':
    main()
