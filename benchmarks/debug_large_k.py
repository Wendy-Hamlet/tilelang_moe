"""
调试Large-k版本的正确性问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from kernels.topk_radix import topk_radix_tilelang
from kernels.topk_radix_large_k import topk_radix_large_k

def debug_correctness():
    N, k, batch = 32768, 2048, 1  # 先测试单个batch
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    x = torch.randn(batch, N, device='cuda', dtype=torch.float32)
    
    print("="*100)
    print("Large-k正确性调试")
    print("="*100)
    print(f"输入: N={N}, k={k}, batch={batch}")
    print(f"数据统计: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
    
    # PyTorch baseline
    print("\n【PyTorch baseline】")
    indices_pt = torch.topk(x, k, dim=-1).indices
    values_pt = torch.gather(x, 1, indices_pt)
    print(f"  前5个索引: {indices_pt[0, :5].cpu().numpy()}")
    print(f"  前5个值: {values_pt[0, :5].cpu().numpy()}")
    print(f"  最后5个索引: {indices_pt[0, -5:].cpu().numpy()}")
    print(f"  最后5个值: {values_pt[0, -5:].cpu().numpy()}")
    print(f"  最大值: {values_pt[0, 0].item():.6f}")
    print(f"  最小值: {values_pt[0, -1].item():.6f}")
    
    # Official TileLang
    print("\n【Official TileLang】")
    indices_official = topk_radix_tilelang(x, k)
    values_official = torch.gather(x, 1, indices_official)
    print(f"  前5个索引: {indices_official[0, :5].cpu().numpy()}")
    print(f"  前5个值: {values_official[0, :5].cpu().numpy()}")
    print(f"  最后5个索引: {indices_official[0, -5:].cpu().numpy()}")
    print(f"  最后5个值: {values_official[0, -5:].cpu().numpy()}")
    print(f"  最大值: {values_official[0, 0].item():.6f}")
    print(f"  最小值: {values_official[0, -1].item():.6f}")
    
    # Large-k version
    print("\n【Large-k TileLang】")
    indices_large_k = topk_radix_large_k(x, k)
    values_large_k = torch.gather(x, 1, indices_large_k)
    print(f"  前5个索引: {indices_large_k[0, :5].cpu().numpy()}")
    print(f"  前5个值: {values_large_k[0, :5].cpu().numpy()}")
    print(f"  最后5个索引: {indices_large_k[0, -5:].cpu().numpy()}")
    print(f"  最后5个值: {values_large_k[0, -5:].cpu().numpy()}")
    print(f"  最大值: {values_large_k[0, 0].item():.6f if values_large_k.numel() > 0 else 'N/A'}")
    print(f"  最小值: {values_large_k[0, -1].item():.6f if values_large_k.numel() > 0 else 'N/A'}")
    
    # 正确性检查
    print("\n" + "="*100)
    print("正确性验证")
    print("="*100)
    
    ref_set = set(indices_pt[0].cpu().numpy())
    official_set = set(indices_official[0].cpu().numpy())
    large_k_set = set(indices_large_k[0].cpu().numpy())
    
    official_overlap = len(ref_set & official_set)
    large_k_overlap = len(ref_set & large_k_set)
    
    print(f"\nOfficial vs PyTorch: {official_overlap}/{k} = {official_overlap/k*100:.2f}%")
    if official_overlap < k * 0.95:
        print("  ❌ Official正确性有问题！")
    else:
        print("  ✅ Official正确")
    
    print(f"\nLarge-k vs PyTorch: {large_k_overlap}/{k} = {large_k_overlap/k*100:.2f}%")
    if large_k_overlap == 0:
        print("  ❌ Large-k返回了完全错误的结果！")
        print(f"  检查索引范围: min={indices_large_k.min().item()}, max={indices_large_k.max().item()}")
        print(f"  索引类型: {indices_large_k.dtype}")
        print(f"  是否有重复: {len(large_k_set)} unique out of {k} total")
    elif large_k_overlap < k * 0.95:
        print(f"  ⚠️ Large-k正确性不足（{large_k_overlap/k*100:.2f}%）")
    else:
        print("  ✅ Large-k正确")
    
    # 值的正确性检查
    print("\n" + "="*100)
    print("值的正确性检查")
    print("="*100)
    
    # 检查返回的值是否都是正确的topk
    if large_k_overlap > 0:
        # 按值排序检查
        pt_values_sorted = torch.sort(values_pt[0], descending=True)[0]
        large_k_values_sorted = torch.sort(values_large_k[0], descending=True)[0]
        
        print(f"\nPyTorch topk值范围: [{pt_values_sorted[0]:.6f}, {pt_values_sorted[-1]:.6f}]")
        print(f"Large-k topk值范围: [{large_k_values_sorted[0]:.6f}, {large_k_values_sorted[-1]:.6f}]")
        
        # 检查Large-k的最小值是否大于threshold
        threshold_value = pt_values_sorted[k-1].item()
        large_k_min = large_k_values_sorted[-1].item()
        
        print(f"\nthreshold (第{k}大的值): {threshold_value:.6f}")
        print(f"Large-k最小值: {large_k_min:.6f}")
        
        if large_k_min < threshold_value - 1e-5:
            print(f"  ❌ Large-k包含了不应该在topk内的值！")
        else:
            print(f"  ✅ Large-k的值都在topk范围内")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    debug_correctness()
