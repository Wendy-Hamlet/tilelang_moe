"""
检查TileLang是否启用了Hopper特性（H100专用优化）

TileLang的Hopper特性包括：
1. TMA (Tensor Memory Accelerator) - 异步内存传输
2. WGMMA (Warp Group Matrix Multiply Accumulate) - 线程束组矩阵乘法
3. Warp Specialization - 线程束专用化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tvm.target import Target
import tilelang
from tilelang.contrib.nvcc import have_tma, is_hopper, have_pdl
from tilelang.utils.target import target_is_hopper


def check_hopper_support():
    """检查硬件和编译器对Hopper的支持"""
    print("=" * 80)
    print("TileLang Hopper特性诊断")
    print("=" * 80)
    
    # 检查GPU硬件
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return
    
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"✅ Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
    
    # 创建Target
    target_str = f"cuda -arch=sm_{compute_capability[0]}{compute_capability[1]}"
    target = Target(target_str)
    print(f"✅ TVM Target: {target_str}")
    
    print("\n" + "=" * 80)
    print("Hopper特性检测")
    print("=" * 80)
    
    # 检查Hopper架构
    is_h100 = is_hopper(target)
    print(f"\n{'✅' if is_h100 else '❌'} is_hopper(target): {is_h100}")
    if is_h100:
        print("   → H100架构 (sm_90)")
    else:
        print(f"   → 非Hopper架构 (sm_{compute_capability[0]}{compute_capability[1]})")
    
    # 检查target_is_hopper
    try:
        is_h100_utils = target_is_hopper(target)
        print(f"{'✅' if is_h100_utils else '❌'} target_is_hopper(target): {is_h100_utils}")
    except Exception as e:
        print(f"❌ target_is_hopper失败: {e}")
    
    # 检查TMA支持
    supports_tma = have_tma(target)
    print(f"\n{'✅' if supports_tma else '❌'} have_tma(target): {supports_tma}")
    if supports_tma:
        print("   → 支持TMA (Tensor Memory Accelerator)")
        print("   → 可用于异步内存传输、高效数据加载")
    else:
        print("   → 不支持TMA (需要Compute Capability >= 9.0)")
    
    # 检查PDL支持
    supports_pdl = have_pdl(target)
    print(f"\n{'✅' if supports_pdl else '❌'} have_pdl(target): {supports_pdl}")
    if supports_pdl:
        print("   → 支持PDL (需要Compute Capability >= 9.0)")
    
    print("\n" + "=" * 80)
    print("编译配置检查")
    print("=" * 80)
    
    # 检查PassContext配置
    from tilelang.transform import PassConfigKey
    
    with tilelang.transform.PassContext():
        pass_ctx = tilelang.transform.get_pass_context()
        
        disable_tma = pass_ctx.config.get(PassConfigKey.TL_DISABLE_TMA_LOWER, False)
        print(f"\n{'❌' if disable_tma else '✅'} TMA Lowering: {'禁用' if disable_tma else '启用'}")
        print(f"   配置键: {PassConfigKey.TL_DISABLE_TMA_LOWER}")
        print(f"   当前值: {disable_tma}")
        
        disable_wgmma = pass_ctx.config.get(PassConfigKey.TL_DISABLE_WGMMA, False)
        print(f"\n{'❌' if disable_wgmma else '✅'} WGMMA: {'禁用' if disable_wgmma else '启用'}")
        print(f"   配置键: {PassConfigKey.TL_DISABLE_WGMMA}")
        print(f"   当前值: {disable_wgmma}")
        
        disable_warp_spec = pass_ctx.config.get(PassConfigKey.TL_DISABLE_WARP_SPECIALIZED, False)
        print(f"\n{'❌' if disable_warp_spec else '✅'} Warp Specialization: {'禁用' if disable_warp_spec else '启用'}")
        print(f"   配置键: {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED}")
        print(f"   当前值: {disable_warp_spec}")
    
    print("\n" + "=" * 80)
    print("可用的Hopper编译Pass")
    print("=" * 80)
    
    # 检查是否有Hopper特定的编译pass
    from tilelang.engine.phase import (
        allow_tma_and_warp_specialized,
        allow_warp_specialized,
        allow_fence_proxy
    )
    
    with tilelang.transform.PassContext():
        pass_ctx = tilelang.transform.get_pass_context()
        
        can_use_tma = allow_tma_and_warp_specialized(pass_ctx, target)
        print(f"\n{'✅' if can_use_tma else '❌'} allow_tma_and_warp_specialized: {can_use_tma}")
        if can_use_tma:
            print("   → 将使用以下优化pass:")
            print("      • WarpSpecialized()")
            print("      • InjectTmaBarrier()")
            print("      • PipelinePlanning()")
            print("      • RewriteWgmmaSync() [如果is_hopper]")
        
        can_warp_spec = allow_warp_specialized(pass_ctx, target)
        print(f"\n{'✅' if can_warp_spec else '❌'} allow_warp_specialized: {can_warp_spec}")
        
        can_fence = allow_fence_proxy(target)
        print(f"\n{'✅' if can_fence else '❌'} allow_fence_proxy: {can_fence}")
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    if is_h100 and supports_tma and can_use_tma:
        print("\n✅ 完整的Hopper特性支持:")
        print("   • 硬件: H100 (sm_90)")
        print("   • TMA异步内存传输: 启用")
        print("   • WGMMA矩阵乘法: 启用")
        print("   • Warp Specialization: 启用")
        print("\n💡 提示: 这些特性主要用于矩阵乘法(GEMM)优化")
        print("   对于TopK/Radix Selection，这些特性的帮助有限，因为:")
        print("   - TopK不是矩阵乘法，无法使用WGMMA")
        print("   - Radix Selection是计算密集型，而非内存带宽瓶颈")
        print("   - Warp Specialization适合异步流水线，但TopK已经高度并行")
    elif is_h100:
        print("\n⚠️ H100硬件存在，但某些特性未启用:")
        if not supports_tma:
            print("   • TMA支持: 未检测到")
        if not can_use_tma:
            print("   • TMA编译优化: 被禁用或不可用")
    else:
        print("\n❌ 非Hopper架构")
        print(f"   当前GPU: sm_{compute_capability[0]}{compute_capability[1]}")
        print("   Hopper特性需要: sm_90 (H100)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_hopper_support()
