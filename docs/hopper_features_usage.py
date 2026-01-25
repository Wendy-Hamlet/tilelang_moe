"""
TileLang Hopper特性启用方式说明

基于TileLang示例代码分析，Hopper特性的启用/禁用是通过编译配置实现的。
"""

# ============================================================================
# 方法1: 使用 @tilelang.jit 的 pass_configs 参数
# ============================================================================

import tilelang
import tilelang.language as T

# 示例1: 禁用TMA和Warp Specialization（强制使用传统方式）
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,         # 禁用TMA
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,  # 禁用Warp Specialization
    }
)
def my_kernel_no_hopper(...):
    # 即使在H100上也不使用Hopper特性
    pass


# 示例2: 启用Hopper特性（默认行为，H100自动启用）
@tilelang.jit()  # 或者 @tilelang.jit(pass_configs={})
def my_kernel_with_hopper(...):
    # 在H100上自动启用TMA、WGMMA、Warp Specialization
    pass


# 示例3: 只禁用WGMMA，保留其他Hopper特性
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,  # 禁用WGMMA矩阵乘法指令
    }
)
def my_kernel_no_wgmma(...):
    pass


# ============================================================================
# 方法2: 在代码中根据硬件选择不同的实现
# ============================================================================

def check_hopper():
    """检测是否为H100"""
    import torch
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor) == (9, 0)


@tilelang.jit()
def my_adaptive_kernel(...):
    is_hopper = check_hopper()
    
    @T.prim_func
    def main(...):
        # 根据硬件选择不同的代码路径
        if is_hopper:
            # 使用Hopper特定的intrinsic
            T.c2d_im2col(...)  # Hopper的im2col硬件加速
        else:
            # 使用通用实现
            for i, j in T.Parallel(...):
                # 手动实现的im2col
                pass
    
    return main


# ============================================================================
# TileLang编译配置键说明
# ============================================================================

"""
主要的Hopper相关配置键（在 tilelang.PassConfigKey 中）:

1. TL_DISABLE_TMA_LOWER: bool (默认 False)
   - 禁用TMA (Tensor Memory Accelerator) lowering
   - TMA用于异步内存传输，Hopper架构专用
   - 适用于: GEMM、Attention等内存密集型操作

2. TL_DISABLE_WGMMA: bool (默认 False)
   - 禁用WGMMA (Warp Group Matrix Multiply Accumulate)
   - Hopper的矩阵乘法专用指令
   - 适用于: GEMM相关操作

3. TL_DISABLE_WARP_SPECIALIZED: bool (默认 False)
   - 禁用Warp Specialization
   - 允许不同warp执行不同任务（生产者/消费者）
   - 适用于: 需要流水线优化的kernel

其他常用配置键:

4. TL_ENABLE_FAST_MATH: bool (默认 False)
   - 启用fast math（传递 --use_fast_math 给nvcc）

5. TL_DISABLE_VECTORIZE_256: bool (默认 False)
   - 禁用256位向量化（LDG.256/STG.256）

6. TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: bool (默认 False)
   - 激进的共享内存合并优化
"""


# ============================================================================
# 实际应用场景
# ============================================================================

# 场景1: FusedMoE（DeepSeek示例）
# 禁用Hopper特性，因为MoE的访存模式不规则，TMA收益有限
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fusedmoe_kernel(...):
    """MoE kernel - 禁用Hopper特性"""
    pass


# 场景2: GEMM（标准矩阵乘法）
# 默认启用所有Hopper特性以获得最佳性能
@tilelang.jit()  # 默认配置，H100自动启用TMA+WGMMA
def gemm_kernel(...):
    """GEMM kernel - 启用所有Hopper特性"""
    @T.prim_func
    def main(A, B, C):
        with T.Kernel(...) as (...):
            A_shared = T.alloc_shared(...)
            B_shared = T.alloc_shared(...)
            C_local = T.alloc_fragment(...)
            
            T.clear(C_local)
            for k in T.Pipelined(..., num_stages=3):
                T.copy(A[...], A_shared)  # 可能使用TMA
                T.copy(B[...], B_shared)  # 可能使用TMA
                T.gemm(A_shared, B_shared, C_local)  # 可能使用WGMMA
            
            T.copy(C_local, C[...])
    
    return main


# ============================================================================
# 对于TopK/Radix Selection的建议
# ============================================================================

"""
TopK算法（Radix Selection）不适合使用Hopper特性，原因:

1. 不使用矩阵乘法
   ❌ WGMMA完全不适用（TopK是排序/选择算法，不是矩阵运算）

2. 计算密集型，非内存瓶颈
   ❌ TMA帮助有限（TopK的瓶颈是计算，不是内存带宽）
   ❌ 已经是随机访问模式，TMA的连续访问优化无用

3. 已经高度并行
   ❌ Warp Specialization意义不大（所有warp都在做相同的计算）

建议配置:
"""

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def topk_radix_selection(...):
    """
    TopK kernel建议禁用Hopper特性
    
    理由:
    1. 算法本质不匹配（排序 vs 矩阵乘法）
    2. 避免不必要的编译复杂度
    3. 减少寄存器压力和共享内存使用
    """
    pass


# ============================================================================
# 如何验证Hopper特性是否生效
# ============================================================================

def verify_hopper_features():
    """
    验证方法:
    1. 查看生成的CUDA代码 - kernel.get_kernel_source()
       - 搜索 "cp.async.bulk" (TMA指令)
       - 搜索 "wgmma" (WGMMA指令)
    
    2. 查看编译日志
       - TileLang会打印使用的pass
    
    3. 性能对比
       - 禁用vs启用Hopper特性的性能差异
       - GEMM: 启用Hopper通常快30-50%
       - TopK: 差异很小或无差异
    """
    import tilelang
    
    # 创建kernel
    kernel = gemm_kernel(1024, 1024, 1024, 128, 128, 32)
    
    # 查看CUDA源码
    source = kernel.get_kernel_source()
    
    # 检查是否包含Hopper指令
    has_tma = "cp.async.bulk" in source
    has_wgmma = "wgmma" in source
    
    print(f"TMA enabled: {has_tma}")
    print(f"WGMMA enabled: {has_wgmma}")
    
    return has_tma, has_wgmma


# ============================================================================
# 总结
# ============================================================================

"""
TileLang Hopper特性启用总结:

✅ 默认行为:
   - H100硬件: 自动启用TMA、WGMMA、Warp Specialization
   - 非H100: 自动使用传统实现

✅ 手动控制:
   - 通过 @tilelang.jit(pass_configs={...}) 配置
   - 可以禁用特定特性，保留其他特性

✅ 适用场景:
   - GEMM/MatMul: 强烈推荐启用所有Hopper特性
   - Attention: 推荐启用TMA和Warp Specialization
   - TopK/Sort: 不推荐启用（收益小，复杂度高）

❌ 对于当前的TopK优化:
   - Hopper特性无法解决occupancy collapse问题
   - 建议使用传统优化方法（减少共享内存使用）
   - 或者使用AutoTune自适应选择kernel
"""
