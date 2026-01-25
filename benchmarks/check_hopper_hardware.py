"""
检查H100硬件是否支持Hopper特性

简化版本 - 仅使用PyTorch，不依赖TileLang构建
"""

import torch


def check_hopper_hardware():
    """检查硬件是否为H100 Hopper架构"""
    print("=" * 80)
    print("H100 Hopper硬件检测")
    print("=" * 80)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    # 获取GPU信息
    device_name = torch.cuda.get_device_name(0)
    print(f"\n✅ GPU名称: {device_name}")
    
    # 获取计算能力
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability
    print(f"✅ Compute Capability: {major}.{minor} (sm_{major}{minor})")
    
    # 获取GPU属性
    props = torch.cuda.get_device_properties(0)
    print(f"\n{'─' * 80}")
    print("GPU硬件规格:")
    print(f"{'─' * 80}")
    print(f"  • 总显存: {props.total_memory / 1024**3:.1f} GB")
    print(f"  • SM数量: {props.multi_processor_count}")
    print(f"  • 最大线程/Block: {props.max_threads_per_block}")
    print(f"  • 最大Block维度: {props.max_block_dim}")
    print(f"  • 最大Grid维度: {props.max_grid_dim}")
    print(f"  • Warp大小: {props.warp_size}")
    print(f"  • 共享内存/Block: {props.shared_memory_per_block / 1024:.1f} KB")
    print(f"  • 共享内存/SM: {props.shared_memory_per_multiprocessor / 1024:.1f} KB")
    print(f"  • 寄存器/Block: {props.regs_per_block}")
    print(f"  • 最大寄存器/Thread: {props.regs_per_multiprocessor // (props.max_threads_per_multiprocessor // props.warp_size)}")
    
    # 检查是否为Hopper架构
    print(f"\n{'=' * 80}")
    print("Hopper架构检测")
    print(f"{'=' * 80}")
    
    is_hopper = (major == 9 and minor == 0)
    
    if is_hopper:
        print("\n✅ 检测到Hopper架构 (H100)")
        print(f"   • Compute Capability: sm_90")
        print(f"   • 硬件支持的Hopper特性:")
        print(f"      ✓ TMA (Tensor Memory Accelerator) - 异步内存传输")
        print(f"      ✓ WGMMA (Warp Group Matrix Multiply) - 线程束组矩阵乘")
        print(f"      ✓ TMEM (Tensor Memory) - 张量内存")
        print(f"      ✓ Warp Specialization - 线程束专用化")
        print(f"      ✓ Async Copy with Barrier - 带屏障的异步拷贝")
    else:
        print(f"\n❌ 非Hopper架构")
        print(f"   当前: sm_{major}{minor}")
        print(f"   Hopper需要: sm_90 (H100)")
        if major >= 9:
            print(f"   → 可能是更新的架构")
        elif major == 8:
            print(f"   → Ampere架构 (A100/A40/A30等)")
            print(f"   → 支持: Async Copy, LDMatrix, Tensor Core")
        elif major == 7:
            print(f"   → Turing/Volta架构")
        else:
            print(f"   → 较旧的架构")
    
    print(f"\n{'=' * 80}")
    print("TileLang Hopper特性说明")
    print(f"{'=' * 80}")
    
    if is_hopper:
        print("\n当H100启用Hopper特性时，TileLang会:")
        print("  1. 自动检测: have_tma(target) = True")
        print("  2. 自动检测: is_hopper(target) = True")
        print("  3. 启用编译pass:")
        print("     • WarpSpecialized() - 线程束专用化")
        print("     • InjectTmaBarrier() - TMA屏障注入")
        print("     • RewriteWgmmaSync() - WGMMA同步重写")
        print("     • PipelinePlanning() - 流水线规划")
        print("\n⚠️ 重要提示: 这些特性主要用于GEMM(矩阵乘法)优化")
        print("  对于TopK/Radix Selection算法:")
        print("  ✗ WGMMA - 不适用(不是矩阵乘法)")
        print("  ✗ TMA - 帮助有限(已经是计算密集型)")
        print("  ✗ Warp Specialization - 可能适用，但TopK已高度并行")
        print("\n📊 结论: Hopper特性对当前TopK优化帮助不大")
        print("  建议策略:")
        print("  • 小k场景 (k<512): 使用Optimized版本 (3-8%加速)")
        print("  • 大k场景 (k≥512): 使用Official版本")
        print("  • 避免Large-k版本 (占用过多共享内存导致occupancy崩溃)")
    
    print(f"\n{'=' * 80}")
    
    return is_hopper


def check_kernel_configs():
    """检查当前内核配置的共享内存占用"""
    print("\n" + "=" * 80)
    print("当前TopK内核配置分析")
    print("=" * 80)
    
    props = torch.cuda.get_device_properties(0)
    smem_per_sm = props.shared_memory_per_multiprocessor
    
    configs = [
        ("Official", 1024, 4096, [
            ("s_histogram", 257, "int32"),  # 包含padding
            ("s_input_idx (Stage 1)", 4096, "int32"),
            ("s_input_idx (Stage 2)", 4096, "int32"),
        ]),
        ("Optimized", 1024, 4096, [
            ("s_histogram", 257, "int32"),
            ("s_input_idx (Stage 1)", 4096, "int32"),
            ("s_input_idx (Stage 2)", 4096, "int32"),
            ("s_cached_uint32", 4097, "uint32"),  # 包含padding
        ]),
        ("Large-k", 1024, 12288, [
            ("s_histogram", 257, "int32"),
            ("s_input_idx (Stage 1)", 12288, "int32"),
            ("s_input_idx (Stage 2)", 12288, "int32"),
            ("s_cached_uint32", 12289, "uint32"),
        ]),
    ]
    
    for name, block_size, input_size, buffers in configs:
        print(f"\n{name}:")
        print(f"  Block Size: {block_size} threads")
        print(f"  Input Size: {input_size}")
        print(f"  共享内存分配:")
        
        total_smem = 0
        for buf_name, size, dtype in buffers:
            bytes_per_elem = 4  # int32 or uint32
            buf_size = size * bytes_per_elem
            total_smem += buf_size
            print(f"    • {buf_name}: {size} × {bytes_per_elem}B = {buf_size/1024:.1f} KB")
        
        print(f"  总共享内存: {total_smem/1024:.1f} KB")
        
        # 计算occupancy
        max_blocks = smem_per_sm // total_smem
        print(f"  理论最大Blocks/SM: {max_blocks}")
        print(f"  Occupancy影响: ", end="")
        if max_blocks >= 6:
            print(f"✅ 很好 ({max_blocks} blocks/SM)")
        elif max_blocks >= 3:
            print(f"⚠️ 中等 ({max_blocks} blocks/SM)")
        else:
            print(f"❌ 差 ({max_blocks} blocks/SM, occupancy崩溃)")
    
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    is_h100 = check_hopper_hardware()
    
    if torch.cuda.is_available():
        check_kernel_configs()
