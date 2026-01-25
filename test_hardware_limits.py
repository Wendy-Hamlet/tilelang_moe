"""
测试H100 GPU的硬件限制
查询最大block size和共享内存大小
"""

import torch
import sys

def test_cuda_limits():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    print("="*80)
    print(f"GPU设备: {props.name}")
    print("="*80)
    
    print("\n【基本信息】")
    print(f"  计算能力 (Compute Capability): {props.major}.{props.minor}")
    print(f"  CUDA核心数: {props.multi_processor_count} SMs")
    print(f"  全局内存: {props.total_memory / 1024**3:.2f} GB")
    
    print("\n【线程限制】")
    print(f"  每个Block最大线程数: {props.max_threads_per_block}")
    print(f"  每个SM最大线程数: {props.max_threads_per_multi_processor}")
    print(f"  每个Block最大维度 (x,y,z): ({props.max_block_dim_x}, {props.max_block_dim_y}, {props.max_block_dim_z})")
    print(f"  Grid最大维度 (x,y,z): ({props.max_grid_dim_x}, {props.max_grid_dim_y}, {props.max_grid_dim_z})")
    
    print("\n【共享内存限制】")
    print(f"  每个Block共享内存: {props.shared_memory_per_block / 1024:.1f} KB")
    print(f"  每个SM共享内存: {props.shared_memory_per_multiprocessor / 1024:.1f} KB")
    
    print("\n【寄存器和Warp】")
    print(f"  每个Block寄存器数: {props.regs_per_block}")
    print(f"  Warp大小: {props.warp_size}")
    
    print("\n【推荐配置建议】")
    print(f"  ✅ 安全的BLOCK_SIZE: 256, 512, 1024 (必须 ≤ {props.max_threads_per_block})")
    print(f"  ✅ 安全的共享内存: ≤ {props.shared_memory_per_block / 1024:.1f} KB/block")
    
    # 计算不同block size下的共享内存预算
    print("\n【不同BLOCK_SIZE下的共享内存预算】")
    for block_size in [256, 512, 1024]:
        max_smem_kb = props.shared_memory_per_block / 1024
        print(f"  BLOCK_SIZE={block_size}: 共享内存预算 {max_smem_kb:.1f} KB")
        
        # 估算可以分配的数组大小
        overhead = 1  # histogram等固定开销约1KB
        available = max_smem_kb - overhead
        
        # 假设我们要分配 s_input_idx[2, SMEM_SIZE] (int32) + s_cached_uint32[SMEM_SIZE] (uint32)
        # 总共需要: SMEM_SIZE * (2*4 + 4) = SMEM_SIZE * 12 bytes
        max_smem_size = int((available * 1024) / 12)
        print(f"    -> 可分配SMEM_INPUT_SIZE ≈ {max_smem_size} (约 {max_smem_size * 12 / 1024:.1f} KB)")
    
    print("\n" + "="*80)
    
    # 测试实际TileLang kernel的共享内存需求
    print("\n【TileLang Kernel共享内存需求计算】")
    
    def calc_smem_usage(smem_input_size):
        """计算给定SMEM_INPUT_SIZE的共享内存使用量"""
        s_histogram = 256 * 4  # [256] int32
        s_threshold = 1 * 4
        s_num_input = 2 * 4
        s_input_idx = 2 * smem_input_size * 4  # [2, SMEM_INPUT_SIZE] int32
        s_cached_uint32 = smem_input_size * 4  # [SMEM_INPUT_SIZE] uint32
        
        total = s_histogram + s_threshold + s_num_input + s_input_idx + s_cached_uint32
        return total / 1024  # KB
    
    for smem_size in [2048, 4096, 6144, 8192]:
        usage = calc_smem_usage(smem_size)
        status = "✅" if usage <= props.shared_memory_per_block / 1024 else "❌"
        print(f"  SMEM_INPUT_SIZE={smem_size}: {usage:.1f} KB {status}")
    
    print("\n【最终推荐配置】")
    max_safe_smem = props.shared_memory_per_block / 1024
    
    # 保守估计
    if max_safe_smem >= 100:
        recommended_smem = 6144
        print(f"  BLOCK_SIZE = 1024")
        print(f"  SMEM_INPUT_SIZE = {recommended_smem}")
        print(f"  预计共享内存使用: {calc_smem_usage(recommended_smem):.1f} KB / {max_safe_smem:.1f} KB")
    elif max_safe_smem >= 64:
        recommended_smem = 4096
        print(f"  BLOCK_SIZE = 1024")
        print(f"  SMEM_INPUT_SIZE = {recommended_smem}")
        print(f"  预计共享内存使用: {calc_smem_usage(recommended_smem):.1f} KB / {max_safe_smem:.1f} KB")
    else:
        recommended_smem = 2048
        print(f"  BLOCK_SIZE = 512")
        print(f"  SMEM_INPUT_SIZE = {recommended_smem}")
        print(f"  预计共享内存使用: {calc_smem_usage(recommended_smem):.1f} KB / {max_safe_smem:.1f} KB")
    
    print("="*80)

if __name__ == '__main__':
    test_cuda_limits()
