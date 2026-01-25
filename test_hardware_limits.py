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
    print(f"  每个Block最大线程数: {props.max_threads_per_block if hasattr(props, 'max_threads_per_block') else 'N/A'}")
    
    # 尝试获取所有可用属性
    print("\n【可用属性列表】")
    for attr in dir(props):
        if not attr.startswith('_'):
            try:
                value = getattr(props, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                pass
    
    # 使用已知的通用限制值（H100标准值）
    max_threads_per_block = 1024  # CUDA标准限制
    warp_size = 32
    
    print("\n【共享内存和线程限制（H100标准值）】")
    print(f"  每个Block最大线程数: {max_threads_per_block}")
    print(f"  Warp大小: {warp_size}")
    
    # 获取共享内存信息（这些属性应该存在）
    try:
        smem_per_block = props.total_memory // 1024 // 1024  # fallback
        if hasattr(props, 'reserved_shared_memory_per_block'):
            smem_per_block = props.reserved_shared_memory_per_block // 1024
        print(f"  每个Block共享内存: 约 {smem_per_block} KB (具体值取决于配置)")
    except:
        print(f"  每个Block共享内存: H100标准为 164 KB (可配置)")
    
    print("\n【推荐配置建议（基于H100 Compute 9.0）】")
    print(f"  ✅ 安全的BLOCK_SIZE: 256, 512, 1024 (必须 ≤ {max_threads_per_block})")
    print(f"  ✅ H100共享内存配置: 最高可达 232 KB/block (需要设置)")
    print(f"  ✅ 默认安全共享内存: 约 96-164 KB/block")
    
    # 计算不同block size下的共享内存预算（H100默认164KB）
    h100_smem_default = 164  # KB
    print("\n【不同BLOCK_SIZE下的共享内存预算（H100默认164KB）】")
    for block_size in [256, 512, 1024]:
        max_smem_kb = h100_smem_default
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
    
    for smem_size in [2048, 4096, 6144, 8192, 10240, 12288]:
        usage = calc_smem_usage(smem_size)
        status = "✅" if usage <= h100_smem_default else "❌"
        print(f"  SMEM_INPUT_SIZE={smem_size}: {usage:.1f} KB {status}")
    
    print("\n【最终推荐配置（H100优化）】")
    max_safe_smem = h100_smem_default
    
    # H100可以使用更大的共享内存
    if max_safe_smem >= 164:
        recommended_smem = 10240  # 约125KB
        print(f"  BLOCK_SIZE = 1024")
        print(f"  SMEM_INPUT_SIZE = {recommended_smem}")
        print(f"  预计共享内存使用: {calc_smem_usage(recommended_smem):.1f} KB / {max_safe_smem:.1f} KB")
        print(f"  ⚡ H100可以支持更大的缓存，适合k=2048场景")
    elif max_safe_smem >= 100:
        recommended_smem = 6144
        print(f"  BLOCK_SIZE = 1024")
        print(f"  SMEM_INPUT_SIZE = {recommended_smem}")
        print(f"  预计共享内存使用: {calc_smem_usage(recommended_smem):.1f} KB / {max_safe_smem:.1f} KB")
    else:
        recommended_smem = 4096
        print(f"  BLOCK_SIZE = 1024")
        print(f"  SMEM_INPUT_SIZE = {recommended_smem}")
        print(f"  预计共享内存使用: {calc_smem_usage(recommended_smem):.1f} KB / {max_safe_smem:.1f} KB")
    
    print("="*80)

if __name__ == '__main__':
    test_cuda_limits()
