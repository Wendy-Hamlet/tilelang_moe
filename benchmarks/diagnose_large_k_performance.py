"""
诊断Large-k性能问题
分析为什么Large-k比Official慢
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

print("="*120)
print("Large-k性能问题诊断")
print("="*120)

# H100 GPU信息
device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)

print(f"\n【H100 GPU规格】")
print(f"  共享内存/Block (默认): {props.shared_memory_per_block / 1024:.0f} KB")
print(f"  共享内存/Block (opt-in): {props.shared_memory_per_block_optin / 1024:.0f} KB")
print(f"  共享内存/SM: {props.shared_memory_per_multiprocessor / 1024:.0f} KB")
print(f"  最大Block/SM: {props.max_blocks_per_multiprocessor}")
print(f"  最大线程/Block: {props.max_threads_per_block}")
print(f"  Warp大小: {props.warp_size}")

# 分析不同版本的资源占用
print(f"\n【Official版本 (SMEM_INPUT_SIZE=4096)】")
official_smem = (
    256 * 4 +           # s_histogram: 256 int32 = 1KB
    4 +                 # s_threshold_bin_id: 1 int32
    8 +                 # s_num_input: 2 int32
    4096 * 2 * 4        # s_input_idx: [2, 4096] int32 = 32KB
)
print(f"  共享内存使用: {official_smem / 1024:.1f} KB")
print(f"  在48KB限制内: {'✅' if official_smem < 48*1024 else '❌'}")

# 估算occupancy
max_blocks_official = int(props.shared_memory_per_multiprocessor / official_smem)
print(f"  理论最大Block/SM: {max_blocks_official}")
print(f"  受限于: {'共享内存' if max_blocks_official < props.max_blocks_per_multiprocessor else 'GPU硬件'}")

print(f"\n【Optimized版本 (SMEM_INPUT_SIZE=4096 + padding)】")
optimized_smem = (
    (256 + 1) * 4 +     # s_histogram: 257 int32 (padding)
    4 +                 # s_threshold_bin_id
    8 +                 # s_num_input
    4096 * 2 * 4 +      # s_input_idx
    (4096 + 1) * 4      # s_cached_uint32: 4097 uint32 = 16KB (padding)
)
print(f"  共享内存使用: {optimized_smem / 1024:.1f} KB")
print(f"  在48KB限制内: {'✅' if optimized_smem < 48*1024 else '❌'}")
max_blocks_optimized = int(props.shared_memory_per_multiprocessor / optimized_smem)
print(f"  理论最大Block/SM: {max_blocks_optimized}")
print(f"  受限于: {'共享内存' if max_blocks_optimized < props.max_blocks_per_multiprocessor else 'GPU硬件'}")

print(f"\n【Large-k版本 (SMEM_INPUT_SIZE=12288)】")
large_k_smem = (
    (256 + 1) * 4 +     # s_histogram: 257 int32
    4 +                 # s_threshold_bin_id
    8 +                 # s_num_input
    12288 * 2 * 4 +     # s_input_idx: [2, 12288] int32 = 96KB
    (12288 + 1) * 4     # s_cached_uint32: 12289 uint32 = 49KB
)
print(f"  共享内存使用: {large_k_smem / 1024:.1f} KB")
print(f"  在48KB限制内: {'✅' if large_k_smem < 48*1024 else '❌'}")
print(f"  在232KB限制内: {'✅' if large_k_smem < 232*1024 else '❌'}")
max_blocks_large_k = int(props.shared_memory_per_multiprocessor / large_k_smem)
print(f"  理论最大Block/SM: {max_blocks_large_k}")
print(f"  受限于: {'共享内存' if max_blocks_large_k < props.max_blocks_per_multiprocessor else 'GPU硬件'}")

# Occupancy对比
print(f"\n【Occupancy对比】")
print(f"  Official:  {max_blocks_official} blocks/SM")
print(f"  Optimized: {max_blocks_optimized} blocks/SM") 
print(f"  Large-k:   {max_blocks_large_k} blocks/SM")

if max_blocks_large_k < max_blocks_official:
    reduction = (max_blocks_official - max_blocks_large_k) / max_blocks_official * 100
    print(f"\n⚠️  Large-k的occupancy降低了 {reduction:.1f}%！")
    print(f"  这意味着SM的利用率降低，隐藏延迟的能力下降")

# Bank conflict分析
print(f"\n【Bank Conflict分析】")
print(f"  共享内存Bank数: 32 (H100)")
print(f"  Bank宽度: 4 bytes (1个int32或uint32)")
print(f"\n  Official: 4096元素缓存")
print(f"    线程0访问[0], 线程1访问[1]... → 可能有conflict")
print(f"\n  Optimized: 4097元素缓存 (padding)")
print(f"    padding可能缓解conflict，但不保证完全消除")
print(f"\n  Large-k: 12289元素缓存 (padding)")
print(f"    缓存太大，1024线程同时访问 → conflict严重")
print(f"    12289 mod 32 = {12289 % 32} (padding到32的倍数+1)")

# 性能预测
print(f"\n【性能预测】")
print(f"  Official: 简单高效，occupancy高，conflict少")
print(f"  Optimized: 小缓存开销，适合小k")
print(f"  Large-k: 大缓存开销 + occupancy降低 + conflict增加")
print(f"\n  预期结果: Official ≥ Optimized > Large-k")
print(f"  实际观察: Official最快，Large-k最慢 ✅ 符合预测")

print(f"\n【结论】")
print("="*120)
print("Large-k的148KB共享内存导致：")
print("  1. ❌ Occupancy降低 - 每个SM能运行的block数减少")
print("  2. ❌ Bank conflict增加 - 12288元素缓存访问冲突严重")
print("  3. ❌ 寄存器压力 - 可能导致register spilling")
print("")
print("官方版本的优势：")
print("  1. ✅ 动态候选集 - 只缓存实际需要的threshold bin元素")
print("  2. ✅ 小共享内存 - 高occupancy")
print("  3. ✅ 简单逻辑 - 编译器优化更好")
print("")
print("建议：")
print("  - 放弃Large-k优化（反而更慢）")
print("  - 保留Optimized用于小k场景")
print("  - 大k场景直接使用Official版本")
print("="*120)
