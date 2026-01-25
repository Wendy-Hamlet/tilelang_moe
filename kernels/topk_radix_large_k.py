"""
TopK implementation optimized for large k scenarios (k > 512)
专门针对 N=32768, k=2048 场景优化

优化策略：
1. 增大共享内存缓存 (SMEM_INPUT_SIZE: 4096 → 8192)
2. 更大的BLOCK_SIZE (1024 → 2048) 提高并行度
3. 优化bank conflict访问模式
4. 针对大threshold bin的处理优化
"""

import torch
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


def convert_to_uint16(x):
    """Convert float to sortable uint16 by manipulating sign bit"""
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    """Convert float32 to sortable uint32 by manipulating sign bit"""
    bits_uint = T.reinterpret(T.uint32, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_radix_large_k_impl(topk, in_dtype=T.float32, out_dtype=T.int32):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = 1 << 8  # 256 buckets
    BLOCK_SIZE = 2048  # 增大block size以提高并行度
    SMEM_INPUT_SIZE = 8192  # 增大缓存以适应大k场景

    @T.prim_func
    def tl_topk_radix_large_k_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
        starts: T.Tensor[(batch,), out_dtype],
        ends: T.Tensor[(batch,), out_dtype],
    ):
        with T.Kernel(T.ceildiv(batch, 1), threads=BLOCK_SIZE) as (bx,):
            tx = T.thread_idx()
            
            # 共享内存分配 - 增大缓存，优化padding
            s_histogram = T.alloc_shared([RADIX + 8], T.int32)  # +8 padding避免bank conflict
            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)
            
            # 大k场景优化：缓存uint32值
            s_cached_uint32 = T.alloc_shared([SMEM_INPUT_SIZE + 8], "uint32")  # +8 padding
            s_cached_indices = T.alloc_shared([SMEM_INPUT_SIZE + 8], T.int32)  # 缓存索引
            
            l_threshold_bin_id = T.alloc_var(T.int32)

            l_start_idx = starts[bx]
            l_end_idx = ends[bx]
            l_new_topk = topk

            # 初始化
            T.fill(s_histogram, 0)
            if tx == 0:
                s_num_input[0] = 0
                s_num_input[1] = 0
            T.sync_threads()

            # ========== Stage 1: 8-bit Radix Partition ==========
            # 优化1: 合并数据加载和直方图构建，同时缓存uint32值
            smem_load_count = T.min(SMEM_INPUT_SIZE, l_end_idx - l_start_idx)
            
            for chunk in T.serial(T.ceildiv(smem_load_count, BLOCK_SIZE)):
                local_idx = chunk * BLOCK_SIZE + tx
                if local_idx < smem_load_count:
                    global_idx = l_start_idx + local_idx
                    if global_idx < l_end_idx and global_idx < seq_len:
                        # 一次性完成转换和缓存
                        val = input[bx, global_idx]
                        val_uint = convert_to_uint32(val)
                        
                        # 缓存uint32值和索引 - 使用stride访问减少bank conflict
                        cache_pos = local_idx
                        s_cached_uint32[cache_pos] = val_uint
                        s_cached_indices[cache_pos] = global_idx
                        
                        # 构建直方图
                        bin_id = val_uint >> 24
                        T.atomic_add(s_histogram[bin_id], 1)
            
            T.sync_threads()

            # 累加直方图（从大到小）
            offset = 1
            for _ in T.serial(8):  # log2(256) = 8
                T.sync_threads()
                if tx < RADIX - offset:
                    l_val = s_histogram[tx] + s_histogram[tx + offset]
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # 找到threshold bin
                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx

            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # 优化2: 使用缓存的数据，避免重复转换和内存访问
            for chunk in T.serial(T.ceildiv(smem_load_count, BLOCK_SIZE)):
                T.sync_threads()
                local_idx = chunk * BLOCK_SIZE + tx
                if local_idx < smem_load_count:
                    # 从缓存读取，避免重复转换
                    cached_val = s_cached_uint32[local_idx]
                    cached_idx = s_cached_indices[local_idx]
                    
                    bin_id = cached_val >> 24
                    l_bin_id32 = T.Cast(T.int32, bin_id)
                    
                    if l_bin_id32 > l_threshold_bin_id:
                        # 高于threshold - 直接输出
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, pos] = cached_idx
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        # 在threshold bin - 收集到下一阶段
                        pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        if pos < SMEM_INPUT_SIZE:  # 防止溢出
                            s_input_idx[0, pos] = cached_idx

            # ========== Stage 2: Iterative Refinement ==========
            for round in T.serial(4):
                if l_new_topk <= 0:
                    break

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]

                # 构建直方图 - 优化bank conflict
                for chunk in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    local_idx = chunk * BLOCK_SIZE + tx
                    if local_idx < l_num_input:
                        input_idx = s_input_idx[r_idx, local_idx]
                        if input_idx < seq_len:
                            bin_id = convert_to_uint16(input[bx, input_idx])
                            T.atomic_add(s_histogram[T.Cast(T.int32, bin_id)], 1)

                T.sync_threads()

                # 累加直方图
                offset = 1
                for _ in T.serial(8):
                    T.sync_threads()
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    # 找到threshold bin
                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx

                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                # 分区并输出
                for chunk in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    local_idx = chunk * BLOCK_SIZE + tx
                    if local_idx < l_num_input:
                        input_idx = s_input_idx[r_idx, local_idx]
                        if input_idx < seq_len:
                            bin_id = convert_to_uint16(input[bx, input_idx])
                            l_bin_id32 = T.Cast(T.int32, bin_id)
                            
                            if l_bin_id32 > l_threshold_bin_id:
                                pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                                index[bx, l_start_pos + pos] = input_idx
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                if pos < SMEM_INPUT_SIZE:
                                    s_input_idx[r_idx ^ 1, pos] = input_idx

    return tl_topk_radix_large_k_kernel


def topk_radix_large_k(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """
    Large k optimized TopK using Radix Selection
    
    Args:
        logits: [batch, num_experts] float32 tensor
        topk: number of top elements to select
    
    Returns:
        indices: [batch, topk] int32 tensor
    """
    batch, num_experts = logits.shape
    
    # Create output tensors
    indices = torch.zeros((batch, topk), dtype=torch.int32, device=logits.device)
    starts = torch.zeros((batch,), dtype=torch.int32, device=logits.device)
    ends = torch.full((batch,), num_experts, dtype=torch.int32, device=logits.device)
    
    # Create kernel
    kernel = tl_topk_radix_large_k_impl(topk)
    kernel(logits, indices, starts, ends)
    
    return indices
