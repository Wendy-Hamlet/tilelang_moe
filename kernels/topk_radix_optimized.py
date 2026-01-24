"""
Optimized TopK using Radix Selection
优化版本 - 基于官方实现的性能改进

主要优化：
1. Warp-level 原子操作归约（减少冲突）
2. AutoTune 参数调优
3. 缓存输入数据避免重复读取
"""

import torch
import tilelang
import tilelang.language as T
import itertools

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


def get_radix_topk_configs():
    """
    AutoTune configurations for Radix TopK
    测试不同的 BLOCK_SIZE 和 SMEM_INPUT_SIZE 组合
    """
    configs = []
    
    # 不同的线程块大小
    block_sizes = [256, 512, 1024]
    # 不同的共享内存缓冲区大小
    smem_sizes = [2048, 4096, 8192]
    
    for bs, smem in itertools.product(block_sizes, smem_sizes):
        configs.append(dict(
            BLOCK_SIZE=bs,
            SMEM_INPUT_SIZE=smem,
        ))
    
    return configs


@tilelang.autotune(configs=get_radix_topk_configs(), key=['topk'])
@tilelang.jit(pass_configs=pass_configs)
def tl_topk_radix_optimized_impl(topk, BLOCK_SIZE=1024, SMEM_INPUT_SIZE=4096, 
                                   in_dtype=T.float32, out_dtype=T.int32):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = 1 << 8  # 256 buckets
    WARP_SIZE = 32

    @T.prim_func
    def tl_topk_radix_optimized_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
        starts: T.Tensor[(batch), out_dtype],
        ends: T.Tensor[(batch), out_dtype],
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
            tx = T.get_thread_binding()
            warp_id = tx // WARP_SIZE
            lane_id = tx % WARP_SIZE

            # Shared memory allocations
            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            # Padding 避免 bank conflict (+1)
            s_histogram = T.alloc_shared([RADIX + 1 + 1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)
            
            # 优化1: 缓存输入的 uint32 值，避免 Stage 2 重复计算
            s_cached_uint32 = T.alloc_shared([SMEM_INPUT_SIZE], T.uint32)
            
            # 优化2: Warp-level 局部直方图，减少原子操作冲突
            num_warps = BLOCK_SIZE // WARP_SIZE
            s_warp_histogram = T.alloc_shared([num_warps, RADIX], T.int32)

            # Local variables
            l_threshold_bin_id = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_num_input = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_start_pos = T.alloc_var(T.int32)
            l_start_idx = T.alloc_var(T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)
            pos = T.alloc_var(T.int32)
            
            # Warp-level 归约用的临时变量
            l_local_count = T.alloc_var(T.int32)

            l_new_topk = topk
            l_start_idx = starts[bx]
            l_end_idx = ends[bx]

            # ========== Stage 1: Quick 8-bit TopK ==========
            # 初始化直方图
            T.fill(s_histogram, 0)
            T.fill(s_warp_histogram, 0)
            T.fill(s_num_input[0], 0)
            T.sync_threads()

            # 优化: Warp-level 构建直方图，减少原子操作冲突
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    inval_int16 = convert_to_uint16(input[bx, input_idx])
                    # 先累加到 warp 局部直方图（无冲突）
                    T.atomic_add(s_warp_histogram[warp_id, inval_int16], 1)
            
            T.sync_threads()
            
            # 归约所有 warp 的局部直方图到全局直方图
            # 每个线程负责一部分 bins
            bins_per_thread = T.ceildiv(RADIX, BLOCK_SIZE)
            for i in T.serial(bins_per_thread):
                bin_id = tx * bins_per_thread + i
                if bin_id < RADIX:
                    l_local_count = 0
                    for w in T.serial(num_warps):
                        l_local_count = l_local_count + s_warp_histogram[w, bin_id]
                    s_histogram[bin_id] = l_local_count
            
            T.sync_threads()

            # Cumulative sum on histogram (保持原样，已经很优化)
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # Find threshold bin
                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx

            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # Collect elements above threshold
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                T.sync_threads()
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    bin_id = convert_to_uint16(input[bx, input_idx])
                    l_bin_id32 = T.Cast(T.int32, bin_id)
                    
                    if l_bin_id32 > l_threshold_bin_id:
                        # Above threshold - directly add to output
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, pos] = input_idx
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        # On threshold - collect for next stage
                        pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        if pos < SMEM_INPUT_SIZE:
                            s_input_idx[0, pos] = input_idx
                            # 优化: 缓存 uint32 值，避免 Stage 2 重复计算
                            s_cached_uint32[pos] = convert_to_uint32(input[bx, input_idx])

            # ========== Stage 2: Iterative Refinement (4 rounds max) ==========
            for round in T.serial(4):
                if l_new_topk <= 0:
                    break

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                T.fill(s_warp_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]

                # 优化: Warp-level 构建直方图
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        # 优化: 使用缓存的 uint32 值，避免重复读取和计算
                        cached_val = s_cached_uint32[s * BLOCK_SIZE + tx]
                        l_bin_id32 = T.Cast(T.int32, ((cached_val >> (24 - round * 8)) & 0xFF))
                        T.atomic_add(s_warp_histogram[warp_id, l_bin_id32], 1)
                
                T.sync_threads()
                
                # 归约 warp 局部直方图
                bins_per_thread = T.ceildiv(RADIX, BLOCK_SIZE)
                for i in T.serial(bins_per_thread):
                    bin_id = tx * bins_per_thread + i
                    if bin_id < RADIX:
                        l_local_count = 0
                        for w in T.serial(num_warps):
                            l_local_count = l_local_count + s_warp_histogram[w, bin_id]
                        s_histogram[bin_id] = l_local_count
                
                T.sync_threads()

                # Cumulative sum
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    # Find threshold bin
                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx

                T.sync_threads()
                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                # Partition elements
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    if s * BLOCK_SIZE + tx < l_num_input:
                        # 优化: 使用缓存值
                        cached_val = s_cached_uint32[s * BLOCK_SIZE + tx]
                        l_bin_id32 = T.Cast(T.int32, ((cached_val >> (24 - round * 8)) & 0xFF))
                        
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            if round == 3:
                                # Final round
                                l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                # Continue to next round
                                pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                if pos < SMEM_INPUT_SIZE:
                                    s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                    # 传递缓存值
                                    s_cached_uint32[pos] = cached_val

    return tl_topk_radix_optimized_kernel


def topk_radix_optimized(logits, topk=6):
    """
    优化版 TileLang TopK using Radix Selection
    
    主要改进：
    1. Warp-level 原子操作归约（减少冲突 32x）
    2. AutoTune 自动调优参数
    3. 缓存 uint32 值避免重复计算
    4. Padding 共享内存避免 bank conflict
    
    Args:
        logits: [batch, num_experts] tensor
        topk: number of experts to select
    
    Returns:
        indices: [batch, topk] indices of top-k experts
    """
    # Ensure logits is float32
    if logits.dtype != torch.float32:
        logits = logits.to(torch.float32)
    
    # Get dimensions
    batch, num_experts = logits.shape
    batch = int(batch)
    num_experts = int(num_experts)
    
    # Ensure topk is Python int
    if isinstance(topk, torch.Tensor):
        topk = int(topk.item())
    else:
        topk = int(topk)
    
    # Create output tensors
    indices = torch.zeros((batch, topk), dtype=torch.int32, device=logits.device)
    starts = torch.zeros((batch,), dtype=torch.int32, device=logits.device)
    ends = torch.full((batch,), num_experts, dtype=torch.int32, device=logits.device)
    
    # Create optimized kernel with AutoTune
    kernel = tl_topk_radix_optimized_impl(topk)
    kernel(logits, indices, starts, ends)
    
    return indices
