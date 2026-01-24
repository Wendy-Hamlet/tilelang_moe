"""
TopK implementation using Radix Selection algorithm
Based on tilelang/examples/deepseek_v32/topk_selector.py
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
    # Flip all bits if negative, else set sign bit
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
def tl_topk_radix_impl(topk, in_dtype=T.float32, out_dtype=T.int32):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = 1 << 8  # 256 buckets
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = 4096  # Assume threshold bucket < 4K after first pass

    @T.prim_func
    def tl_topk_radix_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
        starts: T.Tensor[(batch), out_dtype],
        ends: T.Tensor[(batch), out_dtype],
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
            tx = T.get_thread_binding()

            # Shared memory allocations
            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_histogram = T.alloc_shared([RADIX + 1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

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

            l_new_topk = topk
            l_start_idx = starts[bx]
            l_end_idx = ends[bx]

            # ========== Stage 1: Quick 8-bit TopK ==========
            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)
            T.sync_threads()

            # Build histogram using 8-bit representation
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    inval_int16 = convert_to_uint16(input[bx, input_idx])
                    T.atomic_add(s_histogram[inval_int16], 1)
            T.sync_threads()

            # Cumulative sum on histogram
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # Find threshold bin where cumsum crosses topk
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
                        s_input_idx[0, pos] = input_idx

            # ========== Stage 2: Iterative Refinement (4 rounds max) ==========
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

                # Build histogram for current candidate set
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        l_bin_id32 = T.Cast(
                            T.int32, 
                            ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) 
                              >> (24 - round * 8)) & 0xFF)
                        )
                        T.atomic_add(s_histogram[l_bin_id32], 1)
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
                        l_bin_id32 = T.Cast(
                            T.int32,
                            ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]])
                              >> (24 - round * 8)) & 0xFF)
                        )
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            if round == 3:
                                # Final round - add remaining elements
                                l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                # Continue to next round
                                pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

    return tl_topk_radix_kernel


def topk_radix_tilelang(logits, topk=6):
    """
    TileLang implementation of TopK using Radix Selection
    
    Args:
        logits: [batch, num_experts] tensor
        topk: number of experts to select
    
    Returns:
        indices: [batch, topk] indices of top-k experts
    """
    batch, num_experts = logits.shape
    
    # Ensure all integers are Python int (not torch.int64/Long)
    batch = int(batch)
    num_experts = int(num_experts)
    topk = int(topk)
    
    indices = torch.zeros((batch, topk), dtype=torch.int32, device=logits.device)
    
    # Full range selection
    starts = torch.zeros((batch,), dtype=torch.int32, device=logits.device)
    ends = torch.full((batch,), num_experts, dtype=torch.int32, device=logits.device)
    
    kernel = tl_topk_radix_impl(topk)
    kernel(logits, indices, starts, ends)
    
    return indices
