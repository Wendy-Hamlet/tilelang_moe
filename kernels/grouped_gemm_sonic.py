# ********************************************************************************
# Copyright (c) 2025
# Grouped GEMM implementation in TileLang, inspired by sonic-moe
# Optimized for NVIDIA Hopper (H100) architecture
# ********************************************************************************

"""
Grouped GEMM kernel implementation using TileLang.

This module provides grouped matrix multiplication kernels optimized for MoE (Mixture of Experts)
workloads, inspired by sonic-moe's implementation. It supports:
- Variable-length batch sizes per expert
- Hopper-specific optimizations (TMA, warp specialization, etc.)
- Both forward pass (activation) and backward pass (weight gradient)

Key features:
- Persistent kernel scheduling for better GPU utilization
- Software pipelining with configurable stages
- Warp specialization for overlapping compute and memory operations
- Support for various activation functions (SiLU, ReLU, GELU, etc.)
"""

import math
from typing import List, Tuple, Optional
import torch
import tilelang
import tilelang.language as T


def compute_group_metadata(
    batch_sizes_list: List[int], 
    block_M: int
) -> Tuple[List[int], List[int], List[int], int]:
    """
    Compute metadata for grouped GEMM scheduling.
    
    Args:
        batch_sizes_list: List of sizes for each group/expert
        block_M: Block size along M dimension
        
    Returns:
        Tuple of (batch_offsets, batch_padded_offsets, group_idx_for_bx, total_m_blocks)
    """
    batch_count = len(batch_sizes_list)
    batch_offsets = [0]
    batch_padded_offsets = [0]
    
    for i in range(batch_count - 1):
        batch_offsets.append(batch_offsets[-1] + batch_sizes_list[i])
        
    for i in range(batch_count - 1):
        padded_size = math.ceil(batch_sizes_list[i] / block_M) * block_M
        batch_padded_offsets.append(batch_padded_offsets[-1] + padded_size)
    
    # Compute total M blocks and group index mapping
    total_m_blocks = sum(math.ceil(size / block_M) for size in batch_sizes_list)
    
    # Precompute which group each block belongs to
    group_idx_for_bx = []
    for group_idx, size in enumerate(batch_sizes_list):
        num_blocks = math.ceil(size / block_M)
        group_idx_for_bx.extend([group_idx] * num_blocks)
    
    return batch_offsets, batch_padded_offsets, group_idx_for_bx, total_m_blocks


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        "tl.disable_tma_lower": False,  # Enable TMA for Hopper
        "tl.disable_warp_specialized": False,  # Enable warp specialization
    }
)
def grouped_gemm_forward(
    batch_sizes_list: Tuple[int, ...],
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 256,
    dtype=T.float16,
):
    """
    Grouped GEMM forward kernel.
    
    Computes C[i] = A[offset_i:offset_i+size_i] @ B[i] for each group i.
    
    Args:
        batch_sizes_list: Tuple of sizes for each expert group
        K: Inner dimension (reduction dimension)
        N: Output dimension
        block_M: Block size for M dimension (tokens)
        block_N: Block size for N dimension (output features)
        block_K: Block size for K dimension (hidden features)
        num_stages: Number of pipeline stages
        threads: Number of threads per block
        dtype: Data type (default float16)
    
    Tensor shapes:
        A: (batch_sum, K) - Input activations
        B: (batch_count, K, N) - Expert weights
        C: (batch_sum, N) - Output
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = T.float32
    
    # Compute total M blocks for grid scheduling
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
    
    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            # Shared memory allocations with software pipelining
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            
            # Group index tracking - MUST initialize to 0
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_idx = 0  # Initialize to first expert

            # Compute M offset from block index
            m_start_padded = bx * block_M

            # Find which group this block belongs to (linear search)
            # Note: When experts have 0 tokens, batch_padded_offsets may have repeated values
            # The >= comparison ensures we find the correct (last matching) expert
            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            # Compute actual M coordinates
            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

            # Initialize accumulator
            T.clear(C_local)
            
            # Main GEMM loop with software pipelining
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Load A tile from global to shared memory
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                # Load B tile from global to shared memory
                T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared)
                # Compute GEMM
                T.gemm(A_shared, B_shared, C_local)

            # Write back results with masking for boundary tiles
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local[i, j]

    return kernel


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        "tl.disable_tma_lower": False,
        "tl.disable_warp_specialized": False,
    }
)
def grouped_gemm_forward_with_silu(
    batch_sizes_list: Tuple[int, ...],
    K: int,
    N: int,  # This is the output dimension, gated output will be N//2
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 256,
    dtype=T.float16,
):
    """
    Grouped GEMM with fused SiLU activation (SwiGLU style).
    
    Computes: output = SiLU(gate) * up
    where gate and up are the two halves of the GEMM output.
    
    Args:
        Same as grouped_gemm_forward, but N should be 2 * intermediate_size
        Output will have shape (batch_sum, N//2)
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = T.float32
    output_N = N // 2  # Output dimension after gating
    
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
    
    # Scale for approximate SiLU: silu(x) ≈ x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + 2^(-x * log2(e)))
    scale = 1.44269504  # log2(e)
    
    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
        C: T.Tensor([batch_sum, output_N], dtype),  # type: ignore  # Output is half the N dimension
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(output_N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_gate_shared = T.alloc_shared([block_K, block_N], dtype)  # Gate weights
            B_up_shared = T.alloc_shared([block_K, block_N], dtype)    # Up weights
            
            gate_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            up_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            
            # Group index tracking - MUST initialize to 0
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_idx = 0  # Initialize to first expert

            m_start_padded = bx * block_M

            # Find which group this block belongs to (linear search)
            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

            T.clear(gate_local)
            T.clear(up_local)
            
            # Main GEMM loop - compute both gate and up projections
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                # Gate weights are in first half of N dimension
                T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_gate_shared)
                # Up weights are in second half of N dimension
                T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, output_N + by * block_N : output_N + (by + 1) * block_N], B_up_shared)
                T.gemm(A_shared, B_gate_shared, gate_local)
                T.gemm(A_shared, B_up_shared, up_local)

            # Fused SiLU activation: output = silu(gate) * up
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    # SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
                    gate_val = gate_local[i, j]
                    sigmoid_gate = 1.0 / (1.0 + T.exp2(-gate_val * scale))
                    silu_gate = gate_val * sigmoid_gate
                    C[m_start + i, by * block_N + j] = silu_gate * up_local[i, j]

    return kernel


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        "tl.disable_tma_lower": False,
        "tl.disable_warp_specialized": False,
    }
)
def grouped_gemm_backward_weight(
    batch_sizes_list: Tuple[int, ...],
    M: int,  # Hidden dimension (input feature size)
    N: int,  # Output dimension
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 256,
    dtype=T.float16,
):
    """
    Grouped GEMM for weight gradient computation.
    
    Computes dW[i] = A[offset_i:offset_i+size_i]^T @ dY[offset_i:offset_i+size_i] for each group i.
    
    Args:
        batch_sizes_list: Tuple of sizes for each expert group
        M: Input feature dimension
        N: Output feature dimension
        block_M: Block size for M dimension
        block_N: Block size for N dimension
        block_K: Block size for K dimension (tokens, varies per group)
    
    Tensor shapes:
        A: (batch_sum, M) - Input activations
        dY: (batch_sum, N) - Output gradients
        dW: (batch_count, M, N) - Weight gradients
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = T.float32
    
    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, M], dtype),  # type: ignore - Input activations
        dY: T.Tensor([batch_sum, N], dtype),  # type: ignore - Output gradients
        dW: T.Tensor([batch_count, M, N], dtype),  # type: ignore - Weight gradients
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        # Grid: (M blocks, N blocks, num_groups)
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), batch_count, threads=threads) as (bx, by, bz):
            A_shared = T.alloc_shared([block_K, block_M], dtype)  # Transposed for A^T
            dY_shared = T.alloc_shared([block_K, block_N], dtype)
            dW_local = T.alloc_fragment([block_M, block_N], accum_dtype)

            T.clear(dW_local)
            
            # Loop over tokens in this group
            for k in T.Pipelined(T.ceildiv(batch_sizes[bz], block_K), num_stages=num_stages):
                # Load A^T: read A and transpose on the fly
                for i, j in T.Parallel(block_K, block_M):
                    A_shared[i, j] = T.if_then_else(
                        k * block_K + i < batch_sizes[bz],
                        A[batch_offsets[bz] + k * block_K + i, bx * block_M + j],
                        dtype(0)
                    )
                # Load dY
                for i, j in T.Parallel(block_K, block_N):
                    dY_shared[i, j] = T.if_then_else(
                        k * block_K + i < batch_sizes[bz],
                        dY[batch_offsets[bz] + k * block_K + i, by * block_N + j],
                        dtype(0)
                    )
                # dW += A^T @ dY
                T.gemm(A_shared, dY_shared, dW_local, transpose_A=True)

            # Write back weight gradients
            T.copy(dW_local, dW[bz, bx * block_M, by * block_N])

    return kernel


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        "tl.disable_tma_lower": False,
        "tl.disable_warp_specialized": False,
    }
)
def grouped_gemm_forward_with_indices(
    total_tokens: int,
    num_experts: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 256,
    dtype=T.float16,
):
    """
    Grouped GEMM with scatter/gather indices for MoE.
    
    This version uses token-to-expert indices for routing, which is more flexible
    for dynamic MoE routing where tokens can be assigned to any expert.
    
    Args:
        total_tokens: Total number of tokens (sum of all expert token counts)
        num_experts: Number of experts
        K: Hidden dimension (input features)
        N: Output dimension
        
    Tensor shapes:
        A: (total_tokens, K) - Input activations (already gathered/sorted by expert)
        B: (num_experts, K, N) - Expert weights
        C: (total_tokens, N) - Output activations
        expert_ids: (total_tokens,) - Expert ID for each token
        expert_offsets: (num_experts + 1,) - Cumulative offsets for each expert
    """
    accum_dtype = T.float32
    
    # Estimate total M blocks (will be refined at runtime)
    max_m_blocks = (total_tokens + block_M - 1) // block_M + num_experts
    
    @T.prim_func
    def kernel(
        A: T.Tensor([total_tokens, K], dtype),  # type: ignore
        B: T.Tensor([num_experts, K, N], dtype),  # type: ignore
        C: T.Tensor([total_tokens, N], dtype),  # type: ignore
        expert_offsets: T.Tensor([num_experts + 1], T.int32),  # type: ignore
        block_to_expert: T.Tensor([max_m_blocks], T.int32),  # type: ignore - Precomputed mapping
        block_to_m_offset: T.Tensor([max_m_blocks], T.int32),  # type: ignore - Precomputed M offsets
        num_valid_blocks: T.Tensor([1], T.int32),  # type: ignore - Actual number of blocks
    ):
        with T.Kernel(max_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)

            # Skip invalid blocks
            if bx >= num_valid_blocks[0]:
                return

            # Get expert and M offset from precomputed tables
            expert_idx = block_to_expert[bx]
            m_start = block_to_m_offset[bx]
            
            # Compute actual rows in this block
            expert_start = expert_offsets[expert_idx]
            expert_end = expert_offsets[expert_idx + 1]
            actual_rows = T.min(block_M, expert_end - m_start)

            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[expert_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local[i, j]

    return kernel


@tilelang.jit(
    out_idx=[3],
    pass_configs={
        "tl.disable_tma_lower": False,
        "tl.disable_warp_specialized": False,
    }
)
def grouped_gemm_forward_fused_moe(
    batch_sizes_list: Tuple[int, ...],
    K: int,  # Hidden dimension
    intermediate_size: int,  # FFN intermediate size
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 3,
    threads: int = 256,
    dtype=T.float16,
):
    """
    Fused MoE forward: combines up projection, gate projection, SiLU activation.
    
    Computes: hidden_states = SiLU(gate_proj(x)) * up_proj(x)
    
    This is a common pattern in transformer MoE layers (e.g., DeepSeek, Mixtral).
    
    Args:
        batch_sizes_list: Token counts per expert
        K: Input hidden dimension
        intermediate_size: FFN intermediate dimension
        
    Tensor shapes:
        input: (batch_sum, K) - Input hidden states
        gate_weight: (num_experts, intermediate_size, K) - Gate projection weights (W1)
        up_weight: (num_experts, intermediate_size, K) - Up projection weights (W3)
        output: (batch_sum, intermediate_size) - Output after SiLU gating
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = T.float32
    N = intermediate_size
    
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
    scale = 1.44269504  # log2(e)
    
    @T.prim_func
    def kernel(
        input_tensor: T.Tensor([batch_sum, K], dtype),  # type: ignore
        gate_weight: T.Tensor([batch_count, N, K], dtype),  # type: ignore - (E, N, K)
        up_weight: T.Tensor([batch_count, N, K], dtype),  # type: ignore - (E, N, K)
        output: T.Tensor([batch_sum, N], dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            input_shared = T.alloc_shared([block_M, block_K], dtype)
            gate_shared = T.alloc_shared([block_N, block_K], dtype)  # Transposed for matmul
            up_shared = T.alloc_shared([block_N, block_K], dtype)
            
            gate_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            up_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            
            # Group index tracking - MUST initialize to 0
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_idx = 0  # Initialize to first expert

            m_start_padded = bx * block_M

            # Find which group this block belongs to (linear search)
            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

            T.clear(gate_local)
            T.clear(up_local)
            
            # Compute gate and up projections simultaneously
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(input_tensor[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], input_shared)
                # Load gate weights - need to transpose B
                T.copy(gate_weight[cur_batch_idx, by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K], gate_shared)
                # Load up weights
                T.copy(up_weight[cur_batch_idx, by * block_N : (by + 1) * block_N, k * block_K : (k + 1) * block_K], up_shared)
                # input @ gate^T
                T.gemm(input_shared, gate_shared, gate_local, transpose_B=True)
                # input @ up^T
                T.gemm(input_shared, up_shared, up_local, transpose_B=True)

            # Fused SiLU activation and element-wise multiply
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    gate_val = gate_local[i, j]
                    # SiLU(x) = x * sigmoid(x)
                    sigmoid_gate = 1.0 / (1.0 + T.exp2(-gate_val * scale))
                    silu_gate = gate_val * sigmoid_gate
                    output[m_start + i, by * block_N + j] = silu_gate * up_local[i, j]

    return kernel


# =============================================================================
# Hopper-optimized kernels with warp specialization
# =============================================================================

@tilelang.jit(
    out_idx=[2],
    pass_configs={
        "tl.disable_tma_lower": False,
        "tl.disable_warp_specialized": False,
    }
)
def grouped_gemm_forward_hopper(
    batch_sizes_list: Tuple[int, ...],
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 4,  # More stages for Hopper
    threads: int = 256,
    dtype=T.float16,
):
    """
    Hopper-optimized grouped GEMM with warp specialization.
    
    Uses TMA (Tensor Memory Accelerator) and warp specialization for 
    overlapping memory and compute operations on H100.
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = T.float32
    
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
    
    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),  # type: ignore
        B: T.Tensor([batch_count, K, N], dtype),  # type: ignore
        C: T.Tensor([batch_sum, N], dtype),  # type: ignore
        batch_sizes: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
        batch_padded_offsets: T.Tensor([batch_count], T.int32),  # type: ignore
    ):
        with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
            # Multi-buffered shared memory for pipelining
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            
            # Barriers for warp specialization
            data_is_ready = T.alloc_barrier(arrive_count=128)
            compute_is_done = T.alloc_barrier(arrive_count=128)
            
            # Group index tracking - MUST initialize to 0
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_idx = 0  # Initialize to first expert

            m_start_padded = bx * block_M

            # Find which group this block belongs to (linear search)
            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

            # Compute warps
            with T.ws(0):
                T.clear(C_local)

            # Main loop with warp specialization
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Memory warps: load data
                with T.ws(1):
                    T.barrier_wait(compute_is_done, (k + 1) % 2)
                    T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                    T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared)
                    T.barrier_arrive(data_is_ready)
                
                # Compute warps: matrix multiply
                with T.ws(0):
                    T.barrier_wait(data_is_ready, k % 2)
                    T.gemm(A_shared, B_shared, C_local)
                    T.barrier_arrive(compute_is_done)

            # Write back results
            with T.ws(0):
                for i, j in T.Parallel(block_M, block_N):
                    if i < actual_rows:
                        C[m_start + i, by * block_N + j] = C_local[i, j]

    return kernel


# =============================================================================
# Helper functions for running the kernels
# =============================================================================

def prepare_grouped_gemm_inputs(
    batch_sizes_list: List[int],
    K: int,
    N: int,
    block_M: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Prepare input tensors and metadata for grouped GEMM.
    
    Args:
        batch_sizes_list: List of sizes for each group
        K: Hidden dimension
        N: Output dimension
        block_M: Block size for M dimension
        device: CUDA device
        dtype: Data type
        
    Returns:
        Dictionary containing all tensors needed for the kernel
    """
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    
    # Compute metadata
    batch_offsets, batch_padded_offsets, _, _ = compute_group_metadata(batch_sizes_list, block_M)
    
    # Create tensors
    A = torch.randn(batch_sum, K, device=device, dtype=dtype)
    B = torch.randn(batch_count, K, N, device=device, dtype=dtype)
    C = torch.empty(batch_sum, N, device=device, dtype=dtype)
    
    batch_sizes_tensor = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets_tensor = torch.tensor(batch_offsets, device=device, dtype=torch.int32)
    batch_padded_offsets_tensor = torch.tensor(batch_padded_offsets, device=device, dtype=torch.int32)
    
    return {
        'A': A,
        'B': B,
        'C': C,
        'batch_sizes': batch_sizes_tensor,
        'batch_offsets': batch_offsets_tensor,
        'batch_padded_offsets': batch_padded_offsets_tensor,
    }


def torch_grouped_gemm_reference(
    A: torch.Tensor,
    B: torch.Tensor,
    batch_sizes: List[int],
) -> torch.Tensor:
    """
    Reference PyTorch implementation for grouped GEMM.
    
    Args:
        A: Input tensor of shape (batch_sum, K)
        B: Weight tensor of shape (batch_count, K, N)
        batch_sizes: List of sizes for each group
        
    Returns:
        Output tensor of shape (batch_sum, N)
    """
    batch_sum = A.shape[0]
    N = B.shape[2]
    output = torch.empty(batch_sum, N, device=A.device, dtype=A.dtype)
    
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        output[start:end] = torch.mm(A[start:end], B[i])
        start = end
    
    return output


def run_grouped_gemm_forward(
    batch_sizes_list: List[int],
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 64,
    num_stages: int = 3,
    use_hopper_kernel: bool = True,
    verify: bool = True,
    profile: bool = False,
):
    """
    Run grouped GEMM forward pass.
    
    Args:
        batch_sizes_list: List of token counts per expert
        K: Hidden dimension
        N: Output dimension
        block_M, block_N, block_K: Block sizes
        num_stages: Pipeline stages
        use_hopper_kernel: Whether to use Hopper-optimized kernel
        verify: Whether to verify against PyTorch reference
        profile: Whether to profile performance
        
    Returns:
        Output tensor and optionally performance metrics
    """
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Prepare inputs
    inputs = prepare_grouped_gemm_inputs(batch_sizes_list, K, N, block_M, device, dtype)
    
    # Create kernel
    if use_hopper_kernel:
        kernel = grouped_gemm_forward_hopper(
            tuple(batch_sizes_list), K, N, block_M, block_N, block_K, num_stages
        )
    else:
        kernel = grouped_gemm_forward(
            tuple(batch_sizes_list), K, N, block_M, block_N, block_K, num_stages
        )
    
    # Run kernel
    output = kernel(
        inputs['A'], inputs['B'],
        inputs['batch_sizes'], inputs['batch_offsets'], inputs['batch_padded_offsets']
    )
    
    # Verify correctness
    if verify:
        ref_output = torch_grouped_gemm_reference(inputs['A'], inputs['B'], batch_sizes_list)
        if torch.allclose(output, ref_output, rtol=1e-2, atol=1e-2):
            print("✅ TileLang and PyTorch match")
        else:
            print("❌ TileLang and PyTorch mismatch")
            max_diff = (output - ref_output).abs().max().item()
            print(f"   Max difference: {max_diff}")
    
    # Profile if requested
    if profile:
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
        latency = profiler.do_bench(
            warmup=500,
            input_tensors=[
                inputs['A'], inputs['B'],
                inputs['batch_sizes'], inputs['batch_offsets'], inputs['batch_padded_offsets']
            ]
        )
        batch_sum = sum(batch_sizes_list)
        flops = 2 * batch_sum * K * N
        tflops = flops / latency / 1e9
        print(f"Latency: {latency:.3f} ms")
        print(f"TFLOPs: {tflops:.2f}")
        return output, {'latency': latency, 'tflops': tflops}
    
    return output


# =============================================================================
# Test functions
# =============================================================================

def test_grouped_gemm():
    """Run basic tests for grouped GEMM kernels."""
    print("Testing grouped_gemm_forward...")
    
    # Test case 1: Single group
    run_grouped_gemm_forward([64], 8192, 8192, use_hopper_kernel=False)
    
    # Test case 2: Multiple groups
    run_grouped_gemm_forward([64, 128, 256], 8192, 8192, use_hopper_kernel=False)
    
    # Test case 3: Non-aligned sizes
    run_grouped_gemm_forward([63, 77, 111, 280], 8192, 8192, use_hopper_kernel=False)
    
    print("\nTesting grouped_gemm_forward_hopper...")
    run_grouped_gemm_forward([64, 128, 256], 8192, 8192, use_hopper_kernel=True)
    
    print("\nAll tests completed!")


# =============================================================================
# Benchmark-compatible interface (matches baseline API)
# =============================================================================

# Kernel cache to avoid recompilation
_kernel_cache = {}


def grouped_gemm_sonic(
    X: torch.Tensor,  # (total_tokens, H)
    W: torch.Tensor,  # (E, I, H) - 所有expert权重
    expert_indices: torch.Tensor,  # (total_tokens,) - 每个token对应的expert ID
) -> torch.Tensor:
    """
    Sonic-style grouped GEMM with benchmark-compatible interface.
    
    This is the main entry point for the benchmark. It handles:
    1. Sorting tokens by expert ID
    2. Computing per-expert batch sizes
    3. Calling the optimized grouped GEMM kernel
    4. Scattering results back to original order
    
    Args:
        X: (total_tokens, H) - Input tokens
        W: (E, I, H) - Expert weights, shape is (num_experts, intermediate_dim, hidden_dim)
        expert_indices: (total_tokens,) - Expert ID for each token
    
    Returns:
        Y: (total_tokens, I) - Output
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape
    device = X.device
    dtype = X.dtype
    
    # Step 1: Sort tokens by expert ID for coalesced memory access
    sorted_indices = expert_indices.argsort()
    sorted_expert_ids = expert_indices[sorted_indices]
    X_sorted = X[sorted_indices]  # (total_tokens, H)
    
    # Step 2: Compute per-expert batch sizes
    expert_counts = torch.bincount(expert_indices, minlength=E)
    batch_sizes_list = expert_counts.tolist()
    
    # Skip if no tokens
    if sum(batch_sizes_list) == 0:
        return torch.zeros(total_tokens, I, device=device, dtype=dtype)
    
    # Step 3: Prepare metadata tensors
    block_M = 128
    block_N = 128
    block_K = 64
    num_stages = 3
    
    # Compute offsets
    batch_offsets = [0]
    batch_padded_offsets = [0]
    for i in range(E - 1):
        batch_offsets.append(batch_offsets[-1] + batch_sizes_list[i])
        padded_size = ((batch_sizes_list[i] + block_M - 1) // block_M) * block_M
        batch_padded_offsets.append(batch_padded_offsets[-1] + padded_size)
    
    batch_sizes_tensor = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets_tensor = torch.tensor(batch_offsets, device=device, dtype=torch.int32)
    batch_padded_offsets_tensor = torch.tensor(batch_padded_offsets, device=device, dtype=torch.int32)
    
    # Step 4: Prepare weight tensor in the format expected by kernel
    # Kernel expects B: (E, K, N), but W is (E, I, H) where we compute X @ W^T
    # So we need W transposed: (E, H, I) = (E, K, N)
    W_transposed = W.transpose(1, 2).contiguous()  # (E, H, I)
    
    # Step 5: Get or compile kernel
    cache_key = (tuple(batch_sizes_list), H, I, block_M, block_N, block_K, num_stages)
    
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = grouped_gemm_forward(
            tuple(batch_sizes_list), H, I, block_M, block_N, block_K, num_stages
        )
    
    kernel = _kernel_cache[cache_key]
    
    # Step 6: Run kernel
    # Kernel computes: C = A @ B where A is (batch_sum, K), B is (E, K, N)
    Y_sorted = kernel(
        X_sorted, W_transposed,
        batch_sizes_tensor, batch_offsets_tensor, batch_padded_offsets_tensor
    )
    
    # Step 7: Scatter results back to original order
    Y = torch.empty(total_tokens, I, device=device, dtype=dtype)
    Y[sorted_indices] = Y_sorted
    
    return Y


def grouped_gemm_sonic_v2(
    X: torch.Tensor,  # (total_tokens, H)
    W: torch.Tensor,  # (E, I, H) - 所有expert权重
    expert_indices: torch.Tensor,  # (total_tokens,) - 每个token对应的expert ID
    use_hopper: bool = False,
) -> torch.Tensor:
    """
    Alternative implementation with Hopper optimizations.
    
    Same interface as grouped_gemm_sonic but with optional Hopper-specific
    optimizations (warp specialization, TMA, etc.)
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape
    device = X.device
    dtype = X.dtype
    
    # Sort tokens by expert ID
    sorted_indices = expert_indices.argsort()
    X_sorted = X[sorted_indices]
    
    # Compute per-expert batch sizes
    expert_counts = torch.bincount(expert_indices, minlength=E)
    batch_sizes_list = expert_counts.tolist()
    
    if sum(batch_sizes_list) == 0:
        return torch.zeros(total_tokens, I, device=device, dtype=dtype)
    
    # Prepare metadata
    block_M = 128
    block_N = 128
    block_K = 64
    num_stages = 4 if use_hopper else 3
    
    batch_offsets = [0]
    batch_padded_offsets = [0]
    for i in range(E - 1):
        batch_offsets.append(batch_offsets[-1] + batch_sizes_list[i])
        padded_size = ((batch_sizes_list[i] + block_M - 1) // block_M) * block_M
        batch_padded_offsets.append(batch_padded_offsets[-1] + padded_size)
    
    batch_sizes_tensor = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets_tensor = torch.tensor(batch_offsets, device=device, dtype=torch.int32)
    batch_padded_offsets_tensor = torch.tensor(batch_padded_offsets, device=device, dtype=torch.int32)
    
    W_transposed = W.transpose(1, 2).contiguous()
    
    # Select kernel
    cache_key = (tuple(batch_sizes_list), H, I, block_M, block_N, block_K, num_stages, use_hopper)
    
    if cache_key not in _kernel_cache:
        if use_hopper:
            _kernel_cache[cache_key] = grouped_gemm_forward_hopper(
                tuple(batch_sizes_list), H, I, block_M, block_N, block_K, num_stages
            )
        else:
            _kernel_cache[cache_key] = grouped_gemm_forward(
                tuple(batch_sizes_list), H, I, block_M, block_N, block_K, num_stages
            )
    
    kernel = _kernel_cache[cache_key]
    
    Y_sorted = kernel(
        X_sorted, W_transposed,
        batch_sizes_tensor, batch_offsets_tensor, batch_padded_offsets_tensor
    )
    
    Y = torch.empty(total_tokens, I, device=device, dtype=dtype)
    Y[sorted_indices] = Y_sorted
    
    return Y


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grouped GEMM benchmark")
    parser.add_argument("--batch_sizes", type=str, default="64,128,256", help="Comma-separated batch sizes per expert")
    parser.add_argument("--K", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--N", type=int, default=14336, help="Intermediate/output dimension")
    parser.add_argument("--block_M", type=int, default=128, help="Block size M")
    parser.add_argument("--block_N", type=int, default=128, help="Block size N")
    parser.add_argument("--block_K", type=int, default=64, help="Block size K")
    parser.add_argument("--num_stages", type=int, default=3, help="Pipeline stages")
    parser.add_argument("--profile", action="store_true", help="Profile performance")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--test_api", action="store_true", help="Test benchmark-compatible API")
    args = parser.parse_args()
    
    if args.test:
        test_grouped_gemm()
    elif args.test_api:
        # Test the benchmark-compatible API
        print("Testing benchmark-compatible API (grouped_gemm_sonic)...")
        
        torch.manual_seed(42)
        device = torch.device("cuda")
        
        # Test configuration
        total_tokens = 256
        H = 1024
        I = 768
        E = 8
        
        X = torch.randn(total_tokens, H, device=device, dtype=torch.float16)
        W = torch.randn(E, I, H, device=device, dtype=torch.float16)
        expert_indices = torch.randint(0, E, (total_tokens,), device=device)
        
        # Run our implementation
        Y_sonic = grouped_gemm_sonic(X, W, expert_indices)
        
        # Reference implementation (loop)
        Y_ref = torch.zeros(total_tokens, I, device=device, dtype=torch.float16)
        for expert_id in range(E):
            mask = expert_indices == expert_id
            if mask.any():
                Y_ref[mask] = X[mask] @ W[expert_id].T
        
        # Check correctness
        max_diff = (Y_sonic - Y_ref).abs().max().item()
        if max_diff < 1e-2:
            print(f"✅ API test passed! Max diff: {max_diff:.6f}")
        else:
            print(f"❌ API test failed! Max diff: {max_diff:.6f}")
        
        print(f"Output shape: {Y_sonic.shape}")
    else:
        batch_sizes_list = [int(x) for x in args.batch_sizes.split(",")]
        run_grouped_gemm_forward(
            batch_sizes_list,
            args.K,
            args.N,
            args.block_M,
            args.block_N,
            args.block_K,
            args.num_stages,
            use_hopper_kernel=True,
            verify=True,
            profile=args.profile,
        )
