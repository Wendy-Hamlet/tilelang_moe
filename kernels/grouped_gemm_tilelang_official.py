"""
Grouped GEMM TileLang Official Implementation Wrapper

使用TileLang官方examples/grouped_gemm中的融合grouped_gemm kernel
适配为benchmark接口
"""

import torch
import tilelang
import tilelang.language as T
import math
from typing import Dict, Tuple


# ============ TileLang Official Grouped GEMM Kernel ============
# 源自: tilelang/examples/grouped_gemm/example_grouped_gemm_fwd.py

@tilelang.jit(out_idx=[2])
def grouped_gemm_kernel(batch_sizes_list, K, N, block_M, block_N, block_K, num_stages=2, threads=128, dtype=T.float16):
    """
    TileLang官方grouped GEMM kernel
    
    args:
        A (torch.Tensor): Input tensor of shape (M, K).
        B (torch.Tensor): Input tensor of shape (G, K, N).
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
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            cur_batch_idx = T.alloc_var(dtype=T.int32)
            cur_batch_size = T.alloc_var(dtype=T.int32)

            m_start_padded = bx * block_M

            for i in range(batch_count):
                in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
                cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)

            cur_batch_size = batch_sizes[cur_batch_idx]
            m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
            actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
                T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows:
                    C[m_start + i, by * block_N + j] = C_local[i, j]

    return kernel


# Kernel编译缓存: (batch_sizes_tuple, K, N) -> kernel
_kernel_cache: Dict[Tuple, any] = {}


def grouped_gemm_tilelang_official(
    X: torch.Tensor,  # (total_tokens, H)
    W: torch.Tensor,  # (E, I, H)
    expert_indices: torch.Tensor,  # (total_tokens,)
) -> torch.Tensor:
    """
    使用TileLang官方grouped GEMM kernel的适配器
    
    Args:
        X: (total_tokens, H) - 输入tokens
        W: (E, I, H) - expert权重
        expert_indices: (total_tokens,) - 每个token对应的expert ID
    
    Returns:
        Y: (total_tokens, I) - 输出
    
    Note:
        官方kernel接口:
        - A: (batch_sum, K) - 按expert排序的输入
        - B: (G, K, N) - 权重 (num_groups, hidden_dim, output_dim)
        - batch_sizes: 每个group的token数
        
        我们需要:
        1. 按expert_id排序tokens
        2. 准备权重W转置为(E, H, I)
        3. 调用kernel
        4. 还原到原始顺序
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape  # W is (E, I, H)
    device = X.device
    dtype = X.dtype
    
    # 1. 按expert_id排序tokens
    sorted_indices = expert_indices.argsort()
    X_sorted = X[sorted_indices]  # (total_tokens, H)
    expert_indices_sorted = expert_indices[sorted_indices]
    
    # 2. 计算每个expert的token数量
    expert_counts = torch.bincount(expert_indices, minlength=E)
    batch_sizes_list = expert_counts.tolist()
    
    # 3. 准备权重：W是(E, I, H)，需要转为(E, H, I)
    # 因为官方kernel计算 A @ B，其中B是(G, K, N)
    W_transposed = W.transpose(1, 2).contiguous()  # (E, H, I)
    
    # 4. 计算batch_offsets和batch_padded_offsets
    block_M = 64  # 与官方example相同
    block_N = 128
    block_K = 64
    
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(E - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
        batch_padded_offsets_list.append(
            batch_padded_offsets_list[-1] + 
            math.ceil(batch_sizes_list[i] / block_M) * block_M
        )
    
    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)
    
    # 5. 获取或编译kernel
    cache_key = (tuple(batch_sizes_list), H, I)
    if cache_key not in _kernel_cache:
        kernel = grouped_gemm_kernel(
            tuple(batch_sizes_list), 
            K=H, 
            N=I, 
            block_M=block_M, 
            block_N=block_N, 
            block_K=block_K,
            num_stages=2,
            threads=256,
            dtype=T.float16
        )
        _kernel_cache[cache_key] = kernel
    
    kernel = _kernel_cache[cache_key]
    
    # 6. 调用kernel
    Y_sorted = kernel(X_sorted, W_transposed, batch_sizes, batch_offsets, batch_padded_offsets)
    
    # 7. 还原到原始token顺序
    Y = torch.empty(total_tokens, I, device=device, dtype=dtype)
    Y[sorted_indices] = Y_sorted
    
    return Y


if __name__ == "__main__":
    # 测试
    torch.manual_seed(42)
    device = "cuda"
    
    total_tokens = 256
    H = 512
    I = 384
    E = 8
    
    X = torch.randn(total_tokens, H, device=device, dtype=torch.float16)
    W = torch.randn(E, I, H, device=device, dtype=torch.float16)
    expert_indices = torch.randint(0, E, (total_tokens,), device=device)
    
    print("Testing TileLang Official Grouped GEMM Wrapper...")
    print(f"Config: total_tokens={total_tokens}, H={H}, I={I}, E={E}")
    
    # TileLang official version
    Y_official = grouped_gemm_tilelang_official(X, W, expert_indices)
    print(f"✓ TileLang Official output shape: {Y_official.shape}")
    
    # PyTorch reference
    from kernels.grouped_gemm_baseline import grouped_gemm_loop
    Y_ref = grouped_gemm_loop(X, W, expert_indices)
    
    # Check correctness
    diff = (Y_official - Y_ref).abs().max().item()
    print(f"\nCorrectness:")
    print(f"  TileLang Official vs PyTorch: max_diff = {diff:.6f}")
    
    if diff < 1e-2:
        print("✅ TileLang Official version matches PyTorch reference!")
    else:
        print("❌ Mismatch detected!")
