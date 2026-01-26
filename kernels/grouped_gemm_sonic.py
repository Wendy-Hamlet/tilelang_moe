"""
Grouped GEMM Sonic-style Fused Implementation

仿照Sonic算法的融合grouped GEMM实现：
- 单kernel处理所有expert计算
- 动态tile调度器实现负载均衡
- Shared memory优化减少global memory访问

Note: 这是简化版实现，展示核心思想，但未使用Hopper TMA等高级特性
"""

import os
import torch
import tilelang
import tilelang.language as T
from typing import Tuple


@tilelang.jit(out_idx=[4, 5])
def grouped_gemm_sonic_kernel(
    total_tokens: int,
    H: int,
    I: int,
    E: int,
    total_blocks: int,
    block_M: int = 128,
    block_N: int = 256,
    block_K: int = 32,
    dtype=T.float16,
    accum_dtype=T.float32
):
    """
    Sonic-style Grouped GEMM Kernel
    
    关键变化（参考官方example_dequant_groupedgemm_bf16_mxfp4_hopper.py）：
    - 使用预计算的block_expert_ids tensor，直接查表获取expert_id
    - 不再在kernel中循环计算expert归属
    
    Args:
        total_tokens: 总token数
        H: 隐藏层维度
        I: 中间层维度
        E: expert数量
    
    计算:
        For each token i with expert_id j:
            Y[i] = X[i] @ W[j]^T
    """
    @T.prim_func
    def grouped_gemm_sonic(
        X: T.Tensor((total_tokens, H), dtype),           # 输入tokens (已按expert排序)
        W: T.Tensor((E, I, H), dtype),                   # Expert权重
        block_expert_ids: T.Tensor((total_blocks,), T.int32),  # 预计算：每个block对应的expert_id
        block_m_starts: T.Tensor((total_blocks,), T.int32),    # 预计算：每个block的m_start
        Y: T.Tensor((total_tokens, I), dtype),           # 输出
        debug_info: T.Tensor((total_blocks, 4), T.int32), # Debug
    ):
        with T.Kernel(total_blocks, T.ceildiv(I, block_N), threads=128) as (bx, by):
            # Shared memory buffers
            X_shared = T.alloc_shared([block_M, block_K], dtype)
            W_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            
            # 从预计算tensor中直接读取expert_id和m_start
            # 这是官方推荐的方式，避免在kernel中用循环查找
            expert_id_local = T.alloc_shared([1], T.int32)
            expert_id_local[0] = block_expert_ids[bx]
            
            m_start = block_m_starts[bx]
            
            # Debug: 记录每个block的信息
            if by == 0:
                debug_info[bx, 0] = expert_id_local[0]
                debug_info[bx, 1] = m_start
                debug_info[bx, 2] = block_M  # actual_rows由host计算
                debug_info[bx, 3] = bx
            
            # GEMM计算
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=2):
                # 加载X: [block_M, block_K]
                T.copy(X[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], X_shared)
                
                # 加载W: 根据expert_id加载对应权重
                # 由于TileLang不支持用变量直接索引（W[expert_id_local[0], ...]），
                # 我们遍历所有expert，用if_then_else选择正确的
                for i, j in T.Parallel(block_K, block_N):
                    k_idx = k * block_K + i
                    n_idx = by * block_N + j
                    if k_idx < H and n_idx < I:
                        # 在if内部、for循环外部声明可变变量
                        w_val = T.alloc_var(dtype=dtype)
                        w_val = W[0, n_idx, k_idx]  # 初始化为expert 0的权重
                        for e in range(E):
                            w_val = T.if_then_else(expert_id_local[0] == e, W[e, n_idx, k_idx], w_val)
                        W_shared[i, j] = w_val
                    else:
                        W_shared[i, j] = T.cast(0.0, dtype)
                
                # GEMM: C_local += X_shared @ W_shared
                T.gemm(X_shared, W_shared, C_local)
            
            # Store结果
            for i, j in T.Parallel(block_M, block_N):
                if m_start + i < total_tokens and by * block_N + j < I:
                    Y[m_start + i, by * block_N + j] = T.cast(C_local[i, j], dtype)
    
    return grouped_gemm_sonic


def prepare_expert_metadata(expert_indices: torch.Tensor, E: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备expert metadata: offsets和排序后的索引
    
    Args:
        expert_indices: (total_tokens,) - 原始expert indices
        E: expert数量
    
    Returns:
        sorted_indices: (total_tokens,) - 按expert排序后的token索引
        expert_offsets: (E+1,) - 每个expert的起始位置
    """
    device = expert_indices.device
    
    # 按expert_id排序tokens
    sorted_expert_ids, sorted_indices = torch.sort(expert_indices)
    
    # 计算每个expert的offset
    expert_counts = torch.bincount(expert_indices, minlength=E)
    expert_offsets = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.cumsum(expert_counts, dim=0, dtype=torch.int32)
    ])
    
    return sorted_indices, expert_offsets


def grouped_gemm_sonic(
    X: torch.Tensor,  # (total_tokens, H)
    W: torch.Tensor,  # (E, I, H)
    expert_indices: torch.Tensor,  # (total_tokens,)
) -> torch.Tensor:
    """
    Sonic-style grouped GEMM主函数
    
    Args:
        X: (total_tokens, H)
        W: (E, I, H)
        expert_indices: (total_tokens,)
    
    Returns:
        Y: (total_tokens, I)
    
    Note:
        简化版实现流程：
        1. 按expert排序tokens（gather）
        2. 计算expert offsets
        3. 调用融合kernel
        4. Scatter回原顺序
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape
    device = X.device
    dtype = X.dtype
    
    # 准备metadata
    sorted_indices, expert_offsets = prepare_expert_metadata(expert_indices, E)
    
    # 计算block调度信息
    block_M = 128
    expert_counts = expert_offsets[1:] - expert_offsets[:-1]  # 每个expert的token数
    expert_blocks = torch.ceil(expert_counts.float() / block_M).to(torch.int32)  # 每个expert的block数
    
    # 预计算每个block的expert_id和m_start（参考官方实现）
    # 这是关键：不在kernel中计算，而是预先准备好
    block_expert_ids = []
    block_m_starts = []
    for e in range(E):
        num_blocks = int(expert_blocks[e].item())
        expert_start = int(expert_offsets[e].item())
        for b in range(num_blocks):
            block_expert_ids.append(e)
            block_m_starts.append(expert_start + b * block_M)
    
    block_expert_ids = torch.tensor(block_expert_ids, dtype=torch.int32, device=device)
    block_m_starts = torch.tensor(block_m_starts, dtype=torch.int32, device=device)
    total_blocks_value = len(block_expert_ids)
    
    # Gather: 按expert排序后的tokens
    X_sorted = X[sorted_indices]
    
    # 编译kernel
    cache_key = (total_tokens, H, I, E, total_blocks_value)
    if cache_key not in grouped_gemm_sonic.compile_cache:
        kernel = grouped_gemm_sonic_kernel(
            total_tokens=total_tokens,
            H=H,
            I=I,
            E=E,
            total_blocks=total_blocks_value,
            block_M=128,
            block_N=min(256, I),
            block_K=32,
            dtype=T.float16,
            accum_dtype=T.float32
        )
        grouped_gemm_sonic.compile_cache[cache_key] = kernel
    
    kernel = grouped_gemm_sonic.compile_cache[cache_key]
    
    # 执行kernel (Y和debug_info由kernel返回)
    Y_sorted, debug_info = kernel(X_sorted, W, block_expert_ids, block_m_starts)
    
    # Print debug info
    if os.environ.get('DEBUG_SONIC', '0') == '1':
        print("\n[DEBUG] Sonic kernel execution:")
        print(f"  total_blocks: {total_blocks_value}")
        print(f"  block_expert_ids (precomputed): {block_expert_ids.tolist()}")
        print(f"  block_m_starts (precomputed): {block_m_starts.tolist()}")
        for bx in range(total_blocks_value):
            expert_id = debug_info[bx, 0].item()
            m_start = debug_info[bx, 1].item()
            print(f"  Block {bx}: expert={expert_id}, m_start={m_start}")
    
    # Scatter: 恢复原顺序
    Y = torch.zeros_like(Y_sorted)
    Y[sorted_indices] = Y_sorted
    
    return Y


# Kernel编译缓存
grouped_gemm_sonic.compile_cache = {}


if __name__ == "__main__":
    # 测试
    torch.manual_seed(0)
    device = "cuda"
    
    total_tokens = 128
    H = 256
    I = 192
    E = 4
    
    X = torch.randn(total_tokens, H, device=device, dtype=torch.float16)
    W = torch.randn(E, I, H, device=device, dtype=torch.float16)
    expert_indices = torch.randint(0, E, (total_tokens,), device=device)
    
    print("Testing Sonic-style Grouped GEMM...")
    print(f"Config: total_tokens={total_tokens}, H={H}, I={I}, E={E}")
    
    # Test metadata preparation
    sorted_indices, expert_offsets = prepare_expert_metadata(expert_indices, E)
    print(f"✓ Expert offsets: {expert_offsets.tolist()}")
    
    # Sonic version
    Y_sonic = grouped_gemm_sonic(X, W, expert_indices)
    print(f"✓ Sonic output shape: {Y_sonic.shape}")
    
    # PyTorch reference
    from kernels.grouped_gemm_baseline import grouped_gemm_loop
    Y_ref = grouped_gemm_loop(X, W, expert_indices)
    
    # Check correctness
    diff = (Y_sonic - Y_ref).abs().max().item()
    print(f"\nCorrectness:")
    print(f"  Sonic vs PyTorch: max_diff = {diff:.6f}")
    
    if diff < 1e-2:
        print("✅ Sonic version matches PyTorch reference!")
    else:
        print("❌ Mismatch detected!")
        print(f"  Sample Y_sonic[0]: {Y_sonic[0, :5]}")
        print(f"  Sample Y_ref[0]:   {Y_ref[0, :5]}")
