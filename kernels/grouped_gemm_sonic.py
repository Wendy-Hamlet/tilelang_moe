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


@tilelang.jit(out_idx=[-1])
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
    
    关键特性：
    1. 单kernel处理所有expert
    2. 每个block根据expert_offset动态选择任务
    3. Shared memory缓存权重和输入
    
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
        expert_offsets: T.Tensor((E + 1,), T.int32),     # Expert累计offset (token级别)
        block_offsets: T.Tensor((E + 1,), T.int32),      # Expert累计offset (block级别)
        Y: T.Tensor((total_tokens, I), dtype),           # 输出
        debug_info: T.Tensor((total_blocks, 4), T.int32), # Debug: [expert_id, m_start, actual_rows, local_block_id]
    ):
        # Sonic算法：tokens已按expert排序，每个expert分配若干blocks
        # 参考官方grouped_gemm，使用T.if_then_else查找expert
        # total_blocks在编译时传入（避免运行时访问block_offsets[E]）
        
        with T.Kernel(total_blocks, T.ceildiv(I, block_N), threads=128) as (bx, by):
            # Shared memory buffers
            X_shared = T.alloc_shared([block_M, block_K], dtype)
            W_shared = T.alloc_shared([block_K, block_N], dtype)
            C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
            
            # 找到当前block属于哪个expert
            # block_offsets[e] 是expert e的第一个block ID  
            # 需要找到满足 block_offsets[e] <= bx < block_offsets[e+1] 的e
            expert_id = T.alloc_var(dtype=T.int32)
            
            # 参考官方实现，不显式初始化，让T.if_then_else处理
            # 正向遍历，持续更新直到不满足条件
            for e in range(E):
                in_cur_expert = bx >= block_offsets[e]
                expert_id = T.if_then_else(in_cur_expert, e, expert_id)
            
            # 计算当前block在expert内的局部位置
            local_block_id = bx - block_offsets[expert_id]
            
            # Expert的token范围
            expert_start = expert_offsets[expert_id]
            expert_end = expert_offsets[expert_id + 1]
            expert_tokens = expert_end - expert_start
            
            # 当前block处理的token范围
            m_start = expert_start + local_block_id * block_M
            remaining_tokens = expert_tokens - local_block_id * block_M
            actual_rows = T.min(block_M, remaining_tokens)
            
            # Debug: 记录每个block的信息（只在by=0时写入避免重复）
            if by == 0:
                debug_info[bx, 0] = expert_id
                debug_info[bx, 1] = m_start
                debug_info[bx, 2] = actual_rows
                debug_info[bx, 3] = local_block_id
            
            # GEMM计算
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=2):
                # 加载X: [block_M, block_K]
                T.copy(X[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], X_shared)
                
                # 加载W转置: W[expert_id, by*block_N:(by+1)*block_N, k*block_K:(k+1)*block_K] -> W_shared[block_K, block_N]
                # 注意：W是[E, I, H]，我们需要[H, I]的转置 -> [block_K, block_N]
                for i, j in T.Parallel(block_K, block_N):
                    k_idx = k * block_K + i
                    n_idx = by * block_N + j
                    if k_idx < H and n_idx < I:
                        W_shared[i, j] = W[expert_id, n_idx, k_idx]
                    else:
                        W_shared[i, j] = T.cast(0.0, dtype)
                
                # GEMM: C_local += X_shared @ W_shared (W_shared已经是转置的layout)
                T.gemm(X_shared, W_shared, C_local)
            
            # Store结果
            for i, j in T.Parallel(block_M, block_N):
                if i < actual_rows and by * block_N + j < I:
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
    
    # 计算block_offsets: 每个expert需要多少blocks
    block_M = 128
    expert_counts = expert_offsets[1:] - expert_offsets[:-1]  # 每个expert的token数
    expert_blocks = torch.ceil(expert_counts.float() / block_M).to(torch.int32)  # 每个expert的block数
    block_offsets = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int32),
        torch.cumsum(expert_blocks, dim=0, dtype=torch.int32)
    ])
    
    # Gather: 按expert排序后的tokens
    X_sorted = X[sorted_indices]
    
    # 计算total_blocks (在host端)
    total_blocks_value = int(block_offsets[-1].item())
    
    # 编译kernel
    cache_key = (total_tokens, H, I, E)
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
    
    # 转换参数为int32
    expert_offsets_int32 = expert_offsets.to(torch.int32)
    block_offsets_int32 = block_offsets.to(torch.int32)
    
    # Debug tensor
    debug_info = torch.zeros((total_blocks_value, 4), dtype=torch.int32, device=device)
    
    # 执行kernel（jit会自动返回输出）
    Y_sorted = kernel(X_sorted, W, expert_offsets_int32, block_offsets_int32, debug_info)
    
    # Print debug info
    if os.environ.get('DEBUG_SONIC', '0') == '1':
        print("\n[DEBUG] Sonic kernel execution:")
        print(f"  total_blocks: {total_blocks_value}")
        for bx in range(total_blocks_value):
            expert_id = debug_info[bx, 0].item()
            m_start = debug_info[bx, 1].item()
            actual_rows = debug_info[bx, 2].item()
            local_block_id = debug_info[bx, 3].item()
            print(f"  Block {bx}: expert={expert_id}, m_start={m_start}, actual_rows={actual_rows}, local_block={local_block_id}")
    
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
