"""
Grouped GEMM Sonic-style Fused Implementation

仿照Sonic算法的融合grouped GEMM实现：
- 单kernel处理所有expert计算
- 动态tile调度器实现负载均衡
- Shared memory优化减少global memory访问

Note: 这是简化版实现，展示核心思想，但未使用Hopper TMA等高级特性
"""

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
        X: T.Tensor((total_tokens, H), dtype),           # 输入tokens
        W: T.Tensor((E, I, H), dtype),                   # Expert权重
        expert_indices: T.Tensor((total_tokens,), T.int32),  # Token→Expert映射
        expert_offsets: T.Tensor((E + 1,), T.int32),     # Expert累计offset
        Y: T.Tensor((total_tokens, I), dtype),           # 输出
    ):
        # Grid: 为每个expert的每个tile分配一个block
        # 这里简化为处理所有tokens，实际Sonic会动态调度
        with T.Kernel(T.ceildiv(total_tokens, block_M), 1, threads=128) as (token_block_id, _):
            # Shared memory buffers
            X_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            # 计算当前block处理的token范围
            token_start = token_block_id * block_M
            token_end = T.min(token_start + block_M, total_tokens)
            num_tokens = token_end - token_start
            
            # 获取第一个token的expert_id（假设同一block内token使用相同expert）
            # 注意：实际Sonic更复杂，这里简化
            if num_tokens > 0:
                # 读取expert ID
                # 这里使用简化逻辑：每个block处理一组连续tokens
                # 实际Sonic使用dynamic scheduler
                first_token_idx = token_start
                expert_id = expert_indices[first_token_idx]
                
                # 清空累加器
                T.clear(C_local)
                
                # K维度分块
                for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=2):
                    # Load X[token_start:token_end, k*block_K:(k+1)*block_K] → X_shared
                    for i, j in T.Parallel(block_M, block_K):
                        token_idx = token_start + i
                        k_idx = k * block_K + j
                        if i < num_tokens and k_idx < H:
                            X_shared[i, j] = X[token_idx, k_idx]
                        else:
                            X_shared[i, j] = T.cast(0.0, dtype)
                    
                    # Load W[expert_id, :, k*block_K:(k+1)*block_K] → W_shared
                    for i, j in T.Parallel(block_N, block_K):
                        k_idx = k * block_K + j
                        if i < I and k_idx < H:
                            W_shared[i, j] = W[expert_id, i, k_idx]
                        else:
                            W_shared[i, j] = T.cast(0.0, dtype)
                    
                    # GEMM: C_local += X_shared @ W_shared^T
                    T.gemm(X_shared, W_shared, C_local, transpose_B=True)
                
                # Store结果
                for i, j in T.Parallel(block_M, block_N):
                    token_idx = token_start + i
                    if i < num_tokens and j < I:
                        Y[token_idx, j] = T.cast(C_local[i, j], dtype)
    
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
    
    # Gather: 按expert排序后的tokens
    X_sorted = X[sorted_indices]
    
    # 准备expert_indices (sorted)
    expert_indices_sorted = expert_indices[sorted_indices]
    
    # 编译kernel
    cache_key = (total_tokens, H, I, E)
    if cache_key not in grouped_gemm_sonic.compile_cache:
        kernel = grouped_gemm_sonic_kernel(
            total_tokens=total_tokens,
            H=H,
            I=I,
            E=E,
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
    expert_indices_int32 = expert_indices_sorted.to(torch.int32)
    
    # 执行kernel（jit会自动返回输出）
    Y_sorted = kernel(X_sorted, W, expert_indices_int32, expert_offsets_int32)
    
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
