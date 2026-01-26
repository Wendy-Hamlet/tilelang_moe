"""
Grouped GEMM TileLang Separate Implementation

使用TileLang官方GEMM API，但为每个expert分别调用
（分离版本，非融合）
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def tilelang_gemm_kernel(M, N, K, block_M=128, block_N=256, block_K=32, dtype=T.float16, accum_dtype=T.float32):
    """
    标准TileLang GEMM kernel
    基于官方examples/gemm/example_gemm.py
    
    Computes: C = A @ B^T
    A: (M, K)
    B: (N, K)
    C: (M, N)
    """
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return gemm


def grouped_gemm_tilelang_separate(
    X: torch.Tensor,  # (total_tokens, H)
    W: torch.Tensor,  # (E, I, H)
    expert_indices: torch.Tensor,  # (total_tokens,)
) -> torch.Tensor:
    """
    使用TileLang GEMM但分expert调用
    
    Args:
        X: (total_tokens, H)
        W: (E, I, H)
        expert_indices: (total_tokens,)
    
    Returns:
        Y: (total_tokens, I)
    
    Note:
        这是分离版本，展示TileLang GEMM API的用法
        但不是最优实现（需要为每个expert分别kernel调用）
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape
    device = X.device
    dtype = X.dtype
    
    Y = torch.zeros(total_tokens, I, device=device, dtype=dtype)
    
    # 为每个expert分别调用TileLang GEMM
    for expert_id in range(E):
        # 找到使用该expert的tokens
        mask = expert_indices == expert_id
        if not mask.any():
            continue
        
        # Gather tokens
        expert_tokens = X[mask]  # (n_tokens, H)
        n_tokens = expert_tokens.shape[0]
        
        # TileLang要求M维度是16的倍数，需要padding
        n_tokens_padded = ((n_tokens + 15) // 16) * 16
        if n_tokens_padded != n_tokens:
            expert_tokens_padded = torch.zeros(n_tokens_padded, H, device=device, dtype=dtype)
            expert_tokens_padded[:n_tokens] = expert_tokens
            expert_tokens = expert_tokens_padded
        
        # 准备权重（需要转置因为我们计算 X @ W^T）
        expert_weight = W[expert_id]  # (I, H)
        
        # 编译TileLang kernel（只编译一次通过缓存）
        cache_key = (n_tokens_padded, I, H)
        if cache_key not in grouped_gemm_tilelang_separate.compile_cache:
            # 使用合适的tile大小，确保满足TileLang约束
            # block_M: 必须是16的倍数
            block_M = 64 if n_tokens_padded < 128 else 128
            # block_N: 选择128或256，避免odd numbers导致warp_row_tiles问题
            if I >= 256:
                block_N = 256
            elif I >= 128:
                block_N = 128
            else:
                block_N = 64
            block_K = 32
            
            kernel = tilelang_gemm_kernel(
                M=n_tokens_padded,
                N=I,
                K=H,
                block_M=block_M,
                block_N=block_N,
                block_K=block_K,
                dtype=T.float16,
                accum_dtype=T.float32
            )
            grouped_gemm_tilelang_separate.compile_cache[cache_key] = kernel
        
        kernel = grouped_gemm_tilelang_separate.compile_cache[cache_key]
        
        # 执行GEMM: expert_tokens @ expert_weight^T
        # 注意：jit(out_idx=[-1])会自动返回输出，不需要显式传入
        expert_output_padded = kernel(expert_tokens, expert_weight)
        
        # 去除padding，只取有效结果
        expert_output = expert_output_padded[:n_tokens]
        
        # Scatter回输出
        Y[mask] = expert_output
    
    return Y


# Kernel编译缓存
grouped_gemm_tilelang_separate.compile_cache = {}


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
    
    print("Testing TileLang Separate Grouped GEMM...")
    print(f"Config: total_tokens={total_tokens}, H={H}, I={I}, E={E}")
    
    # TileLang version
    Y_tilelang = grouped_gemm_tilelang_separate(X, W, expert_indices)
    print(f"✓ TileLang output shape: {Y_tilelang.shape}")
    
    # PyTorch reference
    from kernels.grouped_gemm_baseline import grouped_gemm_loop
    Y_ref = grouped_gemm_loop(X, W, expert_indices)
    
    # Check correctness
    diff = (Y_tilelang - Y_ref).abs().max().item()
    print(f"\nCorrectness:")
    print(f"  TileLang vs PyTorch: max_diff = {diff:.6f}")
    
    if diff < 1e-2:
        print("✅ TileLang version matches PyTorch reference!")
    else:
        print("❌ Mismatch detected!")
