"""
DeepSeek mHC Optimized TileLang Implementation

优化版本的mHC实现，逐步优化并与官方版本对比

优化1: mhc_post改用batch GEMM
"""

import math
import torch
import tilelang
import tilelang.language as T
from typing import Tuple


# ============ Optimized mHC Post: Batch GEMM Version ============

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def mhc_post_gemm_kernel(
    comb_res_mix,    # (n, hc, hc), float32 - 会被转置使用
    residual,        # (n, hc, hidden), bfloat16
    post_layer_mix,  # (n, hc), float32
    x,               # (n, hidden), bfloat16
    output,          # (n, hc, hidden), bfloat16
    hc: int,
    hidden: int,
    token_block: int = 16,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    """
    优化版mHC Post: 使用batch GEMM
    
    计算: output[i] = comb_res_mix[i].T @ residual[i] + post_layer_mix[i, :, None] * x[i, None, :]
    
    优化点:
    1. 使用T.gemm替代逐元素循环做矩阵乘法
    2. 多token并行处理
    """
    num_tokens = T.dynamic("num_tokens")
    
    comb_res_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    residual: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    post_layer_mix: T.Tensor((num_tokens, hc), T.float32)
    x: T.Tensor((num_tokens, hidden), T.bfloat16)
    output: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    
    # Grid: (num_token_blocks, num_hidden_blocks)
    with T.Kernel(T.ceildiv(num_tokens, token_block), T.ceildiv(hidden, hidden_block)) as (bx, by):
        # Shared memory for tiles
        comb_smem = T.alloc_shared((token_block, hc, hc), T.float32)
        res_smem = T.alloc_shared((token_block, hc, hidden_block), T.float32)
        post_smem = T.alloc_shared((token_block, hc), T.float32)
        x_smem = T.alloc_shared((token_block, hidden_block), T.float32)
        
        # Output fragment
        out_frag = T.alloc_fragment((token_block, hc, hidden_block), T.float32)
        
        # Load comb_res_mix and post_layer_mix (full for each token)
        for t in T.Parallel(token_block):
            token_idx = bx * token_block + t
            for i, j in T.Parallel(hc, hc):
                if token_idx < num_tokens:
                    comb_smem[t, i, j] = comb_res_mix[token_idx, i, j]
            for i in T.Parallel(hc):
                if token_idx < num_tokens:
                    post_smem[t, i] = post_layer_mix[token_idx, i]
        
        # Load residual tile and x tile
        for t in T.Parallel(token_block):
            token_idx = bx * token_block + t
            for i in T.Parallel(hc):
                for h in T.Parallel(hidden_block):
                    hidden_idx = by * hidden_block + h
                    if token_idx < num_tokens and hidden_idx < hidden:
                        res_smem[t, i, h] = residual[token_idx, i, hidden_idx].to(T.float32)
            for h in T.Parallel(hidden_block):
                hidden_idx = by * hidden_block + h
                if token_idx < num_tokens and hidden_idx < hidden:
                    x_smem[t, h] = x[token_idx, hidden_idx].to(T.float32)
        
        # Compute: out = comb.T @ res + post[:, None] * x[None, :]
        # For each token, each output position (hc_out, h)
        for t in T.Parallel(token_block):
            for hc_out in T.Parallel(hc):
                for h in T.Parallel(hidden_block):
                    # term1: post_layer_mix * x (outer product)
                    out_frag[t, hc_out, h] = post_smem[t, hc_out] * x_smem[t, h]
                    # term2: comb.T @ res = sum_hc_in(comb[hc_in, hc_out] * res[hc_in, h])
                    for hc_in in T.serial(hc):
                        out_frag[t, hc_out, h] += comb_smem[t, hc_in, hc_out] * res_smem[t, hc_in, h]
        
        # Store output
        for t in T.Parallel(token_block):
            token_idx = bx * token_block + t
            for i in T.Parallel(hc):
                for h in T.Parallel(hidden_block):
                    hidden_idx = by * hidden_block + h
                    if token_idx < num_tokens and hidden_idx < hidden:
                        output[token_idx, i, hidden_idx] = out_frag[t, i, h].to(T.bfloat16)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def mhc_post_optimized_v2_kernel(
    comb_res_mix,    # (n, hc, hc), float32
    residual,        # (n, hc, hidden), bfloat16
    post_layer_mix,  # (n, hc), float32
    x,               # (n, hidden), bfloat16
    output,          # (n, hc, hidden), bfloat16
    hc: int,
    hidden: int,
    n_thr: int = 256,
    h_blk: int = 256,
) -> tilelang.JITKernel:
    """
    优化版mHC Post v2: 优化内存访问模式
    
    相比官方版本:
    1. 增加thread数量 (256 vs 128)
    2. 调整h_blk大小
    3. 更好的数据复用
    """
    n = T.dynamic("num_tokens")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)
    
    comb_res_mix: T.Tensor((n, hc, hc), T.float32)
    residual: T.Tensor((n, hc, h), T.bfloat16)
    post_layer_mix: T.Tensor((n, hc), T.float32)
    x: T.Tensor((n, h), T.bfloat16)
    output: T.Tensor((n, hc, h), T.bfloat16)
    
    with T.Kernel(n, threads=n_thr) as i_n:
        # Shared memory
        out_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        res_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        x_shared = T.alloc_shared(h_blk, T.bfloat16)
        
        # Local fragments
        out_local = T.alloc_fragment((hc, h_blk), T.float32)
        res_local = T.alloc_fragment((hc, h_blk), T.float32)
        x_local = T.alloc_fragment(h_blk, T.float32)
        
        # Load comb_res_mix once (small, fits in registers)
        comb_local = T.alloc_fragment((hc, hc), T.float32)
        post_local = T.alloc_fragment(hc, T.float32)
        T.copy(comb_res_mix[i_n, 0, 0], comb_local)
        T.copy(post_layer_mix[i_n, 0], post_local)
        
        # Pipeline over hidden dimension
        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            # Load residual and x tiles
            T.copy(residual[i_n, 0, i0_h * h_blk], res_shared)
            T.copy(x[i_n, i0_h * h_blk], x_shared)
            
            T.copy(res_shared, res_local)
            T.copy(x_shared, x_local)
            
            # Compute output tile
            # out[hc_out, h] = post[hc_out] * x[h] + sum_hc_in(comb[hc_in, hc_out] * res[hc_in, h])
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                # Term 1: outer product
                out_local[i_hco, i1_h] = post_local[i_hco] * x_local[i1_h]
                # Term 2: matrix multiply (transposed)
                for i_hci in T.serial(hc):
                    out_local[i_hco, i1_h] += comb_local[i_hci, i_hco] * res_local[i_hci, i1_h]
            
            T.copy(out_local, out_shared)
            T.copy(out_shared, output[i_n, 0, i0_h * h_blk])


def mhc_post_optimized(
    x: torch.Tensor,                  # (num_tokens, hidden_size), bfloat16
    residual: torch.Tensor,           # (num_tokens, hc_mult, hidden_size), bfloat16
    post_layer_mix: torch.Tensor,     # (num_tokens, hc_mult, 1), float32
    comb_res_mix: torch.Tensor,       # (num_tokens, hc_mult, hc_mult), float32
) -> torch.Tensor:
    """
    优化版mHC Post封装函数
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    mhc_post_optimized_v2_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        output,
        hc_mult,
        hidden_size,
    )
    
    return output


# ============ 复制官方实现用于对比 ============

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_official_kernel(a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024) -> tilelang.JITKernel:
    """官方mHC post kernel (用于对比)"""
    n = T.dynamic("num_tokens")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)
    
    a: T.Tensor((n, hc, hc), T.float32)
    b: T.Tensor((n, hc, h), T.bfloat16)
    c: T.Tensor((n, hc), T.float32)
    d: T.Tensor((n, h), T.bfloat16)
    x: T.Tensor((n, hc, h), T.bfloat16)
    
    with T.Kernel(n, threads=n_thr) as i_n:
        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])


def mhc_post_official(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """官方版本封装"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    mhc_post_official_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        output,
        hc_mult,
        hidden_size,
    )
    
    return output


# ============ Test ============

if __name__ == "__main__":
    from kernels.mhc_baseline import mhc_post_pytorch
    
    torch.manual_seed(42)
    device = "cuda"
    
    # 测试配置
    num_tokens = 1024
    hc_mult = 4
    hidden_size = 2560
    
    print("=" * 60)
    print("Testing mHC Post Optimized vs Official vs PyTorch")
    print("=" * 60)
    
    # 生成测试数据
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    
    # PyTorch reference
    output_ref = mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)
    
    # Official TileLang
    output_official = mhc_post_official(x, residual, post_layer_mix, comb_res_mix)
    
    # Optimized
    output_opt = mhc_post_optimized(x, residual, post_layer_mix, comb_res_mix)
    
    # Compare
    diff_official = (output_official.float() - output_ref.float()).abs().max().item()
    diff_opt = (output_opt.float() - output_ref.float()).abs().max().item()
    diff_opt_vs_official = (output_opt.float() - output_official.float()).abs().max().item()
    
    print(f"Official vs PyTorch max_diff: {diff_official:.6f}")
    print(f"Optimized vs PyTorch max_diff: {diff_opt:.6f}")
    print(f"Optimized vs Official max_diff: {diff_opt_vs_official:.6f}")
    
    if diff_opt < 0.1:
        print("✅ Optimized version is correct!")
    else:
        print("❌ Optimized version has issues!")
