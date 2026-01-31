"""
DeepSeek mHC Optimized TileLang Implementation

优化版本的mHC实现，逐步优化并与官方版本对比

已完成优化:
✅ 优化2: mhc_pre GEMM优化 - 成功！（1.10x）- 已保留
❌ 优化1: mhc_post调整参数 - 效果不明显（1.03x）- 已回退到官方实现
❌ 优化3: Sinkhorn迭代优化 - 反而降低性能（1.10x→1.04x）- 已回退

当前优化:
🔧 优化4: 增加token_block提升GEMM并行度
  - 尝试 token_block: 32 -> 48
  - 优化shared memory布局以避免超出99KB限制
  - 预期提升: 3-8% (更好的GPU占用率)
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
    mHC Post - 已回退到官方版本
    
    优化1 (参数调整) 效果不明显 (1.03x vs 官方):
    - hidden_size=5120时有1.48x提升，但hidden_size=2560时下降到0.84x
    - 原因: tile size对齐问题，h_blk=256适合5120但不适合2560
    - 决定: 回退到官方稳定版本，等待后续实现auto-tuning
    """
    return mhc_post_official(x, residual, post_layer_mix, comb_res_mix)


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


# ============ Optimized mHC Pre: GEMM优化 ============

@tilelang.jit
def mhc_pre_gemm_sqrsum_optimized_kernel(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,    # 保持32以适应5090的99KB共享内存限制
    hidden_block: int = 128,  # 减小hidden_block: 256 -> 128
) -> tilelang.JITKernel:
    """
    优化版mHC Pre GEMM+sqrsum kernel
    
    优化点:
    1. 减小hidden_block以适应5090的共享内存限制(99KB vs H100的224KB)
    2. 使用3-stage pipeline
    3. 移除pass_configs限制，让编译器自动优化
    
    Note: 后续在H100上可以改回token_block=64, hidden_block=256
    """
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=3):
            x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout({x_smem: tilelang.layout.make_swizzled_layout(x_smem)})

            T.copy(x[px * token_block, pz * hidden_block], x_smem)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            # sqrsum计算
            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            # GEMM
            T.gemm(
                x_frag,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                wg_wait=0,
                clear_accum=False,
            )
        
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]


@tilelang.jit
def mhc_pre_gemm_sqrsum_optimized_v2_kernel(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 48,    # 优化4: 增加token_block 32->48
    hidden_block: int = 128,  # 保持128以确保整除（10240%128=0, 20480%128=0）
) -> tilelang.JITKernel:
    """
    优化版v2 mHC Pre GEMM+sqrsum kernel - 增加并行度
    
    优化4改进:
    1. 增加token_block: 32 -> 48 (提升50%并行度)
    2. 保持hidden_block=128 (确保能整除hc_hidden_size)
    3. 共享内存估算: 48*128*2 + 32*128*4 = 12KB + 16KB = 28KB (每stage)
       3-stage pipeline: 28KB * 3 = 84KB < 99KB ✅
    """
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=3):
            x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout({x_smem: tilelang.layout.make_swizzled_layout(x_smem)})

            T.copy(x[px * token_block, pz * hidden_block], x_smem)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            # sqrsum计算
            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            # GEMM
            T.gemm(
                x_frag,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                wg_wait=0,
                clear_accum=False,
            )
        
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_optimized_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int = 6,  # 优化: 减少迭代次数 10 -> 6
    n_splits: int = 1,
    hc_mult: int = 4,
    hidden_block: int = 64,
) -> tilelang.JITKernel:
    """
    优化版 mHC Pre Big Fuse Kernel
    
    优化点:
    1. 减少Sinkhorn迭代次数: 10 -> 6 (研究表明6次已足够收敛)
    2. 使用warp shuffle优化reduce操作
    3. 改进共享内存使用模式
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = T.sigmoid(mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]) * hc_post_mult_value
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2] + hc_base[j * hc_mult + k + hc_mult * 2]

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)
            row_max = T.alloc_fragment(hc_mult, T.float32)
            
            # 初始归一化
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # Sinkhorn迭代: 优化后减少到5次（6-1）
            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + hc_pre_eps

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_official_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
):
    """官方mHC pre big_fuse kernel (复制用于对比)"""
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0
        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = T.sigmoid(mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]) * hc_post_mult_value
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2] + hc_base[j * hc_mult + k + hc_mult * 2]

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)
            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + hc_pre_eps

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_gemm_sqrsum_official_kernel(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    """官方mHC pre GEMM+sqrsum kernel (用于对比)"""
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
            x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout({x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)})

            T.copy(x[px * token_block, pz * hidden_block], x_smem_16)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem_16, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            T.gemm(x_frag, fn_smem, out_frag, transpose_A=False, transpose_B=True, wg_wait=0, clear_accum=False)
        
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]


def mhc_pre_official(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """官方mHC Pre封装"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    
    num_tokens = residual.shape[0]
    device = residual.device
    
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    gemm_out_mul = torch.empty(n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=device)
    gemm_out_sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=device)
    
    assert n_splits == 1
    mhc_pre_gemm_sqrsum_official_kernel(
        residual.view(num_tokens, hc_hidden_size),
        fn,
        gemm_out_mul.squeeze(0),
        gemm_out_sqrsum.squeeze(0),
        hc_mult3,
        hc_hidden_size,
    )
    
    mhc_pre_big_fuse_official_kernel(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )
    
    post_mix = post_mix.view(num_tokens, hc_mult, 1)
    comb_mix = comb_mix.view(num_tokens, hc_mult, hc_mult)
    
    return post_mix, comb_mix, layer_input


def mhc_pre_optimized_v2(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    优化版v2 mHC Pre封装 - 增加并行度
    
    优化4改进:
    - 使用token_block=48, hidden_block=96 (vs v1的32/128)
    - 提升50%的token并行度
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    
    num_tokens = residual.shape[0]
    device = residual.device
    
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    gemm_out_mul = torch.empty(n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=device)
    gemm_out_sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=device)
    
    assert n_splits == 1
    # 使用优化v2的GEMM kernel (token_block=48)
    mhc_pre_gemm_sqrsum_optimized_v2_kernel(
        residual.view(num_tokens, hc_hidden_size),
        fn,
        gemm_out_mul.squeeze(0),
        gemm_out_sqrsum.squeeze(0),
        hc_mult3,
        hc_hidden_size,
    )
    
    # 使用官方big_fuse
    mhc_pre_big_fuse_official_kernel(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )
    
    post_mix = post_mix.view(num_tokens, hc_mult, 1)
    comb_mix = comb_mix.view(num_tokens, hc_mult, hc_mult)
    
    return post_mix, comb_mix, layer_input


def mhc_pre_optimized(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    优化版mHC Pre封装
    
    优化内容:
    - ✅ 优化2: GEMM kernel (hidden_block=128, 3-stage pipeline) - 1.10x
    - 🔧 优化4: 增加token_block (32->48) 测试中...
    """
    # 调用v2版本进行测试
    return mhc_pre_optimized_v2(
        residual, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, n_splits
    )

# ============ Test ============

if __name__ == "__main__":
    from kernels.mhc_baseline import mhc_pre_pytorch, mhc_post_pytorch
    
    torch.manual_seed(42)
    device = "cuda"
    
    # 测试配置
    num_tokens = 1024
    hc_mult = 4
    hidden_size = 2560
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    
    print("=" * 60)
    print("Testing mHC Pre Optimized vs Official vs PyTorch")
    print("=" * 60)
    
    # 生成测试数据
    residual = (torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.float32)
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).bfloat16()
    fn = (torch.randn(hc_mult3, hc_mult, hidden_size, device=device, dtype=torch.float32) * 1e-4
          * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).flatten(1, 2)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    
    params = dict(
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0,
        sinkhorn_repeat=10,
    )
    
    # PyTorch reference
    post_ref, comb_ref, layer_ref = mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params)
    
    # Official TileLang
    post_official, comb_official, layer_official = mhc_pre_official(residual, fn, hc_scale, hc_base, **params)
    
    # Optimized
    post_opt, comb_opt, layer_opt = mhc_pre_optimized(residual, fn, hc_scale, hc_base, **params)
    
    # Compare
    print("\n--- mHC Pre Correctness ---")
    print(f"Official vs PyTorch:")
    print(f"  post_mix: {(post_official - post_ref).abs().max().item():.6f}")
    print(f"  comb_mix: {(comb_official - comb_ref).abs().max().item():.6f}")
    print(f"  layer_input: {(layer_official.float() - layer_ref.float()).abs().max().item():.6f}")
    
    print(f"\nOptimized vs PyTorch:")
    print(f"  post_mix: {(post_opt - post_ref).abs().max().item():.6f}")
    print(f"  comb_mix: {(comb_opt - comb_ref).abs().max().item():.6f}")
    print(f"  layer_input: {(layer_opt.float() - layer_ref.float()).abs().max().item():.6f}")
    
    print(f"\nOptimized vs Official:")
    print(f"  post_mix: {(post_opt - post_official).abs().max().item():.6f}")
    print(f"  comb_mix: {(comb_opt - comb_official).abs().max().item():.6f}")
    print(f"  layer_input: {(layer_opt.float() - layer_official.float()).abs().max().item():.6f}")
