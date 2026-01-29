"""
DeepSeek mHC TileLang Official Implementation Wrapper

封装TileLang官方的mHC实现，适配benchmark接口
"""

import math
import torch
import tilelang
import tilelang.language as T
from typing import Tuple


# ============ TileLang Official Kernels (from examples/deepseek_mhc) ============

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_kernel(
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
    """TileLang官方的mHC pre融合kernel"""
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
        # RMS norm
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
            # Split mixes (post & comb)
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = T.sigmoid(mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]) * hc_post_mult_value
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2] + hc_base[j * hc_mult + k + hc_mult * 2]

            # Sinkhorn
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
            # Split mixes (pre) and apply
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


@tilelang.jit
def mhc_pre_gemm_sqrsum_kernel(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    """TileLang官方的GEMM+sqrsum融合kernel"""
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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_kernel(a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024):
    """TileLang官方的mHC post kernel"""
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


# ============ Wrapper Functions ============

def mhc_pre_tilelang_official(
    residual: torch.Tensor,          # (num_tokens, hc_mult, hidden_size), bfloat16
    fn: torch.Tensor,                 # (hc_mult3, hc_mult * hidden_size), float32
    hc_scale: torch.Tensor,           # (3,), float32
    hc_base: torch.Tensor,            # (hc_mult3,), float32
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TileLang官方mHC Pre实现的封装
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    
    num_tokens = residual.shape[0]
    device = residual.device
    
    # 准备输出tensors
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    # GEMM + sqrsum
    gemm_out_mul = torch.empty(n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=device)
    gemm_out_sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=device)
    
    assert n_splits == 1, "TileLang version doesn't support split-k"
    mhc_pre_gemm_sqrsum_kernel(
        residual.view(num_tokens, hc_hidden_size),
        fn,
        gemm_out_mul.squeeze(0),
        gemm_out_sqrsum.squeeze(0),
        hc_mult3,
        hc_hidden_size,
    )
    
    # Big fuse kernel
    mhc_pre_big_fuse_kernel(
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
    
    # Reshape outputs
    post_mix = post_mix.view(num_tokens, hc_mult, 1)
    comb_mix = comb_mix.view(num_tokens, hc_mult, hc_mult)
    
    return post_mix, comb_mix, layer_input


def mhc_post_tilelang_official(
    x: torch.Tensor,                  # (num_tokens, hidden_size), bfloat16
    residual: torch.Tensor,           # (num_tokens, hc_mult, hidden_size), bfloat16
    post_layer_mix: torch.Tensor,     # (num_tokens, hc_mult, 1), float32
    comb_res_mix: torch.Tensor,       # (num_tokens, hc_mult, hc_mult), float32
) -> torch.Tensor:
    """
    TileLang官方mHC Post实现的封装
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    mhc_post_kernel(
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
    from kernels.mhc_baseline import mhc_pre_pytorch, mhc_post_pytorch
    
    torch.manual_seed(42)
    device = "cuda"
    
    # 测试配置
    num_tokens = 1024
    hc_mult = 4
    hidden_size = 2560
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    
    print("=" * 60)
    print("Testing mHC Pre TileLang Official vs PyTorch")
    print("=" * 60)
    
    # 生成测试数据
    residual = (torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.float32)
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).bfloat16()
    fn = (torch.randn(hc_mult3, hc_mult, hidden_size, device=device, dtype=torch.float32) * 1e-4
          * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).flatten(1, 2)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    
    # PyTorch reference
    post_mix_ref, comb_mix_ref, layer_input_ref = mhc_pre_pytorch(
        residual, fn, hc_scale, hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0, sinkhorn_repeat=10
    )
    
    # TileLang official
    post_mix_tl, comb_mix_tl, layer_input_tl = mhc_pre_tilelang_official(
        residual, fn, hc_scale, hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0, sinkhorn_repeat=10
    )
    
    # Compare
    post_diff = (post_mix_tl - post_mix_ref).abs().max().item()
    comb_diff = (comb_mix_tl - comb_mix_ref).abs().max().item()
    layer_diff = (layer_input_tl.float() - layer_input_ref.float()).abs().max().item()
    
    print(f"post_mix max_diff: {post_diff:.6f}")
    print(f"comb_mix max_diff: {comb_diff:.6f}")
    print(f"layer_input max_diff: {layer_diff:.6f}")
    
    if post_diff < 0.01 and comb_diff < 0.01 and layer_diff < 0.1:
        print("✅ mHC Pre: TileLang matches PyTorch!")
    else:
        print("❌ mHC Pre: Mismatch detected!")
    
    print("\n" + "=" * 60)
    print("Testing mHC Post TileLang Official vs PyTorch")
    print("=" * 60)
    
    # 生成测试数据
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    
    # PyTorch reference
    output_ref = mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)
    
    # TileLang official
    output_tl = mhc_post_tilelang_official(x, residual, post_layer_mix, comb_res_mix)
    
    # Compare
    output_diff = (output_tl.float() - output_ref.float()).abs().max().item()
    print(f"output max_diff: {output_diff:.6f}")
    
    if output_diff < 0.1:
        print("✅ mHC Post: TileLang matches PyTorch!")
    else:
        print("❌ mHC Post: Mismatch detected!")
