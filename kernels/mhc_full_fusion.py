"""
DeepSeek mHC Full Fusion Implementation

算法层面优化：将mHC Pre的所有计算融合到单个kernel中

原始实现 (2 kernels):
- Kernel 1: GEMM + sqrsum
- Kernel 2: big_fuse (RMS + sigmoid + Sinkhorn + pre_mix)

Full Fusion (1 kernel):
- ALL-in-ONE: GEMM + sqrsum + RMS + sigmoid + Sinkhorn + pre_mix

优势：
1. 减少kernel launch开销
2. GEMM结果保留在shared memory，不需要写回global memory
3. 减少global memory带宽需求
"""

import math
import torch
import tilelang
import tilelang.language as T
from typing import Tuple


@tilelang.jit
def mhc_pre_full_fusion_kernel(
    residual,           # (num_tokens, hc_mult, hidden_size), bfloat16
    fn,                 # (hc_mult3, hc_mult * hidden_size), float32
    hc_scale,           # (3,), float32
    hc_base,            # (hc_mult3,), float32
    post_mix_out,       # (num_tokens, hc_mult), float32
    comb_mix_out,       # (num_tokens, hc_mult * hc_mult), float32
    layer_input_out,    # (num_tokens, hidden_size), bfloat16
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int = 10,
    hc_mult: int = 4,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    """
    Full Fusion mHC Pre Kernel
    
    每个block处理一个token，所有计算在一个kernel内完成：
    1. GEMM: mixes = (residual_flat @ fn.T) (分块累加)
    2. sqrsum: 同时累加平方和
    3. RMS: scale = rsqrt(sqrsum/dim + eps)
    4. Apply scale: mixes *= scale
    5. Sigmoid: pre_mix, post_mix
    6. Sinkhorn: comb_mix归一化
    7. pre_mix加权求和: layer_input = (residual * pre_mix).sum(-2)
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # 4+4+16=24
    hc_hidden_size = hc_mult * hidden_size      # 4*5120=20480
    
    residual: T.Tensor((num_tokens, hc_mult, hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    hc_scale: T.Tensor((3,), T.float32)
    hc_base: T.Tensor((hc_mult3,), T.float32)
    post_mix_out: T.Tensor((num_tokens, hc_mult), T.float32)
    comb_mix_out: T.Tensor((num_tokens, hc_mult * hc_mult), T.float32)
    layer_input_out: T.Tensor((num_tokens, hidden_size), T.bfloat16)
    
    # 每个block处理一个token
    with T.Kernel(num_tokens, threads=128) as token_idx:
        # ============ Stage 1: GEMM + sqrsum ============
        # 累加器
        gemm_out = T.alloc_fragment(hc_mult3, T.float32)  # (24,)
        sqrsum_acc = T.alloc_fragment(1, T.float32)
        T.clear(gemm_out)
        sqrsum_acc[0] = 0.0
        
        # 分块计算GEMM
        for h_blk in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
            # Load residual block: (hc_mult, hidden_block)
            x_smem = T.alloc_shared((hc_mult, hidden_block), T.float32)
            x_local_load = T.alloc_fragment((hc_mult, hidden_block), T.float32)
            for i in T.Parallel(hc_mult):
                for j in T.Parallel(hidden_block):
                    h_idx = h_blk * hidden_block + j
                    hc_idx = h_idx // hidden_size
                    h_local = h_idx % hidden_size
                    x_val = residual[token_idx, hc_idx, h_local]
                    x_local_load[i, j] = x_val
            
            T.copy(x_local_load, x_smem)
            
            # Load fn block: (hc_mult3, hidden_block)
            fn_smem = T.alloc_shared((hc_mult3, hidden_block), T.float32)
            T.copy(fn[0, h_blk * hidden_block], fn_smem)
            
            # 本地累加
            x_local = T.alloc_fragment((hc_mult, hidden_block), T.float32)
            T.copy(x_smem, x_local)
            
            # sqrsum累加
            for i in T.Parallel(hc_mult):
                for j in T.serial(hidden_block):
                    sqrsum_acc[0] += x_local[i, j] * x_local[i, j]
            
            # GEMM累加: gemm_out[k] += sum_j(x_flat[j] * fn[k, j])
            # x_flat = x_smem.flatten() = (hc_mult * hidden_block,)
            for k in T.Parallel(hc_mult3):
                for i in T.serial(hc_mult):
                    for j in T.serial(hidden_block):
                        gemm_out[k] += x_local[i, j] * fn_smem[k, i * hidden_block + j]
        
        # ============ Stage 2: RMS Norm ============
        rms_scale = T.alloc_fragment(1, T.float32)
        rms_scale[0] = T.rsqrt(sqrsum_acc[0] / hc_hidden_size + rms_eps)
        
        for k in T.Parallel(hc_mult3):
            gemm_out[k] *= rms_scale[0]
        
        # ============ Stage 3: Apply scale and base ============
        # hc_scale[0] for pre_mix (0:hc_mult)
        # hc_scale[1] for post_mix (hc_mult:2*hc_mult)  
        # hc_scale[2] for comb_mix (2*hc_mult:hc_mult3)
        scale_local = T.alloc_fragment(3, T.float32)
        base_local = T.alloc_fragment(hc_mult3, T.float32)
        T.copy(hc_scale, scale_local)
        T.copy(hc_base, base_local)
        
        for k in T.Parallel(hc_mult):
            gemm_out[k] = gemm_out[k] * scale_local[0] + base_local[k]
        for k in T.Parallel(hc_mult):
            gemm_out[hc_mult + k] = gemm_out[hc_mult + k] * scale_local[1] + base_local[hc_mult + k]
        for k in T.Parallel(hc_mult * hc_mult):
            gemm_out[2 * hc_mult + k] = gemm_out[2 * hc_mult + k] * scale_local[2] + base_local[2 * hc_mult + k]
        
        # ============ Stage 4: Sigmoid for pre_mix and post_mix ============
        pre_mix = T.alloc_fragment(hc_mult, T.float32)
        post_mix = T.alloc_fragment(hc_mult, T.float32)
        
        for k in T.Parallel(hc_mult):
            pre_mix[k] = T.sigmoid(gemm_out[k]) + hc_pre_eps
            post_mix[k] = T.sigmoid(gemm_out[hc_mult + k]) * hc_post_mult_value
        
        # Output post_mix
        for k in T.Parallel(hc_mult):
            post_mix_out[token_idx, k] = post_mix[k]
        
        # ============ Stage 5: Sinkhorn for comb_mix ============
        cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = gemm_out[2 * hc_mult + j * hc_mult + k]
        
        row_sum = T.alloc_fragment(hc_mult, T.float32)
        col_sum = T.alloc_fragment(hc_mult, T.float32)
        row_max = T.alloc_fragment(hc_mult, T.float32)
        
        # 初始归一化: softmax行 + eps + 列归一化
        T.reduce_max(cm, row_max, dim=1)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = T.exp(cm[j, k] - row_max[j])
        T.reduce_sum(cm, row_sum, dim=1)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps
        T.reduce_sum(cm, col_sum, dim=0)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
        
        # Sinkhorn迭代
        for _ in T.serial(sinkhorn_repeat - 1):
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
        
        # Output comb_mix
        for j, k in T.Parallel(hc_mult, hc_mult):
            comb_mix_out[token_idx, j * hc_mult + k] = cm[j, k]
        
        # ============ Stage 6: pre_mix加权求和 ============
        # layer_input = (residual * pre_mix).sum(dim=-2)
        # 需要重新加载residual数据
        for h_blk in T.Pipelined(hidden_size // hidden_block, num_stages=2):
            res_bf16 = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
            T.copy(residual[token_idx, 0, h_blk * hidden_block], res_bf16)
            
            res_local = T.alloc_fragment((hc_mult, hidden_block), T.float32)
            T.copy(res_bf16, res_local)
            
            out_local = T.alloc_fragment(hidden_block, T.float32)
            T.clear(out_local)
            
            for i_hc in T.serial(hc_mult):
                pre = pre_mix[i_hc]
                for i_h in T.Parallel(hidden_block):
                    out_local[i_h] += pre * res_local[i_hc, i_h]
            
            # Output layer_input (convert to bfloat16)
            out_bf16 = T.alloc_fragment(hidden_block, T.bfloat16)
            T.copy(out_local, out_bf16)
            for i_h in T.Parallel(hidden_block):
                layer_input_out[token_idx, h_blk * hidden_block + i_h] = out_bf16[i_h]


def mhc_pre_full_fusion(
    residual: torch.Tensor,           # (num_tokens, hc_mult, hidden_size), bfloat16
    fn: torch.Tensor,                 # (hc_mult3, hc_mult * hidden_size), float32
    hc_scale: torch.Tensor,           # (3,), float32
    hc_base: torch.Tensor,            # (hc_mult3,), float32
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full Fusion mHC Pre封装
    
    相比原始实现(2 kernels)，这个版本只用1个kernel完成所有计算
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    mhc_pre_full_fusion_kernel(
        residual,
        fn,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        hc_mult,
    )
    
    post_mix = post_mix.view(num_tokens, hc_mult, 1)
    comb_mix = comb_mix.view(num_tokens, hc_mult, hc_mult)
    
    return post_mix, comb_mix, layer_input


# ============ 简化版Full Fusion (更高效) ============

@tilelang.jit
def mhc_pre_full_fusion_v2_kernel(
    x_flat,             # (num_tokens, hc_hidden_size), bfloat16 - 预先flatten
    residual,           # (num_tokens, hc_mult, hidden_size), bfloat16
    fn,                 # (hc_mult3, hc_hidden_size), float32
    hc_scale,           # (3,), float32
    hc_base,            # (hc_mult3,), float32
    post_mix_out,       # (num_tokens, hc_mult), float32
    comb_mix_out,       # (num_tokens, hc_mult * hc_mult), float32
    layer_input_out,    # (num_tokens, hidden_size), bfloat16
    hc_mult3: int,
    hc_hidden_size: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int = 10,
    hc_mult: int = 4,
    token_block: int = 32,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    """
    Full Fusion v2: 使用预flatten的输入，支持多token并行
    
    改进:
    1. 输入预先flatten，减少kernel内索引计算
    2. 多token block并行处理
    3. 使用高效的GEMM tiling
    """
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    
    x_flat: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    residual: T.Tensor((num_tokens, hc_mult, hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    hc_scale: T.Tensor((3,), T.float32)
    hc_base: T.Tensor((hc_mult3,), T.float32)
    post_mix_out: T.Tensor((num_tokens, hc_mult), T.float32)
    comb_mix_out: T.Tensor((num_tokens, hc_mult * hc_mult), T.float32)
    layer_input_out: T.Tensor((num_tokens, hidden_size), T.bfloat16)
    
    # Grid: 按token block划分
    with T.Kernel(T.ceildiv(num_tokens, token_block), threads=128) as px:
        # ============ Stage 1: GEMM + sqrsum ============
        gemm_out = T.alloc_fragment((token_block, 32), T.float32)  # (token_block, hc_mult3)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(gemm_out)
        T.clear(sqrsum_part)
        
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=3):
            x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)
            
            T.annotate_layout({x_smem: tilelang.layout.make_swizzled_layout(x_smem)})
            
            T.copy(x_flat[px * token_block, pz * hidden_block], x_smem)
            T.copy(fn[0, pz * hidden_block], fn_smem)
            
            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)
            
            # sqrsum累加
            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]
            
            # GEMM累加
            T.gemm(
                x_frag,
                fn_smem,
                gemm_out,
                transpose_A=False,
                transpose_B=True,
                wg_wait=0,
                clear_accum=False,
            )
        
        # sqrsum reduce
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        
        # ============ Stage 2-5: RMS + Scale + Sigmoid + Sinkhorn ============
        # 每个token独立处理后续计算
        for t in T.serial(token_block):
            token_idx = px * token_block + t
            
            # RMS scale
            rms_scale = T.rsqrt(sqrsum_l[t] / hc_hidden_size + rms_eps)
            
            # Load hc_scale and hc_base
            scale_0 = hc_scale[0]
            scale_1 = hc_scale[1]
            scale_2 = hc_scale[2]
            
            # Apply scale, base, and sigmoid for pre_mix
            pre_mix = T.alloc_fragment(hc_mult, T.float32)
            for k in T.Parallel(hc_mult):
                val = gemm_out[t, k] * rms_scale * scale_0 + hc_base[k]
                pre_mix[k] = T.sigmoid(val) + hc_pre_eps
            
            # Apply scale, base, and sigmoid for post_mix, then output
            for k in T.Parallel(hc_mult):
                val = gemm_out[t, hc_mult + k] * rms_scale * scale_1 + hc_base[hc_mult + k]
                post_mix_out[token_idx, k] = T.sigmoid(val) * hc_post_mult_value
            
            # Sinkhorn for comb_mix
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = gemm_out[t, 2 * hc_mult + j * hc_mult + k] * rms_scale * scale_2 + hc_base[2 * hc_mult + j * hc_mult + k]
            
            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)
            row_max = T.alloc_fragment(hc_mult, T.float32)
            
            # 初始softmax + eps + col norm
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
            
            # Sinkhorn iterations
            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
            
            # Output comb_mix
            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix_out[token_idx, j * hc_mult + k] = cm[j, k]
            
            # ============ Stage 6: pre_mix weighted sum for layer_input ============
            # 这部分需要分离成另一个loop来处理hidden dimension
            # 暂时在这里标记，实际可能需要另一个pipeline
        
        # Stage 6: layer_input = (residual * pre_mix).sum(dim=-2)
        # 重新加载pre_mix和residual来计算
        for t in T.serial(token_block):
            token_idx = px * token_block + t
            
            # 重新计算pre_mix (因为上面的scope不共享)
            rms_scale = T.rsqrt(sqrsum_l[t] / hc_hidden_size + rms_eps)
            pre_mix = T.alloc_fragment(hc_mult, T.float32)
            for k in T.Parallel(hc_mult):
                val = gemm_out[t, k] * rms_scale * hc_scale[0] + hc_base[k]
                pre_mix[k] = T.sigmoid(val) + hc_pre_eps
            
            # 分块计算layer_input
            for h_blk in T.serial(hidden_size // 64):
                out_local = T.alloc_fragment(64, T.float32)
                T.clear(out_local)
                
                for i_hc in T.serial(hc_mult):
                    pre = pre_mix[i_hc]
                    # Load residual和计算
                    for i_h in T.Parallel(64):
                        res_val = residual[token_idx, i_hc, h_blk * 64 + i_h]
                        out_local[i_h] += pre * res_val
                
                out_bf16 = T.alloc_fragment(64, T.bfloat16)
                T.copy(out_local, out_bf16)
                for i_h in T.Parallel(64):
                    layer_input_out[token_idx, h_blk * 64 + i_h] = out_bf16[i_h]


def mhc_pre_full_fusion_v2(
    residual: torch.Tensor,           # (num_tokens, hc_mult, hidden_size), bfloat16
    fn: torch.Tensor,                 # (hc_mult3, hc_mult * hidden_size), float32
    hc_scale: torch.Tensor,           # (3,), float32
    hc_base: torch.Tensor,            # (hc_mult3,), float32
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full Fusion v2 mHC Pre封装
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    num_tokens = residual.shape[0]
    device = residual.device
    
    # 预先flatten residual
    x_flat = residual.view(num_tokens, hc_hidden_size)
    
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    mhc_pre_full_fusion_v2_kernel(
        x_flat,
        residual,
        fn,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        layer_input,
        hc_mult3,
        hc_hidden_size,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        hc_mult,
    )
    
    post_mix = post_mix.view(num_tokens, hc_mult, 1)
    comb_mix = comb_mix.view(num_tokens, hc_mult, hc_mult)
    
    return post_mix, comb_mix, layer_input


# ============ Test ============

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    
    # Test parameters
    num_tokens = 256
    hc_mult = 4
    hidden_size = 2560
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # 24
    hc_hidden_size = hc_mult * hidden_size  # 10240
    
    # Create test inputs
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device=device) * 0.01
    hc_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device=device)
    
    print("Testing Full Fusion mHC Pre...")
    
    # Test v2
    try:
        post_mix, comb_mix, layer_input = mhc_pre_full_fusion_v2(
            residual, fn, hc_scale, hc_base,
            rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=1.0, sinkhorn_repeat=10
        )
        print(f"✅ Full Fusion v2 output shapes:")
        print(f"   post_mix: {post_mix.shape}")
        print(f"   comb_mix: {comb_mix.shape}")
        print(f"   layer_input: {layer_input.shape}")
    except Exception as e:
        print(f"❌ Full Fusion v2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")
