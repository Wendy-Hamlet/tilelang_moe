"""
DeepSeek mHC Full Fusion - Simplified Version

简化版Full Fusion：每个block处理一个token，使用简单循环避免TensorCore layout问题

优势：
1. 避免TensorCore layout与串行访问的冲突
2. 所有计算在一个kernel内完成
3. GEMM结果保留在registers/shared memory
"""

import torch
import tilelang
import tilelang.language as T
from typing import Tuple


@tilelang.jit
def mhc_pre_full_fusion_simple_kernel(
    x_flat,             # (num_tokens, hc_hidden_size), bfloat16 - 预先flatten的residual
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
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    """
    Simplified Full Fusion: 每个block处理一个token
    
    使用简单的循环累加，避免TensorCore layout问题
    """
    num_tokens = T.dynamic("num_tokens")
    
    x_flat: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    residual: T.Tensor((num_tokens, hc_mult, hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    hc_scale: T.Tensor((3,), T.float32)
    hc_base: T.Tensor((hc_mult3,), T.float32)
    post_mix_out: T.Tensor((num_tokens, hc_mult), T.float32)
    comb_mix_out: T.Tensor((num_tokens, hc_mult * hc_mult), T.float32)
    layer_input_out: T.Tensor((num_tokens, hidden_size), T.bfloat16)
    
    # 每个block处理一个token
    with T.Kernel(num_tokens, threads=256) as token_idx:
        # ============ Stage 1: GEMM + sqrsum ============
        # 累加器: gemm_out[k] = sum_j(x_flat[j] * fn[k, j])
        gemm_acc = T.alloc_fragment(hc_mult3, T.float32)
        sqrsum_acc = T.alloc_fragment(1, T.float32)
        T.clear(gemm_acc)
        sqrsum_acc[0] = 0.0
        
        # 分块计算GEMM和sqrsum
        for h_blk in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
            # Load x block
            x_smem = T.alloc_shared(hidden_block, T.float32)
            for j in T.Parallel(hidden_block):
                x_smem[j] = x_flat[token_idx, h_blk * hidden_block + j]
            
            # Load fn block: (hc_mult3, hidden_block)
            fn_smem = T.alloc_shared((hc_mult3, hidden_block), T.float32)
            T.copy(fn[0, h_blk * hidden_block], fn_smem)
            
            # 计算sqrsum
            for j in T.serial(hidden_block):
                sqrsum_acc[0] += x_smem[j] * x_smem[j]
            
            # 计算GEMM: gemm_acc[k] += sum_j(x_smem[j] * fn_smem[k, j])
            for k in T.serial(hc_mult3):
                acc = T.alloc_fragment(1, T.float32)
                acc[0] = 0.0
                for j in T.serial(hidden_block):
                    acc[0] += x_smem[j] * fn_smem[k, j]
                gemm_acc[k] += acc[0]
        
        # ============ Stage 2: RMS Norm ============
        rms_scale = T.rsqrt(sqrsum_acc[0] / hc_hidden_size + rms_eps)
        
        # ============ Stage 3: Apply scale, base, sigmoid ============
        # Load scales
        scale_0 = hc_scale[0]
        scale_1 = hc_scale[1]
        scale_2 = hc_scale[2]
        
        # pre_mix: sigmoid(gemm_acc[0:4] * rms_scale * scale_0 + base) + eps
        pre_mix = T.alloc_fragment(hc_mult, T.float32)
        for k in T.serial(hc_mult):
            val = gemm_acc[k] * rms_scale * scale_0 + hc_base[k]
            pre_mix[k] = T.sigmoid(val) + hc_pre_eps
        
        # post_mix: sigmoid(gemm_acc[4:8] * rms_scale * scale_1 + base) * mult
        for k in T.serial(hc_mult):
            val = gemm_acc[hc_mult + k] * rms_scale * scale_1 + hc_base[hc_mult + k]
            post_mix_out[token_idx, k] = T.sigmoid(val) * hc_post_mult_value
        
        # ============ Stage 4: Sinkhorn for comb_mix ============
        # comb_mix: (hc_mult, hc_mult) = gemm_acc[8:24]
        cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
        for j in T.serial(hc_mult):
            for k in T.serial(hc_mult):
                cm[j, k] = gemm_acc[2 * hc_mult + j * hc_mult + k] * rms_scale * scale_2 + hc_base[2 * hc_mult + j * hc_mult + k]
        
        # Sinkhorn: softmax行 + eps + 列归一化
        row_max = T.alloc_fragment(hc_mult, T.float32)
        row_sum = T.alloc_fragment(hc_mult, T.float32)
        col_sum = T.alloc_fragment(hc_mult, T.float32)
        
        # Row max
        for j in T.serial(hc_mult):
            row_max[j] = cm[j, 0]
            for k in T.serial(hc_mult):
                if cm[j, k] > row_max[j]:
                    row_max[j] = cm[j, k]
        
        # Softmax
        for j in T.serial(hc_mult):
            row_sum[j] = 0.0
            for k in T.serial(hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
                row_sum[j] += cm[j, k]
        
        for j in T.serial(hc_mult):
            for k in T.serial(hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps
        
        # Column normalization
        for k in T.serial(hc_mult):
            col_sum[k] = 0.0
            for j in T.serial(hc_mult):
                col_sum[k] += cm[j, k]
        
        for j in T.serial(hc_mult):
            for k in T.serial(hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
        
        # Sinkhorn iterations
        for _ in T.serial(sinkhorn_repeat - 1):
            # Row normalization
            for j in T.serial(hc_mult):
                row_sum[j] = 0.0
                for k in T.serial(hc_mult):
                    row_sum[j] += cm[j, k]
            
            for j in T.serial(hc_mult):
                for k in T.serial(hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
            
            # Column normalization
            for k in T.serial(hc_mult):
                col_sum[k] = 0.0
                for j in T.serial(hc_mult):
                    col_sum[k] += cm[j, k]
            
            for j in T.serial(hc_mult):
                for k in T.serial(hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
        
        # Output comb_mix
        for j in T.serial(hc_mult):
            for k in T.serial(hc_mult):
                comb_mix_out[token_idx, j * hc_mult + k] = cm[j, k]
        
        # ============ Stage 5: layer_input = (residual * pre_mix).sum(dim=-2) ============
        for h_blk in T.Pipelined(hidden_size // 64, num_stages=2):
            # Load residual block
            res_smem = T.alloc_shared((hc_mult, 64), T.bfloat16)
            T.copy(residual[token_idx, 0, h_blk * 64], res_smem)
            
            # Compute weighted sum
            out_local = T.alloc_fragment(64, T.float32)
            T.clear(out_local)
            
            for i_hc in T.serial(hc_mult):
                pre = pre_mix[i_hc]
                for i_h in T.serial(64):
                    out_local[i_h] += pre * res_smem[i_hc, i_h]
            
            # Output
            out_bf16 = T.alloc_fragment(64, T.bfloat16)
            T.copy(out_local, out_bf16)
            for i_h in T.Parallel(64):
                layer_input_out[token_idx, h_blk * 64 + i_h] = out_bf16[i_h]


def mhc_pre_full_fusion_simple(
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
    Simplified Full Fusion mHC Pre封装
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    num_tokens = residual.shape[0]
    device = residual.device
    
    # 预先flatten residual
    x_flat = residual.reshape(num_tokens, hc_hidden_size)
    
    post_mix = torch.empty(num_tokens, hc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    
    mhc_pre_full_fusion_simple_kernel(
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
    num_tokens = 64
    hc_mult = 4
    hidden_size = 2560
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult  # 24
    hc_hidden_size = hc_mult * hidden_size
    
    # Create test inputs
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device=device) * 0.01
    hc_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device=device)
    
    print("Testing Simplified Full Fusion mHC Pre...")
    
    try:
        post_mix, comb_mix, layer_input = mhc_pre_full_fusion_simple(
            residual, fn, hc_scale, hc_base,
            rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=1.0, sinkhorn_repeat=10
        )
        print(f"✅ Simplified Full Fusion output shapes:")
        print(f"   post_mix: {post_mix.shape}")
        print(f"   comb_mix: {comb_mix.shape}")
        print(f"   layer_input: {layer_input.shape}")
    except Exception as e:
        print(f"❌ Simplified Full Fusion failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")
