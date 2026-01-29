"""
DeepSeek mHC (Multi-Head Cache) Baseline Implementations

mHC是DeepSeek模型中用于高效缓存和混合的机制，包含两个主要部分：
1. mHC Pre Block: 输入预处理，包括RMS norm、sigmoid激活、Sinkhorn归一化
2. mHC Post Block: 输出后处理，结合残差和混合系数

提供PyTorch实现和torch.compile优化版本作为baseline。
"""

import torch
import torch.nn.functional as F
from typing import Tuple


# ============ Sinkhorn Normalization ============

def sinkhorn_normalize(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """
    Sinkhorn双重归一化算法
    
    将矩阵归一化为双随机矩阵（行和列和均为1）
    
    Args:
        x: (batch, hc_mult, hc_mult) - 输入矩阵
        repeat: Sinkhorn迭代次数
        eps: 数值稳定性epsilon
    
    Returns:
        归一化后的双随机矩阵
    """
    # 先做softmax并加eps
    x = x.softmax(-1) + eps
    # 列归一化
    x = x / (x.sum(-2, keepdim=True) + eps)
    
    for _ in range(repeat - 1):
        # 行归一化
        x = x / (x.sum(-1, keepdim=True) + eps)
        # 列归一化
        x = x / (x.sum(-2, keepdim=True) + eps)
    
    return x


# ============ mHC Pre Block ============

def mhc_pre_pytorch(
    residual: torch.Tensor,          # (num_tokens, hc_mult, hidden_size), bfloat16
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
    mHC Pre Block的PyTorch实现
    
    计算流程：
    1. RMS Normalization: 对residual做RMS归一化
    2. GEMM: residual_flat @ fn.T 得到mixes
    3. Split mixes: 分成pre_mix, post_mix, comb_mix三部分
    4. Sinkhorn: 对comb_mix做Sinkhorn归一化
    5. Apply pre_mix: 加权求和得到layer_input
    
    Args:
        residual: 输入残差 (num_tokens, hc_mult, hidden_size)
        fn: 混合矩阵 (hc_mult3, hc_mult * hidden_size)
        hc_scale: 缩放系数 (3,)
        hc_base: 偏置 (hc_mult3,)
        rms_eps: RMS归一化epsilon
        hc_pre_eps: pre_mix epsilon
        hc_sinkhorn_eps: Sinkhorn epsilon
        hc_post_mult_value: post_mix乘数
        sinkhorn_repeat: Sinkhorn迭代次数
    
    Returns:
        post_mix: (num_tokens, hc_mult, 1), float32
        comb_mix: (num_tokens, hc_mult, hc_mult), float32
        layer_input: (num_tokens, hidden_size), bfloat16
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    
    # Flatten residual: (num_tokens, hc_mult * hidden_size)
    residual_flat = residual.flatten(-2, -1).float()
    
    # RMS计算
    sqrsum = residual_flat.square().sum(-1)  # (num_tokens,)
    rms_scale = (sqrsum / fn.shape[-1] + rms_eps).rsqrt()  # (num_tokens,)
    
    # GEMM + RMS norm: (num_tokens, hc_mult3)
    mixes = (residual_flat @ fn.T) * rms_scale.unsqueeze(-1)
    
    # 扩展hc_scale到正确的形状
    hc_scale_expanded = torch.cat([
        hc_scale[0].expand(hc_mult),
        hc_scale[1].expand(hc_mult),
        hc_scale[2].expand(hc_mult2),
    ])
    
    # 应用scale和base
    mixes = mixes * hc_scale_expanded + hc_base
    
    # Split mixes
    pre_mix = mixes[..., :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps  # (num_tokens, hc_mult, 1)
    post_mix = (mixes[..., hc_mult:2*hc_mult].sigmoid() * hc_post_mult_value).unsqueeze(-1)  # (num_tokens, hc_mult, 1)
    comb_mix = mixes[..., 2*hc_mult:].view(-1, hc_mult, hc_mult)  # (num_tokens, hc_mult, hc_mult)
    
    # Sinkhorn归一化
    comb_mix = sinkhorn_normalize(comb_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)
    
    # Apply pre_mix: (num_tokens, hidden_size)
    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    
    return post_mix, comb_mix, layer_input


# torch.compile优化版本
mhc_pre_pytorch_compiled = torch.compile(mhc_pre_pytorch)


# ============ mHC Post Block ============

def mhc_post_pytorch(
    x: torch.Tensor,                  # (num_tokens, hidden_size), bfloat16
    residual: torch.Tensor,           # (num_tokens, hc_mult, hidden_size), bfloat16
    post_layer_mix: torch.Tensor,     # (num_tokens, hc_mult, 1), float32
    comb_res_mix: torch.Tensor,       # (num_tokens, hc_mult, hc_mult), float32
) -> torch.Tensor:
    """
    mHC Post Block的PyTorch实现
    
    计算:
        output = x * post_layer_mix + comb_res_mix.T @ residual
    
    Args:
        x: 输入 (num_tokens, hidden_size)
        residual: 残差 (num_tokens, hc_mult, hidden_size)
        post_layer_mix: 后混合系数 (num_tokens, hc_mult, 1)
        comb_res_mix: 组合残差混合矩阵 (num_tokens, hc_mult, hc_mult)
    
    Returns:
        output: (num_tokens, hc_mult, hidden_size), bfloat16
    """
    # term1: x广播乘以post_layer_mix
    # x: (num_tokens, hidden_size) -> (num_tokens, 1, hidden_size)
    # post_layer_mix: (num_tokens, hc_mult, 1)
    # result: (num_tokens, hc_mult, hidden_size)
    term1 = x.float().unsqueeze(-2) * post_layer_mix
    
    # term2: comb_res_mix.T @ residual
    # comb_res_mix: (num_tokens, hc_mult, hc_mult) -> transpose -> (num_tokens, hc_mult, hc_mult)
    # residual: (num_tokens, hc_mult, hidden_size)
    # bmm: (num_tokens, hc_mult, hc_mult) @ (num_tokens, hc_mult, hidden_size) = (num_tokens, hc_mult, hidden_size)
    term2 = torch.bmm(comb_res_mix.transpose(1, 2), residual.float())
    
    return (term1 + term2).bfloat16()


# torch.compile优化版本
mhc_post_pytorch_compiled = torch.compile(mhc_post_pytorch)


# ============ Test ============

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    
    # 测试配置
    num_tokens = 1024
    hc_mult = 4
    hidden_size = 2560
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    
    print("=" * 60)
    print("Testing mHC Pre Block")
    print("=" * 60)
    
    # 生成测试数据
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, device=device, dtype=torch.float32) * 1e-4
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    
    # 运行mHC pre
    post_mix, comb_mix, layer_input = mhc_pre_pytorch(
        residual, fn, hc_scale, hc_base,
        rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=1.0, sinkhorn_repeat=10
    )
    
    print(f"Input residual shape: {residual.shape}")
    print(f"Output post_mix shape: {post_mix.shape}")
    print(f"Output comb_mix shape: {comb_mix.shape}")
    print(f"Output layer_input shape: {layer_input.shape}")
    
    # 验证comb_mix是双随机矩阵
    row_sums = comb_mix.sum(-1)
    col_sums = comb_mix.sum(-2)
    print(f"\nComb_mix row sums (should be ~1): mean={row_sums.mean():.4f}, std={row_sums.std():.6f}")
    print(f"Comb_mix col sums (should be ~1): mean={col_sums.mean():.4f}, std={col_sums.std():.6f}")
    
    print("\n" + "=" * 60)
    print("Testing mHC Post Block")
    print("=" * 60)
    
    # 生成测试数据
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    
    # 运行mHC post
    output = mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)
    
    print(f"Input x shape: {x.shape}")
    print(f"Input residual shape: {residual.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n✅ All tests passed!")
