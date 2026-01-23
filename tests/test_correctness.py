"""
正确性测试
验证TileLang实现与PyTorch baseline的输出一致性
"""
import torch
import pytest
import sys
sys.path.insert(0, '/home/chjin/CUDA/tilelang_moe')

from kernels.topk_gating import topk_gating_tilelang
from baselines.pytorch.moe_layer import MoELayer


@pytest.mark.parametrize("batch_size,num_experts,k", [
    (4, 8, 2),
    (8, 16, 4),
    (16, 64, 6),
])
def test_topk_gating_correctness(batch_size, num_experts, k):
    """测试TopK门控的正确性"""
    logits = torch.randn(batch_size, num_experts, device='cuda')
    
    # TileLang实现
    weights_tl, indices_tl = topk_gating_tilelang(logits, k)
    
    # PyTorch baseline
    weights_pt, indices_pt = torch.topk(logits, k, dim=-1)
    weights_pt = torch.softmax(weights_pt, dim=-1)
    
    # 验证
    assert torch.allclose(weights_tl, weights_pt, rtol=1e-4, atol=1e-4)
    assert torch.equal(indices_tl, indices_pt)


def test_moe_layer_shape():
    """测试MoE层的输出形状"""
    batch, seq_len = 2, 128
    hidden_size = 1024
    
    moe = MoELayer(
        hidden_size=hidden_size,
        intermediate_size=2048,
        num_experts=8,
        num_experts_per_tok=2
    ).cuda()
    
    x = torch.randn(batch, seq_len, hidden_size, device='cuda')
    output = moe(x)
    
    assert output.shape == x.shape
