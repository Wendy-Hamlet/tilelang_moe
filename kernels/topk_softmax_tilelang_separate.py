"""
TileLang无融合版本（两个算子分开）：
1) 对所有logits做全局softmax
2) 用TileLang官方TopK (Radix Selection) 选出top-k
"""
import torch
from typing import Tuple
from .topk_radix import topk_radix_tilelang


def topk_softmax_tilelang_separate(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang无融合版本（两个算子分开，方式A：标准MoE路由）：
    1) 对所有E个logits做全局softmax（得到权重分布，sum=1）
    2) 用TileLang官方TopK (Radix Selection) 选出top-k个
    
    注：与pytorch-baseline和torch.compile对齐，用方式A（sum<1.0）。
    
    Args:
        logits: (batch, num_experts) float32
        k: top-k count
    Returns:
        (scores, indices): both shape (batch, k)
    """
    # 1) 全局softmax
    import torch.nn.functional as F
    scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    # 2) TileLang TopK
    topk_indices = topk_radix_tilelang(logits, k)
    
    # 3) Gather top-k scores
    topk_scores = torch.gather(scores, dim=-1, index=topk_indices)
    
    return topk_scores, topk_indices
