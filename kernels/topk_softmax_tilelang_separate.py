"""
TileLang无融合版本（两个算子分开）：
1) 用TileLang官方TopK (Radix Selection) 选出k个索引
2) 用PyTorch对选出的k个logit做softmax（k很小，PyTorch已足够快）
"""
import torch
from typing import Tuple
from .topk_radix import topk_radix_tilelang


def topk_softmax_tilelang_separate(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang无融合版本（两个算子分开）：
    1) 用TileLang官方TopK (Radix Selection) 选出k个索引
    2) 用PyTorch对选出的k个logit做softmax
    
    注：原计划使用TileLang online_softmax，但其对小N值（k=6）有padding bug。
        由于k很小（6-16），PyTorch softmax已经很快且正确。
    
    Args:
        logits: (batch, num_experts) float32
        k: top-k count
    Returns:
        (scores, indices): both shape (batch, k)
    """
    # 1) TileLang TopK (indices)
    indices = topk_radix_tilelang(logits, k)
    
    # 2) Gather logits and PyTorch softmax
    topk_logits = torch.gather(logits, dim=-1, index=indices)
    scores = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
    
    return scores, indices
