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
    2) 在PyTorch侧对选出的k个logit做softmax
    
    Args:
        logits: (batch, num_experts) float32
        k: top-k count
    Returns:
        (scores, indices): both shape (batch, k)
    """
    # 1) TileLang TopK (indices)
    indices = topk_radix_tilelang(logits, k)
    # 2) Gather logits and softmax on k
    topk_logits = torch.gather(logits, dim=-1, index=indices)
    scores = torch.softmax(topk_logits, dim=-1, dtype=torch.float32)
    return scores, indices
