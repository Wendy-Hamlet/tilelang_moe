"""
TileLang无融合版本（两个算子分开）：
1) 用TileLang官方TopK (Radix Selection) 选出k个索引
2) 用TileLang官方Online Softmax对选出的k个logit做softmax
"""
import torch
from typing import Tuple
import tilelang
import tilelang.language as T
from .topk_radix import topk_radix_tilelang


@tilelang.jit(out_idx=[1])
def online_softmax_kernel(
    M: int,
    N: int,
    dtype = T.float32,
):
    """
    TileLang官方Online Softmax实现
    
    两遍扫描算法：
    1. 第一遍：计算log-sum-exp (LSE)
    2. 第二遍：应用归一化
    """
    BN = min(tilelang.next_power_of_2(N), 8192)
    NN = tilelang.cdiv(N, BN)
    accum_dtype = T.float32
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(M, threads=128) as i_m:
            x = T.alloc_fragment((BN,), dtype)
            y = T.alloc_fragment((BN,), dtype)
            lse = T.alloc_fragment((1,), accum_dtype)
            max_x = T.alloc_fragment((1,), dtype)
            exp_x = T.alloc_fragment((BN,), accum_dtype)
            sum_exp_x = T.alloc_fragment((1,), accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))

            # Pass 1: 计算LSE
            for i_n in T.Pipelined(NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                T.reduce_max(x, max_x, dim=0, clear=True)

                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)

                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)
                lse[0] = max_x[0] * scale + T.log2(T.exp2(lse[0] - max_x[0] * scale) + sum_exp_x[0])

            # Pass 2: 应用归一化
            for i_n in T.Pipelined(NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)

                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])

                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main


def topk_softmax_tilelang_separate(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang无融合版本（两个算子分开，全TileLang实现）：
    1) 用TileLang官方TopK (Radix Selection) 选出k个索引
    2) 用TileLang官方Online Softmax对选出的k个logit做softmax
    
    Args:
        logits: (batch, num_experts) float32
        k: top-k count
    Returns:
        (scores, indices): both shape (batch, k)
    """
    batch, num_experts = logits.shape
    
    # 1) TileLang TopK (indices)
    indices = topk_radix_tilelang(logits, k)
    
    # 2) Gather logits
    topk_logits = torch.gather(logits, dim=-1, index=indices)
    
    # 3) TileLang Online Softmax on k elements
    softmax_fn = online_softmax_kernel(batch, k, dtype=T.float32)
    scores = torch.empty_like(topk_logits)
    softmax_fn(topk_logits, scores)
    
    return scores, indices
