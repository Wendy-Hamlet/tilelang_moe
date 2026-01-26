"""
PyTorch Baseline: TopK + Softmax 实现

对比三个版本：
1. 分离版本：先全局softmax，再topk
2. 融合版本：先topk，再对top-k做softmax（对齐Sonic）
3. Compiled版本：torch.compile优化
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def topk_softmax_separate(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准MoE路由流程：先全局softmax，再topk
    
    Args:
        logits: (batch_size, num_experts) - 未归一化的logits
        k: top-k数量
    
    Returns:
        scores: (batch_size, k) - 归一化的概率分数
        indices: (batch_size, k) - top-k的索引
    """
    # 全局softmax
    scores = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    # 选top-k
    topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
    
    return topk_scores, topk_indices


def topk_softmax_fused(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sonic优化流程：先topk，再对top-k做softmax
    
    数学等价性：
        标准: score_i = exp(logit_i) / Σⱼ₌₁ᴱ exp(logit_j)
        融合: score_i = exp(logit_i) / Σⱼ₌₁ᵏ exp(topk_logit_j)
    
    因为topk选出最大的k个，剩余(E-k)个exp值接近0，两者几乎等价。
    
    Args:
        logits: (batch_size, num_experts) - 未归一化的logits
        k: top-k数量
    
    Returns:
        scores: (batch_size, k) - 归一化的概率分数
        indices: (batch_size, k) - top-k的索引
    """
    # 先找top-k的logits（未归一化）
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    
    # 只对top-k做softmax
    topk_scores = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
    
    return topk_scores, topk_indices


# torch.compile版本：在模块级别定义，避免重复编译
@torch.compile(mode='max-autotune', fullgraph=True)
def _topk_softmax_compiled_impl(logits: torch.Tensor, k: int):
    """编译后的实现，会被缓存复用"""
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    topk_scores = F.softmax(topk_logits, dim=-1, dtype=torch.float32)
    return topk_scores, topk_indices


def topk_softmax_fused_compiled(
    logits: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    torch.compile优化的融合版本
    
    使用max-autotune模式获得最佳性能。
    编译后的函数会被缓存，避免重复编译开销。
    """
    return _topk_softmax_compiled_impl(logits, k)


class TopKSoftmaxRouter(torch.nn.Module):
    """
    MoE Router模块 - 封装topk+softmax逻辑
    
    对齐DeepSeek配置：
        - num_experts: 160 (V2) 或 256 (V3)
        - top_k: 6 (V2) 或 8 (V3)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        fusion_mode: str = 'fused',  # 'separate', 'fused', 'compiled'
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.fusion_mode = fusion_mode
        
        # Router权重：将hidden_size映射到num_experts个logit
        self.router = torch.nn.Linear(hidden_size, num_experts, bias=False)
        
        # 选择实现方式
        if fusion_mode == 'separate':
            self.topk_softmax_fn = topk_softmax_separate
        elif fusion_mode == 'fused':
            self.topk_softmax_fn = topk_softmax_fused
        elif fusion_mode == 'compiled':
            self.topk_softmax_fn = topk_softmax_fused_compiled
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch_size, hidden_size)
        
        Returns:
            router_scores: (batch_size, top_k) - 归一化的路由概率
            selected_experts: (batch_size, top_k) - 选中的expert索引
        """
        # 计算logits
        logits = self.router(hidden_states)  # (batch_size, num_experts)
        
        # TopK + Softmax
        router_scores, selected_experts = self.topk_softmax_fn(logits, self.top_k)
        
        return router_scores, selected_experts


def verify_equivalence(batch_size: int = 64, num_experts: int = 256, top_k: int = 8):
    """验证分离版本和融合版本的数学等价性"""
    print("验证TopK+Softmax融合的数学等价性")
    print("=" * 80)
    
    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_experts, device='cuda')
    
    # 分离版本（需要重新归一化以对齐融合版本）
    scores_sep_raw, indices_sep = topk_softmax_separate(logits, top_k)
    # 重新归一化topk的分数，使其和为1
    scores_sep = scores_sep_raw / scores_sep_raw.sum(dim=-1, keepdim=True)
    
    # 融合版本
    scores_fused, indices_fused = topk_softmax_fused(logits, top_k)
    
    # 检查索引是否一致
    indices_match = (indices_sep == indices_fused).all()
    print(f"索引一致性: {indices_match}")
    
    # 检查分数差异（重新归一化后）
    score_diff = (scores_sep - scores_fused).abs()
    max_diff = score_diff.max().item()
    mean_diff = score_diff.mean().item()
    
    print(f"分数最大差异（重归一化后）: {max_diff:.2e}")
    print(f"分数平均差异（重归一化后）: {mean_diff:.2e}")
    
    # 显示原始topk分数的和
    sum_sep_raw = scores_sep_raw.sum(dim=-1)
    print(f"\n分离版本原始topk分数和: min={sum_sep_raw.min():.6f}, max={sum_sep_raw.max():.6f}")
    print(f"  说明: 从全局softmax选topk，和<1是正常的")
    
    # 验证归一化
    sum_sep = scores_sep.sum(dim=-1)
    sum_fused = scores_fused.sum(dim=-1)
    
    print(f"\n重归一化后:")
    print(f"分离版本归一化检查: min={sum_sep.min():.6f}, max={sum_sep.max():.6f}")
    print(f"融合版本归一化检查: min={sum_fused.min():.6f}, max={sum_fused.max():.6f}")
    
    # 理论分析：融合版本的误差
    # 误差来源于分母差异: Σ(1..E) exp vs Σ(1..k) exp
    with torch.no_grad():
        all_scores = F.softmax(logits, dim=-1)
        # 计算top-k之外的概率和
        remaining_prob = 1.0 - all_scores.gather(-1, indices_fused).sum(dim=-1)
        print(f"\nTop-k之外的概率和: mean={remaining_prob.mean():.2e}, max={remaining_prob.max():.2e}")
        print(f"理论误差界: O({remaining_prob.max():.2e})")
    
    print("\n结论:")
    if max_diff < 1e-5:
        print("✅ 融合版本与分离版本几乎完全等价（误差<1e-5）")
        print("   两种方法在重归一化后得到相同结果")
    elif max_diff < 1e-3:
        print("✅ 融合版本与分离版本高度一致（误差<1e-3）")
        print("   数值差异在可接受范围内")
    else:
        print("⚠️  融合版本与分离版本存在明显差异")
        print("   可能的原因：浮点精度、索引选择顺序等")
    
    print("\n注意:")
    print("• 分离版本：全局softmax → topk → 重归一化")
    print("• 融合版本：topk logits → 局部softmax")
    print("• Sonic优势：避免计算所有E个expert的softmax，只计算k个")
    print("=" * 80)


if __name__ == "__main__":
    # 验证数学等价性
    print("\n测试1: DeepSeek-V2配置")
    verify_equivalence(batch_size=64, num_experts=160, top_k=6)
    
    print("\n测试2: DeepSeek-V3配置")
    verify_equivalence(batch_size=64, num_experts=256, top_k=8)
    
    print("\n测试3: 大规模配置")
    verify_equivalence(batch_size=128, num_experts=512, top_k=16)
