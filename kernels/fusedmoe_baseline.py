"""
FusedMoE PyTorch Baseline Implementation

完整的MoE FFN计算，包含:
- Shared Experts: 所有token共享的专家
- Routed Experts: 根据gate分配的稀疏专家

基于 tilelang/examples/fusedmoe/example_fusedmoe_torch.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class FusedMoEPyTorch(nn.Module):
    """
    PyTorch实现的FusedMoE
    
    计算流程:
    1. Router: logits -> softmax -> topk
    2. Shared Expert: x -> gate*up -> down (所有token)
    3. Routed Experts: x -> gate*up -> down (按expert分组)
    4. 输出 = shared_output + routed_output
    """
    
    def __init__(
        self,
        d_hidden: int,
        d_expert: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_experts_per_token: int,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_expert = d_expert
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_experts_per_token = n_experts_per_token
        
        self.act_fn = nn.SiLU()
        
    def forward(
        self,
        x: torch.Tensor,
        router_weight: torch.Tensor,
        shared_gate: torch.Tensor,
        shared_up: torch.Tensor,
        shared_down: torch.Tensor,
        routed_gate: torch.Tensor,  # (n_routed_experts, d_expert, d_hidden)
        routed_up: torch.Tensor,    # (n_routed_experts, d_expert, d_hidden)
        routed_down: torch.Tensor,  # (n_routed_experts, d_hidden, d_expert)
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_hidden)
            router_weight: (n_routed_experts, d_hidden)
            shared_*: shared expert权重
            routed_*: routed expert权重 (stacked)
        
        Returns:
            output: (batch_size, seq_len, d_hidden)
        """
        batch_size, seq_len, d_hidden = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, d_hidden)
        
        # 1. Router: 计算expert分配
        logits = x_flat @ router_weight.T  # (B*S, n_routed_experts)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(
            scores, k=self.n_experts_per_token, dim=-1, sorted=False
        )
        
        # 2. Shared Expert计算 (所有token)
        shared_output = self._compute_expert(
            x_flat, shared_gate, shared_up, shared_down
        )
        
        # 3. Routed Expert计算 (分组)
        routed_output = self._compute_routed_experts(
            x_flat, topk_indices, topk_scores,
            routed_gate, routed_up, routed_down
        )
        
        # 4. 合并输出
        output = (shared_output + routed_output).view(*orig_shape)
        return output
    
    def _compute_expert(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
    ) -> torch.Tensor:
        """单个Expert的FFN计算: SiLU(x @ gate) * (x @ up) @ down"""
        gate_out = self.act_fn(x @ gate.T)
        up_out = x @ up.T
        return (gate_out * up_out) @ down.T
    
    def _compute_routed_experts(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        routed_gate: torch.Tensor,
        routed_up: torch.Tensor,
        routed_down: torch.Tensor,
    ) -> torch.Tensor:
        """
        分组计算Routed Experts
        
        使用排序+分组的方式减少随机访问
        """
        num_tokens = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # 展平indices和scores
        flat_indices = topk_indices.view(-1)  # (num_tokens * k,)
        flat_scores = topk_scores.view(-1, 1)  # (num_tokens * k, 1)
        
        # 按expert_id排序
        sorted_idx = flat_indices.argsort()
        counts = flat_indices.bincount(minlength=self.n_routed_experts).cpu().numpy()
        offsets = counts.cumsum()
        
        # token索引映射
        num_per_tok = self.n_experts_per_token
        token_idxs = sorted_idx // num_per_tok
        
        # 输出缓存
        expert_cache = torch.zeros(num_tokens, self.d_hidden, device=device, dtype=dtype)
        
        # 按expert分组计算
        for expert_id in range(self.n_routed_experts):
            start_idx = 0 if expert_id == 0 else offsets[expert_id - 1]
            end_idx = offsets[expert_id]
            
            if start_idx == end_idx:
                continue
            
            # 获取该expert的token
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            
            # Expert FFN计算
            gate_out = self.act_fn(expert_tokens @ routed_gate[expert_id].T)
            up_out = expert_tokens @ routed_up[expert_id].T
            expert_out = (gate_out * up_out) @ routed_down[expert_id].T
            
            # 加权
            weights = flat_scores[sorted_idx[start_idx:end_idx]]
            expert_out = expert_out * weights
            
            # scatter_reduce累加
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).expand(-1, self.d_hidden),
                expert_out,
                reduce="sum"
            )
        
        return expert_cache


def fusedmoe_pytorch(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor, 
    shared_down: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_up: torch.Tensor,
    routed_down: torch.Tensor,
    n_experts_per_token: int,
) -> torch.Tensor:
    """
    Functional API for FusedMoE PyTorch
    """
    batch_size, seq_len, d_hidden = x.shape
    n_routed_experts = routed_gate.shape[0]
    d_expert = routed_gate.shape[1]
    n_shared_experts = shared_gate.shape[0] // d_expert
    
    model = FusedMoEPyTorch(
        d_hidden=d_hidden,
        d_expert=d_expert,
        n_routed_experts=n_routed_experts,
        n_shared_experts=n_shared_experts,
        n_experts_per_token=n_experts_per_token,
    )
    
    with torch.no_grad():
        return model(
            x, router_weight,
            shared_gate, shared_up, shared_down,
            routed_gate, routed_up, routed_down
        )


def _fusedmoe_impl(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_up: torch.Tensor,
    routed_down: torch.Tensor,
    n_experts_per_token: int,
) -> torch.Tensor:
    """
    用于torch.compile的实现函数
    """
    batch_size, seq_len, d_hidden = x.shape
    n_routed_experts = routed_gate.shape[0]
    d_expert = routed_gate.shape[1]
    
    x_flat = x.view(-1, d_hidden)
    num_tokens = x_flat.shape[0]
    device = x.device
    dtype = x.dtype
    
    # 1. Router
    logits = x_flat @ router_weight.T
    scores = F.softmax(logits, dim=-1)
    topk_scores, topk_indices = torch.topk(
        scores, k=n_experts_per_token, dim=-1, sorted=False
    )
    
    # 2. Shared Expert
    shared_gate_out = F.silu(x_flat @ shared_gate.T)
    shared_up_out = x_flat @ shared_up.T
    shared_output = (shared_gate_out * shared_up_out) @ shared_down.T
    
    # 3. Routed Experts (分组)
    flat_indices = topk_indices.view(-1)
    flat_scores = topk_scores.view(-1, 1)
    
    sorted_idx = flat_indices.argsort()
    counts = flat_indices.bincount(minlength=n_routed_experts)
    offsets = counts.cumsum(dim=0)
    
    num_per_tok = n_experts_per_token
    token_idxs = sorted_idx // num_per_tok
    
    expert_cache = torch.zeros(num_tokens, d_hidden, device=device, dtype=dtype)
    
    for expert_id in range(n_routed_experts):
        start_idx = 0 if expert_id == 0 else offsets[expert_id - 1].item()
        end_idx = offsets[expert_id].item()
        
        if start_idx == end_idx:
            continue
        
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x_flat[exp_token_idxs]
        
        gate_out = F.silu(expert_tokens @ routed_gate[expert_id].T)
        up_out = expert_tokens @ routed_up[expert_id].T
        expert_out = (gate_out * up_out) @ routed_down[expert_id].T
        
        weights = flat_scores[sorted_idx[start_idx:end_idx]]
        expert_out = expert_out * weights
        
        expert_cache.scatter_reduce_(
            0,
            exp_token_idxs.view(-1, 1).expand(-1, d_hidden),
            expert_out,
            reduce="sum"
        )
    
    output = (shared_output + expert_cache).view(batch_size, seq_len, d_hidden)
    return output


# Pre-compiled version (will be compiled on first call)
_compiled_fusedmoe = None

def fusedmoe_pytorch_compiled(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_up: torch.Tensor,
    routed_down: torch.Tensor,
    n_experts_per_token: int,
) -> torch.Tensor:
    """
    torch.compile version of FusedMoE
    """
    global _compiled_fusedmoe
    if _compiled_fusedmoe is None:
        _compiled_fusedmoe = torch.compile(_fusedmoe_impl)
    
    return _compiled_fusedmoe(
        x, router_weight,
        shared_gate, shared_up, shared_down,
        routed_gate, routed_up, routed_down,
        n_experts_per_token
    )


def generate_fusedmoe_inputs(
    batch_size: int,
    seq_len: int,
    d_hidden: int,
    d_expert: int,
    n_routed_experts: int,
    n_shared_experts: int,
    n_experts_per_token: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    生成FusedMoE测试数据
    
    Returns:
        x: (batch_size, seq_len, d_hidden)
        weights: dict containing all weight tensors
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    x = torch.randn(
        batch_size, seq_len, d_hidden,
        device=device, dtype=dtype, generator=gen
    ).contiguous()
    
    weights = {}
    
    # Router权重
    weights["router_weight"] = torch.randn(
        n_routed_experts, d_hidden,
        device=device, dtype=dtype, generator=gen
    ) / math.sqrt(d_hidden)
    
    # Shared Expert权重
    shared_dim = d_expert * n_shared_experts
    weights["shared_gate"] = torch.randn(
        shared_dim, d_hidden,
        device=device, dtype=dtype, generator=gen
    ) / math.sqrt(d_hidden)
    weights["shared_up"] = torch.randn(
        shared_dim, d_hidden,
        device=device, dtype=dtype, generator=gen
    ) / math.sqrt(d_hidden)
    weights["shared_down"] = torch.randn(
        d_hidden, shared_dim,
        device=device, dtype=dtype, generator=gen
    ) / math.sqrt(shared_dim)
    
    # Routed Expert权重 (stacked)
    weights["routed_gate"] = torch.randn(
        n_routed_experts, d_expert, d_hidden,
        device=device, dtype=dtype, generator=gen
    ) / math.sqrt(d_hidden)
    weights["routed_up"] = torch.randn(
        n_routed_experts, d_expert, d_hidden,
        device=device, dtype=dtype, generator=gen
    ) / math.sqrt(d_hidden)
    weights["routed_down"] = torch.randn(
        n_routed_experts, d_hidden, d_expert,
        device=device, dtype=dtype, generator=gen
    ) / math.sqrt(d_expert)
    
    return x, weights


if __name__ == "__main__":
    # Test basic functionality
    torch.manual_seed(42)
    
    config = {
        "batch_size": 1,
        "seq_len": 1024,
        "d_hidden": 2048,
        "d_expert": 512,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_experts_per_token": 2,
    }
    
    x, weights = generate_fusedmoe_inputs(**config)
    
    print(f"Input shape: {x.shape}")
    print(f"Router weight: {weights['router_weight'].shape}")
    print(f"Routed gate: {weights['routed_gate'].shape}")
    
    # Test PyTorch version
    output = fusedmoe_pytorch(
        x, weights["router_weight"],
        weights["shared_gate"], weights["shared_up"], weights["shared_down"],
        weights["routed_gate"], weights["routed_up"], weights["routed_down"],
        config["n_experts_per_token"]
    )
    print(f"Output shape: {output.shape}")
    print("✅ PyTorch FusedMoE works!")
    
    # Test compiled version
    output_compiled = fusedmoe_pytorch_compiled(
        x, weights["router_weight"],
        weights["shared_gate"], weights["shared_up"], weights["shared_down"],
        weights["routed_gate"], weights["routed_up"], weights["routed_down"],
        config["n_experts_per_token"]
    )
    print(f"Compiled output shape: {output_compiled.shape}")
    
    # Check consistency
    diff = (output - output_compiled).abs().max().item()
    print(f"Max diff: {diff}")
    print("✅ torch.compile FusedMoE works!")
