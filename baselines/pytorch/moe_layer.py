"""
PyTorch baseline实现的MoE Layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    标准的Mixture of Experts层
    参考DeepSeek-V2/V3的实现
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        # 门控网络
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # 专家网络 (简化的FFN)
        self.experts_w1 = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size)
        )
        self.experts_w2 = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        x = x.view(-1, hidden_size)  # [batch*seq_len, hidden_size]
        
        # 门控选择
        router_logits = self.gate(x)  # [batch*seq_len, num_experts]
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # 专家计算（简化版本，未优化）
        final_output = torch.zeros_like(x)
        
        for i in range(self.num_experts_per_tok):
            expert_idx = selected_experts[:, i]
            expert_weight = routing_weights[:, i:i+1]
            
            # 为每个token选择对应的专家
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    token_input = x[mask]
                    # FFN计算
                    hidden = F.silu(token_input @ self.experts_w1[expert_id].T)
                    expert_output = hidden @ self.experts_w2[expert_id].T
                    final_output[mask] += expert_weight[mask] * expert_output
        
        return final_output.view(batch_size, seq_len, hidden_size)
