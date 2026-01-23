"""
DeepSeek MoE 使用示例
演示如何使用实现的MoE层
"""
import torch
import sys
sys.path.insert(0, '/home/chjin/CUDA/tilelang_moe')

from baselines.pytorch.moe_layer import MoELayer


def main():
    """简单的MoE使用示例"""
    
    # 配置
    batch_size = 4
    seq_len = 128
    hidden_size = 1024
    intermediate_size = 2048
    num_experts = 8
    num_experts_per_tok = 2
    
    print("=" * 80)
    print("DeepSeek MoE Demo")
    print("=" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Hidden size: {hidden_size}")
    print(f"Number of experts: {num_experts}")
    print(f"Experts per token: {num_experts_per_tok}")
    print("=" * 80)
    
    # 创建MoE层
    moe = MoELayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok
    ).cuda()
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    print(f"\nInput shape: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = moe(x)
    
    print(f"Output shape: {output.shape}")
    print(f"\nInput stats: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
