"""
Grouped GEMM PyTorch Baseline Implementations

提供3个版本：
1. grouped_gemm_loop: 简单for循环实现（低效）
2. grouped_gemm_padded: 填充后批量GEMM（通用baseline）
3. grouped_gemm_loop_compiled: torch.compile优化的循环版本
"""

import torch
import torch.nn.functional as F


def grouped_gemm_loop(
    X: torch.Tensor,  # (total_tokens, H)
    W: torch.Tensor,  # (E, I, H) - 所有expert权重
    expert_indices: torch.Tensor,  # (total_tokens,) - 每个token对应的expert ID
) -> torch.Tensor:
    """
    简单循环实现grouped GEMM
    
    Args:
        X: (total_tokens, H) - 输入tokens
        W: (E, I, H) - expert权重，shape为(num_experts, intermediate_dim, hidden_dim)
        expert_indices: (total_tokens,) - 每个token对应的expert ID
    
    Returns:
        Y: (total_tokens, I) - 输出
    
    Note:
        这是最简单的实现，每个expert分别计算，效率低但正确性高
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape
    device = X.device
    dtype = X.dtype
    
    Y = torch.zeros(total_tokens, I, device=device, dtype=dtype)
    
    # 为每个expert分别计算
    for expert_id in range(E):
        # 找到使用该expert的tokens
        mask = expert_indices == expert_id
        if not mask.any():
            continue
        
        # Gather该expert的tokens
        expert_tokens = X[mask]  # (n_tokens, H)
        
        # GEMM: expert_tokens @ W[expert_id].T
        expert_weight = W[expert_id]  # (I, H)
        expert_output = expert_tokens @ expert_weight.T  # (n_tokens, I)
        
        # Scatter回输出
        Y[mask] = expert_output
    
    return Y


def grouped_gemm_padded(
    X: torch.Tensor,  # (total_tokens, H)
    W: torch.Tensor,  # (E, I, H)
    expert_indices: torch.Tensor,  # (total_tokens,)
) -> torch.Tensor:
    """
    填充对齐后使用批量GEMM实现
    
    Args:
        X: (total_tokens, H)
        W: (E, I, H)
        expert_indices: (total_tokens,)
    
    Returns:
        Y: (total_tokens, I)
    
    Note:
        这是常见的baseline实现：
        1. 计算每个expert的token数量
        2. 填充到最大值，形成(E, max_tokens, H)
        3. 批量GEMM: (E, max_tokens, H) @ (E, H, I)
        4. 去padding并scatter回原位置
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape
    device = X.device
    dtype = X.dtype
    
    # 统计每个expert的token数量
    expert_counts = torch.bincount(expert_indices, minlength=E)
    max_tokens = expert_counts.max().item()
    
    if max_tokens == 0:
        return torch.zeros(total_tokens, I, device=device, dtype=dtype)
    
    # 为每个expert创建填充后的张量
    X_padded = torch.zeros(E, max_tokens, H, device=device, dtype=dtype)
    
    # Gather tokens到对应expert的位置
    expert_offsets = torch.zeros(E, dtype=torch.long, device=device)
    for token_id in range(total_tokens):
        expert_id = expert_indices[token_id].item()
        offset = expert_offsets[expert_id]
        X_padded[expert_id, offset] = X[token_id]
        expert_offsets[expert_id] += 1
    
    # 批量GEMM: (E, max_tokens, H) @ (E, H, I) → (E, max_tokens, I)
    Y_padded = torch.bmm(X_padded, W.transpose(1, 2))
    
    # Scatter回原token位置
    Y = torch.zeros(total_tokens, I, device=device, dtype=dtype)
    expert_offsets = torch.zeros(E, dtype=torch.long, device=device)
    for token_id in range(total_tokens):
        expert_id = expert_indices[token_id].item()
        offset = expert_offsets[expert_id]
        Y[token_id] = Y_padded[expert_id, offset]
        expert_offsets[expert_id] += 1
    
    return Y


def _grouped_gemm_loop_impl(
    X: torch.Tensor,
    W: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """
    torch.compile用的实现函数
    与grouped_gemm_loop相同，但分离出来方便编译
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape
    device = X.device
    dtype = X.dtype
    
    Y = torch.zeros(total_tokens, I, device=device, dtype=dtype)
    
    for expert_id in range(E):
        mask = expert_indices == expert_id
        if not mask.any():
            continue
        
        expert_tokens = X[mask]
        expert_weight = W[expert_id]
        expert_output = expert_tokens @ expert_weight.T
        Y[mask] = expert_output
    
    return Y


# 模块级编译（仿照topk+softmax的成功经验）
grouped_gemm_loop_compiled = torch.compile(_grouped_gemm_loop_impl)


if __name__ == "__main__":
    # 简单测试
    torch.manual_seed(0)
    device = "cuda"
    
    # DeepSeek-V3配置: H=7168, I=5376
    total_tokens = 256
    H = 512  # 简化测试
    I = 384
    E = 8
    
    X = torch.randn(total_tokens, H, device=device, dtype=torch.float16)
    W = torch.randn(E, I, H, device=device, dtype=torch.float16)
    expert_indices = torch.randint(0, E, (total_tokens,), device=device)
    
    print("Testing Grouped GEMM Baseline Implementations...")
    print(f"Config: total_tokens={total_tokens}, H={H}, I={I}, E={E}")
    
    # Test loop version
    Y_loop = grouped_gemm_loop(X, W, expert_indices)
    print(f"✓ Loop version output shape: {Y_loop.shape}")
    
    # Test padded version
    Y_padded = grouped_gemm_padded(X, W, expert_indices)
    print(f"✓ Padded version output shape: {Y_padded.shape}")
    
    # Test compiled version
    Y_compiled = grouped_gemm_loop_compiled(X, W, expert_indices)
    print(f"✓ Compiled version output shape: {Y_compiled.shape}")
    
    # Check correctness
    diff_loop_padded = (Y_loop - Y_padded).abs().max().item()
    diff_loop_compiled = (Y_loop - Y_compiled).abs().max().item()
    
    print(f"\nCorrectness:")
    print(f"  Loop vs Padded: max_diff = {diff_loop_padded:.6f}")
    print(f"  Loop vs Compiled: max_diff = {diff_loop_compiled:.6f}")
    
    if diff_loop_padded < 1e-2 and diff_loop_compiled < 1e-2:
        print("✅ All versions produce consistent results!")
    else:
        print("❌ Inconsistent results detected!")
