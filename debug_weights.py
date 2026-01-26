"""
Debug weight loading in Sonic kernel
检查权重是否被正确加载
"""

import os
os.environ['DEBUG_SONIC'] = '1'

import torch
from kernels.grouped_gemm_baseline import grouped_gemm_loop
from kernels.grouped_gemm_sonic import grouped_gemm_sonic

def test_weight_loading():
    """测试权重加载"""
    print("="*80)
    print("Weight Loading Debug Test")
    print("="*80)
    
    # 超简单配置
    total_tokens = 4
    H = 8
    I = 8
    E = 4
    
    # 每个expert一个token
    expert_indices = torch.tensor([0, 1, 2, 3], device='cuda')
    
    # 创建可识别的权重：每个expert的权重不同
    # Expert 0: 全1
    # Expert 1: 全2
    # Expert 2: 全3
    # Expert 3: 全4
    W = torch.zeros((E, I, H), dtype=torch.float16, device='cuda')
    for e in range(E):
        W[e, :, :] = float(e + 1)
    
    # 输入全1
    X = torch.ones((total_tokens, H), dtype=torch.float16, device='cuda')
    
    print(f"\nConfig: total_tokens={total_tokens}, H={H}, I={I}, E={E}")
    print(f"Expert distribution: {expert_indices.tolist()}")
    print("\nWeights:")
    for e in range(E):
        print(f"  W[{e}] = {W[e, 0, 0].item()} (all values)")
    
    print("\nExpected output:")
    print("  Token 0 (expert 0): X @ W[0]^T = 1*1*8 = 8.0")
    print("  Token 1 (expert 1): X @ W[1]^T = 1*2*8 = 16.0")
    print("  Token 2 (expert 2): X @ W[2]^T = 1*3*8 = 24.0")
    print("  Token 3 (expert 3): X @ W[3]^T = 1*4*8 = 32.0")
    
    # Baseline
    Y_baseline = grouped_gemm_loop(X, W, expert_indices)
    print("\nBaseline results:")
    for i in range(total_tokens):
        print(f"  Token {i} (expert {expert_indices[i].item()}): {Y_baseline[i, 0].item():.1f}")
    
    # Sonic
    print("\n" + "="*80)
    Y_sonic = grouped_gemm_sonic(X, W, expert_indices)
    print("="*80)
    
    print("\nSonic results:")
    for i in range(total_tokens):
        expected = (expert_indices[i].item() + 1) * 8.0
        actual = Y_sonic[i, 0].item()
        status = "✓" if abs(actual - expected) < 0.1 else "✗"
        print(f"  Token {i} (expert {expert_indices[i].item()}): {actual:.1f} (expected {expected:.1f}) {status}")
    
    # 检查差异
    diff = (Y_sonic - Y_baseline).abs()
    max_diff = diff.max().item()
    
    print(f"\nMax diff: {max_diff:.6f}")
    
    if max_diff < 0.1:
        print("\n✅ PASS: All outputs correct!")
        return True
    else:
        print("\n❌ FAIL: Outputs don't match")
        print("\nDetailed diff:")
        for i in range(total_tokens):
            print(f"  Token {i}: baseline={Y_baseline[i, 0].item():.1f}, sonic={Y_sonic[i, 0].item():.1f}, diff={diff[i, 0].item():.1f}")
        return False

if __name__ == '__main__':
    test_weight_loading()
