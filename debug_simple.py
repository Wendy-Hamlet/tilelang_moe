"""
Simplest test case - 1 token per expert
"""
import torch
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Enable debug mode
os.environ['DEBUG_SONIC'] = '1'

from kernels.grouped_gemm_baseline import grouped_gemm_loop
from kernels.grouped_gemm_sonic import grouped_gemm_sonic

torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

# Ultra simple: 4 tokens, 4 experts, 1 token each
total_tokens = 4
H = 8
I = 8
E = 4

print("=" * 80)
print("Simplest Test: 1 token per expert")
print("=" * 80)
print(f"Config: total_tokens={total_tokens}, H={H}, I={I}, E={E}")

# Generate test data - use ones for easy verification
X = torch.ones(total_tokens, H, device=device, dtype=dtype)
W = torch.ones(E, I, H, device=device, dtype=dtype)

# Each expert gets exactly 1 token
expert_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int64)

print(f"Expert distribution: {expert_indices.tolist()}")

# Expected: Y[i] = X[i] @ W[expert_indices[i]].T
# Since X=ones(4,8), W=ones(4,8,8)
# Y[i] = ones(1,8) @ ones(8,8).T = ones(1,8) * 8 = [8, 8, 8, 8, 8, 8, 8, 8]

print(f"\nExpected output (all should be 8.0):")
Y_expected = torch.ones(total_tokens, I, device=device, dtype=dtype) * H
print(f"  Y_expected shape: {Y_expected.shape}")
print(f"  Y_expected[0]: {Y_expected[0].tolist()}")

# Compute baseline
Y_baseline = grouped_gemm_loop(X, W, expert_indices)
print(f"\nBaseline:")
for i in range(total_tokens):
    print(f"  Y_baseline[{i}] (expert {expert_indices[i].item()}): {Y_baseline[i].tolist()}")

# Compute Sonic
try:
    Y_sonic = grouped_gemm_sonic(X, W, expert_indices)
    
    print(f"\nSonic:")
    for i in range(total_tokens):
        print(f"  Y_sonic[{i}] (expert {expert_indices[i].item()}): {Y_sonic[i].tolist()}")
    
    # Compare
    max_diff = (Y_sonic - Y_baseline).abs().max().item()
    mean_diff = (Y_sonic - Y_baseline).abs().mean().item()
    
    print(f"\nComparison:")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 0.1:
        print("\n✅ PASS")
    else:
        print(f"\n❌ FAIL")
        
        # Show which ones are wrong
        for i in range(total_tokens):
            diff = (Y_sonic[i] - Y_baseline[i]).abs().max().item()
            if diff > 0.1:
                print(f"  Token {i} (expert {expert_indices[i].item()}): WRONG (diff={diff:.3f})")
            else:
                print(f"  Token {i} (expert {expert_indices[i].item()}): OK")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
