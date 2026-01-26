"""
Debug Sonic grouped GEMM implementation
"""
import torch
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from kernels.grouped_gemm_baseline import grouped_gemm_loop
from kernels.grouped_gemm_sonic import grouped_gemm_sonic

torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

# Small test case
total_tokens = 16
H = 64
I = 32
E = 4

print("=" * 80)
print("Debug Sonic Grouped GEMM")
print("=" * 80)
print(f"Config: total_tokens={total_tokens}, H={H}, I={I}, E={E}")

# Generate test data
X = torch.randn(total_tokens, H, device=device, dtype=dtype)
W = torch.randn(E, I, H, device=device, dtype=dtype)

# Simple expert distribution
expert_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=device, dtype=torch.int64)

print(f"\nExpert distribution: {expert_indices.tolist()}")

# Compute baseline
Y_baseline = grouped_gemm_loop(X, W, expert_indices)

# Compute Sonic
try:
    Y_sonic = grouped_gemm_sonic(X, W, expert_indices)
    
    # Compare
    max_diff = (Y_sonic - Y_baseline).abs().max().item()
    mean_diff = (Y_sonic - Y_baseline).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Baseline output shape: {Y_baseline.shape}")
    print(f"  Sonic output shape: {Y_sonic.shape}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    if max_diff < 0.1:
        print("\n✅ PASS: Outputs match!")
    else:
        print(f"\n❌ FAIL: Large difference detected!")
        
        # Debug: print more details for each expert
        print("\nDetailed comparison by expert:")
        for expert_id in range(E):
            mask = expert_indices == expert_id
            indices = torch.where(mask)[0]
            if len(indices) == 0:
                continue
            
            expert_baseline = Y_baseline[mask]
            expert_sonic = Y_sonic[mask]
            expert_diff = (expert_sonic - expert_baseline).abs()
            
            print(f"\n  Expert {expert_id} (tokens {indices.tolist()}):")
            print(f"    Max diff: {expert_diff.max().item():.6f}")
            print(f"    Mean diff: {expert_diff.mean().item():.6f}")
            print(f"    Baseline[{indices[0].item()}][0:4]: {expert_baseline[0, :4].tolist()}")
            print(f"    Sonic[{indices[0].item()}][0:4]:    {expert_sonic[0, :4].tolist()}")
            if len(indices) > 1:
                print(f"    Baseline[{indices[-1].item()}][0:4]: {expert_baseline[-1, :4].tolist()}")
                print(f"    Sonic[{indices[-1].item()}][0:4]:    {expert_sonic[-1, :4].tolist()}")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
