"""
Debug Sonic with print statements
Check intermediate values
"""
import torch
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from kernels.grouped_gemm_baseline import grouped_gemm_loop
from kernels.grouped_gemm_sonic import prepare_expert_metadata

torch.manual_seed(42)
device = "cuda"
dtype = torch.float16

# Small test case
total_tokens = 16
H = 64
I = 32
E = 4
block_M = 128

print("=" * 80)
print("Debug Metadata Preparation")
print("=" * 80)

# Generate test data
X = torch.randn(total_tokens, H, device=device, dtype=dtype)
W = torch.randn(E, I, H, device=device, dtype=dtype)
expert_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=device, dtype=torch.int64)

print(f"\nExpert distribution: {expert_indices.tolist()}")

# Compute metadata
sorted_indices, expert_offsets = prepare_expert_metadata(expert_indices, E)

print(f"\nMetadata:")
print(f"  sorted_indices: {sorted_indices.tolist()}")
print(f"  expert_offsets: {expert_offsets.tolist()}")

# Compute block_offsets
expert_counts = expert_offsets[1:] - expert_offsets[:-1]
expert_blocks = torch.ceil(expert_counts.float() / block_M).to(torch.int32)
block_offsets = torch.cat([
    torch.tensor([0], device=device, dtype=torch.int32),
    torch.cumsum(expert_blocks, dim=0, dtype=torch.int32)
])

print(f"  expert_counts: {expert_counts.tolist()}")
print(f"  expert_blocks (block_M={block_M}): {expert_blocks.tolist()}")
print(f"  block_offsets: {block_offsets.tolist()}")

print(f"\nExpected block assignments:")
for bx in range(int(block_offsets[-1])):
    # Find expert
    expert_id = 0
    for e in range(E):
        if bx >= block_offsets[e]:
            expert_id = e
    
    local_block_id = bx - block_offsets[expert_id]
    expert_start = expert_offsets[expert_id]
    expert_end = expert_offsets[expert_id + 1]
    expert_tokens = expert_end - expert_start
    
    m_start = expert_start + local_block_id * block_M
    remaining = expert_tokens - local_block_id * block_M
    actual_rows = min(block_M, remaining)
    
    print(f"  bx={bx}: expert={expert_id}, local_block={local_block_id}, "
          f"m_start={m_start}, actual_rows={actual_rows}, "
          f"tokens=[{m_start}:{m_start+actual_rows}]")

# Check sorted X
X_sorted = X[sorted_indices]
print(f"\nSorted X:")
for i in range(total_tokens):
    orig_idx = sorted_indices[i].item()
    expert = expert_indices[orig_idx].item()
    print(f"  X_sorted[{i}] = X[{orig_idx}] (expert {expert})")
