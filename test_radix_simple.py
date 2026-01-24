"""
Simple test for radix topk - minimal reproduction
"""
import torch
from kernels.topk_radix import topk_radix_tilelang

# Test with same config as official example
batch = 64
seq_len = 32 * 1024
topk = 2048

print(f"Testing with batch={batch}, seq_len={seq_len}, topk={topk}")

torch.manual_seed(1)
input_data = torch.randn(batch, seq_len, dtype=torch.float32).cuda()

print("Running TileLang topk...")
try:
    indices = topk_radix_tilelang(input_data, topk)
    print(f"✓ Success! indices.shape = {indices.shape}, dtype = {indices.dtype}")
    
    # Compare with PyTorch
    indices_pt = torch.topk(input_data, topk, dim=-1)[1]
    print(f"✓ PyTorch: indices.shape = {indices_pt.shape}, dtype = {indices_pt.dtype}")
    
    # Check correctness
    for i in range(min(5, batch)):
        ref_np = indices_pt[i].cpu().to(torch.int32).numpy()
        tl_np = indices[i].cpu().to(torch.int32).numpy()
        set_ref = set(ref_np)
        set_tl = set(tl_np)
        intersection = set_ref & set_tl
        print(f"  Batch {i}: selected/all: {len(intersection)}/{len(set_ref)} = {len(intersection)/len(set_ref):.4f}")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
