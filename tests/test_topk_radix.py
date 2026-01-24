"""
Quick test to verify TopK Radix implementation correctness
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.topk_radix import topk_radix_tilelang


def test_correctness(batch=4, num_experts=160, topk=6):
    """Quick correctness test"""
    print(f"Testing: batch={batch}, num_experts={num_experts}, topk={topk}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available! Test requires GPU.")
        return False
    
    torch.manual_seed(42)
    logits = torch.randn(batch, num_experts, dtype=torch.float32, device=device)
    
    # PyTorch reference
    indices_torch = torch.topk(logits, topk, dim=-1, largest=True, sorted=False)[1]
    
    # TileLang Radix
    indices_tl = topk_radix_tilelang(logits, topk)
    
    # Get values
    values_torch = logits.gather(1, indices_torch.long())
    values_tl = logits.gather(1, indices_tl.long())
    
    # Sort and compare
    values_torch_sorted, _ = torch.sort(values_torch, dim=-1, descending=True)
    values_tl_sorted, _ = torch.sort(values_tl, dim=-1, descending=True)
    
    max_diff = (values_torch_sorted - values_tl_sorted).abs().max().item()
    
    # Calculate intersection
    total_intersection = 0
    for i in range(batch):
        ref_set = set(indices_torch[i].cpu().numpy())
        tl_set = set(indices_tl[i].cpu().numpy())
        intersection = len(ref_set & tl_set)
        total_intersection += intersection
        print(f"  Batch {i}: {intersection}/{topk} match ({intersection/topk*100:.1f}%)")
    
    avg_intersection = total_intersection / (batch * topk)
    is_correct = max_diff < 1e-5
    
    print(f"\nResults:")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Avg intersection: {avg_intersection*100:.1f}%")
    print(f"  Correct: {is_correct}")
    
    if is_correct:
        print("\n✅ Test PASSED!")
    else:
        print("\n❌ Test FAILED!")
        print(f"\nPyTorch indices:\n{indices_torch}")
        print(f"\nTileLang indices:\n{indices_tl}")
    
    return is_correct


if __name__ == '__main__':
    # Test multiple configurations
    test_cases = [
        (1, 160, 6),
        (4, 160, 6),
        (16, 160, 6),
        (64, 160, 6),
    ]
    
    print("=" * 60)
    print("TopK Radix Correctness Test")
    print("=" * 60)
    
    all_passed = True
    for batch, num_experts, topk in test_cases:
        print(f"\n{'='*60}")
        if not test_correctness(batch, num_experts, topk):
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
