"""
Quick test for SonicMoE-style TopK implementations
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.topk_sonic_style import topk_sonic_style


def test_sonic_style(batch=4, num_experts=160, topk=6, test_warp=False):
    """Quick correctness test"""
    print(f"Testing SonicMoE-style TopK: batch={batch}, num_experts={num_experts}, topk={topk}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available! Test requires GPU.")
        return False
    
    torch.manual_seed(42)
    logits = torch.randn(batch, num_experts, dtype=torch.float32, device=device)
    
    # PyTorch reference
    values_torch, indices_torch = torch.topk(logits, topk, dim=-1, largest=True, sorted=False)
    
    print("\n" + "=" * 60)
    print("Testing Thread-Level Version")
    print("=" * 60)
    
    # Thread-level version
    values_thread, indices_thread = topk_sonic_style(logits, topk, use_warp=False)
    
    # Check correctness
    def check_and_report(values, indices, name):
        values_sorted, _ = torch.sort(values, dim=-1, descending=True)
        values_torch_sorted, _ = torch.sort(values_torch, dim=-1, descending=True)
        
        max_diff = (values_sorted - values_torch_sorted).abs().max().item()
        
        print(f"\n{name}:")
        for i in range(min(batch, 4)):
            ref_set = set(indices_torch[i].cpu().numpy())
            test_set = set(indices[i].cpu().numpy())
            intersection = len(ref_set & test_set)
            print(f"  Batch {i}: {intersection}/{topk} match ({intersection/topk*100:.1f}%)")
            if intersection < topk:
                print(f"    PyTorch: {sorted(indices_torch[i].cpu().numpy())}")
                print(f"    {name}:   {sorted(indices[i].cpu().numpy())}")
        
        is_correct = max_diff < 1e-5
        print(f"\n  Max value diff: {max_diff:.2e}")
        print(f"  Status: {'✅ PASSED' if is_correct else '❌ FAILED'}")
        
        return is_correct
    
    thread_ok = check_and_report(values_thread, indices_thread, "Thread-Level")
    
    warp_ok = True
    if test_warp:
        print("\n" + "=" * 60)
        print("Testing Warp-Level Version")
        print("=" * 60)
        
        # Warp-level version
        values_warp, indices_warp = topk_sonic_style(logits, topk, use_warp=True)
        warp_ok = check_and_report(values_warp, indices_warp, "Warp-Level")
    else:
        print("\n" + "=" * 60)
        print("Skipping Warp-Level Version (use test_warp=True to enable)")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    if thread_ok and warp_ok:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
    
    return thread_ok and warp_ok


if __name__ == '__main__':
    test_cases = [
        (1, 160, 6),
        (4, 160, 6),
        (16, 160, 6),
        (64, 160, 6),
    ]
    
    print("=" * 60)
    print("SonicMoE-Style TopK Correctness Test")
    print("=" * 60)
    
    all_passed = True
    for batch, num_experts, topk in test_cases:
        print(f"\n{'='*60}")
        if not test_sonic_style(batch, num_experts, topk):
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("=" * 60)
