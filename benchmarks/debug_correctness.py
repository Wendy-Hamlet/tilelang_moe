"""
Debug correctness issues in TopK implementation
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.topk_gating import topk_gating_pytorch, topk_gating_tilelang


def debug_topk(batch, num_experts=160, k=6):
    """Debug specific batch size to find correctness issue"""
    print(f"\n{'='*80}")
    print(f"Debugging batch={batch}, num_experts={num_experts}, k={k}")
    print('='*80)
    
    torch.manual_seed(42)  # Fixed seed for reproducibility
    scores = torch.randn(batch, num_experts, device="cuda", dtype=torch.float32)
    
    # PyTorch reference
    indices_pt, weights_pt = topk_gating_pytorch(scores, k)
    
    # TileLang
    indices_tl, weights_tl = topk_gating_tilelang(scores, k, use_optimized=False)
    
    # Detailed comparison
    print(f"\n📊 Shape check:")
    print(f"  PyTorch:  indices={indices_pt.shape}, weights={weights_pt.shape}")
    print(f"  TileLang: indices={indices_tl.shape}, weights={weights_tl.shape}")
    
    # Check if values match (sorted)
    weights_pt_sorted = weights_pt.sort(descending=True)[0]
    weights_tl_sorted = weights_tl.sort(descending=True)[0]
    
    print(f"\n🔍 Value comparison (first 5 rows):")
    for i in range(min(5, batch)):
        pt_vals = weights_pt_sorted[i].cpu().numpy()
        tl_vals = weights_tl_sorted[i].cpu().numpy()
        match = torch.allclose(weights_pt_sorted[i], weights_tl_sorted[i], rtol=1e-4, atol=1e-4)
        status = "✓" if match else "✗"
        print(f"  Row {i}: {status}")
        print(f"    PyTorch:  {pt_vals}")
        print(f"    TileLang: {tl_vals}")
        if not match:
            diff = abs(pt_vals - tl_vals)
            print(f"    Diff:     {diff}")
            print(f"    Max diff: {diff.max():.6f}")
    
    # Overall match
    overall_match = torch.allclose(weights_pt_sorted, weights_tl_sorted, rtol=1e-4, atol=1e-4)
    print(f"\n{'✅' if overall_match else '❌'} Overall match: {overall_match}")
    
    if not overall_match:
        diff = (weights_pt_sorted - weights_tl_sorted).abs()
        print(f"\n📈 Error statistics:")
        print(f"  Max error: {diff.max().item():.6e}")
        print(f"  Mean error: {diff.mean().item():.6e}")
        print(f"  Median error: {diff.median().item():.6e}")
        
        # Find problematic rows
        row_max_errors = diff.max(dim=1)[0]
        bad_rows = (row_max_errors > 1e-3).nonzero().squeeze()
        print(f"  Bad rows (error > 1e-3): {bad_rows.numel()}/{batch}")
        if bad_rows.numel() > 0 and bad_rows.numel() < 10:
            print(f"  Bad row indices: {bad_rows.cpu().tolist()}")
    
    return overall_match


if __name__ == "__main__":
    # Test problematic batch sizes
    problematic_batches = [4096, 65536]
    
    print("Testing problematic batch sizes...")
    for batch in problematic_batches:
        try:
            result = debug_topk(batch)
            print(f"\nbatch={batch}: {'PASS ✅' if result else 'FAIL ❌'}")
        except Exception as e:
            print(f"\nbatch={batch}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    # Also test a known-good batch
    print("\n" + "="*80)
    print("Testing known-good batch size for comparison...")
    debug_topk(1024)
