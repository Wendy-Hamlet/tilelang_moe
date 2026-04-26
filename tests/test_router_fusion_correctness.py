"""Quick correctness test for all router_fusion variants."""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import tilelang.language as T
from kernels.router_gemm_topk_softmax import (
    _next_pow2,
    fused_router_cpasync_selection_kernel,
    fused_router_cpasync_bitonic_kernel,
    fused_router_tma_selection_kernel,
    fused_router_tma_bitonic_kernel,
    fused_router_tma_auto_selection_kernel,
    fused_router_tma_auto_bitonic_kernel,
)


def ref_topk_softmax(X, W, K):
    logits = (X.float() @ W.float())
    vals, idx = torch.topk(logits, K, dim=-1)
    scores = torch.softmax(vals, dim=-1)
    return scores, idx


def test_variant(name, factory, T_tokens, H, E, K, threads=256, block_M=64):
    E_padded = _next_pow2(E)
    torch.manual_seed(0)
    X = torch.randn(T_tokens, H, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(H, E, device="cuda", dtype=torch.bfloat16)
    W_p = torch.zeros(H, E_padded, device="cuda", dtype=torch.bfloat16) if E != E_padded else W
    if E != E_padded:
        W_p[:, :E] = W

    try:
        kern = factory(T_tokens, H, E, E_padded, K,
                       block_M=block_M, block_K=64, num_stages=2, threads=threads,
                       dtype=T.bfloat16)
    except Exception as e:
        print(f"  [{name}] COMPILE FAIL: {e}")
        return

    try:
        scores, idx = kern(X, W_p)
    except Exception as e:
        print(f"  [{name}] RUN FAIL: {e}")
        return

    ref_s, ref_i = ref_topk_softmax(X, W, K)

    idx_match = (torch.sort(idx, dim=-1).values == torch.sort(ref_i, dim=-1).values).all(dim=-1)
    n_ok = int(idx_match.sum().item())
    nan_rows = int(torch.isnan(scores).any(dim=-1).sum().item())
    score_diff = (scores - ref_s).abs().max().item() if n_ok > 0 else float('nan')

    status = "OK" if n_ok == T_tokens else "FAIL"
    print(f"  [{name}] {status}  idx_match={n_ok}/{T_tokens}  NaN_rows={nan_rows}  max_score_diff={score_diff:.4f}")

    if n_ok != T_tokens:
        failing = (~idx_match).nonzero().flatten()[:3].tolist()
        for r in failing:
            print(f"    row {r}: got idx={idx[r].tolist()}, ref idx={ref_i[r].tolist()}, scores[:2]={scores[r, :2].tolist()}")


def main():
    configs = [
        # (name, T, H, E, K)
        ("Qwen2-57B",   256, 3584,  64, 8),
        ("Qwen3-235B",  256, 4096, 128, 8),
        ("DeepSeek-V2", 256, 5120, 160, 6),
    ]
    variants = [
        ("cpasync_sel",  fused_router_cpasync_selection_kernel,  256,  64),
        ("cpasync_bit",  fused_router_cpasync_bitonic_kernel,    256,  64),
        ("tma_sel",      fused_router_tma_selection_kernel,      384, 128),
        ("tma_bit",      fused_router_tma_bitonic_kernel,        384, 128),
        ("tma_auto_sel", fused_router_tma_auto_selection_kernel, 256,  64),
        ("tma_auto_bit", fused_router_tma_auto_bitonic_kernel,   256,  64),
    ]
    for cfg_name, T_tokens, H, E, K in configs:
        print(f"\n=== {cfg_name}  T={T_tokens} H={H} E={E} K={K} ===")
        for vname, factory, threads, block_M in variants:
            test_variant(vname, factory, T_tokens, H, E, K, threads=threads, block_M=block_M)


if __name__ == "__main__":
    main()
