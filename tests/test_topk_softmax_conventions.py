"""
TopK+Softmax Convention Correctness Verification
=================================================

Tests the three convention types against each other:

  A-norenorm : softmax(E) -> topk           [scores sum < 1]
  A-renorm   : softmax(E) -> topk -> renorm [scores sum = 1]
  B          : topk(logits) -> softmax(K)   [scores sum = 1]

Expected outcomes (sec 3.1-3.5):
  1. Index equivalence: all three conventions select identical top-K experts (exact).
  2. A-renorm vs B: scores diff ~1e-8 (fp32 rounding only, algebraically exact).
  3. A-norenorm vs B: systematic uniform scale 1/(1-delta); delta typically 0.8-0.9
     for random logits with large E.

Model configs:
  A-renorm  : Mixtral-8x7B(HF) E=8 K=2, DBRX E=16 K=4, Qwen3-30B E=128 K=8
  A-norenorm: DeepSeek-V2 E=160 K=6, OLMoE E=64 K=8, Qwen2-57B E=64 K=8
  B         : Mixtral(Mistral-official) E=8 K=2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from tabulate import tabulate

from kernels.topk_softmax_baseline import (
    topk_softmax_separate       as conv_a_norenorm,
    topk_softmax_conv_a_renorm  as conv_a_renorm,
    topk_softmax_conv_b_unfused as conv_b,
)

torch.manual_seed(42)
NUM_TOKENS = 512


# ---------------------------------------------------------------------------
# Convention configs  (name, E, K, convention, renorm, source_model)
# ---------------------------------------------------------------------------
CONFIGS = [
    # --- A-renorm ---
    ("Mixtral-HF",    8,   2, "A", True,  "mistralai/Mixtral-8x7B-v0.1 (HuggingFace)"),
    ("DBRX",          16,  4, "A", True,  "databricks/dbrx"),
    ("Qwen3-30B",     128, 8, "A", True,  "Qwen/Qwen3-30B-A3B"),
    # --- A-norenorm ---
    ("DeepSeek-V2",   160, 6, "A", False, "deepseek-ai/DeepSeek-V2"),
    ("OLMoE",         64,  8, "A", False, "allenai/OLMoE-1B-7B"),
    ("Qwen2-57B",     64,  8, "A", False, "Qwen/Qwen2-57B-A14B"),
    # --- B ---
    ("Mixtral-Mist",  8,   2, "B", False, "mistral-inference (official)"),
]


def run_conventions(logits, k):
    s_a_no, idx_a_no = conv_a_norenorm(logits, k)
    s_a_re, idx_a_re = conv_a_renorm(logits, k)
    s_b,    idx_b    = conv_b(logits, k)
    return (s_a_no, idx_a_no), (s_a_re, idx_a_re), (s_b, idx_b)


def check_index_equiv(idx_a, idx_b):
    """Indices may be in different order; check set equality per row."""
    a_sorted = idx_a.sort(dim=-1).values
    b_sorted = idx_b.sort(dim=-1).values
    return (a_sorted == b_sorted).all().item()


def summary_table():
    print()
    print("=" * 90)
    print("Convention Correctness Verification")
    print("=" * 90)

    rows_idx   = []   # index equivalence
    rows_score = []   # score differences

    for name, E, K, conv, renorm, model in CONFIGS:
        logits = torch.randn(NUM_TOKENS, E, device="cuda", dtype=torch.float32)
        (s_ano, idx_ano), (s_are, idx_are), (s_b, idx_b) = run_conventions(logits, K)

        # --- index equivalence (all three should agree) ---
        ano_vs_b  = check_index_equiv(idx_ano, idx_b)
        are_vs_b  = check_index_equiv(idx_are, idx_b)
        ano_vs_are= check_index_equiv(idx_ano, idx_are)

        rows_idx.append([
            name, f"E={E} K={K}", conv, str(renorm),
            "YES" if ano_vs_b   else "NO",
            "YES" if are_vs_b   else "NO",
            "YES" if ano_vs_are else "NO",
        ])

        # --- score differences ---
        # A-renorm vs B (expect ~1e-8)
        diff_are_b = (s_are - s_b).abs()
        max_are_b  = diff_are_b.max().item()
        mean_are_b = diff_are_b.mean().item()

        # A-norenorm vs B (expect systematic scale)
        # Re-sort indices to align before comparing
        sorted_b_vals = s_b.gather(1, s_b.argsort(dim=1))
        sorted_ano    = s_ano.sort(dim=-1).values
        ratio_min = (sorted_b_vals / sorted_ano.clamp(min=1e-12)).min().item()
        ratio_max = (sorted_b_vals / sorted_ano.clamp(min=1e-12)).max().item()

        # delta: probability mass outside top-K
        with torch.no_grad():
            all_probs = F.softmax(logits, dim=-1)
            topk_mass = all_probs.gather(1, idx_b.to(torch.int64)).sum(dim=-1)
            delta_mean = (1 - topk_mass).mean().item()

        sum_ano = s_ano.sum(dim=-1).mean().item()
        sum_are = s_are.sum(dim=-1).mean().item()
        sum_b   = s_b.sum(dim=-1).mean().item()

        rows_score.append([
            name, f"E={E} K={K}",
            f"{sum_ano:.4f}", f"{sum_are:.4f}", f"{sum_b:.4f}",
            f"{max_are_b:.2e}", f"{mean_are_b:.2e}",
            f"{delta_mean:.3f}",
            f"{ratio_min:.2f}-{ratio_max:.2f}",
        ])

    print()
    print("Table 1: Index Equivalence  (all three conventions should select identical experts)")
    print(tabulate(rows_idx,
        headers=["Model", "Config", "Conv", "Renorm",
                 "A-norenorm==B", "A-renorm==B", "A-norenorm==A-renorm"],
        tablefmt="grid"))

    print()
    print("Table 2: Score Differences")
    print("  'A-renorm vs B max diff': expect ~1e-8 (fp32 rounding, algebraically exact)")
    print("  'delta': probability mass outside top-K;  'B/A-norenorm ratio': expect 1/(1-delta)")
    print(tabulate(rows_score,
        headers=["Model", "Config",
                 "sum(A-norenorm)", "sum(A-renorm)", "sum(B)",
                 "A-renorm vs B\nmax diff", "A-renorm vs B\nmean diff",
                 "delta", "B/A-norenorm\nratio range"],
        tablefmt="grid"))


def per_config_detail():
    """Detailed per-config report for A-norenorm vs B (the non-equivalent case)."""
    print()
    print("=" * 90)
    print("Detail: A-norenorm vs B  (systematic uniform scale, not random noise)")
    print("=" * 90)

    for name, E, K, conv, renorm, model in CONFIGS:
        if conv == "B":
            continue
        logits = torch.randn(NUM_TOKENS, E, device="cuda", dtype=torch.float32)
        (s_ano, idx_ano), _, (s_b, idx_b) = run_conventions(logits, K)

        with torch.no_grad():
            all_probs = F.softmax(logits, dim=-1)
            delta = (1 - all_probs.gather(1, idx_b.to(torch.int64)).sum(dim=-1)).mean().item()

        # Score comparison (align by sorting)
        s_ano_s = s_ano.sort(dim=-1).values
        s_b_s   = s_b.sort(dim=-1).values
        ratio   = (s_b_s / s_ano_s.clamp(min=1e-12))

        print(f"\n  {name}  (E={E}, K={K}, renorm={renorm})  [{model}]")
        print(f"    delta (mass outside top-K) = {delta:.4f}")
        print(f"    theoretical 1/(1-delta)    = {1/(1-delta):.4f}")
        print(f"    actual B/A-norenorm ratio  : min={ratio.min():.4f}  max={ratio.max():.4f}  "
              f"mean={ratio.mean():.4f}")
        print(f"    sum(A-norenorm) mean = {s_ano.sum(-1).mean():.4f}  "
              f"sum(B) mean = {s_b.sum(-1).mean():.4f}")
        if renorm:
            s_are = s_ano / s_ano.sum(dim=-1, keepdim=True)
            diff  = (s_are - s_b).abs()
            print(f"    A-renorm vs B: max diff = {diff.max():.2e}  (fp32 rounding only)")


def main():
    print("TopK+Softmax Convention Correctness Verification")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Num tokens: {NUM_TOKENS}")
    summary_table()
    per_config_detail()
    print()
    print("Done.")


if __name__ == "__main__":
    main()
