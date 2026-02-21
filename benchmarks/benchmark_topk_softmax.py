"""
TopK+Softmax Experiment Benchmark
==================================

Convention definitions:
  A-norenorm : softmax(E experts) -> topk           [scores sum < 1]
  A-renorm   : softmax(E experts) -> topk -> renorm [scores sum = 1]
  B          : topk(logits)       -> softmax(K)     [scores sum = 1]

Equivalence (sec 3.2): A-renorm == B  (exact, algebraic; fp32 diff ~1e-8)
                        A-norenorm != B (uniform scale 1/(1-delta))

Experiment design
-----------------
Exp 1 -- Convention + programming model chain  (algorithm fixed to radix)
  BL-1 : Conv A, PyTorch          -[Q1: convention swap]->
  BL-4 : Conv B, PyTorch unfused  -[Q2: programming model]->
  BL-5 : Conv B, TileLang radix unfused

Exp 2 -- Algorithm comparison  (Conv B + TileLang, all configs)
  radix (unfused, = BL-5) | bitonic | selection
  Key: bitonic/selection produce scores inside a single kernel not because
  of classical fusion but because the algorithm itself never creates an
  intermediate index tensor in global memory.

Q4 -- Total speedup: BL-1 vs Ours-best

Performance configs cover representative (E, K) from real models.
Correctness verification is in test_topk_softmax_conventions.py.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tabulate import tabulate
from tilelang.profiler import do_bench

from kernels.topk_softmax_baseline import (
    topk_softmax_separate       as bl1_conv_a_pytorch,
    topk_softmax_conv_b_unfused as bl4_conv_b_pytorch,
)
from kernels.topk_softmax_tilelang import (
    topk_softmax_conv_b_radix_unfused as bl5_conv_b_radix,
    topk_softmax_bitonic               as ours_bitonic,
    topk_softmax_selection             as ours_selection,
    topk_softmax_best                  as ours_best,
)

torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Configs: (display_name, E, K, representative_models)
# We benchmark across all (E, K) from real models. Performance is determined
# by E and K; the source model's convention only matters for correctness.
# ---------------------------------------------------------------------------
PERF_CONFIGS = [
    ("Mixtral",     8,   2, "Mixtral-8x7B/8x22B, Grok-1"),
    ("DBRX",        16,  4, "databricks/dbrx"),
    ("OLMoE/Qwen2", 64,  8, "allenai/OLMoE, Qwen/Qwen2-57B-A14B"),
    ("Qwen3",       128, 8, "Qwen/Qwen3-30B-A3B, Qwen3-235B-A22B"),
    ("DeepSeek-V2", 160, 6, "deepseek-ai/DeepSeek-V2/V2.5"),
    ("DeepSeek-V3", 256, 8, "deepseek-ai/DeepSeek-V3 routing (softmax proxy)"),
]

BATCH_SIZE = 256
WARMUP     = 50
REP        = 500


def bench(fn, logits):
    try:
        for _ in range(10):
            fn(logits)
        torch.cuda.synchronize()
        return do_bench(lambda: fn(logits), warmup=WARMUP, rep=REP, return_mode="median")
    except Exception as e:
        print(f"    [ERR] {e}")
        return None


def make_logits(batch, num_experts):
    return torch.randn(batch, num_experts, device="cuda", dtype=torch.float32)


def fmt_ms(t):
    return f"{t:.4f}" if t is not None else "ERR"


def fmt_speedup(ref, target):
    if ref and target:
        return f"{ref / target:.2f}x"
    return "N/A"


# ---------------------------------------------------------------------------
# Experiment 1: Convention + programming model chain  (Q1, Q2)
# ---------------------------------------------------------------------------
def experiment_1(batch_size):
    print()
    print("=" * 95)
    print("Experiment 1  |  Convention + Programming Model Chain  (algorithm = radix throughout)")
    print("  BL-1 (Conv A, PyTorch)  --[Q1: convention]-->  BL-4 (Conv B, PyTorch unfused)")
    print("  BL-4                    --[Q2: prog model]-->  BL-5 (Conv B, TileLang radix unfused)")
    print(f"  batch_size = {batch_size}")
    print("=" * 95)

    rows = []
    for name, E, K, _ in PERF_CONFIGS:
        logits = make_logits(batch_size, E)
        t1 = bench(lambda x: bl1_conv_a_pytorch(x, K), logits)
        t4 = bench(lambda x: bl4_conv_b_pytorch(x, K), logits)
        t5 = bench(lambda x: bl5_conv_b_radix(x, K),   logits)
        rows.append([
            name, f"E={E} K={K}",
            fmt_ms(t1), fmt_ms(t4), fmt_ms(t5),
            fmt_speedup(t1, t4),
            fmt_speedup(t4, t5),
        ])

    print(tabulate(rows,
        headers=["Model", "Config",
                 "BL-1 A-PyTorch(ms)", "BL-4 B-PyTorch(ms)", "BL-5 B-TL-radix(ms)",
                 "Q1 speedup", "Q2 speedup"],
        tablefmt="grid"))


# ---------------------------------------------------------------------------
# Experiment 2: Algorithm comparison  (Conv B + TileLang)
# ---------------------------------------------------------------------------
def experiment_2(batch_size):
    print()
    print("=" * 95)
    print("Experiment 2  |  Algorithm Comparison  (Conv B + TileLang)")
    print("  radix [unfused = BL-5]  |  bitonic  |  selection")
    print("  Note: bitonic/selection are single-kernel because the algorithm never")
    print("        writes intermediate indices to global memory -- not classical fusion.")
    print(f"  batch_size = {batch_size}")
    print("=" * 95)

    rows = []
    for name, E, K, _ in PERF_CONFIGS:
        logits     = make_logits(batch_size, E)
        t_radix    = bench(lambda x: bl5_conv_b_radix(x, K),  logits)
        t_bitonic  = bench(lambda x: ours_bitonic(x, K),      logits) if E <= 1024 else None
        t_selection= bench(lambda x: ours_selection(x, K),    logits)

        valid = [t for t in [t_bitonic, t_selection] if t is not None]
        best_t = min(valid) if valid else None
        winner = ""
        if t_bitonic is not None and t_selection is not None:
            winner = "bitonic" if t_bitonic <= t_selection else "selection"
        elif t_bitonic is not None:
            winner = "bitonic"
        elif t_selection is not None:
            winner = "selection"

        rows.append([
            name, f"E={E} K={K}",
            fmt_ms(t_radix),
            fmt_ms(t_bitonic) if E <= 1024 else "N/A(E>1024)",
            fmt_ms(t_selection),
            fmt_speedup(t_radix, best_t),
            winner,
        ])

    print(tabulate(rows,
        headers=["Model", "Config",
                 "radix-unfused(ms)", "bitonic(ms)", "selection(ms)",
                 "best/radix speedup", "winner"],
        tablefmt="grid"))


# ---------------------------------------------------------------------------
# Q4: Total speedup  BL-1 vs Ours-best
# ---------------------------------------------------------------------------
def experiment_q4(batch_size):
    print()
    print("=" * 95)
    print("Q4  |  Total Speedup  --  BL-1 (Conv A, PyTorch) vs Ours-best (Conv B, TileLang)")
    print(f"  batch_size = {batch_size}")
    print("=" * 95)

    rows = []
    for name, E, K, models in PERF_CONFIGS:
        logits = make_logits(batch_size, E)
        t_bl1  = bench(lambda x: bl1_conv_a_pytorch(x, K), logits)
        t_ours = bench(lambda x: ours_best(x, K),           logits)
        rows.append([
            name, f"E={E} K={K}", models[:42],
            fmt_ms(t_bl1), fmt_ms(t_ours),
            fmt_speedup(t_bl1, t_ours),
        ])

    print(tabulate(rows,
        headers=["Model", "Config", "Representative models",
                 "BL-1(ms)", "Ours-best(ms)", "Total speedup"],
        tablefmt="grid"))


def main():
    print("TopK+Softmax Experiment Benchmark")
    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch   : {torch.__version__}")
    print(f"batch_size: {BATCH_SIZE}  (tokens)")

    experiment_1(BATCH_SIZE)
    experiment_2(BATCH_SIZE)
    experiment_q4(BATCH_SIZE)

    print()
    print("Correctness verification: python tests/test_topk_softmax_conventions.py")


if __name__ == "__main__":
    main()
