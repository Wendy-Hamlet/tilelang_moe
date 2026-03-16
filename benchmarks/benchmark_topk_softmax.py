"""
TopK+Softmax Experiment Benchmark  (TileLang environment)
==========================================================

Run in: conda activate tilelang

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

Q4 -- Total speedup matrix: BL-1 vs Ours-best across (num_tokens x model)

Token sweep
-----------
  256   : small decode batch (latency-bound)
  1024  : medium decode batch
  4096  : small prefill / training micro-batch
  16384 : large prefill (memory-bandwidth-bound)

At large num_tokens, the Q1 convention swap matters more:
  Conv A softmax reads E values/token  (e.g. 256 floats for DeepSeek-V3)
  Conv B softmax reads K values/token  (e.g. 8 floats for DeepSeek-V3)
  -> 32x less data for softmax at E=256,K=8; benefit grows with num_tokens.

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
# ---------------------------------------------------------------------------
PERF_CONFIGS = [
    ("Mixtral",     8,   2, "Mixtral-8x7B/8x22B, Grok-1"),
    ("DBRX",        16,  4, "databricks/dbrx"),
    ("OLMoE/Qwen2", 64,  8, "allenai/OLMoE, Qwen/Qwen2-57B-A14B"),
    ("Qwen3",       128, 8, "Qwen/Qwen3-30B-A3B, Qwen3-235B-A22B"),
    ("DeepSeek-V2", 160, 6, "deepseek-ai/DeepSeek-V2/V2.5"),
    ("DeepSeek-V3", 256, 8, "deepseek-ai/DeepSeek-V3 routing (softmax proxy)"),
]

# Token sweep: decode (256, 1024) -> prefill/training (4096, 16384)
TOKEN_SWEEP = [256, 1024, 4096, 16384]

WARMUP = 50
REP    = 500


def bench(fn, logits):
    try:
        for _ in range(10):
            fn(logits)
        torch.cuda.synchronize()
        return do_bench(lambda: fn(logits), warmup=WARMUP, rep=REP, return_mode="median")
    except Exception as e:
        print(f"    [ERR] {e}")
        return None


def make_logits(num_tokens, num_experts):
    return torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)


def fmt_ms(t):
    return f"{t:.4f}" if t is not None else "ERR"


def fmt_speedup(ref, target):
    if ref and target:
        return f"{ref / target:.2f}x"
    return "N/A"


# ---------------------------------------------------------------------------
# Experiment 1: Convention + programming model chain  (Q1, Q2)
# Shows per-token-count sub-tables to reveal when convention swap matters.
# ---------------------------------------------------------------------------
def experiment_1(token_sweep):
    print()
    print("=" * 95)
    print("Experiment 1  |  Convention + Programming Model Chain  (algorithm = radix throughout)")
    print("  BL-1 (Conv A, PyTorch)  --[Q1: convention]-->  BL-4 (Conv B, PyTorch unfused)")
    print("  BL-4                    --[Q2: prog model]-->  BL-5 (Conv B, TileLang radix unfused)")
    print("  Hypothesis: Q1 speedup grows with num_tokens (Conv A reads E vals, Conv B reads K vals)")
    print("=" * 95)

    for num_tokens in token_sweep:
        print(f"\n  -- num_tokens = {num_tokens} --")
        rows = []
        for name, E, K, _ in PERF_CONFIGS:
            logits = make_logits(num_tokens, E)
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
def experiment_2(token_sweep):
    print()
    print("=" * 95)
    print("Experiment 2  |  Algorithm Comparison  (Conv B + TileLang)")
    print("  radix [unfused = BL-5]  |  bitonic  |  selection")
    print("  Note: bitonic/selection are single-kernel because the algorithm never")
    print("        writes intermediate indices to global memory -- not classical fusion.")
    print("=" * 95)

    for num_tokens in token_sweep:
        print(f"\n  -- num_tokens = {num_tokens} --")
        rows = []
        for name, E, K, _ in PERF_CONFIGS:
            logits      = make_logits(num_tokens, E)
            t_radix     = bench(lambda x: bl5_conv_b_radix(x, K),  logits)
            t_bitonic   = bench(lambda x: ours_bitonic(x, K),      logits) if E <= 1024 else None
            t_selection = bench(lambda x: ours_selection(x, K),    logits)

            valid  = [t for t in [t_bitonic, t_selection] if t is not None]
            best_t = min(valid) if valid else None
            if t_bitonic is not None and t_selection is not None:
                winner = "bitonic" if t_bitonic <= t_selection else "selection"
            elif t_bitonic is not None:
                winner = "bitonic"
            elif t_selection is not None:
                winner = "selection"
            else:
                winner = "N/A"

            rows.append([
                name, f"E={E} K={K}",
                fmt_ms(t_radix),
                fmt_ms(t_bitonic) if E <= 1024 else "N/A",
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
# Q4: Total speedup matrix  --  BL-1 vs Ours-best
# Rows: num_tokens (decode -> prefill sweep)
# Cols: model config
# Shows how total speedup scales with token count.
# ---------------------------------------------------------------------------
def experiment_q4(token_sweep):
    print()
    print("=" * 95)
    print("Q4  |  Total Speedup Matrix  --  BL-1 (Conv A, PyTorch) vs Ours-best (Conv B, TileLang)")
    print("  Rows: num_tokens (decode -> prefill)  |  Cols: model config")
    print("=" * 95)

    detail_rows  = []
    speedup_rows = []

    for num_tokens in token_sweep:
        speedup_row = [str(num_tokens)]
        for name, E, K, _ in PERF_CONFIGS:
            logits = make_logits(num_tokens, E)
            t_bl1  = bench(lambda x: bl1_conv_a_pytorch(x, K), logits)
            t_ours = bench(lambda x: ours_best(x, K),           logits)
            speedup_row.append(fmt_speedup(t_bl1, t_ours))
            detail_rows.append([
                num_tokens, name, f"E={E} K={K}",
                fmt_ms(t_bl1), fmt_ms(t_ours),
                fmt_speedup(t_bl1, t_ours),
            ])
        speedup_rows.append(speedup_row)

    # Compact speedup matrix
    print()
    print("  Speedup matrix (BL-1 / Ours-best):")
    col_headers = ["num_tokens"] + [f"{name}\nE={E},K={K}" for name, E, K, _ in PERF_CONFIGS]
    print(tabulate(speedup_rows, headers=col_headers, tablefmt="grid"))

    # Full latency detail
    print()
    print("  Full latency detail:")
    print(tabulate(detail_rows,
        headers=["num_tokens", "Model", "Config", "BL-1(ms)", "Ours-best(ms)", "Speedup"],
        tablefmt="grid"))


def main():
    print("TopK+Softmax Experiment Benchmark")
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"token_sweep: {TOKEN_SWEEP}")
    print(f"  256   = small decode batch (latency-bound)")
    print(f"  1024  = medium decode batch")
    print(f"  4096  = small prefill / training")
    print(f"  16384 = large prefill (bandwidth-bound)")

    experiment_1(TOKEN_SWEEP)
    experiment_2(TOKEN_SWEEP)
    experiment_q4(TOKEN_SWEEP)

    print()
    print("Correctness verification: python tests/test_topk_softmax_conventions.py")


if __name__ == "__main__":
    main()
