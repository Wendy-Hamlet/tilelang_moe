"""
TopK+Softmax Benchmark -- Sonic CuTe-DSL  (sonic environment)
==============================================================

Run in: conda activate sonic
Usage:  cd /cephfs/xhluo/chjin/palette/3rdparty/tilelang_moe
        python benchmarks/benchmark_topk_softmax_sonic.py

Benchmarks Sonic's CuTe-DSL TopK+Softmax kernel against PyTorch baselines.
Uses the same parameter grid as benchmark_topk_softmax.py (TileLang env)
so results can be compared side-by-side.

Sonic implementation: CuTe-DSL bitonic sort + bit-packing index encoding,
  single-kernel fused TopK+Softmax.  Convention B (topk-first).
  Constraints: E <= 4096, K <= 16, E % 8 == 0.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from tabulate import tabulate
from triton.testing import do_bench

from sonicmoe.functional.forward import _softmax_topk_fwd


torch.set_float32_matmul_precision("high")

# ---------------------------------------------------------------------------
# Configs: MUST match benchmark_topk_softmax.py exactly
# ---------------------------------------------------------------------------
PERF_CONFIGS = [
    ("Mixtral",     8,   2, "Mixtral-8x7B/8x22B, Grok-1"),
    ("DBRX",        16,  4, "databricks/dbrx"),
    ("OLMoE/Qwen2", 64,  8, "allenai/OLMoE, Qwen/Qwen2-57B-A14B"),
    ("Qwen3",       128, 8, "Qwen/Qwen3-30B-A3B, Qwen3-235B-A22B"),
    ("DeepSeek-V2", 160, 6, "deepseek-ai/DeepSeek-V2/V2.5"),
    ("DeepSeek-V3", 256, 8, "deepseek-ai/DeepSeek-V3 routing (softmax proxy)"),
]

TOKEN_SWEEP = [256, 1024, 4096, 16384]

WARMUP = 50
REP    = 500


# ---------------------------------------------------------------------------
# Baseline implementations (pure PyTorch, no tilelang dependency)
# ---------------------------------------------------------------------------
def bl1_conv_a_pytorch(logits, k):
    """BL-1: Conv A -- softmax(E) -> topk.  scores sum < 1."""
    scores = F.softmax(logits, dim=-1)
    return torch.topk(scores, k, dim=-1)


def bl4_conv_b_pytorch(logits, k):
    """BL-4: Conv B -- topk(logits) -> softmax(K).  scores sum = 1."""
    topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
    return F.softmax(topk_logits, dim=-1), topk_indices


def sonic_topk_softmax(logits, k):
    """Sonic CuTe-DSL: Conv B, single-kernel fused TopK+Softmax."""
    T, E = logits.shape
    scores  = torch.empty(T, k, dtype=torch.float32, device=logits.device)
    indices = torch.empty(T, k, dtype=torch.int32,    device=logits.device)
    _softmax_topk_fwd(logits, scores, indices, E, k)
    return scores, indices


# ---------------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------------
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
# Experiment: BL-1 (Conv A PyTorch) vs BL-4 (Conv B PyTorch) vs Sonic
# ---------------------------------------------------------------------------
def experiment_sonic(token_sweep):
    print()
    print("=" * 95)
    print("Sonic Benchmark  |  BL-1 (Conv A PyTorch) vs BL-4 (Conv B PyTorch) vs Sonic (CuTe-DSL)")
    print("  BL-1 : softmax(E) -> topk           (Conv A, PyTorch)")
    print("  BL-4 : topk(logits) -> softmax(K)   (Conv B, PyTorch unfused)")
    print("  Sonic: topk+softmax fused            (Conv B, CuTe-DSL bitonic)")
    print("=" * 95)

    for num_tokens in token_sweep:
        print(f"\n  -- num_tokens = {num_tokens} --")
        rows = []
        for name, E, K, _ in PERF_CONFIGS:
            logits  = make_logits(num_tokens, E)
            t_bl1   = bench(lambda x: bl1_conv_a_pytorch(x, K), logits)
            t_bl4   = bench(lambda x: bl4_conv_b_pytorch(x, K), logits)
            t_sonic = bench(lambda x: sonic_topk_softmax(x, K), logits)

            rows.append([
                name, f"E={E} K={K}",
                fmt_ms(t_bl1), fmt_ms(t_bl4), fmt_ms(t_sonic),
                fmt_speedup(t_bl1, t_bl4),
                fmt_speedup(t_bl4, t_sonic),
                fmt_speedup(t_bl1, t_sonic),
            ])

        print(tabulate(rows,
            headers=["Model", "Config",
                     "BL-1 A-Torch(ms)", "BL-4 B-Torch(ms)", "Sonic(ms)",
                     "BL1/BL4", "BL4/Sonic", "BL1/Sonic"],
            tablefmt="grid"))


# ---------------------------------------------------------------------------
# Speedup matrix  --  BL-1 vs Sonic
# ---------------------------------------------------------------------------
def speedup_matrix(token_sweep):
    print()
    print("=" * 95)
    print("Speedup Matrix  --  BL-1 (Conv A, PyTorch) vs Sonic (CuTe-DSL)")
    print("  Rows: num_tokens (decode -> prefill)  |  Cols: model config")
    print("=" * 95)

    detail_rows  = []
    speedup_rows = []

    for num_tokens in token_sweep:
        speedup_row = [str(num_tokens)]
        for name, E, K, _ in PERF_CONFIGS:
            logits  = make_logits(num_tokens, E)
            t_bl1   = bench(lambda x: bl1_conv_a_pytorch(x, K), logits)
            t_sonic = bench(lambda x: sonic_topk_softmax(x, K), logits)
            speedup_row.append(fmt_speedup(t_bl1, t_sonic))
            detail_rows.append([
                num_tokens, name, f"E={E} K={K}",
                fmt_ms(t_bl1), fmt_ms(t_sonic),
                fmt_speedup(t_bl1, t_sonic),
            ])
        speedup_rows.append(speedup_row)

    print()
    print("  Speedup matrix (BL-1 / Sonic):")
    col_headers = ["num_tokens"] + [f"{name}\nE={E},K={K}" for name, E, K, _ in PERF_CONFIGS]
    print(tabulate(speedup_rows, headers=col_headers, tablefmt="grid"))

    print()
    print("  Full latency detail:")
    print(tabulate(detail_rows,
        headers=["num_tokens", "Model", "Config", "BL-1(ms)", "Sonic(ms)", "Speedup"],
        tablefmt="grid"))


def main():
    print("TopK+Softmax Benchmark -- Sonic CuTe-DSL")
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"token_sweep: {TOKEN_SWEEP}")
    print(f"  256   = small decode batch (latency-bound)")
    print(f"  1024  = medium decode batch")
    print(f"  4096  = small prefill / training")
    print(f"  16384 = large prefill (bandwidth-bound)")

    experiment_sonic(TOKEN_SWEEP)
    speedup_matrix(TOKEN_SWEEP)

    print()
    print("Compare with TileLang results: python benchmarks/benchmark_topk_softmax.py  (tilelang env)")


if __name__ == "__main__":
    main()
