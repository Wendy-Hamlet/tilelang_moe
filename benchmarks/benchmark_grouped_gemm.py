"""
Grouped GEMM Benchmark v2 — Real MoE Prefill Scenarios

Key changes from v1:
- Variable: input_tokens (not T), routed_tokens = input_tokens * top_k
- Prefill only: input_tokens = [1024, 2048, 4096, 8192, 16384]
- For E>32 models, simulate single-GPU with E_local = 8, 16, 32 (expert parallel)
- tokens_per_expert = input_tokens * top_k / E (same regardless of parallelism)
- routed_tokens_local = E_local * tokens_per_expert

Implementations compared:
- PyTorch: for-loop torch.mm (cuBLAS per expert)
- torch._grouped_mm: CUTLASS grouped GEMM (single kernel)
- TL-v1: TileLang official grouped GEMM
- TL-v2: TileLang optimized (padded output + TMA store + O(1) expert lookup)
"""

import sys
import os
import argparse
import warnings
import math
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from typing import Dict, List, Tuple
from tabulate import tabulate

torch.set_float32_matmul_precision('high')

# ============ Import implementations ============
from kernels.grouped_gemm_tilelang import (
    grouped_gemm_pytorch,
    grouped_gemm_tilelang,
)
try:
    from kernels.grouped_gemm_tilelang import grouped_gemm_tilelang_v2
    HAS_V2 = True
except ImportError:
    HAS_V2 = False

HAS_TORCH_GROUPED_MM = hasattr(torch, "_grouped_mm")


# ============ Benchmark infra ============
def bench_fn(fn, warmup=50, rep=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


# ============ Real MoE model configs ============
# (name, H, I, E_total, top_k)
MODEL_CONFIGS = [
    ("OLMoE-1B",      2048,  1024,  64, 8),
    ("Mixtral-8x7B",  4096, 14336,   8, 2),
    ("Qwen3-235B",    4096,  3072, 128, 8),
    ("DeepSeek-V2",   5120,  1536, 160, 6),
    ("DBRX",          6144, 10752,  16, 4),
    ("DeepSeek-V3",   7168,  2048, 256, 8),
]

INPUT_TOKENS_SWEEP = [1024, 2048, 4096, 8192, 16384]

# For models with E>32, simulate single-GPU expert parallel
E_LOCAL_OPTIONS = [8, 16, 32]


def get_e_local_list(E_total):
    """Get E_local values to test for a model."""
    if E_total <= 32:
        return [E_total]  # fits on one GPU
    return [e for e in E_LOCAL_OPTIONS if e <= E_total]


def run_single_config(name, H, I, E_local, top_k, E_total, input_tokens,
                      warmup=20, rep=200):
    """Run benchmark for one (model, E_local, input_tokens) config.

    Returns dict of {version_name: latency_ms}.
    """
    device = "cuda"
    tokens_per_expert = (input_tokens * top_k) // E_total
    if tokens_per_expert == 0:
        tokens_per_expert = 1
    routed_tokens_local = E_local * tokens_per_expert

    # Uniform distribution on local experts
    expert_indices = torch.arange(routed_tokens_local, device=device) % E_local

    # BF16 data (for fair grouped_mm comparison)
    X_bf16 = torch.randn(routed_tokens_local, H, device=device, dtype=torch.bfloat16)
    W_bf16_HI = torch.randn(E_local, H, I, device=device, dtype=torch.bfloat16)
    W_bf16_IH = W_bf16_HI.transpose(1, 2).contiguous()

    # FP16 data (for TileLang)
    X_fp16 = X_bf16.to(torch.float16)
    W_fp16_IH = W_bf16_IH.to(torch.float16)  # (E, I, H) for TileLang wrapper

    sorted_indices = expert_indices.argsort()
    X_sorted_bf16 = X_bf16[sorted_indices]
    X_sorted_fp16 = X_fp16[sorted_indices]

    counts = torch.bincount(expert_indices, minlength=E_local)
    offs = torch.cumsum(counts, dim=0).to(torch.int32)
    batch_sizes = counts.tolist()

    Y_bf16 = torch.empty(routed_tokens_local, I, device=device, dtype=torch.bfloat16)

    results = {}

    # --- PyTorch for-loop (bf16, cuBLAS) ---
    def loop_fn():
        start = 0
        for eid in range(E_local):
            sz = batch_sizes[eid]
            if sz == 0:
                continue
            torch.mm(X_sorted_bf16[start:start+sz], W_bf16_IH[eid].T,
                     out=Y_bf16[start:start+sz])
            start += sz

    results['PyTorch'] = bench_fn(loop_fn, warmup=warmup, rep=rep)

    # --- torch._grouped_mm (CUTLASS) ---
    if HAS_TORCH_GROUPED_MM:
        try:
            torch._grouped_mm(X_sorted_bf16, W_bf16_HI, offs)  # verify
            results['grouped_mm'] = bench_fn(
                lambda: torch._grouped_mm(X_sorted_bf16, W_bf16_HI, offs),
                warmup=warmup, rep=rep)
        except Exception as e:
            results['grouped_mm'] = float('inf')

    # --- TileLang v1 (fp16) ---
    try:
        grouped_gemm_tilelang(X_fp16, W_fp16_IH, expert_indices)
        results['TL-v1'] = bench_fn(
            lambda: grouped_gemm_tilelang(X_fp16, W_fp16_IH, expert_indices),
            warmup=warmup, rep=rep)
    except Exception:
        results['TL-v1'] = float('inf')

    # --- TileLang v2 (fp16) ---
    if HAS_V2:
        try:
            grouped_gemm_tilelang_v2(X_fp16, W_fp16_IH, expert_indices)
            results['TL-v2'] = bench_fn(
                lambda: grouped_gemm_tilelang_v2(X_fp16, W_fp16_IH, expert_indices),
                warmup=warmup, rep=rep)
        except Exception:
            results['TL-v2'] = float('inf')

    return results, tokens_per_expert, routed_tokens_local


def main():
    parser = argparse.ArgumentParser(description="Grouped GEMM Benchmark v2")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--input-tokens", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()

    if args.models:
        model_filter = set(args.models.split(","))
        configs = [c for c in MODEL_CONFIGS if c[0] in model_filter]
    else:
        configs = MODEL_CONFIGS

    if args.input_tokens:
        input_sweep = [int(t) for t in args.input_tokens.split(",")]
    else:
        input_sweep = INPUT_TOKENS_SWEEP

    print("=" * 100)
    print("Grouped GEMM Benchmark v2 — Real MoE Prefill Scenarios")
    print("=" * 100)
    print("Device:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("torch._grouped_mm:", "available" if HAS_TORCH_GROUPED_MM else "NOT available")
    print("TL-v2:", "available" if HAS_V2 else "NOT available")
    print()

    all_rows = []

    for name, H, I, E_total, top_k in configs:
        e_locals = get_e_local_list(E_total)

        for E_local in e_locals:
            print("\n{} (H={}, I={}, E_total={}, top_k={}, E_local={})".format(
                name, H, I, E_total, top_k, E_local))
            print("-" * 80)

            for input_tokens in input_sweep:
                results, tpe, rt_local = run_single_config(
                    name, H, I, E_local, top_k, E_total, input_tokens,
                    warmup=args.warmup, rep=args.rep)

                # Print inline
                parts = []
                for v, lat in results.items():
                    if lat == float('inf'):
                        parts.append("{}=FAIL".format(v))
                    else:
                        parts.append("{}={:.4f}ms".format(v, lat))
                print("  input_tokens={:<6} tpe={:<4} rt_local={:<6} {}".format(
                    input_tokens, tpe, rt_local, "  ".join(parts)))

                all_rows.append({
                    'model': name, 'H': H, 'I': I,
                    'E_total': E_total, 'E_local': E_local,
                    'top_k': top_k, 'input_tokens': input_tokens,
                    'tpe': tpe, 'rt_local': rt_local,
                    **results
                })

    # ============ Summary table ============
    print("\n\n" + "=" * 100)
    print("SUMMARY: Latency (ms) and Speedup vs PyTorch for-loop")
    print("=" * 100)

    version_keys = ['PyTorch', 'grouped_mm', 'TL-v1', 'TL-v2']
    version_keys = [v for v in version_keys
                    if any(v in r for r in all_rows)]

    headers = ["Model", "E_loc", "inp_tok", "tpe", "rt_loc"] + \
              [v + "(ms)" for v in version_keys] + \
              [v + "/Py" for v in version_keys if v != 'PyTorch']

    table = []
    for r in all_rows:
        py_lat = r.get('PyTorch', float('inf'))
        row = [r['model'], r['E_local'], r['input_tokens'],
               r['tpe'], r['rt_local']]
        for v in version_keys:
            lat = r.get(v, float('inf'))
            row.append("{:.4f}".format(lat) if lat != float('inf') else "FAIL")
        for v in version_keys:
            if v == 'PyTorch':
                continue
            lat = r.get(v, float('inf'))
            if lat == float('inf') or py_lat == float('inf'):
                row.append("FAIL")
            else:
                sp = py_lat / lat
                row.append("{:.2f}x".format(sp))
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="pipe"))


if __name__ == "__main__":
    main()
