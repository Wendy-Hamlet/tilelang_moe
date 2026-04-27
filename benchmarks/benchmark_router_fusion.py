"""
Router Fusion Benchmark — Fused GEMM + TopK + Softmax vs Baselines

Measures: X(T,H) @ W(H,E) -> topk_softmax -> (scores, indices)(T,K)

All versions measured with do_bench (per-iteration CUDA events + L2 flush).
First-time compilation/JIT cost is excluded by an explicit dry-run pass before
do_bench enters its timing loop.

Five comparison points reported in the summary:
  PyTorch    eager torch.mm + torch.topk + torch.softmax
  Inductor   torch.compile(pytorch_sep) (first compile NOT timed)
  Triton     official triton_kernels.matmul + triton_kernels.topk(apply_softmax=True)
  TL-sep     TileLang separate GEMM + topk_softmax_best
  Ours       best of six fused variants (cpasync/tma/tma_auto x sel/bit)
"""

import sys
import math
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from tabulate import tabulate
from tilelang.profiler import do_bench


# ============ Import TileLang kernels ============
try:
    from kernels.topk_softmax_tilelang import topk_softmax_best
    HAS_TL_TOPK = True
except ImportError:
    HAS_TL_TOPK = False

try:
    from kernels.router_gemm_topk_softmax import (
        _next_pow2, router_gemm_kernel,
        fused_router_cpasync_selection_kernel,
        fused_router_cpasync_bitonic_kernel,
        fused_router_tma_selection_kernel,
        fused_router_tma_bitonic_kernel,
        fused_router_tma_auto_selection_kernel,
        fused_router_tma_auto_bitonic_kernel,
    )
    import tilelang.language as T
    HAS_FUSED = True
except ImportError:
    HAS_FUSED = False


# ============ Import official triton_kernels (matmul + topk) ============
try:
    from triton_kernels.matmul import matmul as triton_matmul, PrecisionConfig as TritonPrecisionConfig
    from triton_kernels.topk import topk as triton_topk
    HAS_TRITON_KERNELS = True
except ImportError:
    HAS_TRITON_KERNELS = False


# ============ Model configs (from HuggingFace) ============
MODEL_CONFIGS = [
    ("Qwen2-57B",   3584,  64,  8),
    ("Qwen3-235B",  4096,  128, 8),
    ("DeepSeek-V2", 5120,  160, 6),
]

INPUT_TOKENS_SWEEP = [256, 1024, 4096, 16384]


FUSED_VARIANTS = [
    "cpasync_selection",
    "cpasync_bitonic",
    "tma_selection",
    "tma_bitonic",
    "tma_auto_selection",
    "tma_auto_bitonic",
]


def _make_colmajor(t):
    """Force column-major layout (required by triton_kernels.matmul)."""
    return t.transpose(-1, -2).contiguous().transpose(-1, -2)


def run_single_config(name, H, E, K, num_tokens, warmup=25, rep=100):
    """Run benchmark for one (model, num_tokens) config."""
    device = "cuda"

    X_bf16 = torch.randn(num_tokens, H, device=device, dtype=torch.bfloat16)
    W_bf16 = torch.randn(H, E, device=device, dtype=torch.bfloat16)

    E_padded = _next_pow2(E)
    if E != E_padded:
        W_padded = torch.zeros(H, E_padded, device=device, dtype=torch.bfloat16)
        W_padded[:, :E] = W_bf16
    else:
        W_padded = W_bf16

    # Compile fused kernels
    raw_fused = {}
    raw_gemm = None
    if HAS_FUSED:
        factory_map = {
            "cpasync_selection":  (fused_router_cpasync_selection_kernel,  256,  64),
            "cpasync_bitonic":    (fused_router_cpasync_bitonic_kernel,    256,  64),
            "tma_selection":      (fused_router_tma_selection_kernel,      384, 128),
            "tma_bitonic":        (fused_router_tma_bitonic_kernel,        384, 128),
            "tma_auto_selection": (fused_router_tma_auto_selection_kernel, 256,  64),
            "tma_auto_bitonic":   (fused_router_tma_auto_bitonic_kernel,   256,  64),
        }
        for algo, (factory, th, bm) in factory_map.items():
            try:
                print(f"    Compiling {algo} T={num_tokens} E={E}...", end=" ", flush=True)
                raw_fused[algo] = factory(
                    num_tokens, H, E, E_padded, K,
                    block_M=bm, block_K=64, num_stages=2, threads=th,
                    dtype=T.bfloat16)
                print("OK", flush=True)
            except Exception as e:
                print(f"FAIL: {e}", flush=True)

        try:
            print(f"    Compiling TL-GEMM T={num_tokens} E={E}...", end=" ", flush=True)
            raw_gemm = router_gemm_kernel(
                num_tokens, H, E_padded,
                block_M=64, block_K=64, num_stages=2, threads=256,
                dtype=T.bfloat16)
            print("OK", flush=True)
        except Exception as e:
            print(f"FAIL: {e}", flush=True)

    if HAS_TL_TOPK:
        try:
            print(f"    Compiling TL-topk T={num_tokens} E={E}...", end=" ", flush=True)
            test_logits = torch.randn(num_tokens, E, device=device, dtype=torch.float32)
            topk_softmax_best(test_logits, K)
            print("OK", flush=True)
        except Exception as e:
            print(f"FAIL: {e}", flush=True)

    results = {}
    raw_variants = {}

    # ---------- 1) PyTorch eager ----------
    def pytorch_sep():
        logits = torch.mm(X_bf16, W_bf16).float()
        topk_vals, topk_idx = torch.topk(logits, K, dim=-1)
        scores = torch.softmax(topk_vals, dim=-1)
        return scores, topk_idx

    print(f"    [PyTorch] timing...", end=" ", flush=True)
    results['PyTorch'] = do_bench(pytorch_sep, warmup=warmup, rep=rep)
    print(f"{results['PyTorch']:.4f}ms", flush=True)

    # ---------- 2) torch.compile / Inductor ----------
    try:
        compiled_fn = torch.compile(pytorch_sep, mode="reduce-overhead", fullgraph=False)
        # warm the JIT cache once OUTSIDE do_bench so first-compile time is not counted
        print(f"    [Inductor] warming compile cache...", end=" ", flush=True)
        for _ in range(3):
            compiled_fn()
        torch.cuda.synchronize()
        print("OK", flush=True)
        print(f"    [Inductor] timing...", end=" ", flush=True)
        results['Inductor'] = do_bench(compiled_fn, warmup=warmup, rep=rep)
        print(f"{results['Inductor']:.4f}ms", flush=True)
    except Exception as e:
        print(f"    [Inductor] FAIL: {e}", flush=True)
        results['Inductor'] = float('inf')

    # ---------- 3) Triton (triton_kernels matmul + topk) ----------
    if HAS_TRITON_KERNELS:
        try:
            W_triton = _make_colmajor(W_bf16)
            pc = TritonPrecisionConfig()

            def triton_glue():
                l = triton_matmul(X_bf16, W_triton, None, precision_config=pc)
                return triton_topk(l, K, apply_softmax=True,
                                   all_gather=False, symm_mem_pool=None)

            triton_glue()
            torch.cuda.synchronize()
            print(f"    [Triton] timing...", end=" ", flush=True)
            results['Triton'] = do_bench(triton_glue, warmup=warmup, rep=rep)
            print(f"{results['Triton']:.4f}ms", flush=True)
        except Exception as e:
            print(f"    [Triton] FAIL: {e}", flush=True)
            results['Triton'] = float('inf')
    else:
        results['Triton'] = float('inf')

    # ---------- 4) TileLang separate ----------
    if HAS_TL_TOPK and raw_gemm is not None:
        def tilelang_sep():
            logits_p = raw_gemm(X_bf16, W_padded)
            return topk_softmax_best(logits_p[:, :E].contiguous(), K)

        try:
            tilelang_sep()
            torch.cuda.synchronize()
            print(f"    [TL-sep] timing...", end=" ", flush=True)
            results['TL-sep'] = do_bench(tilelang_sep, warmup=warmup, rep=rep)
            print(f"{results['TL-sep']:.4f}ms", flush=True)
        except Exception as e:
            print(f"    [TL-sep] FAIL: {e}", flush=True)
            results['TL-sep'] = float('inf')

    # ---------- 5) Our fused variants (all 6, summary uses min as "Ours") ----------
    for algo in FUSED_VARIANTS:
        if algo not in raw_fused:
            continue
        kern = raw_fused[algo]
        try:
            kern(X_bf16, W_padded)
            torch.cuda.synchronize()
            print(f"    [{algo}] timing...", end=" ", flush=True)
            raw_variants[algo] = do_bench(
                lambda k=kern: k(X_bf16, W_padded),
                warmup=warmup, rep=rep)
            print(f"{raw_variants[algo]:.4f}ms", flush=True)
        except Exception as e:
            print(f"    [{algo}] FAIL: {e}", flush=True)
            raw_variants[algo] = float('inf')

    # Pick best fused variant
    if raw_variants:
        best_algo = min(raw_variants, key=raw_variants.get)
        results['Ours'] = raw_variants[best_algo]
        results['_ours_variant'] = best_algo
    else:
        results['Ours'] = float('inf')
        results['_ours_variant'] = "N/A"

    # Stash all raw variants for JSON dump
    results['_variants'] = raw_variants

    return results


def main():
    parser = argparse.ArgumentParser(description="Router Fusion Benchmark")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--input-tokens", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=25, help="Warmup time budget in ms")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark time budget in ms")
    parser.add_argument("--json", type=str, default=None,
                        help="If set, dump structured results to this JSON path")
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
    print("Router Fusion Benchmark — PyTorch / Inductor / Triton / TL-sep / Ours")
    print("=" * 100)
    print("Device:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("TL topk_softmax:", "available" if HAS_TL_TOPK else "NOT available")
    print("Fused kernels:", "available" if HAS_FUSED else "NOT available")
    print("triton_kernels:", "available" if HAS_TRITON_KERNELS else "NOT available")
    print("Timing: do_bench (per-iteration events + L2 flush) for ALL versions")
    print("Inductor: first-compile time excluded (3 warm passes before do_bench)")
    print()

    all_rows = []
    summary_keys = ['PyTorch', 'Inductor', 'Triton', 'TL-sep', 'Ours']

    for name, H, E, K in configs:
        print(f"\n{name} (H={H}, E={E}, K={K})")
        print("-" * 80)

        for num_tokens in input_sweep:
            results = run_single_config(
                name, H, E, K, num_tokens,
                warmup=args.warmup, rep=args.rep)

            parts = []
            for v in summary_keys:
                lat = results.get(v, float('inf'))
                if lat == float('inf'):
                    parts.append(f"{v}=FAIL")
                else:
                    parts.append(f"{v}={lat:.4f}ms")
            ours_var = results.get('_ours_variant', 'N/A')
            print(f"  T={num_tokens:<6} {', '.join(parts)}  [ours={ours_var}]")

            row = {
                'model': name, 'H': H, 'E': E, 'K': K, 'T': num_tokens,
                'ours_variant': results.get('_ours_variant', 'N/A'),
            }
            for v in summary_keys:
                row[v] = results.get(v, float('inf'))
            row['variants'] = results.get('_variants', {})
            all_rows.append(row)

    # ============ Summary table ============
    print("\n\n" + "=" * 100)
    print("SUMMARY: Latency (ms) and Speedup vs PyTorch eager")
    print("=" * 100)

    headers = ["Model", "T"] + \
              [f"{v}(ms)" for v in summary_keys] + \
              [f"{v}/Py" for v in summary_keys if v != 'PyTorch'] + \
              ["Best", "Ours-Variant"]

    table = []
    for r in all_rows:
        py_lat = r.get('PyTorch', float('inf'))
        row = [r['model'], r['T']]
        for v in summary_keys:
            lat = r.get(v, float('inf'))
            row.append(f"{lat:.4f}" if lat != float('inf') else "FAIL")
        for v in summary_keys:
            if v == 'PyTorch':
                continue
            lat = r.get(v, float('inf'))
            if lat == float('inf') or py_lat == float('inf'):
                row.append("FAIL")
            else:
                row.append(f"{py_lat / lat:.2f}x")
        best_name, best_lat = "N/A", float('inf')
        for v in summary_keys:
            lat = r.get(v, float('inf'))
            if lat < best_lat:
                best_lat = lat
                best_name = v
        row.append(best_name)
        row.append(r.get('ours_variant', 'N/A'))
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="pipe"))

    # ============ Optional JSON dump ============
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = []
        for r in all_rows:
            row_clean = {}
            for k, v in r.items():
                if k == 'variants':
                    row_clean[k] = {
                        kk: (None if (isinstance(vv, float) and math.isinf(vv)) else vv)
                        for kk, vv in v.items()
                    }
                elif isinstance(v, float) and math.isinf(v):
                    row_clean[k] = None
                else:
                    row_clean[k] = v
            serialized.append(row_clean)
        payload = {
            "device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "warmup": args.warmup,
            "rep": args.rep,
            "summary_keys": summary_keys,
            "fused_variants": FUSED_VARIANTS,
            "rows": serialized,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[json] wrote {out_path}")


if __name__ == "__main__":
    main()
