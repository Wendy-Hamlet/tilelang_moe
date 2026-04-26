"""
Router Fusion Benchmark — Fused GEMM + TopK + Softmax vs Separate Kernels

Measures: X(T,H) @ W(H,E) -> topk_softmax -> (scores, indices)(T,K)

All versions measured with do_bench (per-iteration CUDA events + L2 flush).
Data preparation (padding, allocation) is done ONCE outside the timing loop.

Kernel variants (3 GEMM strategies × 2 top-K algorithms = 6 kernels):
  cpasync_selection / cpasync_bitonic   cp.async GEMM, threads=256, block_M=64
  tma_selection     / tma_bitonic       Manual 1:2 WS (1 prod + 2 M-split cons),
                                        threads=384, block_M=128            (Hopper+)
  tma_auto_selection/ tma_auto_bitonic  TMA + auto-WS, threads=256, block_M=64,
                                        N-split via default Square policy   (Hopper+)
"""

import sys
import math
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


# ============ Model configs (from HuggingFace) ============
MODEL_CONFIGS = [
    ("Qwen2-57B",   3584,  64,  8),
    ("Qwen3-235B",  4096,  128, 8),
    ("DeepSeek-V2", 5120,  160, 6),
]

INPUT_TOKENS_SWEEP = [256, 1024, 4096, 16384]


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

    def pytorch_sep():
        logits = torch.mm(X_bf16, W_bf16).float()
        topk_vals, topk_idx = torch.topk(logits, K, dim=-1)
        scores = torch.softmax(topk_vals, dim=-1)
        return scores, topk_idx

    print(f"    [PyTorch] timing...", end=" ", flush=True)
    results['PyTorch'] = do_bench(pytorch_sep, warmup=warmup, rep=rep)
    print(f"{results['PyTorch']:.4f}ms", flush=True)

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

    algo_display = {
        "cpasync_selection":  "CpAsync-Sel",
        "cpasync_bitonic":    "CpAsync-Bit",
        "tma_selection":      "TMA-Sel",
        "tma_bitonic":        "TMA-Bit",
        "tma_auto_selection": "TMA-Auto-Sel",
        "tma_auto_bitonic":   "TMA-Auto-Bit",
    }
    for algo, label in algo_display.items():
        if algo in raw_fused:
            kern = raw_fused[algo]
            try:
                kern(X_bf16, W_padded)
                torch.cuda.synchronize()
                print(f"    [{label}] timing...", end=" ", flush=True)
                results[label] = do_bench(
                    lambda k=kern: k(X_bf16, W_padded),
                    warmup=warmup, rep=rep)
                print(f"{results[label]:.4f}ms", flush=True)
            except Exception as e:
                print(f"    [{label}] FAIL: {e}", flush=True)
                results[label] = float('inf')

    return results


def main():
    parser = argparse.ArgumentParser(description="Router Fusion Benchmark")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--input-tokens", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=25, help="Warmup time budget in ms")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark time budget in ms")
    parser.add_argument("--algos", type=str, default=None,
                        help="Comma-separated algos to run")
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
    print("Router Fusion Benchmark — Fused GEMM + TopK + Softmax (BF16, do_bench uniform)")
    print("=" * 100)
    print("Device:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("TL topk_softmax:", "available" if HAS_TL_TOPK else "NOT available")
    print("Fused kernels:", "available" if HAS_FUSED else "NOT available")
    print("Timing: do_bench (per-iteration events + L2 flush) for ALL versions")
    print()

    all_rows = []

    for name, H, E, K in configs:
        print(f"\n{name} (H={H}, E={E}, K={K})")
        print("-" * 80)

        for num_tokens in input_sweep:
            results = run_single_config(
                name, H, E, K, num_tokens,
                warmup=args.warmup, rep=args.rep)

            parts = []
            for v, lat in results.items():
                if lat == float('inf'):
                    parts.append(f"{v}=FAIL")
                else:
                    parts.append(f"{v}={lat:.4f}ms")
            print(f"  T={num_tokens:<6} {', '.join(parts)}")

            all_rows.append({
                'model': name, 'H': H, 'E': E, 'K': K,
                'T': num_tokens, **results
            })

    print("\n\n" + "=" * 100)
    print("SUMMARY: Latency (ms) and Speedup vs PyTorch separate")
    print("=" * 100)

    version_keys = ['PyTorch', 'TL-sep',
                    'CpAsync-Sel', 'CpAsync-Bit', 'TMA-Sel', 'TMA-Bit',
                    'TMA-Auto-Sel', 'TMA-Auto-Bit']
    version_keys = [v for v in version_keys if any(v in r for r in all_rows)]

    headers = ["Model", "T"] + \
              [f"{v}(ms)" for v in version_keys] + \
              [f"{v}/Py" for v in version_keys if v != 'PyTorch'] + \
              ["Best"]

    table = []
    for r in all_rows:
        py_lat = r.get('PyTorch', float('inf'))
        row = [r['model'], r['T']]
        for v in version_keys:
            lat = r.get(v, float('inf'))
            row.append(f"{lat:.4f}" if lat != float('inf') else "FAIL")
        for v in version_keys:
            if v == 'PyTorch':
                continue
            lat = r.get(v, float('inf'))
            if lat == float('inf') or py_lat == float('inf'):
                row.append("FAIL")
            else:
                row.append(f"{py_lat / lat:.2f}x")
        best_name, best_lat = "N/A", float('inf')
        for v in version_keys:
            lat = r.get(v, float('inf'))
            if lat < best_lat:
                best_lat = lat
                best_name = v
        row.append(best_name)
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="pipe"))


if __name__ == "__main__":
    main()
