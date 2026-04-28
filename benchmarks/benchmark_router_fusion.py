"""
Router Fusion Benchmark — Fused GEMM + TopK + Softmax vs Baselines

Measures: X(T,H) @ W(H,E) -> topk_softmax -> (scores, indices)(T,K)

All versions measured with do_bench (per-iteration CUDA events + L2 flush),
wrapped in stable_bench (outer median over N repeats; default 3) to defend
against autotune flakiness and GPU clock drift. min/max also reported per
config to expose run-to-run spread.

First-time compilation/JIT cost is excluded by an explicit dry-run pass before
do_bench enters its timing loop.

Five comparison points reported in the summary:
  PyTorch    eager torch.mm + torch.topk + torch.softmax
  Inductor   torch.compile(pytorch_sep) (first compile NOT timed)
  Triton     Triton-tutorial-03 matmul (inline, autotuned) + torch.topk + torch.softmax
             (vanilla `triton`/`triton.language`, NO `triton_kernels` package needed)
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


def stable_bench(fn, warmup, rep, n_repeats=3):
    """Run do_bench n_repeats times, return (median, min, max).

    do_bench already averages many in-loop iterations, but cross-call variance
    (autotune flakiness, GPU clock drift, L2/SM placement state) can still be
    large. Outer N-run median + reported spread defends against that.
    """
    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        times.append(do_bench(fn, warmup=warmup, rep=rep))
    times.sort()
    median = times[len(times) // 2]
    return median, times[0], times[-1]


def _fmt_bench(median, lo, hi):
    spread_pct = (hi - lo) / median * 100.0 if median > 0 else 0.0
    return f"{median:.4f}ms (min {lo:.4f} / max {hi:.4f}, spread {spread_pct:.1f}%)"


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


# ============ Inline Triton matmul (tutorial 03, autotuned) ============
# NOTE: deliberately uses only `triton` / `triton.language` so the benchmark
# runs on any environment with vanilla Triton — no `triton_kernels` package.
try:
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _triton_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.bfloat16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def triton_matmul(a, b):
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        _triton_matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
        )
        return c

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


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


def run_single_config(name, H, E, K, num_tokens, warmup=25, rep=100, n_repeats=3):
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
    med, lo, hi = stable_bench(pytorch_sep, warmup, rep, n_repeats)
    results['PyTorch'] = med
    results['_PyTorch_minmax'] = (lo, hi)
    print(_fmt_bench(med, lo, hi), flush=True)

    # ---------- 2) torch.compile / Inductor ----------
    # mode="reduce-overhead" -> Inductor + CUDA graphs. The first few calls are
    # special: pass 1 = Dynamo trace + eager run, pass 2 = Inductor compile
    # (Triton codegen, mostly CPU), pass 3 = CUDA-graph capture,
    # pass 4+ = steady-state graph replay.
    # We run 5 warm passes OUTSIDE do_bench and print the per-pass wall time
    # (CPU+GPU, via time.perf_counter + cuda.synchronize) so it's auditable
    # from the log that compile/capture cost is excluded — expected pattern:
    # first pass O(seconds), last 1-2 passes converge to ~ms (= steady state).
    import time
    try:
        compiled_fn = torch.compile(pytorch_sep, mode="reduce-overhead", fullgraph=False)
        print(f"    [Inductor] warmup (per-pass ms, wall): ", end="", flush=True)
        warm_times = []
        for i in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            compiled_fn()
            torch.cuda.synchronize()
            warm_times.append((time.perf_counter() - t0) * 1e3)
        print(", ".join(f"#{i+1}={t:.1f}" for i, t in enumerate(warm_times)), flush=True)
        print(f"    [Inductor] timing...", end=" ", flush=True)
        med, lo, hi = stable_bench(compiled_fn, warmup, rep, n_repeats)
        results['Inductor'] = med
        results['_Inductor_minmax'] = (lo, hi)
        print(_fmt_bench(med, lo, hi), flush=True)
    except Exception as e:
        print(f"    [Inductor] FAIL: {e}", flush=True)
        results['Inductor'] = float('inf')

    # ---------- 3) Triton (inline tutorial-03 matmul + torch topk/softmax) ----------
    if HAS_TRITON:
        try:
            def triton_glue():
                logits = triton_matmul(X_bf16, W_bf16).float()
                topk_vals, topk_idx = torch.topk(logits, K, dim=-1)
                scores = torch.softmax(topk_vals, dim=-1)
                return scores, topk_idx

            # Burn autotune passes outside do_bench so picking-the-best-config
            # is not part of the timed loop.
            for _ in range(3):
                triton_glue()
            torch.cuda.synchronize()
            print(f"    [Triton] timing...", end=" ", flush=True)
            med, lo, hi = stable_bench(triton_glue, warmup, rep, n_repeats)
            results['Triton'] = med
            results['_Triton_minmax'] = (lo, hi)
            print(_fmt_bench(med, lo, hi), flush=True)
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
            med, lo, hi = stable_bench(tilelang_sep, warmup, rep, n_repeats)
            results['TL-sep'] = med
            results['_TL-sep_minmax'] = (lo, hi)
            print(_fmt_bench(med, lo, hi), flush=True)
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
            med, lo, hi = stable_bench(
                lambda k=kern: k(X_bf16, W_padded),
                warmup, rep, n_repeats)
            raw_variants[algo] = med
            print(_fmt_bench(med, lo, hi), flush=True)
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
    parser.add_argument("--repeats", type=int, default=3,
                        help="Outer-loop repetitions of do_bench; result is median, "
                             "min/max also reported. Defends against autotune flakiness "
                             "and GPU clock drift.")
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
    print("Triton (vanilla):", "available" if HAS_TRITON else "NOT available")
    print("Timing: do_bench (per-iteration events + L2 flush) for ALL versions")
    print(f"        wrapped in stable_bench (median of {args.repeats} repeats, min/max also shown)")
    print("Inductor: first-compile time excluded (5 warm passes before do_bench)")
    print()

    all_rows = []
    summary_keys = ['PyTorch', 'Inductor', 'Triton', 'TL-sep', 'Ours']

    for name, H, E, K in configs:
        print(f"\n{name} (H={H}, E={E}, K={K})")
        print("-" * 80)

        for num_tokens in input_sweep:
            results = run_single_config(
                name, H, E, K, num_tokens,
                warmup=args.warmup, rep=args.rep,
                n_repeats=args.repeats)

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
            spreads = {}
            for v in summary_keys:
                mm = results.get(f'_{v}_minmax')
                if mm is not None:
                    spreads[v] = {'min': mm[0], 'max': mm[1]}
            row['spreads'] = spreads
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
            "repeats": args.repeats,
            "summary_keys": summary_keys,
            "fused_variants": FUSED_VARIANTS,
            "rows": serialized,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[json] wrote {out_path}")


if __name__ == "__main__":
    main()
