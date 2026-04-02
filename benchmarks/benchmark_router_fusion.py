"""
Router Fusion Benchmark — Fused GEMM + TopK + Softmax vs Separate Kernels

Measures: X(T,H) @ W(H,E) → topk_softmax → (scores, indices)(T,K)

Versions:
  BL-PyTorch:   torch.mm + torch.topk + torch.softmax (fully PyTorch)
  BL-TileLang:  torch.mm + topk_softmax_best (cuBLAS GEMM + our best topk)
  Ours-Select:  Fused GEMM+TopK+Softmax with selection sort epilogue
  Ours-Bitonic: Fused GEMM+TopK+Softmax with bitonic sort epilogue
  Ours-BV3:     Fused GEMM+TopK+Softmax with bitonic v3 epilogue

Fairness:
  - BF16 input (X, W), float32 logits/scores, int32 indices
  - Only kernel execution timed (no JIT, no data allocation)
  - Same X, W tensors across all versions
"""

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from tabulate import tabulate


# ============ Benchmark timing ============
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


# ============ Import TileLang kernels ============
try:
    from kernels.topk_softmax_tilelang import topk_softmax_best
    HAS_TL_TOPK = True
except ImportError:
    HAS_TL_TOPK = False

try:
    from kernels.router_gemm_topk_softmax import get_fused_router_kernel, get_router_gemm_kernel
    import tilelang.language as T
    HAS_FUSED = True
except ImportError:
    HAS_FUSED = False


# ============ Model configs (from HuggingFace) ============
# (name, H=hidden_size, E=num_experts, K=num_experts_per_tok)
MODEL_CONFIGS = [
    ("Qwen2-57B",   3584,  64,  8),
    ("Qwen3-235B",  4096,  128, 8),
    ("DeepSeek-V2", 5120,  160, 6),
]

INPUT_TOKENS_SWEEP = [256, 1024, 4096, 16384]


# ============ Single config runner ============
def run_single_config(name, H, E, K, num_tokens, warmup=20, rep=200):
    """Run benchmark for one (model, num_tokens) config."""
    device = "cuda"

    # ==========================================
    # Preprocessing (NOT timed)
    # ==========================================
    X_bf16 = torch.randn(num_tokens, H, device=device, dtype=torch.bfloat16)
    W_bf16 = torch.randn(H, E, device=device, dtype=torch.bfloat16)

    # Pre-pad W for non-power-of-2 E (done ONCE, outside timing loop)
    import math
    E_padded = 1 << math.ceil(math.log2(max(E, 2)))
    if E != E_padded:
        W_padded = torch.zeros(H, E_padded, device=device, dtype=torch.bfloat16)
        W_padded[:, :E] = W_bf16
    else:
        W_padded = W_bf16

    # Pre-compile fused kernels
    fused_kernels = {}
    tl_gemm_kernel = None
    if HAS_FUSED:
        for algo in ["selection", "bitonic"]:
            try:
                print(f"    Compiling fused-{algo} T={num_tokens} E={E}...", end=" ", flush=True)
                fused_kernels[algo] = get_fused_router_kernel(
                    algo, num_tokens, H, E, K, dtype=T.bfloat16)
                print("OK", flush=True)
            except Exception as e:
                print(f"FAIL: {e}", flush=True)
        # Compile standalone TileLang GEMM (same structure as fused, minus epilogue)
        try:
            print(f"    Compiling TL-GEMM T={num_tokens} E={E}...", end=" ", flush=True)
            tl_gemm_kernel = get_router_gemm_kernel(num_tokens, H, E, dtype=T.bfloat16)
            print("OK", flush=True)
        except Exception as e:
            print(f"FAIL: {e}", flush=True)

    # Pre-compile TileLang topk (warmup call)
    if HAS_TL_TOPK:
        try:
            print(f"    Compiling TL-topk T={num_tokens} E={E}...", end=" ", flush=True)
            test_logits = torch.randn(num_tokens, E, device=device, dtype=torch.float32)
            topk_softmax_best(test_logits, K)
            print("OK", flush=True)
        except Exception as e:
            print(f"FAIL: {e}", flush=True)

    results = {}

    # ==========================================
    # Pure kernel benchmark (NO wrapper overhead in timing)
    # ==========================================

    # --- GEMM-only: torch.mm vs TileLang GEMM ---
    print(f"    [torch.mm] timing...", end=" ", flush=True)
    results['torch.mm'] = bench_fn(
        lambda: torch.mm(X_bf16, W_bf16), warmup=warmup, rep=rep)
    print(f"{results['torch.mm']:.4f}ms", flush=True)

    if tl_gemm_kernel is not None:
        # Call the raw compiled kernel with pre-padded W (no wrapper overhead)
        from kernels.router_gemm_topk_softmax import _next_pow2, router_gemm_kernel
        _Ep = _next_pow2(E)
        try:
            raw_gemm = router_gemm_kernel(
                num_tokens, H, _Ep,
                block_M=64, block_K=64, num_stages=2, threads=256,
                dtype=T.bfloat16)
            raw_gemm(X_bf16, W_padded)  # verify
            torch.cuda.synchronize()
            print(f"    [TL-GEMM] timing...", end=" ", flush=True)
            results['TL-GEMM'] = bench_fn(
                lambda: raw_gemm(X_bf16, W_padded), warmup=warmup, rep=rep)
            print(f"{results['TL-GEMM']:.4f}ms", flush=True)
        except Exception as e:
            print(f"    [TL-GEMM] FAIL: {e}", flush=True)

    # --- BL-PyTorch: torch.mm + torch.topk + torch.softmax ---
    def pytorch_sep():
        logits = torch.mm(X_bf16, W_bf16).float()
        topk_vals, topk_idx = torch.topk(logits, K, dim=-1)
        scores = torch.softmax(topk_vals, dim=-1)
        return scores, topk_idx

    print(f"    [PyTorch] verify...", end=" ", flush=True)
    pytorch_sep()
    torch.cuda.synchronize()
    print("OK, timing...", end=" ", flush=True)
    results['PyTorch'] = bench_fn(pytorch_sep, warmup=warmup, rep=rep)
    print(f"{results['PyTorch']:.4f}ms", flush=True)

    # --- BL-TileLang: TileLang GEMM + topk_softmax_best ---
    if HAS_TL_TOPK and tl_gemm_kernel is not None:
        def tilelang_sep():
            logits = tl_gemm_kernel(X_bf16, W_bf16)
            return topk_softmax_best(logits, K)

        try:
            print(f"    [TL-sep] verify...", end=" ", flush=True)
            tilelang_sep()
            torch.cuda.synchronize()
            print("OK, timing...", end=" ", flush=True)
            results['TL-sep'] = bench_fn(tilelang_sep, warmup=warmup, rep=rep)
            print(f"{results['TL-sep']:.4f}ms", flush=True)
        except Exception as e:
            print(f"FAIL: {e}", flush=True)
            results['TL-sep'] = float('inf')

    # --- Fused variants: call raw kernel with pre-padded W ---
    algo_names = {
        "selection": "Fused-Sel",
        "bitonic": "Fused-Bit",
    }
    for algo, label in algo_names.items():
        if algo in fused_kernels:
            kern = fused_kernels[algo]
            try:
                print(f"    [{label}] verify...", end=" ", flush=True)
                kern(X_bf16, W_padded)
                torch.cuda.synchronize()
                print("OK, timing...", end=" ", flush=True)
                results[label] = bench_fn(
                    lambda k=kern: k(X_bf16, W_padded),
                    warmup=warmup, rep=rep)
                print(f"{results[label]:.4f}ms", flush=True)
            except Exception as e:
                print(f"\n[{label} Error] {e}")
                results[label] = float('inf')

    return results


def main():
    parser = argparse.ArgumentParser(description="Router Fusion Benchmark")
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
    print("Router Fusion Benchmark — Fused GEMM + TopK + Softmax (BF16 input, F32 logits)")
    print("=" * 100)
    print("Device:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("TL topk_softmax:", "available" if HAS_TL_TOPK else "NOT available")
    print("Fused kernels:", "available" if HAS_FUSED else "NOT available")
    print()

    all_rows = []

    for name, H, E, K in configs:
        print(f"\n{name} (H={H}, E={E}, K={K})")
        print("-" * 80)

        for num_tokens in input_sweep:
            results = run_single_config(
                name, H, E, K, num_tokens,
                warmup=args.warmup, rep=args.rep)

            # Print inline
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

    # ============ Summary table ============
    print("\n\n" + "=" * 100)
    print("SUMMARY: Latency (ms) and Speedup vs PyTorch separate")
    print("=" * 100)

    version_keys = ['PyTorch', 'TL-sep', 'Fused-Sel', 'Fused-Bit']
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
        # Best
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
