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
    from kernels.router_gemm_topk_softmax import get_fused_router_kernel
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

    # Pre-allocate outputs for PyTorch baseline
    logits_buf = torch.empty(num_tokens, E, device=device, dtype=torch.float32)

    # Pre-compile fused kernels
    fused_kernels = {}
    if HAS_FUSED:
        for algo in ["selection", "bitonic", "bitonic_v3"]:
            try:
                fused_kernels[algo] = get_fused_router_kernel(
                    algo, num_tokens, H, E, K, dtype=T.bfloat16)
            except Exception as e:
                print(f"\n[Fused-{algo} compile error] {e}")

    # Pre-compile TileLang topk (warmup call)
    if HAS_TL_TOPK:
        try:
            test_logits = torch.randn(num_tokens, E, device=device, dtype=torch.float32)
            topk_softmax_best(test_logits, K)
        except Exception:
            pass

    results = {}

    # ==========================================
    # Pure kernel benchmark
    # ==========================================

    # --- BL-PyTorch: torch.mm + torch.topk + torch.softmax ---
    def pytorch_sep():
        logits = torch.mm(X_bf16, W_bf16).float()
        topk_vals, topk_idx = torch.topk(logits, K, dim=-1)
        scores = torch.softmax(topk_vals, dim=-1)
        return scores, topk_idx

    results['PyTorch'] = bench_fn(pytorch_sep, warmup=warmup, rep=rep)

    # --- BL-TileLang: torch.mm + topk_softmax_best ---
    if HAS_TL_TOPK:
        def tilelang_sep():
            logits = torch.mm(X_bf16, W_bf16).float()
            return topk_softmax_best(logits, K)

        try:
            tilelang_sep()  # verify
            results['TL-sep'] = bench_fn(tilelang_sep, warmup=warmup, rep=rep)
        except Exception as e:
            print(f"\n[TL-sep Error] {e}")
            results['TL-sep'] = float('inf')

    # --- Fused variants ---
    algo_names = {
        "selection": "Fused-Sel",
        "bitonic": "Fused-Bit",
        "bitonic_v3": "Fused-BV3",
    }
    for algo, label in algo_names.items():
        if algo in fused_kernels:
            kern = fused_kernels[algo]
            try:
                kern(X_bf16, W_bf16)  # verify
                results[label] = bench_fn(
                    lambda k=kern: k(X_bf16, W_bf16),
                    warmup=warmup, rep=rep)
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

    version_keys = ['PyTorch', 'TL-sep', 'Fused-Sel', 'Fused-Bit', 'Fused-BV3']
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
