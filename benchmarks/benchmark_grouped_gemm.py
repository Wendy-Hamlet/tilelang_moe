"""
Grouped GEMM Performance Benchmark — Real MoE Model Configurations

使用 HuggingFace 上真实 MoE 模型的参数进行 benchmark。
Up projection: (tokens_per_expert, H) @ (H, I) → (tokens_per_expert, I)

模型参数来源:
- OLMoE-1B-7B:   H=2048, I=1024,  E=64,  K=8
- Mixtral-8x7B:  H=4096, I=14336, E=8,   K=2
- Qwen3-235B:    H=4096, I=3072,  E=128, K=8  (SwiGLU: I_gemm=2*1536)
- DeepSeek-V2:   H=5120, I=1536,  E=160, K=6
- DBRX:          H=6144, I=10752, E=16,  K=4
- DeepSeek-V3:   H=7168, I=2048,  E=256, K=8
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import math
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate

torch.set_float32_matmul_precision('high')


# ============ Import implementations ============
from kernels.grouped_gemm_tilelang import (
    grouped_gemm_pytorch,
    grouped_gemm_tilelang,
    grouped_gemm_best,
)

# Try importing v2 (will be added later)
try:
    from kernels.grouped_gemm_tilelang import grouped_gemm_tilelang_v2
    HAS_V2 = True
except ImportError:
    HAS_V2 = False

# Try importing v3 (will be added later)
try:
    from kernels.grouped_gemm_tilelang import grouped_gemm_tilelang_v3
    HAS_V3 = True
except ImportError:
    HAS_V3 = False

# Check torch._grouped_mm availability (PyTorch 2.6+, CUTLASS grouped GEMM)
HAS_TORCH_GROUPED_MM = hasattr(torch, "_grouped_mm")


# ============ torch._grouped_mm wrapper (CUTLASS grouped GEMM) ============

def grouped_gemm_torch_grouped_mm(
    X: torch.Tensor,
    W: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """
    torch._grouped_mm baseline (PyTorch 2.6+, CUTLASS Hopper grouped GEMM).
    Requires BFloat16. Single kernel launch for all experts.

    Args:
        X: (total_tokens, H) fp16 input
        W: (E, I, H) fp16 weights
        expert_indices: (total_tokens,) expert IDs
    Returns:
        Y: (total_tokens, I) fp16 output
    """
    total_tokens, H = X.shape
    E, I, _ = W.shape
    device = X.device

    sorted_indices = expert_indices.argsort()
    X_sorted = X[sorted_indices].to(torch.bfloat16)

    expert_counts = torch.bincount(expert_indices, minlength=E)
    offs = torch.cumsum(expert_counts, dim=0).to(torch.int32)

    # W: (E, I, H) -> need (E, H, I) for A @ B = (tokens, H) @ (H, I)
    W_transposed = W.transpose(1, 2).contiguous().to(torch.bfloat16)

    # torch._grouped_mm(A, B, offs) -> C
    Y_sorted_bf16 = torch._grouped_mm(X_sorted, W_transposed, offs)

    Y = torch.empty(total_tokens, I, device=device, dtype=X.dtype)
    Y[sorted_indices] = Y_sorted_bf16.to(X.dtype)
    return Y


# ============ Benchmark infra ============
try:
    from tilelang.profiler import do_bench

    def benchmark_function(func, *args, warmup=100, rep=1000):
        def wrapper():
            return func(*args)
        return do_bench(wrapper, warmup=warmup, rep=rep, return_mode='median')
except ImportError:
    def benchmark_function(func, *args, warmup=100, rep=1000):
        for _ in range(warmup):
            func(*args)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            func(*args)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / rep


# ============ Real MoE model configs ============
# (name, H, I, E, top_k, description)
MODEL_CONFIGS = [
    ("OLMoE-1B",      2048,  1024,  64, 8, "allenai/OLMoE-1B-7B"),
    ("Mixtral-8x7B",  4096, 14336,   8, 2, "mistralai/Mixtral-8x7B"),
    ("Qwen3-235B",    4096,  3072, 128, 8, "Qwen/Qwen3-235B (SwiGLU)"),
    ("DeepSeek-V2",   5120,  1536, 160, 6, "deepseek-ai/DeepSeek-V2"),
    ("DBRX",          6144, 10752,  16, 4, "databricks/dbrx-instruct"),
    ("DeepSeek-V3",   7168,  2048, 256, 8, "deepseek-ai/DeepSeek-V3"),
]

# total_tokens = input_tokens * top_k (routed tokens)
TOKEN_SWEEP = [256, 1024, 4096, 16384]


def generate_test_data(total_tokens: int, H: int, I: int, E: int,
                       device="cuda", seed=42, distribution="zipf"):
    """Generate test data with specified expert distribution."""
    torch.manual_seed(seed)

    X = torch.randn(total_tokens, H, device=device, dtype=torch.float16)
    W = torch.randn(E, I, H, device=device, dtype=torch.float16)

    if distribution == "uniform":
        # Each expert gets exactly total_tokens // E tokens
        expert_indices = torch.arange(total_tokens, device=device) % E
    else:
        # Zipf distribution for realistic expert load imbalance
        expert_probs = torch.tensor([1.0 / (i + 1) for i in range(E)],
                                    device=device)
        expert_probs = expert_probs / expert_probs.sum()
        expert_indices = torch.multinomial(expert_probs, total_tokens,
                                          replacement=True)

    return X, W, expert_indices


def get_versions() -> Dict[str, callable]:
    """Build version dict based on available implementations."""
    versions = {
        'PyTorch': grouped_gemm_pytorch,
    }
    if HAS_TORCH_GROUPED_MM:
        versions['torch.grouped_mm'] = grouped_gemm_torch_grouped_mm
    versions['TL-v1'] = grouped_gemm_tilelang
    if HAS_V2:
        versions['TL-v2'] = grouped_gemm_tilelang_v2
    if HAS_V3:
        versions['TL-v3'] = grouped_gemm_tilelang_v3
    return versions


def benchmark_single(H: int, I: int, E: int, total_tokens: int,
                     config_name: str, versions: Dict,
                     warmup=50, rep=500, verbose=True, distribution="zipf"):
    """Run benchmark for a single (model, token_count) config."""
    X, W, expert_indices = generate_test_data(total_tokens, H, I, E,
                                              distribution=distribution)

    # Correctness
    baseline = grouped_gemm_pytorch(X, W, expert_indices)
    correctness = {}
    for name, func in versions.items():
        try:
            out = func(X, W, expert_indices)
            max_diff = (out - baseline).abs().max().item()
            correctness[name] = max_diff
        except Exception as e:
            correctness[name] = str(e)

    # Performance
    latencies = {}
    for name, func in versions.items():
        try:
            _ = func(X, W, expert_indices)
            torch.cuda.synchronize()
            lat = benchmark_function(func, X, W, expert_indices,
                                     warmup=warmup, rep=rep)
            latencies[name] = lat
        except Exception as e:
            latencies[name] = float('inf')
            if verbose:
                print(f"  {config_name} T={total_tokens} {name}: FAILED ({e})")

    return latencies, correctness


def main():
    parser = argparse.ArgumentParser(description="Grouped GEMM Benchmark")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to benchmark (default: all)")
    parser.add_argument("--tokens", type=str, default=None,
                        help="Comma-separated token counts (default: 256,1024,4096,16384)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--dist", type=str, default="zipf",
                        choices=["zipf", "uniform"],
                        help="Expert token distribution (default: zipf)")
    args = parser.parse_args()

    # Filter models
    if args.models:
        model_filter = set(args.models.split(","))
        configs = [c for c in MODEL_CONFIGS if c[0] in model_filter]
    else:
        configs = MODEL_CONFIGS

    # Token sweep
    if args.tokens:
        token_sweep = [int(t) for t in args.tokens.split(",")]
    else:
        token_sweep = TOKEN_SWEEP

    versions = get_versions()
    version_names = list(versions.keys())

    print("=" * 80)
    print("Grouped GEMM Benchmark — Real MoE Model Configurations")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Versions: {', '.join(version_names)}")
    print(f"Token sweep: {token_sweep}")
    print(f"Distribution: {args.dist}")
    print()

    # ============ Run all benchmarks ============
    all_results = []  # (model_name, T, latencies, correctness)

    for model_name, H, I, E, K, desc in configs:
        print(f"\n{'='*80}")
        print(f"Model: {model_name}  ({desc})")
        print(f"  H={H}, I={I}, E={E}, top_k={K}")
        print(f"{'='*80}")

        for T in token_sweep:
            print(f"\n  T={T}:")
            latencies, correctness = benchmark_single(
                H, I, E, T, model_name, versions,
                warmup=args.warmup, rep=args.rep,
                distribution=args.dist)

            # Print inline
            parts = []
            for v in version_names:
                lat = latencies.get(v, float('inf'))
                corr = correctness.get(v, "N/A")
                if isinstance(corr, float):
                    corr_str = f"diff={corr:.4f}"
                else:
                    corr_str = f"ERR:{corr[:30]}"
                if lat == float('inf'):
                    parts.append(f"    {v}: FAILED ({corr_str})")
                else:
                    parts.append(f"    {v}: {lat:.4f} ms  ({corr_str})")
            print("\n".join(parts))

            all_results.append((model_name, H, I, E, T, latencies))

    # ============ Summary: speedup matrix ============
    print("\n\n" + "=" * 80)
    print("SUMMARY: Latency Matrix (ms)")
    print("=" * 80)

    headers = ["Model", "H", "I", "E", "T"] + version_names
    table = []
    for model_name, H, I, E, T, latencies in all_results:
        row = [model_name, H, I, E, T]
        for v in version_names:
            lat = latencies.get(v, float('inf'))
            row.append(f"{lat:.4f}" if lat != float('inf') else "FAIL")
        table.append(row)
    print(tabulate(table, headers=headers, tablefmt="pipe"))

    # ============ Summary: speedup vs PyTorch ============
    print("\n" + "=" * 80)
    print("SUMMARY: Speedup vs PyTorch")
    print("=" * 80)

    speedup_versions = [v for v in version_names if v != 'PyTorch']
    headers2 = ["Model", "T"] + [f"{v}/PyTorch" for v in speedup_versions]
    table2 = []
    for model_name, H, I, E, T, latencies in all_results:
        pytorch_lat = latencies.get('PyTorch', float('inf'))
        if pytorch_lat == float('inf'):
            continue
        row = [model_name, T]
        for v in speedup_versions:
            lat = latencies.get(v, float('inf'))
            if lat == float('inf'):
                row.append("FAIL")
            else:
                speedup = pytorch_lat / lat
                marker = ""
                if speedup >= 1.1:
                    marker = " +"
                elif speedup <= 0.9:
                    marker = " -"
                row.append(f"{speedup:.2f}x{marker}")
        table2.append(row)
    print(tabulate(table2, headers=headers2, tablefmt="pipe"))

    # ============ Best version per config ============
    print("\n" + "=" * 80)
    print("SUMMARY: Best Version per Config")
    print("=" * 80)

    headers3 = ["Model", "T", "Best", "Latency (ms)", "vs PyTorch"]
    table3 = []
    for model_name, H, I, E, T, latencies in all_results:
        best_name = min(latencies, key=latencies.get)
        best_lat = latencies[best_name]
        pytorch_lat = latencies.get('PyTorch', float('inf'))
        speedup = pytorch_lat / best_lat if best_lat > 0 else 0
        table3.append([model_name, T, best_name, f"{best_lat:.4f}",
                       f"{speedup:.2f}x"])
    print(tabulate(table3, headers=headers3, tablefmt="pipe"))


if __name__ == "__main__":
    main()
