"""
Grouped GEMM Benchmark v2 — Real MoE Prefill Scenarios (Pure Kernel Latency, BF16)

Key changes:
- Strip host-side preprocessing overhead (sorting, bincount, metadata allocation, JIT compilation)
- Measure only pure kernel launch and execution latency
- Use BFloat16 uniformly for fair comparison (PyTorch vs TileLang vs torch._grouped_mm)
- TileLang kernels use profiler.do_bench (adapter-level, no JIT wrapper overhead)
- PyTorch / grouped_mm / Triton use do_bench (function-level)
"""

import sys
import os
import argparse
import warnings
import math
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import tilelang.language as T
from typing import Dict, List, Tuple
from tabulate import tabulate
from tilelang.profiler import do_bench

torch.set_float32_matmul_precision('high')

# ============ Import implementations ============
from kernels.grouped_gemm_tilelang import (
    _grouped_gemm_kernel,  # low-level v1 kernel
)
try:
    from kernels.grouped_gemm_tilelang import (
        _grouped_gemm_v2_kernel,  # low-level v2 kernel
        _get_v2_block_config
    )
    HAS_V2 = True
except ImportError:
    HAS_V2 = False

try:
    from kernels.grouped_gemm_tilelang import (
        _grouped_gemm_v4_kernel,
        _get_v4_block_config
    )
    HAS_V4 = True
except ImportError:
    HAS_V4 = False

try:
    from kernels.grouped_gemm_tilelang import _grouped_gemm_v5_kernel
    HAS_V5 = True
except ImportError:
    HAS_V5 = False

HAS_TORCH_GROUPED_MM = hasattr(torch, "_grouped_mm")

try:
    from kernels.grouped_gemm_triton import (
        prepare_triton_metadata,
        prepare_triton_tma_metadata,
        triton_grouped_gemm,
        triton_grouped_gemm_tma,
        _supports_tma,
    )
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============ Real MoE model configs ============
# (name, H, I, E_total, top_k)
MODEL_CONFIGS = [
    ("Qwen3-235B",    4096,  3072, 128, 8),
    ("DeepSeek-V2",   5120,  1536, 160, 6),
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
                      warmup=25, rep=100):
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

    # Unified BF16 data for fair comparison
    X_bf16 = torch.randn(routed_tokens_local, H, device=device, dtype=torch.bfloat16)
    W_bf16_HI = torch.randn(E_local, H, I, device=device, dtype=torch.bfloat16)
    W_bf16_IH = W_bf16_HI.transpose(1, 2).contiguous()

    # ==========================================
    # Preprocessing (host-side, NOT included in latency)
    # ==========================================
    sorted_indices = expert_indices.argsort()
    X_sorted_bf16 = X_bf16[sorted_indices]

    counts = torch.bincount(expert_indices, minlength=E_local)
    batch_sizes = counts.tolist()
    offs = torch.cumsum(counts, dim=0).to(torch.int32)

    Y_bf16 = torch.empty(routed_tokens_local, I, device=device, dtype=torch.bfloat16)

    # Pre-extract views for PyTorch for-loop to eliminate per-call transpose overhead
    W_bf16_IH_views = [W_bf16_IH[eid].T for eid in range(E_local)]

    # --- Prepare TL-v1 metadata ---
    block_M, block_N, block_K = 64, 64, 64
    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(E_local - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes[i])
        batch_padded_offsets_list.append(
            batch_padded_offsets_list[-1] +
            math.ceil(batch_sizes[i] / block_M) * block_M
        )

    batch_sizes_t_v1 = torch.tensor(batch_sizes, device=device, dtype=torch.int32)
    batch_offsets_t_v1 = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets_t_v1 = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)

    # Pre-compile TL-v1 kernel (dtype=T.bfloat16)
    kernel_v1 = _grouped_gemm_kernel(
        tuple(batch_sizes), K=H, N=I,
        block_M=block_M, block_N=block_N, block_K=block_K,
        num_stages=2, threads=256, dtype=T.bfloat16
    )

    # --- Prepare TL-v2 metadata ---
    if HAS_V2:
        v2_bm, v2_bn, v2_bk, v2_stages, v2_threads = _get_v2_block_config(H, I, E_local, routed_tokens_local)

        batch_offsets_v2 = [0]
        batch_padded_offsets_v2 = [0]
        for i in range(E_local - 1):
            batch_offsets_v2.append(batch_offsets_v2[-1] + batch_sizes[i])
            batch_padded_offsets_v2.append(
                batch_padded_offsets_v2[-1]
                + math.ceil(batch_sizes[i] / v2_bm) * v2_bm
            )

        tile_expert_ids_list = []
        for expert_id, size in enumerate(batch_sizes):
            n_blocks = math.ceil(size / v2_bm) if size > 0 else 0
            tile_expert_ids_list.extend([expert_id] * n_blocks)

        batch_offsets_t_v2 = torch.tensor(batch_offsets_v2, device=device, dtype=torch.int32)
        batch_padded_offsets_t_v2 = torch.tensor(batch_padded_offsets_v2, device=device, dtype=torch.int32)
        tile_expert_ids_t_v2 = torch.tensor(tile_expert_ids_list, device=device, dtype=torch.int32)

        # Pre-compile TL-v2 kernel (dtype=T.bfloat16)
        kernel_v2 = _grouped_gemm_v2_kernel(
            tuple(batch_sizes), K=H, N=I,
            block_M=v2_bm, block_N=v2_bn, block_K=v2_bk,
            num_stages=v2_stages, threads=v2_threads, dtype=T.bfloat16
        )

    # --- Prepare TL-v4 metadata ---
    v4_cfg = ""
    if HAS_V4:
        v4_bm, v4_bn, v4_bk, v4_stages, v4_threads = _get_v4_block_config(H, I, E_local, routed_tokens_local)
        v4_cfg = "{}x{}x{}_s{}_t{}".format(v4_bm, v4_bn, v4_bk, v4_stages, v4_threads)

        batch_offsets_v4 = [0]
        batch_padded_offsets_v4 = [0]
        for i in range(E_local - 1):
            batch_offsets_v4.append(batch_offsets_v4[-1] + batch_sizes[i])
            batch_padded_offsets_v4.append(
                batch_padded_offsets_v4[-1]
                + math.ceil(batch_sizes[i] / v4_bm) * v4_bm
            )

        tile_expert_ids_v4_list = []
        for expert_id, size in enumerate(batch_sizes):
            n_blocks = math.ceil(size / v4_bm) if size > 0 else 0
            tile_expert_ids_v4_list.extend([expert_id] * n_blocks)

        batch_offsets_t_v4 = torch.tensor(batch_offsets_v4, device=device, dtype=torch.int32)
        batch_padded_offsets_t_v4 = torch.tensor(batch_padded_offsets_v4, device=device, dtype=torch.int32)
        tile_expert_ids_t_v4 = torch.tensor(tile_expert_ids_v4_list, device=device, dtype=torch.int32)

        kernel_v4 = _grouped_gemm_v4_kernel(
            tuple(batch_sizes), K=H, N=I,
            block_M=v4_bm, block_N=v4_bn, block_K=v4_bk,
            num_stages=v4_stages, threads=v4_threads, dtype=T.bfloat16
        )

    # --- Prepare TL-v5 (persistent v4) ---
    if HAS_V5 and HAS_V4:
        sm_num = torch.cuda.get_device_properties(device).multi_processor_count
        kernel_v5 = _grouped_gemm_v5_kernel(
            tuple(batch_sizes), K=H, N=I, sm_num=sm_num,
            block_M=v4_bm, block_N=v4_bn, block_K=v4_bk,
            num_stages=v4_stages, threads=v4_threads, dtype=T.bfloat16
        )

    results = {}

    # ==========================================
    # Pure kernel benchmark
    # - PyTorch / grouped_mm / Triton: do_bench (function-level)
    # - TileLang kernels: profiler.do_bench (adapter-level, no JIT wrapper overhead)
    # ==========================================

    # --- PyTorch for-loop (bf16, cuBLAS) ---
    def loop_fn():
        start = 0
        for eid in range(E_local):
            sz = batch_sizes[eid]
            if sz == 0:
                continue
            torch.mm(X_sorted_bf16[start:start+sz], W_bf16_IH_views[eid],
                     out=Y_bf16[start:start+sz])
            start += sz

    results['PyTorch'] = do_bench(loop_fn, warmup=warmup, rep=rep)

    # --- torch._grouped_mm (CUTLASS, bf16) ---
    if HAS_TORCH_GROUPED_MM:
        try:
            torch._grouped_mm(X_sorted_bf16, W_bf16_HI, offs)  # verify/warmup
            results['grouped_mm'] = do_bench(
                lambda: torch._grouped_mm(X_sorted_bf16, W_bf16_HI, offs),
                warmup=warmup, rep=rep)
        except Exception:
            results['grouped_mm'] = float('inf')

    # --- TileLang v1 (bf16) — profiler.do_bench via adapter ---
    try:
        kernel_v1(X_sorted_bf16, W_bf16_HI, batch_sizes_t_v1, batch_offsets_t_v1, batch_padded_offsets_t_v1)
        profiler_v1 = kernel_v1.get_profiler()
        results['TL-v1'] = profiler_v1.do_bench(
            input_tensors=[X_sorted_bf16, W_bf16_HI, batch_sizes_t_v1, batch_offsets_t_v1, batch_padded_offsets_t_v1],
            warmup=warmup, rep=rep)
    except Exception as e:
        print("\n[TL-v1 Error] {}".format(e))
        results['TL-v1'] = float('inf')

    # --- TileLang v2 (bf16) — profiler.do_bench via adapter ---
    if HAS_V2:
        try:
            kernel_v2(X_sorted_bf16, W_bf16_HI, batch_offsets_t_v2, batch_padded_offsets_t_v2, tile_expert_ids_t_v2)
            profiler_v2 = kernel_v2.get_profiler()
            results['TL-v2'] = profiler_v2.do_bench(
                input_tensors=[X_sorted_bf16, W_bf16_HI, batch_offsets_t_v2, batch_padded_offsets_t_v2, tile_expert_ids_t_v2],
                warmup=warmup, rep=rep)
        except Exception as e:
            print("\n[TL-v2 Error] {}".format(e))
            results['TL-v2'] = float('inf')

    # --- TileLang v4 (bf16) — profiler.do_bench via adapter ---
    if HAS_V4:
        try:
            kernel_v4(X_sorted_bf16, W_bf16_HI, batch_offsets_t_v4, batch_padded_offsets_t_v4, tile_expert_ids_t_v4)
            profiler_v4 = kernel_v4.get_profiler()
            results['TL-v4'] = profiler_v4.do_bench(
                input_tensors=[X_sorted_bf16, W_bf16_HI, batch_offsets_t_v4, batch_padded_offsets_t_v4, tile_expert_ids_t_v4],
                warmup=warmup, rep=rep)
        except Exception as e:
            print("\n[TL-v4 Error] {}".format(e))
            results['TL-v4'] = float('inf')

    # --- TileLang v5 (persistent v4, bf16) — profiler.do_bench via adapter ---
    if HAS_V5 and HAS_V4:
        try:
            kernel_v5(X_sorted_bf16, W_bf16_HI, batch_offsets_t_v4, batch_padded_offsets_t_v4, tile_expert_ids_t_v4)
            profiler_v5 = kernel_v5.get_profiler()
            results['TL-v5'] = profiler_v5.do_bench(
                input_tensors=[X_sorted_bf16, W_bf16_HI, batch_offsets_t_v4, batch_padded_offsets_t_v4, tile_expert_ids_t_v4],
                warmup=warmup, rep=rep)
        except Exception as e:
            print("\n[TL-v5 Error] {}".format(e))
            results['TL-v5'] = float('inf')

    # --- Triton grouped GEMM (bf16, non-TMA) ---
    if HAS_TRITON:
        try:
            tri_a, tri_b, tri_c, tri_sizes, tri_lds, tri_Y = prepare_triton_metadata(
                X_sorted_bf16, W_bf16_HI, batch_sizes, device=device)
            group_size = E_local
            # Warmup (includes autotune)
            triton_grouped_gemm(tri_a, tri_b, tri_c, tri_sizes, tri_lds, group_size)
            results['Triton'] = do_bench(
                lambda: triton_grouped_gemm(tri_a, tri_b, tri_c, tri_sizes, tri_lds, group_size),
                warmup=warmup, rep=rep)
        except Exception as e:
            print("\n[Triton Error] {}".format(e))
            results['Triton'] = float('inf')

        # --- Triton grouped GEMM (bf16, TMA, Hopper+ only) ---
        if _supports_tma():
            try:
                tma_a, tma_b, tma_c, tma_sizes, tma_lds, tma_Y = prepare_triton_tma_metadata(
                    X_sorted_bf16, W_bf16_HI, batch_sizes, device=device)
                triton_grouped_gemm_tma(tma_a, tma_b, tma_c, tma_sizes, tma_lds, group_size)
                results['Triton-TMA'] = do_bench(
                    lambda: triton_grouped_gemm_tma(tma_a, tma_b, tma_c, tma_sizes, tma_lds, group_size),
                    warmup=warmup, rep=rep)
            except Exception as e:
                print("\n[Triton-TMA Error] {}".format(e))
                results['Triton-TMA'] = float('inf')

    return results, tokens_per_expert, routed_tokens_local, v4_cfg


def main():
    parser = argparse.ArgumentParser(description="Grouped GEMM Benchmark v2")
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--input-tokens", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=25, help="Warmup time budget in ms")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark time budget in ms")
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
    print("Grouped GEMM Benchmark v2 — Real MoE Prefill Scenarios (Pure Kernel BF16)")
    print("=" * 100)
    print("Device:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__)
    print("torch._grouped_mm:", "available" if HAS_TORCH_GROUPED_MM else "NOT available")
    print("TL-v2:", "available" if HAS_V2 else "NOT available")
    print("TL-v4:", "available" if HAS_V4 else "NOT available")
    print("TL-v5:", "available" if HAS_V5 else "NOT available")
    print("Triton:", "available" if HAS_TRITON else "NOT available")
    if HAS_TRITON:
        print("Triton-TMA:", "available" if _supports_tma() else "NOT available (need Hopper+)")
    print()

    all_rows = []

    for name, H, I, E_total, top_k in configs:
        e_locals = get_e_local_list(E_total)

        for E_local in e_locals:
            print("\n{} (H={}, I={}, E_total={}, top_k={}, E_local={})".format(
                name, H, I, E_total, top_k, E_local))
            print("-" * 80)

            for input_tokens in input_sweep:
                results, tpe, rt_local, v4_tag = run_single_config(
                    name, H, I, E_local, top_k, E_total, input_tokens,
                    warmup=args.warmup, rep=args.rep)

                # Print inline
                parts = []
                for v, lat in results.items():
                    if lat == float('inf'):
                        parts.append("{}=FAIL".format(v))
                    else:
                        parts.append("{}={:.4f}ms".format(v, lat))
                cfg_str = " [v4:{}]".format(v4_tag) if v4_tag else ""
                print("  input_tokens={:<6} tpe={:<4} rt_local={:<6} {}{}".format(
                    input_tokens, tpe, rt_local, "  ".join(parts), cfg_str))

                all_rows.append({
                    'model': name, 'H': H, 'I': I,
                    'E_total': E_total, 'E_local': E_local,
                    'top_k': top_k, 'input_tokens': input_tokens,
                    'tpe': tpe, 'rt_local': rt_local,
                    **results
                })

    # ============ Summary table ============
    print("\n\n" + "=" * 100)
    print("SUMMARY: Pure Kernel Latency (ms) and Speedup vs PyTorch for-loop (BF16)")
    print("=" * 100)

    version_keys = ['PyTorch', 'grouped_mm', 'TL-v1', 'TL-v2', 'TL-v4', 'TL-v5', 'Triton', 'Triton-TMA']
    version_keys = [v for v in version_keys
                    if any(v in r for r in all_rows)]

    headers = ["Model", "E_loc", "inp_tok", "tpe", "rt_loc"] + \
              [v + "(ms)" for v in version_keys] + \
              [v + "/Py" for v in version_keys if v != 'PyTorch'] + \
              ["Best"]

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
        # Find best implementation
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
