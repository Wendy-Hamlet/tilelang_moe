"""Dump CUDA source AND TileLang IR of all router_fusion kernel variants."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tilelang.language as T
from kernels.router_gemm_topk_softmax import (
    _next_pow2,
    fused_router_cpasync_selection_kernel,
    fused_router_cpasync_bitonic_kernel,
    fused_router_tma_selection_kernel,
    fused_router_tma_bitonic_kernel,
    fused_router_tma2_selection_kernel,
    fused_router_tma2_bitonic_kernel,
    fused_router_tma2n_selection_kernel,
    fused_router_tma2n_bitonic_kernel,
    fused_router_tma3_selection_kernel,
    fused_router_tma3_bitonic_kernel,
)

# Use DeepSeek-V2 shape: most complex (non-pow2 E, largest H)
T_TOKENS = 256
H = 5120
E = 160
K = 6
E_padded = _next_pow2(E)  # 256

VARIANTS = [
    ("cpasync_selection", fused_router_cpasync_selection_kernel, 256,  64),
    ("cpasync_bitonic",   fused_router_cpasync_bitonic_kernel,   256,  64),
    ("tma_selection",     fused_router_tma_selection_kernel,     256,  64),
    ("tma_bitonic",       fused_router_tma_bitonic_kernel,       256,  64),
    ("tma2_selection",    fused_router_tma2_selection_kernel,    384,  64),
    ("tma2_bitonic",      fused_router_tma2_bitonic_kernel,      384,  64),
    ("tma2n_selection",   fused_router_tma2n_selection_kernel,   384,  64),
    ("tma2n_bitonic",     fused_router_tma2n_bitonic_kernel,     384,  64),
    ("tma3_selection",    fused_router_tma3_selection_kernel,    384, 128),
    ("tma3_bitonic",      fused_router_tma3_bitonic_kernel,      384, 128),
]

SEPARATOR = "=" * 100

docs_dir = Path(__file__).resolve().parent.parent / "docs"
docs_dir.mkdir(exist_ok=True)
cuda_path = docs_dir / "router_fusion_cuda_dump.cu"
tir_path  = docs_dir / "router_fusion_tir_dump.py"

cuda_lines = [
    "// Router Fusion CUDA Dump",
    f"// Shape: T={T_TOKENS} H={H} E={E} E_padded={E_padded} K={K}",
    "// Model: DeepSeek-V2",
    "",
]
tir_lines = [
    "# Router Fusion TileLang IR Dump",
    f"# Shape: T={T_TOKENS} H={H} E={E} E_padded={E_padded} K={K}",
    "# Model: DeepSeek-V2",
    "",
]

for name, factory, threads, block_M in VARIANTS:
    print(f"Compiling {name} (threads={threads}, block_M={block_M})...", flush=True)
    try:
        kern = factory(
            T_TOKENS, H, E, E_padded, K,
            block_M=block_M, block_K=64, num_stages=2, threads=threads,
            dtype=T.bfloat16)

        cuda_src = kern.get_kernel_source()
        tir_src  = kern.prim_func.script()
        n_cu  = len(cuda_src.splitlines())
        n_tir = len(tir_src.splitlines())
        print(f"  OK  CUDA={n_cu} lines, TIR={n_tir} lines", flush=True)

        cuda_lines += [
            SEPARATOR,
            f"// VARIANT: {name}  threads={threads}  block_M={block_M}",
            f"// Lines: {n_cu}",
            SEPARATOR,
            cuda_src,
            "",
        ]
        tir_lines += [
            "# " + "=" * 96,
            f"# VARIANT: {name}  threads={threads}  block_M={block_M}",
            f"# Lines: {n_tir}",
            "# " + "=" * 96,
            tir_src,
            "",
        ]
    except Exception as ex:
        print(f"  FAIL: {ex}", flush=True)
        cuda_lines += [SEPARATOR, f"// VARIANT: {name}  FAILED: {ex}", SEPARATOR, ""]
        tir_lines  += ["# " + "=" * 96, f"# VARIANT: {name}  FAILED: {ex}",
                       "# " + "=" * 96, ""]

cuda_out = "\n".join(cuda_lines)
tir_out  = "\n".join(tir_lines)
cuda_path.write_text(cuda_out, encoding="utf-8")
tir_path.write_text(tir_out,  encoding="utf-8")
print(f"\nWritten CUDA: {len(cuda_out)} chars → {cuda_path}")
print(f"Written TIR : {len(tir_out)} chars → {tir_path}")
