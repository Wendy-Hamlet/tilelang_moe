"""Dump CUDA source of all router_fusion kernel variants for inspection."""
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
    ("cpasync_selection", fused_router_cpasync_selection_kernel, 256),
    ("cpasync_bitonic",   fused_router_cpasync_bitonic_kernel,   256),
    ("tma_selection",     fused_router_tma_selection_kernel,     256),
    ("tma_bitonic",       fused_router_tma_bitonic_kernel,       256),
    ("tma2_selection",    fused_router_tma2_selection_kernel,    384),
    ("tma2_bitonic",      fused_router_tma2_bitonic_kernel,      384),
    ("tma2n_selection",   fused_router_tma2n_selection_kernel,   384),
    ("tma2n_bitonic",     fused_router_tma2n_bitonic_kernel,     384),
    ("tma3_selection",    fused_router_tma3_selection_kernel,    384),
    ("tma3_bitonic",      fused_router_tma3_bitonic_kernel,      384),
]

SEPARATOR = "=" * 100

out_path = Path(__file__).resolve().parent.parent / "docs" / "router_fusion_cuda_dump.cu"
out_path.parent.mkdir(exist_ok=True)

lines = []
lines.append(f"// Router Fusion CUDA Dump")
lines.append(f"// Shape: T={T_TOKENS} H={H} E={E} E_padded={E_padded} K={K}")
lines.append(f"// Model: DeepSeek-V2")
lines.append("")

for name, factory, threads in VARIANTS:
    print(f"Compiling {name} (threads={threads})...", flush=True)
    try:
        kern = factory(
            T_TOKENS, H, E, E_padded, K,
            block_M=64, block_K=64, num_stages=2, threads=threads,
            dtype=T.bfloat16)
        src = kern.get_kernel_source()
        n_lines = len(src.splitlines())
        print(f"  OK  ({n_lines} lines)", flush=True)

        lines.append(SEPARATOR)
        lines.append(f"// VARIANT: {name}  threads={threads}")
        lines.append(f"// Lines: {n_lines}")
        lines.append(SEPARATOR)
        lines.append(src)
        lines.append("")
    except Exception as ex:
        print(f"  FAIL: {ex}", flush=True)
        lines.append(SEPARATOR)
        lines.append(f"// VARIANT: {name}  FAILED: {ex}")
        lines.append(SEPARATOR)
        lines.append("")

output = "\n".join(lines)
out_path.write_text(output, encoding="utf-8")
print(f"\nWritten {len(output)} chars to {out_path}")
