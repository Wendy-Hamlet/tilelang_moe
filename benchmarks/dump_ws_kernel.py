"""Dump CUDA source of ws selection kernel for debugging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tilelang
import tilelang.language as T
from kernels.router_gemm_topk_softmax import fused_router_ws_selection_kernel, _next_pow2

E = 160
E_padded = _next_pow2(E)

k = fused_router_ws_selection_kernel(
    256, 5120, E, E_padded, 6,
    block_M=64, block_K=64, num_stages=2, threads=256, dtype=T.bfloat16)

src = k.get_kernel_source()
with open("/tmp/ws_kernel.cu", "w") as f:
    f.write(src)
print(f"Written {len(src)} chars, {len(src.splitlines())} lines")

# Print key structural lines
for i, line in enumerate(src.splitlines()):
    s = line.strip()
    if any(kw in s for kw in [
        "__global__", "__syncthreads", "__barrier_sync", "barrier_sync",
        "mbarrier", "tl_shuffle_elect", "warpgroup_reg",
        "tl::tma_load", "tl::wgmma", "for (int row", "for (int ko",
    ]):
        print(f"L{i+1}: {line.rstrip()}")
