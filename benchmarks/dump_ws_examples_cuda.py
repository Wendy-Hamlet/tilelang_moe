"""Dump CUDA source of TileLang official WS examples to check warpgroup_reg_alloc."""
import sys
from pathlib import Path

# Run from tilelang_moe root — uses conda tilelang, not dev build
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import tilelang
import tilelang.language as T
import tilelang as tl

# Find the official WS examples
tilelang_pkg = Path(tilelang.__file__).parent
examples_dir = tilelang_pkg.parent.parent / "examples" / "warp_specialize"
if not examples_dir.exists():
    # Try alternate location (installed package)
    import site
    for sp in site.getsitepackages():
        candidate = Path(sp) / "tilelang" / "examples" / "warp_specialize"
        if candidate.exists():
            examples_dir = candidate
            break

print(f"TileLang location: {tilelang.__file__}")
print(f"TileLang version:  {tilelang.__version__}")
print(f"Examples dir:      {examples_dir}")
print(f"Examples exist:    {examples_dir.exists()}")
if examples_dir.exists():
    print(f"Files: {[f.name for f in examples_dir.glob('*.py')]}")

print()

# ── Example 1: gemm_copy_0_gemm_1 ──────────────────────────────────────────
# Simple 2-WG WS: WG0=producer(copy), WG1=consumer(gemm)
def make_ws_gemm_copy_0_gemm_1(M=256, N=128, K=256, block_M=64, block_N=128, block_K=64,
                                num_stages=2, threads=256, dtype=T.float16):
    """Reproduce the pattern from example_warp_specialize_gemm_copy_0_gemm_1."""
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), T.float32)
            barrier  = T.alloc_barrier(arrive_count=128)  # one warp group

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                with T.ws(0):  # producer: threads 0..127
                    T.copy(A[bx*block_M:(bx+1)*block_M, ko*block_K:(ko+1)*block_K], A_shared)
                    T.copy(B[ko*block_K:(ko+1)*block_K, by*block_N:(by+1)*block_N], B_shared)
                    T.barrier_arrive(barrier)

                with T.ws(1):  # consumer: threads 128..255
                    T.barrier_wait(barrier)
                    T.gemm(A_shared, B_shared, C_local)

            with T.ws(1):
                T.copy(C_local, C[bx*block_M:(bx+1)*block_M, by*block_N:(by+1)*block_N])

    return tilelang.jit(main, out_idx=[2], supply_type=tilelang.TensorSupplyType.Normal,
                        ref_prog=None, profiler="auto", target="auto", execution_backend="cython")


# ── Example 2: gemm_barrierpipe_stage2 ─────────────────────────────────────
# TMA + mbarrier pipeline WS (producer=WG0, consumer=WG1)
def make_ws_gemm_barrierpipe(M=256, N=128, K=256, block_M=64, block_N=128, block_K=64,
                              num_stages=2, threads=256, dtype=T.float16):
    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), T.float32)

            mbarrier_list = T.create_list_of_mbarrier([128] * num_stages + [128] * num_stages)
            LOAD_DONE = 0
            GEMM_DONE = num_stages

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                with T.ws(0):
                    T.copy(A[bx*block_M:(bx+1)*block_M, ko*block_K:(ko+1)*block_K], A_shared)
                    T.copy(B[ko*block_K:(ko+1)*block_K, by*block_N:(by+1)*block_N], B_shared)
                    T.mbarrier_arrive(mbarrier_list, LOAD_DONE + ko % num_stages)

                with T.ws(1):
                    T.mbarrier_wait_parity(mbarrier_list, LOAD_DONE + ko % num_stages, ko // num_stages % 2)
                    T.gemm(A_shared, B_shared, C_local)
                    T.mbarrier_arrive(mbarrier_list, GEMM_DONE + ko % num_stages)

                with T.ws(0):
                    T.mbarrier_wait_parity(mbarrier_list, GEMM_DONE + ko % num_stages, ko // num_stages % 2)

            with T.ws(1):
                T.copy(C_local, C[bx*block_M:(bx+1)*block_M, by*block_N:(by+1)*block_N])

    return tilelang.jit(main, out_idx=[2], supply_type=tilelang.TensorSupplyType.Normal,
                        ref_prog=None, profiler="auto", target="auto", execution_backend="cython")


SEPARATOR = "=" * 100

out_path = project_root / "docs" / "ws_examples_cuda_dump.cu"
out_path.parent.mkdir(exist_ok=True)

lines = []
lines.append("// TileLang WS Examples CUDA Dump")
lines.append("// Purpose: check for warpgroup_reg_alloc / warpgroup_reg_dealloc")
lines.append("")

examples = [
    ("ws_gemm_copy_0_gemm_1", make_ws_gemm_copy_0_gemm_1),
    ("ws_gemm_barrierpipe",   make_ws_gemm_barrierpipe),
]

for name, factory in examples:
    print(f"Compiling {name}...", flush=True)
    try:
        kern = factory()
        src = kern.get_kernel_source()
        n_lines = len(src.splitlines())
        has_alloc   = "warpgroup_reg_alloc"   in src
        has_dealloc = "warpgroup_reg_dealloc" in src
        print(f"  OK ({n_lines} lines) | reg_alloc={has_alloc} | reg_dealloc={has_dealloc}", flush=True)

        lines.append(SEPARATOR)
        lines.append(f"// EXAMPLE: {name}")
        lines.append(f"// Lines: {n_lines}  | warpgroup_reg_alloc: {has_alloc}  | warpgroup_reg_dealloc: {has_dealloc}")
        lines.append(SEPARATOR)
        lines.append(src)
        lines.append("")
    except Exception as ex:
        print(f"  FAIL: {ex}", flush=True)
        lines.append(SEPARATOR)
        lines.append(f"// EXAMPLE: {name}  FAILED: {ex}")
        lines.append(SEPARATOR)
        lines.append("")

# Also grep router_fusion dump for reg_alloc
router_dump = project_root / "docs" / "router_fusion_cuda_dump.cu"
if router_dump.exists():
    router_src = router_dump.read_text(encoding="utf-8")
    print(f"\nRouter fusion dump grep:")
    for variant in ["cpasync_selection", "cpasync_bitonic", "tma_selection",
                    "tma_bitonic", "tma2_selection", "tma2_bitonic",
                    "tma2n_selection", "tma2n_bitonic"]:
        has_alloc   = "warpgroup_reg_alloc"   in router_src
        has_dealloc = "warpgroup_reg_dealloc" in router_src
    # Per-variant check
    import re
    blocks = re.split(r"={100}", router_src)
    for block in blocks:
        m = re.search(r"VARIANT: (\w+)", block)
        if m:
            vname = m.group(1)
            has_a = "warpgroup_reg_alloc" in block
            has_d = "warpgroup_reg_dealloc" in block
            print(f"  {vname:<25} reg_alloc={has_a}  reg_dealloc={has_d}")

output = "\n".join(lines)
out_path.write_text(output, encoding="utf-8")
print(f"\nWritten {len(output)} chars to {out_path}")
