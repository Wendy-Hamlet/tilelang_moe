"""Test whether T.annotate_consumer_reg_alloc emits setmaxnreg PTX."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tilelang
import tilelang.language as T

M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 64


@tilelang.jit(out_idx=[2])
def ws_with_reg_alloc(M, N, K, block_M, block_N, block_K,
                      dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            data_ready   = T.alloc_barrier(arrive_count=128)
            compute_done = T.alloc_barrier(arrive_count=128)

            # Register reallocation: consumer gets max, producer gives away
            with T.ws(1):
                T.annotate_consumer_reg_alloc(240)
                T.clear(C_local)

            with T.ws(0):
                T.annotate_producer_reg_dealloc(24)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                with T.ws(0):
                    T.barrier_wait(compute_done, (ko + 1) % 2)
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.barrier_arrive(data_ready)
                with T.ws(1):
                    T.barrier_wait(data_ready, ko % 2)
                    T.gemm(A_shared, B_shared, C_local)
                    T.barrier_arrive(compute_done)

            with T.ws(1):
                T.copy(C_local, C[by * block_M, bx * block_N])
    return main


@tilelang.jit(out_idx=[2])
def ws_without_reg_alloc(M, N, K, block_M, block_N, block_K,
                         dtype=T.float16, accum_dtype=T.float32):
    """Same kernel WITHOUT register reallocation — baseline for comparison."""
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            data_ready   = T.alloc_barrier(arrive_count=128)
            compute_done = T.alloc_barrier(arrive_count=128)

            with T.ws(1):
                T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                with T.ws(0):
                    T.barrier_wait(compute_done, (ko + 1) % 2)
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.barrier_arrive(data_ready)
                with T.ws(1):
                    T.barrier_wait(data_ready, ko % 2)
                    T.gemm(A_shared, B_shared, C_local)
                    T.barrier_arrive(compute_done)

            with T.ws(1):
                T.copy(C_local, C[by * block_M, bx * block_N])
    return main


if __name__ == "__main__":
    for name, factory in [("with_reg_alloc", ws_with_reg_alloc),
                          ("without_reg_alloc", ws_without_reg_alloc)]:
        print(f"\nCompiling {name}...")
        try:
            kern = factory(M, N, K, block_M, block_N, block_K)
            src = kern.get_kernel_source()
            n = len(src.splitlines())
            has_alloc   = "warpgroup_reg_alloc"   in src
            has_dealloc = "warpgroup_reg_dealloc" in src
            has_setmax  = "setmaxnreg"             in src
            has_inc     = "set_max_nreg"           in src
            print(f"  lines={n}  reg_alloc={has_alloc}  reg_dealloc={has_dealloc}"
                  f"  setmaxnreg={has_setmax}  set_max_nreg_call={has_inc}")
            for i, line in enumerate(src.splitlines(), 1):
                if any(k in line for k in ["warpgroup_reg", "setmaxnreg", "set_max_nreg", "max_nreg"]):
                    print(f"    L{i}: {line.rstrip()}")
        except Exception as e:
            print(f"  FAIL: {e}")
