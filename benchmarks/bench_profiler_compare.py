"""Use TileLang's built-in profiler (adapter-level, bypasses Python JIT wrapper)."""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def std_gemm(M, N, K, block_M=128, block_N=128, block_K=64,
             num_stages=3, threads=128, dtype=T.bfloat16):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), T.float32)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            As = T.alloc_shared((block_M, block_K), dtype)
            Bs = T.alloc_shared((block_K, block_N), dtype)
            Cl = T.alloc_fragment((block_M, block_N), T.float32)
            Cs = T.alloc_shared((block_M, block_N), T.float32)
            T.use_swizzle(10)
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], As)
                T.copy(B[k * block_K, bx * block_N], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, Cs)
            T.copy(Cs, C[by * block_M, bx * block_N])
    return kernel


def our_bench(fn, warmup=50, rep=500):
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


def main():
    K, N = 5120, 256
    print(f"GEMM profiler comparison (K={K}, N={N}, BF16->F32)")
    print(f"profiler.do_bench uses adapter (C++ level, no Python JIT wrapper)")
    print(f"{'M':>6} {'torch.mm':>10} {'ours(jit)':>10} {'TL-event':>10} {'TL-cupti':>10} {'event/mm':>10}")
    print("-" * 65)

    for M in [256, 1024, 4096, 16384]:
        X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

        k = std_gemm(M, N, K)

        # torch.mm
        t_mm = our_bench(lambda: torch.mm(X, W))

        # Our bench (Python JIT wrapper call)
        t_ours = our_bench(lambda: k(X, W))

        # TileLang profiler: uses adapter directly (C++ level)
        profiler = k.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

        # event backend
        t_event = profiler.do_bench(backend="event")

        # cupti backend (measures actual CUDA kernel time via PyTorch profiler)
        try:
            t_cupti = profiler.do_bench(backend="cupti")
        except Exception as e:
            t_cupti = -1
            print(f"  cupti err: {e}")

        ratio_e = f"{t_mm/t_event:.2f}x" if t_event > 0 else "ERR"
        ratio_c = f"{t_mm/t_cupti:.2f}x" if t_cupti > 0 else "ERR"
        print(f"{M:>6} {t_mm:>9.4f}ms {t_ours:>9.4f}ms {t_event:>9.4f}ms {t_cupti:>9.4f}ms {ratio_e:>10} {ratio_c:>10}")


if __name__ == "__main__":
    main()
