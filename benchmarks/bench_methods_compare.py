"""Compare bench methods: our bench_fn vs TileLang do_bench (event/cudagraph)."""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler.bench import do_bench


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
    """Our original bench_fn: one event pair wrapping all reps."""
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
    print(f"GEMM Benchmark Method Comparison (K={K}, N={N}, BF16)")
    print(f"{'M':>6} {'torch.mm':>10} {'ours':>10} {'TL-event':>10} {'TL-graph':>10} {'graph/mm':>10}")
    print("-" * 65)

    for M in [256, 1024, 4096, 16384]:
        X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

        k = std_gemm(M, N, K)

        # torch.mm baseline (various methods)
        t_mm = our_bench(lambda: torch.mm(X, W))

        # Our bench_fn
        t_ours = our_bench(lambda: k(X, W))

        # TileLang do_bench: event backend (per-call events + L2 flush)
        t_event = do_bench(lambda: k(X, W), backend="event")

        # TileLang do_bench: cudagraph backend (graph replay, no host overhead)
        try:
            t_graph = do_bench(lambda: k(X, W), backend="cudagraph")
        except Exception as e:
            t_graph = -1
            print(f"  cudagraph err: {e}")

        ratio = f"{t_mm/t_graph:.2f}x" if t_graph > 0 else "ERR"
        print(f"{M:>6} {t_mm:>9.4f}ms {t_ours:>9.4f}ms {t_event:>9.4f}ms {t_graph:>9.4f}ms {ratio:>10}")


if __name__ == "__main__":
    main()
