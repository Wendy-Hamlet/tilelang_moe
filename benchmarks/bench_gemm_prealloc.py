"""Test TileLang GEMM with pre-allocated output vs JIT allocation."""
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


def bench(fn, warmup=50, rep=500):
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
    for M in [256, 1024, 4096]:
        K, N = 5120, 256
        X = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float32)

        k = std_gemm(M, N, K)

        # JIT call (allocates output each time)
        t_jit = bench(lambda: k(X, W))

        # Call with pre-allocated output via adapter
        try:
            k.adapter(X, W, C)
            torch.cuda.synchronize()
            t_pre = bench(lambda: k.adapter(X, W, C))
        except Exception as e:
            t_pre = -1
            print(f"  adapter failed: {e}")

        # torch.mm
        t_mm = bench(lambda: torch.mm(X, W))

        print(f"M={M:>5}: torch.mm={t_mm:.4f}ms  TL-jit={t_jit:.4f}ms  TL-prealloc={t_pre:.4f}ms  "
              f"jit/mm={t_mm/t_jit:.2f}x  pre/mm={t_mm/t_pre:.2f}x")


if __name__ == "__main__":
    main()
