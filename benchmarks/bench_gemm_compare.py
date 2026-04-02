"""Quick GEMM comparison: standard 2D-tiled vs our 1D-tiled vs torch.mm"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def standard_gemm(M, N, K, block_M=128, block_N=128, block_K=64,
                  num_stages=3, threads=128, dtype=T.bfloat16):
    """Standard 2D-tiled GEMM from TileLang examples (TMA + warp spec enabled)."""
    accum_dtype = T.float32

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype),
               B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)
            T.use_swizzle(10)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return kernel


@tilelang.jit(out_idx=[1])
def router_gemm_1d(num_tokens, H, E_padded,
                   block_M=64, block_K=64, num_stages=2, threads=256,
                   dtype=T.bfloat16):
    """Our 1D-tiled GEMM (BN=E_padded, untiled over N)."""
    num_m_blocks = (num_tokens + block_M - 1) // block_M
    accum_dtype = T.float32

    @T.prim_func
    def kernel(X: T.Tensor([num_tokens, H], dtype),
               logits: T.Tensor([num_tokens, E_padded], accum_dtype),
               W: T.Tensor([H, E_padded], dtype)):
        with T.Kernel(num_m_blocks, threads=threads) as (bx):
            A_shared = T.alloc_shared([block_M, block_K], dtype)
            B_shared = T.alloc_shared([block_K, E_padded], dtype)
            C_local = T.alloc_fragment([block_M, E_padded], accum_dtype)
            C_shared = T.alloc_shared([block_M, E_padded], accum_dtype)
            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })
            m_start = bx * block_M
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(H, block_K), num_stages=num_stages):
                T.copy(X[m_start, k * block_K], A_shared)
                T.copy(W[k * block_K, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, logits[m_start, 0])

    return kernel


def bench(fn, warmup=50, rep=200):
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
    H, E = 5120, 160
    E_padded = 256
    print(f"H100 GEMM Comparison: K={H}, N_orig={E}, N_padded={E_padded}")
    print(f"{'M':>6} {'torch.mm':>10} {'TL-2D':>10} {'TL-1D':>10} {'2D/mm':>8} {'1D/mm':>8}")
    print("-" * 60)

    for M in [256, 1024, 4096]:
        X = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        W = torch.randn(H, E, device="cuda", dtype=torch.bfloat16)
        W_pad = torch.zeros(H, E_padded, device="cuda", dtype=torch.bfloat16)
        W_pad[:, :E] = W

        # torch.mm (uses original E=160)
        t_mm = bench(lambda: torch.mm(X, W))

        # Standard 2D GEMM (TMA + warp spec auto-enabled, padded E=256)
        try:
            k2d = standard_gemm(M, E_padded, H, block_M=128, block_N=128, block_K=64,
                                num_stages=3, threads=128, dtype=T.bfloat16)
            k2d(X, W_pad)
            torch.cuda.synchronize()
            t_2d = bench(lambda: k2d(X, W_pad))
        except Exception as ex:
            t_2d = float("inf")
            print(f"  2D err: {ex}")

        # Our 1D GEMM (TMA + warp spec auto-enabled, padded E=256)
        try:
            k1d = router_gemm_1d(M, H, E_padded)
            k1d(X, W_pad)
            torch.cuda.synchronize()
            t_1d = bench(lambda: k1d(X, W_pad))
        except Exception as ex:
            t_1d = float("inf")
            print(f"  1D err: {ex}")

        r2d = f"{t_mm/t_2d:.2f}x" if t_2d != float("inf") else "FAIL"
        r1d = f"{t_mm/t_1d:.2f}x" if t_1d != float("inf") else "FAIL"
        print(f"{M:>6} {t_mm:>9.4f}ms {t_2d:>9.4f}ms {t_1d:>9.4f}ms {r2d:>8} {r1d:>8}")


if __name__ == "__main__":
    main()
