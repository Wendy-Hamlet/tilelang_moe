"""
mHC Pre GEMM+sqrsum Optimized Versions

优化策略:
v2: 分离sqrsum到独立计算阶段
v3: 使用更大的hidden_block=256 (需要测试是否适合5090)
v4: 合并sqrsum到GEMM后处理
"""

import torch
import tilelang
import tilelang.language as T


# ============ v2: 分离sqrsum ============

@tilelang.jit
def mhc_pre_gemm_sqrsum_v2_kernel(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    """
    v2: 分离sqrsum - 先做完整GEMM，再单独计算sqrsum
    
    原理: GEMM和sqrsum分开可以让GEMM流水线更高效
    """
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        T.clear(out_frag)
        
        # Phase 1: Pure GEMM with pipeline
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=3):
            x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout({x_smem: tilelang.layout.make_swizzled_layout(x_smem)})

            T.copy(x[px * token_block, pz * hidden_block], x_smem)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            T.gemm(
                x_smem,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                wg_wait=0,
                clear_accum=False,
            )
        
        # Write GEMM output
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]
        
        # Phase 2: Compute sqrsum separately
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(sqrsum_part)
        
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
            x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            T.copy(x[px * token_block, pz * hidden_block], x_smem)
            
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_smem, x_frag)
            
            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]
        
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]


# ============ v3: 更大的hidden_block ============

@tilelang.jit
def mhc_pre_gemm_sqrsum_v3_kernel(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    """
    v3: 使用更大的hidden_block=256
    
    减少迭代次数，每次处理更多数据
    """
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
            x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout({x_smem: tilelang.layout.make_swizzled_layout(x_smem)})

            T.copy(x[px * token_block, pz * hidden_block], x_smem)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            # sqrsum
            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            # GEMM
            T.gemm(
                x_frag,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                wg_wait=0,
                clear_accum=False,
            )
        
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]


# ============ v4: 4-stage pipeline ============

@tilelang.jit
def mhc_pre_gemm_sqrsum_v4_kernel(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    """
    v4: 4-stage pipeline
    
    更深的流水线可能在5090上有益
    """
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=4):
            x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout({x_smem: tilelang.layout.make_swizzled_layout(x_smem)})

            T.copy(x[px * token_block, pz * hidden_block], x_smem)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            T.gemm(
                x_frag,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                wg_wait=0,
                clear_accum=False,
            )
        
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]


# ============ v5: 更小的sqrsum累加器 ============

@tilelang.jit
def mhc_pre_gemm_sqrsum_v5_kernel(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    """
    v5: 优化sqrsum累加 - 使用更大的并行度
    
    将sqrsum计算改为直接并行累加
    """
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        # 使用更大的累加器来提高并行度
        sqrsum_part = T.alloc_fragment((token_block, 8), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=3):
            x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout({x_smem: tilelang.layout.make_swizzled_layout(x_smem)})

            T.copy(x[px * token_block, pz * hidden_block], x_smem)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            # sqrsum with larger accumulator
            for jj in T.serial(hidden_block // 8):
                for i, j in T.Parallel(token_block, 8):
                    sqrsum_part[i, j] += x_frag[i, jj * 8 + j] * x_frag[i, jj * 8 + j]

            T.gemm(
                x_frag,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                wg_wait=0,
                clear_accum=False,
            )
        
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]


# ============ Wrapper functions ============

def mhc_pre_gemm_sqrsum_v2(x, fn, out, sqrsum, hc_mult3, hc_hidden_size):
    mhc_pre_gemm_sqrsum_v2_kernel(x, fn, out, sqrsum, hc_mult3, hc_hidden_size)

def mhc_pre_gemm_sqrsum_v3(x, fn, out, sqrsum, hc_mult3, hc_hidden_size):
    mhc_pre_gemm_sqrsum_v3_kernel(x, fn, out, sqrsum, hc_mult3, hc_hidden_size)

def mhc_pre_gemm_sqrsum_v4(x, fn, out, sqrsum, hc_mult3, hc_hidden_size):
    mhc_pre_gemm_sqrsum_v4_kernel(x, fn, out, sqrsum, hc_mult3, hc_hidden_size)

def mhc_pre_gemm_sqrsum_v5(x, fn, out, sqrsum, hc_mult3, hc_hidden_size):
    mhc_pre_gemm_sqrsum_v5_kernel(x, fn, out, sqrsum, hc_mult3, hc_hidden_size)


# ============ Test ============

if __name__ == "__main__":
    import time
    
    torch.manual_seed(42)
    device = "cuda"
    
    num_tokens = 256
    hc_mult = 4
    hidden_size = 5120
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    
    x = torch.randn(num_tokens, hc_hidden_size, dtype=torch.bfloat16, device=device)
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device=device)
    out = torch.empty(num_tokens, hc_mult3, dtype=torch.float32, device=device)
    sqrsum = torch.empty(num_tokens, dtype=torch.float32, device=device)
    
    print("Testing mHC Pre GEMM+sqrsum versions...")
    
    versions = [
        ("v2 (separate sqrsum)", mhc_pre_gemm_sqrsum_v2),
        ("v3 (h_blk=256)", mhc_pre_gemm_sqrsum_v3),
        ("v4 (4-stage)", mhc_pre_gemm_sqrsum_v4),
        ("v5 (sqrsum_8)", mhc_pre_gemm_sqrsum_v5),
    ]
    
    for name, fn_ver in versions:
        try:
            fn_ver(x, fn, out, sqrsum, hc_mult3, hc_hidden_size)
            print(f"✅ {name}: out shape = {out.shape}, sqrsum shape = {sqrsum.shape}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")
