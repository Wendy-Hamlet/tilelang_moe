"""
mHC Post Optimized Implementation

算法层面优化：
1. 分离计算：将 comb.T @ res 和 post * x 分开计算
2. 利用矩阵结构：将小矩阵乘法重组
3. 减少访存：优化数据复用

原始计算:
  output[n, hc_out, h] = sum_hc_in(comb[hc_in, hc_out] * res[hc_in, h]) + post[hc_out] * x[h]
  
重组为:
  term1 = comb.T @ res   # (n, hc, hidden) = (n, hc, hc).T @ (n, hc, hidden)
  term2 = post @ x       # (n, hc, hidden) = (n, hc, 1) @ (n, 1, hidden)  (outer product)
  output = term1 + term2
"""

import math
import torch
import tilelang
import tilelang.language as T


@tilelang.jit
def mhc_post_optimized_v2_kernel(
    comb_res_mix,    # (n, hc, hc), float32
    residual,        # (n, hc, hidden), bfloat16
    post_layer_mix,  # (n, hc), float32
    x,               # (n, hidden), bfloat16
    output,          # (n, hc, hidden), bfloat16
    hc: int,
    hidden: int,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    """
    优化版mHC Post v2: 多token batch + 更大hidden块
    
    每个block处理一个token的全部hidden维度
    优化点:
    1. 加载一次comb_res_mix，复用于所有hidden位置
    2. 更大的hidden_block提高并行度
    3. 流水线加载减少延迟
    """
    num_tokens = T.dynamic("num_tokens")
    
    comb_res_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    residual: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    post_layer_mix: T.Tensor((num_tokens, hc), T.float32)
    x: T.Tensor((num_tokens, hidden), T.bfloat16)
    output: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    
    # 每个block处理一个token
    with T.Kernel(num_tokens, threads=256) as token_idx:
        # 加载comb_res_mix到local (只需加载一次，复用于所有hidden)
        comb_local = T.alloc_fragment((hc, hc), T.float32)
        T.copy(comb_res_mix[token_idx, 0, 0], comb_local)
        
        # 加载post_layer_mix
        post_local = T.alloc_fragment(hc, T.float32)
        T.copy(post_layer_mix[token_idx, 0], post_local)
        
        # 分块处理hidden维度
        for h_blk in T.Pipelined(hidden // hidden_block, num_stages=2):
            # 加载residual block: (hc, hidden_block)
            res_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
            T.copy(residual[token_idx, 0, h_blk * hidden_block], res_smem)
            
            # 加载x block: (hidden_block,)
            x_smem = T.alloc_shared(hidden_block, T.bfloat16)
            T.copy(x[token_idx, h_blk * hidden_block], x_smem)
            
            # 转换到local fragments
            res_local = T.alloc_fragment((hc, hidden_block), T.float32)
            T.copy(res_smem, res_local)
            
            x_local = T.alloc_fragment(hidden_block, T.float32)
            T.copy(x_smem, x_local)
            
            # 计算 output = comb.T @ res + post * x
            out_local = T.alloc_fragment((hc, hidden_block), T.float32)
            
            for hc_out in T.serial(hc):
                post_val = post_local[hc_out]
                for h in T.Parallel(hidden_block):
                    # term2: outer product
                    out_local[hc_out, h] = post_val * x_local[h]
                    # term1: matrix multiply
                    for hc_in in T.serial(hc):
                        out_local[hc_out, h] += comb_local[hc_in, hc_out] * res_local[hc_in, h]
            
            # 输出
            out_bf16 = T.alloc_fragment((hc, hidden_block), T.bfloat16)
            T.copy(out_local, out_bf16)
            
            out_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
            T.copy(out_bf16, out_smem)
            T.copy(out_smem, output[token_idx, 0, h_blk * hidden_block])


@tilelang.jit
def mhc_post_optimized_v3_kernel(
    comb_res_mix,    # (n, hc, hc), float32
    residual,        # (n, hc, hidden), bfloat16
    post_layer_mix,  # (n, hc), float32
    x,               # (n, hidden), bfloat16
    output,          # (n, hc, hidden), bfloat16
    hc: int,
    hidden: int,
    token_block: int = 8,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    """
    优化版mHC Post v3: 分离term1和term2的计算
    
    思路：
    - term1 = comb.T @ res 是矩阵乘法
    - term2 = post * x 是向量outer product (独立计算)
    
    这个版本先计算term2，再累加term1
    """
    num_tokens = T.dynamic("num_tokens")
    
    comb_res_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    residual: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    post_layer_mix: T.Tensor((num_tokens, hc), T.float32)
    x: T.Tensor((num_tokens, hidden), T.bfloat16)
    output: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    
    # Grid: (num_token_blocks, num_hidden_blocks)
    with T.Kernel(T.ceildiv(num_tokens, token_block), T.ceildiv(hidden, hidden_block), threads=256) as (bx, by):
        # Allocate output accumulator
        out_frag = T.alloc_fragment((token_block, hc, hidden_block), T.float32)
        
        # Load post_layer_mix for all tokens in block
        post_smem = T.alloc_shared((token_block, hc), T.float32)
        for t in T.Parallel(token_block):
            token_idx = bx * token_block + t
            if token_idx < num_tokens:
                for i in T.Parallel(hc):
                    post_smem[t, i] = post_layer_mix[token_idx, i]
        
        # Load x block for all tokens
        x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
        for t in T.Parallel(token_block):
            token_idx = bx * token_block + t
            for h in T.Parallel(hidden_block):
                hidden_idx = by * hidden_block + h
                if token_idx < num_tokens and hidden_idx < hidden:
                    x_smem[t, h] = x[token_idx, hidden_idx]
        
        x_local = T.alloc_fragment((token_block, hidden_block), T.float32)
        T.copy(x_smem, x_local)
        
        # Step 1: Compute term2 = post * x (outer product)
        for t in T.Parallel(token_block):
            for hc_out in T.Parallel(hc):
                for h in T.Parallel(hidden_block):
                    out_frag[t, hc_out, h] = post_smem[t, hc_out] * x_local[t, h]
        
        # Load comb_res_mix for all tokens in block
        comb_smem = T.alloc_shared((token_block, hc, hc), T.float32)
        for t in T.Parallel(token_block):
            token_idx = bx * token_block + t
            if token_idx < num_tokens:
                for i, j in T.Parallel(hc, hc):
                    comb_smem[t, i, j] = comb_res_mix[token_idx, i, j]
        
        # Load residual block
        res_smem = T.alloc_shared((token_block, hc, hidden_block), T.bfloat16)
        for t in T.Parallel(token_block):
            token_idx = bx * token_block + t
            for i in T.Parallel(hc):
                for h in T.Parallel(hidden_block):
                    hidden_idx = by * hidden_block + h
                    if token_idx < num_tokens and hidden_idx < hidden:
                        res_smem[t, i, h] = residual[token_idx, i, hidden_idx]
        
        res_local = T.alloc_fragment((token_block, hc, hidden_block), T.float32)
        T.copy(res_smem, res_local)
        
        # Step 2: Add term1 = comb.T @ res
        for t in T.Parallel(token_block):
            for hc_out in T.Parallel(hc):
                for h in T.Parallel(hidden_block):
                    for hc_in in T.serial(hc):
                        out_frag[t, hc_out, h] += comb_smem[t, hc_in, hc_out] * res_local[t, hc_in, h]
        
        # Store output
        out_smem = T.alloc_shared((token_block, hc, hidden_block), T.bfloat16)
        out_bf16 = T.alloc_fragment((token_block, hc, hidden_block), T.bfloat16)
        T.copy(out_frag, out_bf16)
        T.copy(out_bf16, out_smem)
        
        for t in T.Parallel(token_block):
            token_idx = bx * token_block + t
            for i in T.Parallel(hc):
                for h in T.Parallel(hidden_block):
                    hidden_idx = by * hidden_block + h
                    if token_idx < num_tokens and hidden_idx < hidden:
                        output[token_idx, i, hidden_idx] = out_smem[t, i, h]


@tilelang.jit
def mhc_post_optimized_v4_kernel(
    comb_res_mix,    # (n, hc, hc), float32
    residual,        # (n, hc, hidden), bfloat16
    post_layer_mix,  # (n, hc), float32
    x,               # (n, hidden), bfloat16
    output,          # (n, hc, hidden), bfloat16
    hc: int,
    hidden: int,
    hidden_block: int = 512,
) -> tilelang.JITKernel:
    """
    优化版mHC Post v4: 优化数据加载模式
    
    关键优化：
    1. 一次加载comb和post，全局复用
    2. 对residual和x做流水线加载
    3. 减少shared memory到global memory的写回次数
    """
    num_tokens = T.dynamic("num_tokens")
    
    comb_res_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    residual: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    post_layer_mix: T.Tensor((num_tokens, hc), T.float32)
    x: T.Tensor((num_tokens, hidden), T.bfloat16)
    output: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    
    # 每个block处理一个token
    with T.Kernel(num_tokens, threads=256) as token_idx:
        # 加载comb到register (16个float32，只需一次)
        comb_local = T.alloc_fragment((hc, hc), T.float32)
        T.copy(comb_res_mix[token_idx, 0, 0], comb_local)
        
        # 加载post到register (4个float32，只需一次)
        post_local = T.alloc_fragment(hc, T.float32)
        T.copy(post_layer_mix[token_idx, 0], post_local)
        
        # 流水线处理hidden维度
        for h_blk in T.Pipelined(hidden // hidden_block, num_stages=3):
            # 加载res和x
            res_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
            x_smem = T.alloc_shared(hidden_block, T.bfloat16)
            
            T.copy(residual[token_idx, 0, h_blk * hidden_block], res_smem)
            T.copy(x[token_idx, h_blk * hidden_block], x_smem)
            
            res_local = T.alloc_fragment((hc, hidden_block), T.float32)
            x_local = T.alloc_fragment(hidden_block, T.float32)
            T.copy(res_smem, res_local)
            T.copy(x_smem, x_local)
            
            # 计算输出
            out_local = T.alloc_fragment((hc, hidden_block), T.float32)
            
            # 展开hc循环，因为hc=4很小
            # hc_out = 0
            for h in T.Parallel(hidden_block):
                out_local[0, h] = post_local[0] * x_local[h]
                out_local[0, h] += comb_local[0, 0] * res_local[0, h]
                out_local[0, h] += comb_local[1, 0] * res_local[1, h]
                out_local[0, h] += comb_local[2, 0] * res_local[2, h]
                out_local[0, h] += comb_local[3, 0] * res_local[3, h]
            
            # hc_out = 1
            for h in T.Parallel(hidden_block):
                out_local[1, h] = post_local[1] * x_local[h]
                out_local[1, h] += comb_local[0, 1] * res_local[0, h]
                out_local[1, h] += comb_local[1, 1] * res_local[1, h]
                out_local[1, h] += comb_local[2, 1] * res_local[2, h]
                out_local[1, h] += comb_local[3, 1] * res_local[3, h]
            
            # hc_out = 2
            for h in T.Parallel(hidden_block):
                out_local[2, h] = post_local[2] * x_local[h]
                out_local[2, h] += comb_local[0, 2] * res_local[0, h]
                out_local[2, h] += comb_local[1, 2] * res_local[1, h]
                out_local[2, h] += comb_local[2, 2] * res_local[2, h]
                out_local[2, h] += comb_local[3, 2] * res_local[3, h]
            
            # hc_out = 3
            for h in T.Parallel(hidden_block):
                out_local[3, h] = post_local[3] * x_local[h]
                out_local[3, h] += comb_local[0, 3] * res_local[0, h]
                out_local[3, h] += comb_local[1, 3] * res_local[1, h]
                out_local[3, h] += comb_local[2, 3] * res_local[2, h]
                out_local[3, h] += comb_local[3, 3] * res_local[3, h]
            
            # 输出
            out_bf16 = T.alloc_fragment((hc, hidden_block), T.bfloat16)
            T.copy(out_local, out_bf16)
            
            out_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
            T.copy(out_bf16, out_smem)
            T.copy(out_smem, output[token_idx, 0, h_blk * hidden_block])


def mhc_post_optimized_v2(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """v2封装"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    # 确定hidden_block
    hidden_block = 256
    if hidden_size % hidden_block != 0:
        hidden_block = math.gcd(hidden_size, hidden_block)
    
    mhc_post_optimized_v2_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        output,
        hc_mult,
        hidden_size,
        hidden_block,
    )
    
    return output


def mhc_post_optimized_v3(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """v3封装"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    mhc_post_optimized_v3_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        output,
        hc_mult,
        hidden_size,
    )
    
    return output


def mhc_post_optimized_v4(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """v4封装 - 展开hc循环"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    assert hc_mult == 4, "v4 requires hc_mult=4"
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    # 确定hidden_block
    hidden_block = 512
    if hidden_size % hidden_block != 0:
        hidden_block = math.gcd(hidden_size, 512)
    
    mhc_post_optimized_v4_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        output,
        hc_mult,
        hidden_size,
        hidden_block,
    )
    
    return output


# ============ Test ============

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"
    
    num_tokens = 256
    hc_mult = 4
    hidden_size = 5120
    
    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=device)
    
    print("Testing mHC Post optimized versions...")
    
    # Test v2
    try:
        output_v2 = mhc_post_optimized_v2(x, residual, post_layer_mix, comb_res_mix)
        print(f"✅ v2 output shape: {output_v2.shape}")
    except Exception as e:
        print(f"❌ v2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test v3
    try:
        output_v3 = mhc_post_optimized_v3(x, residual, post_layer_mix, comb_res_mix)
        print(f"✅ v3 output shape: {output_v3.shape}")
    except Exception as e:
        print(f"❌ v3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test v4
    try:
        output_v4 = mhc_post_optimized_v4(x, residual, post_layer_mix, comb_res_mix)
        print(f"✅ v4 output shape: {output_v4.shape}")
    except Exception as e:
        print(f"❌ v4 failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nDone!")
