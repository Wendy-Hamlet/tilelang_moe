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

v2-v4: 反向优化，比官方慢
v5: 2D Grid并行化 (token, hidden)
v6: 更大的hidden_block=1024 匹配官方
v7: 直接register计算，减少shared memory
"""

import math
import torch
import tilelang
import tilelang.language as T


# ============ v5: 2D Grid并行化 ============

@tilelang.jit
def mhc_post_optimized_v5_kernel(
    comb_res_mix,    # (n, hc, hc), float32
    residual,        # (n, hc, hidden), bfloat16
    post_layer_mix,  # (n, hc), float32
    x,               # (n, hidden), bfloat16
    output,          # (n, hc, hidden), bfloat16
    hc: int,
    hidden: int,
    hidden_block: int = 1024,
) -> tilelang.JITKernel:
    """
    v5: 2D Grid并行化
    Grid: (num_tokens, hidden // hidden_block)
    每个block处理一个token的一部分hidden
    """
    num_tokens = T.dynamic("num_tokens")
    
    comb_res_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    residual: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    post_layer_mix: T.Tensor((num_tokens, hc), T.float32)
    x: T.Tensor((num_tokens, hidden), T.bfloat16)
    output: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    
    # 2D Grid: (token_idx, h_blk_idx)
    with T.Kernel(num_tokens, hidden // hidden_block, threads=256) as (token_idx, h_blk_idx):
        h_offset = h_blk_idx * hidden_block
        
        # 加载comb_res_mix (小数据，每个block独立加载)
        comb_local = T.alloc_fragment((hc, hc), T.float32)
        T.copy(comb_res_mix[token_idx, 0, 0], comb_local)
        
        # 加载post_layer_mix
        post_local = T.alloc_fragment(hc, T.float32)
        T.copy(post_layer_mix[token_idx, 0], post_local)
        
        # 加载residual和x
        res_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
        T.copy(residual[token_idx, 0, h_offset], res_smem)
        
        x_smem = T.alloc_shared(hidden_block, T.bfloat16)
        T.copy(x[token_idx, h_offset], x_smem)
        
        res_local = T.alloc_fragment((hc, hidden_block), T.float32)
        T.copy(res_smem, res_local)
        
        x_local = T.alloc_fragment(hidden_block, T.float32)
        T.copy(x_smem, x_local)
        
        # 计算
        out_local = T.alloc_fragment((hc, hidden_block), T.float32)
        
        for hc_out in T.serial(hc):
            post_val = post_local[hc_out]
            for h in T.Parallel(hidden_block):
                out_local[hc_out, h] = post_val * x_local[h]
                for hc_in in T.serial(hc):
                    out_local[hc_out, h] += comb_local[hc_in, hc_out] * res_local[hc_in, h]
        
        # 输出
        out_bf16 = T.alloc_fragment((hc, hidden_block), T.bfloat16)
        T.copy(out_local, out_bf16)
        
        out_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
        T.copy(out_bf16, out_smem)
        T.copy(out_smem, output[token_idx, 0, h_offset])


# ============ v6: 匹配官方的h_blk=1024 ============

@tilelang.jit
def mhc_post_optimized_v6_kernel(
    comb_res_mix,    # (n, hc, hc), float32
    residual,        # (n, hc, hidden), bfloat16
    post_layer_mix,  # (n, hc), float32
    x,               # (n, hidden), bfloat16
    output,          # (n, hc, hidden), bfloat16
    hc: int,
    hidden: int,
    hidden_block: int = 1024,
) -> tilelang.JITKernel:
    """
    v6: 使用与官方相同的h_blk=1024，但优化计算顺序
    先计算outer product，再累加矩阵乘法
    """
    num_tokens = T.dynamic("num_tokens")
    
    comb_res_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    residual: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    post_layer_mix: T.Tensor((num_tokens, hc), T.float32)
    x: T.Tensor((num_tokens, hidden), T.bfloat16)
    output: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    
    with T.Kernel(num_tokens, threads=256) as token_idx:
        # 加载小数据到register
        comb_local = T.alloc_fragment((hc, hc), T.float32)
        T.copy(comb_res_mix[token_idx, 0, 0], comb_local)
        
        post_local = T.alloc_fragment(hc, T.float32)
        T.copy(post_layer_mix[token_idx, 0], post_local)
        
        # 流水线处理hidden
        for h_blk in T.Pipelined(hidden // hidden_block, num_stages=2):
            res_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
            x_smem = T.alloc_shared(hidden_block, T.bfloat16)
            
            T.copy(residual[token_idx, 0, h_blk * hidden_block], res_smem)
            T.copy(x[token_idx, h_blk * hidden_block], x_smem)
            
            res_local = T.alloc_fragment((hc, hidden_block), T.float32)
            x_local = T.alloc_fragment(hidden_block, T.float32)
            T.copy(res_smem, res_local)
            T.copy(x_smem, x_local)
            
            out_local = T.alloc_fragment((hc, hidden_block), T.float32)
            
            # 先计算outer product (post * x)
            for hc_out in T.serial(hc):
                post_val = post_local[hc_out]
                for h in T.Parallel(hidden_block):
                    out_local[hc_out, h] = post_val * x_local[h]
            
            # 再累加矩阵乘法 (comb.T @ res)
            for hc_in in T.serial(hc):
                for hc_out in T.serial(hc):
                    comb_val = comb_local[hc_in, hc_out]
                    for h in T.Parallel(hidden_block):
                        out_local[hc_out, h] += comb_val * res_local[hc_in, h]
            
            out_bf16 = T.alloc_fragment((hc, hidden_block), T.bfloat16)
            T.copy(out_local, out_bf16)
            
            out_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
            T.copy(out_bf16, out_smem)
            T.copy(out_smem, output[token_idx, 0, h_blk * hidden_block])


# ============ v7: 优化内存访问模式 ============

@tilelang.jit
def mhc_post_optimized_v7_kernel(
    comb_res_mix,    # (n, hc, hc), float32
    residual,        # (n, hc, hidden), bfloat16
    post_layer_mix,  # (n, hc), float32
    x,               # (n, hidden), bfloat16
    output,          # (n, hc, hidden), bfloat16
    hc: int,
    hidden: int,
    hidden_block: int = 1024,
) -> tilelang.JITKernel:
    """
    v7: 优化循环结构 - 将hc_out放在最外层
    这样每个hc_out的输出可以连续写入
    """
    num_tokens = T.dynamic("num_tokens")
    
    comb_res_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    residual: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    post_layer_mix: T.Tensor((num_tokens, hc), T.float32)
    x: T.Tensor((num_tokens, hidden), T.bfloat16)
    output: T.Tensor((num_tokens, hc, hidden), T.bfloat16)
    
    with T.Kernel(num_tokens, threads=256) as token_idx:
        comb_local = T.alloc_fragment((hc, hc), T.float32)
        T.copy(comb_res_mix[token_idx, 0, 0], comb_local)
        
        post_local = T.alloc_fragment(hc, T.float32)
        T.copy(post_layer_mix[token_idx, 0], post_local)
        
        for h_blk in T.Pipelined(hidden // hidden_block, num_stages=2):
            res_smem = T.alloc_shared((hc, hidden_block), T.bfloat16)
            x_smem = T.alloc_shared(hidden_block, T.bfloat16)
            
            T.copy(residual[token_idx, 0, h_blk * hidden_block], res_smem)
            T.copy(x[token_idx, h_blk * hidden_block], x_smem)
            
            res_local = T.alloc_fragment((hc, hidden_block), T.float32)
            x_local = T.alloc_fragment(hidden_block, T.float32)
            T.copy(res_smem, res_local)
            T.copy(x_smem, x_local)
            
            # 每次处理一个hc_out，完成后立即写出
            for hc_out in T.serial(hc):
                out_row = T.alloc_fragment(hidden_block, T.float32)
                post_val = post_local[hc_out]
                
                # outer product
                for h in T.Parallel(hidden_block):
                    out_row[h] = post_val * x_local[h]
                
                # matrix multiply
                for hc_in in T.serial(hc):
                    comb_val = comb_local[hc_in, hc_out]
                    for h in T.Parallel(hidden_block):
                        out_row[h] += comb_val * res_local[hc_in, h]
                
                # 写出这一行
                out_row_bf16 = T.alloc_fragment(hidden_block, T.bfloat16)
                T.copy(out_row, out_row_bf16)
                
                out_row_smem = T.alloc_shared(hidden_block, T.bfloat16)
                T.copy(out_row_bf16, out_row_smem)
                T.copy(out_row_smem, output[token_idx, hc_out, h_blk * hidden_block])


# ============ Wrapper Functions ============

def mhc_post_optimized_v5(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """v5封装 - 2D Grid"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    hidden_block = 1024
    if hidden_size % hidden_block != 0:
        hidden_block = math.gcd(hidden_size, 1024)
    
    mhc_post_optimized_v5_kernel(
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


def mhc_post_optimized_v6(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """v6封装 - h_blk=1024"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    hidden_block = 1024
    if hidden_size % hidden_block != 0:
        hidden_block = math.gcd(hidden_size, 1024)
    
    mhc_post_optimized_v6_kernel(
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


def mhc_post_optimized_v7(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """v7封装 - 行级写出"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    hidden_block = 1024
    if hidden_size % hidden_block != 0:
        hidden_block = math.gcd(hidden_size, 1024)
    
    mhc_post_optimized_v7_kernel(
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


# ============ Legacy v2-v4 (反向优化，保留供参考) ============

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
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    """
    优化版mHC Post v3: 更大的hidden_block + 3-stage流水线
    
    与v2区别: hidden_block更大，流水线级数更多
    每个block处理一个token
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
        
        # 3-stage流水线处理hidden维度
        for h_blk in T.Pipelined(hidden // hidden_block, num_stages=3):
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
    优化版mHC Post v4: 更大的hidden_block (512)
    
    关键优化：
    1. 一次加载comb和post，全局复用
    2. 对residual和x做流水线加载
    3. hidden_block=512提高并行度
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
            
            # 计算输出 - 使用serial循环避免TileLang索引检查问题
            out_local = T.alloc_fragment((hc, hidden_block), T.float32)
            
            for hc_out in T.serial(hc):
                post_val = post_local[hc_out]
                for h in T.Parallel(hidden_block):
                    # term2: outer product
                    out_local[hc_out, h] = post_val * x_local[h]
                    # term1: matrix multiply (展开内层循环)
                    for hc_in in T.serial(hc):
                        out_local[hc_out, h] += comb_local[hc_in, hc_out] * res_local[hc_in, h]
            
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
    """v3封装 - 3-stage流水线"""
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    num_tokens = residual.shape[0]
    device = residual.device
    
    output = torch.empty(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    
    # 确定hidden_block
    hidden_block = 256
    if hidden_size % hidden_block != 0:
        hidden_block = math.gcd(hidden_size, hidden_block)
    
    mhc_post_optimized_v3_kernel(
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
