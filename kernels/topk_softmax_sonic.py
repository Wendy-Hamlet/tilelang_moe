"""
TileLang实现：Sonic式TopK+Softmax融合

参考Sonic的核心优化：
1. 先TopK选出最大的k个logit
2. 只对这k个做softmax归一化
3. 单kernel融合，减少内存访问

实现策略：
- 使用共享内存进行TopK（简化版，避免复杂的Bitonic Sort）
- 在寄存器中完成Softmax
- 每个block处理多行
"""

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[1, 2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def topk_softmax_sonic(
    batch_size: int,
    num_experts: int,
    top_k: int,
    block_size: int = 256,
    dtype = T.float32,
):
    """
    Sonic式TopK+Softmax融合kernel
    
    算法流程：
    1. 每个block处理一行(batch)
    2. 所有线程协作找到top-k最大值和索引
    3. 每个线程计算softmax（exp和归一化）
    
    Args:
        batch_size: batch大小
        num_experts: expert数量（N）
        top_k: 选取的top-k数量
        block_size: 线程块大小
        dtype: 数据类型
    
    Memory:
        logits: (batch_size, num_experts) - 输入logits
        scores: (batch_size, top_k) - 输出归一化分数
        indices: (batch_size, top_k) - 输出索引
    """
    
    @T.prim_func
    def main(
        logits: T.Tensor((batch_size, num_experts), dtype),
        scores: T.Tensor((batch_size, top_k), T.float32),
        indices: T.Tensor((batch_size, top_k), T.int32),
    ):
        # 每个block处理一个batch
        with T.Kernel(batch_size, threads=block_size) as bx:
            # 共享内存：存储当前找到的top-k值和索引
            s_topk_vals = T.alloc_shared((top_k,), T.float32)
            s_topk_idx = T.alloc_shared((top_k,), T.int32)
            
            # 线程局部变量
            tx = T.get_thread_binding()
            
            # ============================================================
            # Stage 1: 找Top-K（使用简化的选择排序）
            # ============================================================
            
            # 初始化：每个线程负责的元素范围
            elems_per_thread = T.ceildiv(num_experts, block_size)
            
            # 初始化topk为负无穷
            for i in T.serial(top_k):
                if tx == 0:
                    s_topk_vals[i] = T.float32(-1e38)
                    s_topk_idx[i] = T.int32(-1)
            T.sync_threads()
            
            # 逐个找top-k
            for k_iter in T.serial(top_k):
                # 每个线程找自己负责范围内的最大值
                thread_max_val = T.alloc_var(T.float32)
                thread_max_idx = T.alloc_var(T.int32)
                thread_max_val = T.float32(-1e38)
                thread_max_idx = T.int32(-1)
                
                for i in T.serial(elems_per_thread):
                    idx = tx * elems_per_thread + i
                    if idx < num_experts:
                        val = logits[bx, idx]
                        
                        # 检查这个值是否已经在topk中
                        already_selected = T.alloc_var(T.bool)
                        already_selected = T.bool(False)
                        for j in T.serial(k_iter):
                            if s_topk_idx[j] == idx:
                                already_selected = T.bool(True)
                        
                        # 如果未被选择且大于当前最大值
                        if (not already_selected) and (val > thread_max_val):
                            thread_max_val = val
                            thread_max_idx = T.int32(idx)
                
                # 使用共享内存规约找全局最大值
                s_thread_vals = T.alloc_shared((block_size,), T.float32)
                s_thread_idx = T.alloc_shared((block_size,), T.int32)
                
                s_thread_vals[tx] = thread_max_val
                s_thread_idx[tx] = thread_max_idx
                T.sync_threads()
                
                # 树状规约
                stride = block_size // 2
                while stride > 0:
                    if tx < stride:
                        if s_thread_vals[tx + stride] > s_thread_vals[tx]:
                            s_thread_vals[tx] = s_thread_vals[tx + stride]
                            s_thread_idx[tx] = s_thread_idx[tx + stride]
                    T.sync_threads()
                    stride = stride // 2
                
                # 线程0写入结果
                if tx == 0:
                    s_topk_vals[k_iter] = s_thread_vals[0]
                    s_topk_idx[k_iter] = s_thread_idx[0]
                T.sync_threads()
            
            # ============================================================
            # Stage 2: Softmax归一化（只对top-k）
            # ============================================================
            
            # 找最大值（数值稳定性）
            if tx == 0:
                max_val = T.float32(-1e38)
                for i in T.serial(top_k):
                    if s_topk_vals[i] > max_val:
                        max_val = s_topk_vals[i]
                
                # 计算exp和sum
                exp_sum = T.float32(0.0)
                for i in T.serial(top_k):
                    s_topk_vals[i] = T.exp(s_topk_vals[i] - max_val)
                    exp_sum = exp_sum + s_topk_vals[i]
                
                # 归一化
                for i in T.serial(top_k):
                    s_topk_vals[i] = s_topk_vals[i] / exp_sum
                    
                    # 写回全局内存
                    scores[bx, i] = s_topk_vals[i]
                    indices[bx, i] = s_topk_idx[i]
    
    return main


# 简化版本：使用并行化的堆排序
@tilelang.jit(
    out_idx=[1, 2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def topk_softmax_sonic_v2(
    batch_size: int,
    num_experts: int,
    top_k: int,
    block_size: int = 256,
    dtype = T.float32,
):
    """
    改进版本：更高效的TopK算法
    
    使用局部排序 + 全局归并
    """
    
    @T.prim_func
    def main(
        logits: T.Tensor((batch_size, num_experts), dtype),
        scores: T.Tensor((batch_size, top_k), T.float32),
        indices: T.Tensor((batch_size, top_k), T.int32),
    ):
        with T.Kernel(batch_size, threads=block_size) as bx:
            tx = T.get_thread_binding()
            
            # 共享内存
            SMEM_SIZE = 2048  # 足够存储中间结果
            s_vals = T.alloc_shared((SMEM_SIZE,), T.float32)
            s_idx = T.alloc_shared((SMEM_SIZE,), T.int32)
            
            # ============================================================
            # Stage 1: 每个线程读取自己的数据
            # ============================================================
            elems_per_thread = T.ceildiv(num_experts, block_size)
            
            # 每个线程找自己范围内的top-k
            LOCAL_K = top_k if elems_per_thread >= top_k else elems_per_thread
            
            local_vals = T.alloc_local((LOCAL_K,), T.float32)
            local_idx = T.alloc_local((LOCAL_K,), T.int32)
            
            # 初始化
            for i in T.serial(LOCAL_K):
                local_vals[i] = T.float32(-1e38)
                local_idx[i] = T.int32(-1)
            
            # 读取并维护局部top-k
            for i in T.serial(elems_per_thread):
                idx = tx * elems_per_thread + i
                if idx < num_experts:
                    val = logits[bx, idx]
                    
                    # 插入排序：维护局部top-k
                    for k in T.serial(LOCAL_K):
                        if val > local_vals[k]:
                            # 后移
                            for j in T.serial(LOCAL_K - 1, 0, -1):
                                if j >= k:
                                    if j + 1 < LOCAL_K:
                                        local_vals[j + 1] = local_vals[j]
                                        local_idx[j + 1] = local_idx[j]
                            # 插入
                            local_vals[k] = val
                            local_idx[k] = T.int32(idx)
                            break
            
            # ============================================================
            # Stage 2: 写入共享内存
            # ============================================================
            for i in T.serial(LOCAL_K):
                if tx * LOCAL_K + i < SMEM_SIZE:
                    s_vals[tx * LOCAL_K + i] = local_vals[i]
                    s_idx[tx * LOCAL_K + i] = local_idx[i]
            T.sync_threads()
            
            # ============================================================
            # Stage 3: 归并得到全局top-k
            # ============================================================
            if tx == 0:
                # 简单的归并：从所有局部top-k中选全局top-k
                for k in T.serial(top_k):
                    max_val = T.float32(-1e38)
                    max_pos = T.int32(-1)
                    
                    for i in T.serial(T.min(block_size * LOCAL_K, SMEM_SIZE)):
                        if s_vals[i] > max_val:
                            max_val = s_vals[i]
                            max_pos = T.int32(i)
                    
                    # 记录结果
                    local_vals[k] = max_val
                    local_idx[k] = s_idx[max_pos]
                    
                    # 标记已选择
                    s_vals[max_pos] = T.float32(-1e38)
                
                # ============================================================
                # Stage 4: Softmax
                # ============================================================
                max_val = T.float32(-1e38)
                for i in T.serial(top_k):
                    if local_vals[i] > max_val:
                        max_val = local_vals[i]
                
                exp_sum = T.float32(0.0)
                for i in T.serial(top_k):
                    local_vals[i] = T.exp(local_vals[i] - max_val)
                    exp_sum = exp_sum + local_vals[i]
                
                for i in T.serial(top_k):
                    scores[bx, i] = local_vals[i] / exp_sum
                    indices[bx, i] = local_idx[i]
    
    return main


if __name__ == "__main__":
    import torch
    
    print("TileLang Sonic TopK+Softmax测试")
    print("=" * 80)
    
    # 测试配置
    batch_size = 64
    num_experts = 256
    top_k = 8
    
    # 准备数据
    torch.manual_seed(42)
    logits = torch.randn(batch_size, num_experts, device='cuda', dtype=torch.float32)
    
    # 编译kernel
    print("编译kernel...")
    kernel = topk_softmax_sonic(batch_size, num_experts, top_k)
    
    # 分配输出
    scores = torch.zeros(batch_size, top_k, device='cuda', dtype=torch.float32)
    indices = torch.zeros(batch_size, top_k, device='cuda', dtype=torch.int32)
    
    # 运行kernel
    print("运行kernel...")
    kernel(logits, scores, indices)
    
    # 验证正确性
    print("\n验证正确性...")
    ref_topk = torch.topk(logits, top_k, dim=-1)
    ref_scores = torch.softmax(ref_topk.values, dim=-1, dtype=torch.float32)
    ref_indices = ref_topk.indices.to(torch.int32)
    
    # 检查索引
    indices_match = (indices == ref_indices).sum() / indices.numel()
    print(f"索引匹配率: {indices_match * 100:.2f}%")
    
    # 检查分数
    score_diff = (scores - ref_scores).abs().mean()
    print(f"分数平均差异: {score_diff:.2e}")
    
    # 检查归一化
    sum_check = scores.sum(dim=-1)
    print(f"归一化检查: min={sum_check.min():.6f}, max={sum_check.max():.6f}")
    
    print("=" * 80)
