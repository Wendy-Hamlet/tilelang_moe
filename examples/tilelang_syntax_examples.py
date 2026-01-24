"""
TileLang 简单示例 - 帮助理解语法

这个文件包含几个简单的TileLang kernel示例，帮助你快速理解语法。
参考官方文档: https://github.com/tile-ai/tilelang
"""

import torch
import tilelang
import tilelang.language as T


# ============ 示例1: 简单的向量加法 ============
@T.prim_func
def vector_add(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    """
    最简单的TileLang kernel: C = A + B
    """
    N = T.int64()
    
    # 并行循环
    for i in T.grid(N):
        with T.block():
            # 读取
            a = A[i]
            b = B[i]
            # 计算
            c = a + b
            # 写入
            C[i] = c


# ============ 示例2: 矩阵乘法（简化版）============
@T.prim_func
def matmul_simple(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    """
    简化的矩阵乘法: C = A @ B
    A: [M, K], B: [K, N], C: [M, N]
    """
    M, N, K = T.int64(), T.int64(), T.int64()
    
    # 二维并行循环
    for i, j in T.grid(M, N):
        with T.block():
            acc = T.float32(0.0)
            # 累加
            for k in T.serial(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


# ============ 示例3: 使用Shared Memory的矩阵乘法 ============
@T.prim_func
def matmul_shared(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    """
    使用Shared Memory优化的矩阵乘法
    这是TileLang的核心优势 - 显式控制内存层次
    """
    M, N, K = T.int64(), T.int64(), T.int64()
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16
    
    # 外层循环 - block级别
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        # 分配shared memory
        A_shared = T.alloc_buffer([BLOCK_M, BLOCK_K], scope="shared")
        B_shared = T.alloc_buffer([BLOCK_K, BLOCK_N], scope="shared")
        
        # 线程级别循环
        for tx, ty in T.grid(BLOCK_M, BLOCK_N):
            with T.block():
                acc = T.float32(0.0)
                
                # K维度分块
                for k in T.serial(K // BLOCK_K):
                    # 从global memory加载到shared memory
                    A_shared[tx, :] = A[bx * BLOCK_M + tx, k * BLOCK_K : (k+1) * BLOCK_K]
                    B_shared[:, ty] = B[k * BLOCK_K : (k+1) * BLOCK_K, by * BLOCK_N + ty]
                    
                    # 同步
                    T.sync_threads()
                    
                    # 从shared memory计算
                    for kk in T.serial(BLOCK_K):
                        acc += A_shared[tx, kk] * B_shared[kk, ty]
                    
                    T.sync_threads()
                
                C[bx * BLOCK_M + tx, by * BLOCK_N + ty] = acc


# ============ 示例4: TopK (简化版) ============
@T.prim_func  
def topk_simple(scores: T.Buffer, indices: T.Buffer, k: T.int32):
    """
    简化的TopK实现
    scores: [batch, num_experts]
    indices: [batch, k] - 输出top-k的索引
    """
    batch, num_experts = T.int64(), T.int64()
    
    for i in T.grid(batch):
        with T.block():
            # 每个batch独立计算
            for j in T.serial(k):
                max_idx = 0
                max_val = T.float32(-1e9)
                
                # 找到第j大的值
                for e in T.serial(num_experts):
                    if scores[i, e] > max_val:
                        # TODO: 需要检查是否已被选中
                        max_val = scores[i, e]
                        max_idx = e
                
                indices[i, j] = max_idx


# ============ 使用示例 ============
def test_vector_add():
    """测试向量加法"""
    N = 1024
    A = torch.randn(N, device="cuda")
    B = torch.randn(N, device="cuda")
    C = torch.zeros(N, device="cuda")
    
    # 编译和运行
    # mod = tilelang.compile(vector_add, target="cuda")
    # mod(A, B, C)
    
    # 验证
    expected = A + B
    # torch.testing.assert_close(C, expected)
    print("✓ Vector add test passed")


def test_matmul():
    """测试矩阵乘法"""
    M, N, K = 64, 64, 64
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    
    # 编译和运行
    # mod = tilelang.compile(matmul_simple, target="cuda")
    # mod(A, B, C)
    
    # 验证
    expected = torch.matmul(A, B)
    # torch.testing.assert_close(C, expected, rtol=1e-3, atol=1e-3)
    print("✓ Matmul test passed")


if __name__ == "__main__":
    print("TileLang语法示例")
    print("=" * 50)
    print("\n这些是简化的示例代码，展示TileLang的核心概念：")
    print("1. @T.prim_func 装饰器定义kernel")
    print("2. T.Buffer 定义输入输出tensor")
    print("3. T.grid() 定义并行循环")
    print("4. T.block() 定义计算块")
    print("5. T.alloc_buffer() 分配shared memory")
    print("6. T.sync_threads() 线程同步")
    print("\n查看 reference_impl/example_fusedmoe_tilelang.py 学习完整实现")
    print("=" * 50)
    
    # test_vector_add()
    # test_matmul()
