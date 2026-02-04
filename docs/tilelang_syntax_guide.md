# TileLang 语法指南 (完整版)

> 本文档基于对 TileLang 库源代码的深入分析整理，涵盖了库的全部语法特性。

## 目录

1. [概述](#1-概述)
2. [核心语法结构](#2-核心语法结构)
3. [数据类型系统](#3-数据类型系统)
4. [内存管理](#4-内存管理)
5. [数据移动](#5-数据移动)
6. [计算原语](#6-计算原语)
7. [并行与循环控制](#7-并行与循环控制)
8. [线程同步与原子操作](#8-线程同步与原子操作)
9. [数学内置函数](#9-数学内置函数)
10. [高级特性](#10-高级特性)
11. [调试工具](#11-调试工具)
12. [完整示例](#12-完整示例)
13. [API 速查表](#13-api-速查表)

---

## 1. 概述

**TileLang** (tile-lang) 是一个简洁的领域特定语言 (DSL)，专为开发高性能 GPU/CPU kernel 设计。它基于 TVM 编译器基础设施，提供 Pythonic 语法。

### 1.1 核心特点

- **Pythonic 语法**: 使用 Python 装饰器和语法糖
- **自动优化**: 自动启用 TMA (Tensor Memory Accelerator) 和 Warp Specialization
- **多后端支持**: CUDA, HIP (AMD), CPU, Metal (Apple), Ascend, WebGPU
- **Tile 级编程**: 以 tile/block 为单位的并行计算抽象

### 1.2 基本导入

```python
import tilelang
import tilelang.language as T
```

---

## 2. 核心语法结构

### 2.1 Kernel 定义 - `@tilelang.jit`

```python
@tilelang.jit
def my_kernel(M, N, K, block_M=128, block_N=128, block_K=32, dtype=T.float16):
    @T.prim_func
    def kernel_impl(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # kernel 实现
        ...
    return kernel_impl
```

#### JIT 装饰器参数

| 参数 | 说明 |
|------|------|
| `target` | 目标平台: `"cuda"`, `"hip"`, `"cpu"` (默认自动推断) |
| `out_idx` | 输出张量的索引列表，如 `[2]` 或 `[-1]` |
| `pass_configs` | 编译器配置字典 |

#### Pass Configs 完整选项

```python
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,     # 禁用 Warp Specialization
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,            # 禁用 TMA
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,  # 禁用线程存储同步
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,     # 寄存器使用级别
    }
)
```

### 2.2 Kernel 上下文 - `T.Kernel`

```python
# 基本用法
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    tx = T.get_thread_binding()
    ...

# 完整参数
with T.Kernel(
    *blocks,              # Grid 维度 (1-3个整数)
    threads=128,          # 每个 block 的线程数 (int 或 list)
    is_cpu=False,         # 是否为 CPU kernel
    prelude=None,         # 导入的 C 代码
) as (bx, by):
    ...

# 2D 线程块
with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
    tx, ty = T.get_thread_bindings()
    ...

# CPU kernel
with T.Kernel(loop_extent, is_cpu=True) as i:
    ...
```

#### 线程/Block 信息获取

```python
tx = T.get_thread_binding(dim=0)      # threadIdx.x
tx, ty, tz = T.get_thread_bindings()  # 所有 threadIdx

bx = T.get_block_binding(dim=0)       # blockIdx.x
bx, by, bz = T.get_block_bindings()   # 所有 blockIdx

# 获取 extent
thread_extent = T.get_thread_extent(dim=0)
block_extent = T.get_block_extent(dim=0)
```

---

## 3. 数据类型系统

### 3.1 标量类型

| 类型 | 说明 |
|------|------|
| `T.float16` / `T.float` / `T.float32` / `T.float64` | 浮点类型 |
| `T.bfloat16` | Brain Float 16 |
| `T.int8` / `T.int16` / `T.int32` / `T.int64` | 有符号整数 |
| `T.uint8` / `T.uint16` / `T.uint32` / `T.uint64` | 无符号整数 |
| `T.bool` | 布尔类型 |

### 3.2 张量声明

```python
# 静态形状
A: T.Tensor((M, K), T.float16)

# 动态形状
batch = T.dynamic("batch")
seq_len = T.dynamic("seq_len")
X: T.Tensor((batch, seq_len, hidden), T.float32)

# 符号形状
n = T.symbolic("n")
```

---

## 4. 内存管理

### 4.1 完整内存层次

| API | 内存类型 | Scope | 说明 |
|-----|----------|-------|------|
| `T.alloc_shared` | Shared Memory | `shared.dyn` | Block 内线程共享 |
| `T.alloc_fragment` | Fragment | `local.fragment` | Tensor Core 友好的线程私有存储 |
| `T.alloc_local` | Local Memory | `local` | 线程本地数组 |
| `T.alloc_var` | Register | `local.var` | 标量变量 |
| `T.alloc_barrier` | Barrier | `shared.barrier` | 同步屏障 |
| `T.alloc_tmem` | Tensor Memory | `shared.tmem` | Hopper+ TMEM |
| `T.alloc_reducer` | Reducer | `local.fragment` | 并行归约缓冲区 |
| `T.alloc_descriptor` | Descriptor | `local.descriptor.*` | WGMMA/TCGEN5 描述符 |

### 4.2 详细分配示例

```python
with T.Kernel(...) as (bx, by):
    # Shared Memory
    A_shared = T.alloc_shared((block_M, block_K), T.float16)
    B_shared = T.alloc_shared((block_K, block_N), T.float16, scope="shared.dyn")
    
    # Fragment (用于 Tensor Core)
    C_local = T.alloc_fragment((block_M, block_N), T.float32)
    
    # Local Memory
    buffer = T.alloc_local((16,), T.float32)
    
    # 标量变量 (多种初始化方式)
    counter = T.alloc_var(T.int32)                    # 未初始化
    max_val = T.alloc_var(T.float32, 0.0)             # 初始化为 0.0
    idx = T.alloc_var(T.int32, 1, "local.var")        # 初始化 + scope
    
    # 同步屏障
    barrier = T.alloc_barrier(arrive_count=128)
    
    # Tensor Memory (Hopper+)
    tmem_buf = T.alloc_tmem((128, 64), T.float16)
    
    # Reducer (用于并行归约)
    sum_reducer = T.alloc_reducer((16,), T.float32, op="sum")
    max_reducer = T.alloc_reducer((16,), T.float32, op="max", replication="all")
    
    # WGMMA 描述符
    desc = T.alloc_wgmma_desc()
    tcgen_desc = T.alloc_tcgen05_smem_desc()
```

### 4.3 输出张量声明

```python
# 在 kernel 内部声明输出
output = T.empty((M, N), dtype=T.float32)
```

---

## 5. 数据移动

### 5.1 T.copy - 通用拷贝

```python
# 基本用法
T.copy(src, dst)

# 完整参数
T.copy(
    src,                                    # 源: Buffer/BufferLoad/BufferRegion
    dst,                                    # 目标: Buffer/BufferLoad/BufferRegion
    coalesced_width=None,                   # 合并访问宽度
    disable_tma=False,                      # 禁用 TMA 加速
    eviction_policy=None,                   # 缓存驱逐策略: "evict_normal"/"evict_first"/"evict_last"
    loop_layout=None,                       # 并行循环 layout hint
)

# 示例
T.copy(A[by * block_M, ko * block_K], A_shared)  # Global → Shared
T.copy(A_shared, A_frag)                          # Shared → Fragment
T.copy(C_local, C[by * block_M, bx * block_N])   # Fragment → Global

# 带驱逐策略
T.copy(A[...], A_shared, eviction_policy="evict_first")
```

### 5.2 Im2Col 变换 (卷积用)

```python
T.c2d_im2col(
    img,              # 输入图像 buffer
    col,              # 输出 column buffer
    nhw_step,         # batch 和空间维度步长
    c_step,           # channel 步长
    kernel=3,         # 卷积核大小
    stride=1,         # 步长
    dilation=1,       # 膨胀率
    pad=0,            # padding
    eviction_policy=None,
)
```

---

## 6. 计算原语

### 6.1 GEMM - 矩阵乘法

```python
# 基本用法: C += A @ B
T.gemm(A_shared, B_shared, C_local)

# 完整参数
T.gemm(
    A, B, C,
    transpose_A=False,                      # 是否转置 A
    transpose_B=False,                      # 是否转置 B
    policy=T.GemmWarpPolicy.Square,         # Warp 分区策略
    clear_accum=False,                      # 是否清零累加器
    k_pack=1,                               # ROCm 专用: packed matrix cores
    wg_wait=0,                              # Warp Group 等待标识
    mbar=None,                              # Blackwell mbarrier
)

# 可用版本
T.gemm_v1(...)  # 使用 tl.gemm op
T.gemm_v2(...)  # 使用 tl.gemm_py op (更快编译)
```

### 6.2 稀疏 GEMM (2:4 Sparse)

```python
T.gemm_sp(A, B, C, ...)      # 2:4 稀疏 Tensor Core
T.gemm_sp_v2(A, B, C, ...)
```

### 6.3 Reduction 操作

```python
# 求和
T.reduce_sum(src, dst, dim=-1, clear=True)

# 最大值
T.reduce_max(src, dst, dim=-1, clear=True)

# 最小值
T.reduce_min(src, dst, dim=-1, clear=True)

# 绝对值求和
T.reduce_abssum(src, dst, dim=-1)

# 绝对值最大
T.reduce_absmax(src, dst, dim=-1, clear=True)

# 位运算归约
T.reduce_bitand(src, dst, dim=-1, clear=True)
T.reduce_bitor(src, dst, dim=-1, clear=True)
T.reduce_bitxor(src, dst, dim=-1, clear=True)

# Warp 级归约 (标量)
result = T.warp_reduce_sum(value)
result = T.warp_reduce_max(value)
result = T.warp_reduce_min(value)
result = T.warp_reduce_bitand(value)
result = T.warp_reduce_bitor(value)
```

### 6.4 累积求和

```python
# 基本用法 (原地)
T.cumsum(src, dim=0, reverse=False)

# 带输出
T.cumsum(src, dst, dim=0, reverse=False)

# 逆序累积
T.cumsum(buffer, dim=1, reverse=True)
```

### 6.5 初始化操作

```python
T.clear(buffer)           # 清零
T.fill(buffer, value)     # 填充指定值

# Reducer 使用
T.fill(reducer, 0)        # 初始化 reducer
T.finalize_reducer(reducer)  # 完成归约
```

### 6.6 Buffer 操作

```python
# Reshape (不改变数据)
new_buf = T.reshape(buffer, [new_shape])

# View (可选新 dtype)
new_buf = T.view(buffer, shape=[...], dtype=T.float16)

# Clamp
result = T.clamp(value, min_val, max_val)
```

---

## 7. 并行与循环控制

### 7.1 T.Parallel - 并行循环

```python
# 基本用法
for i in T.Parallel(block_M):
    output[i] = input[i] * 2

# 多维并行
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = T.max(C_local[i, j], 0)

# 带选项
for i, j in T.Parallel(M, N, coalesced_width=4, loop_layout=layout):
    ...
```

### 7.2 T.serial / T.Serial - 串行循环

```python
# 基本用法
for k in T.serial(top_k):
    ...

# 带步长
for i in T.serial(0, 100, step=2):
    ...

# 带注解
for i in T.serial(N, annotations={"pragma_unroll": 4}):
    ...
```

### 7.3 T.unroll / T.Unroll - 展开循环

```python
# 基本展开
for i in T.unroll(8):
    ...

# 显式展开
for i in T.unroll(8, explicit=True):
    ...

# 指定展开因子
for i in T.unroll(N, unroll_factor=4):
    ...

# 带步长
for i in T.unroll(0, 16, step=2):
    ...
```

### 7.4 T.Pipelined - 流水线循环

```python
# 基本用法
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[...], A_shared)
    T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_local)

# 完整参数
for k in T.Pipelined(
    start=0,
    stop=K // block_K,
    num_stages=3,           # 流水线阶段数
    order=[...],            # 操作顺序
    stage=[...],            # 阶段分配
    sync=[...],             # 同步点
    group=[...],            # 分组
):
    ...
```

### 7.5 T.Persistent - 持久化循环

```python
for tile_idx in T.Persistent(
    domain=[grid_x, grid_y],
    wave_size=wave_size,
    index=tile_idx,
    group_size=8,
):
    ...
```

### 7.6 T.vectorized / T.Vectorized - 向量化循环

```python
for i in T.vectorized(0, 4):
    ...
```

### 7.7 控制流

```python
# 条件分支
if condition:
    ...
else:
    ...

# 三元表达式
value = T.if_then_else(condition, true_value, false_value)

# 循环中断
for i in T.serial(max_iter):
    if done:
        T.loop_break()
```

---

## 8. 线程同步与原子操作

### 8.1 线程同步

```python
T.sync_threads()              # Block 内所有线程同步
T.sync_threads(scope, count)  # 部分同步
```

### 8.2 原子操作 (完整)

```python
# 原子加法
T.atomic_add(dst, value)
T.atomic_add(dst, value, memory_order="relaxed")
prev = T.atomic_add(dst, value, return_prev=True)
T.atomic_add(dst, value, use_tma=True)  # 使用 TMA (sm90+)

# 向量原子加法
T.atomic_addx2(dst, value)  # 双宽度 (FP16 pairs)
T.atomic_addx4(dst, value)  # 四宽度 (float4)

# 原子最大/最小
T.atomic_max(dst, value, memory_order="acquire", return_prev=False)
T.atomic_min(dst, value, memory_order="release", return_prev=False)

# 原子加载/存储
value = T.atomic_load(src, memory_order="seq_cst")
T.atomic_store(dst, value, memory_order="release")

# Memory Order 选项
# "relaxed", "consume", "acquire", "release", "acq_rel", "seq_cst"
```

---

## 9. 数学内置函数

### 9.1 标准数学函数

| 函数 | 说明 |
|------|------|
| `T.exp(x)` | 自然指数 |
| `T.exp2(x)` | 2^x |
| `T.log(x)` | 自然对数 |
| `T.log2(x)` | log₂(x) |
| `T.sqrt(x)` | 平方根 |
| `T.rsqrt(x)` | 1/√x |
| `T.abs(x)` | 绝对值 |
| `T.max(a, b)` | 最大值 |
| `T.min(a, b)` | 最小值 |
| `T.sigmoid(x)` | Sigmoid 函数 |
| `T.sin(x)`, `T.cos(x)`, `T.tan(x)` | 三角函数 |

### 9.2 快速数学函数 (Fast Math)

```python
T.__log(x)    # 快速 log
T.__log2(x)   # 快速 log2
T.__log10(x)  # 快速 log10
T.__exp(x)    # 快速 exp
T.__exp10(x)  # 快速 10^x
T.__sin(x)    # 快速 sin
T.__cos(x)    # 快速 cos
T.__tan(x)    # 快速 tan
```

### 9.3 IEEE 精确运算 (带舍入模式)

```python
# 舍入模式: "rn" (最近), "rz" (向零), "ru" (向上), "rd" (向下)

T.ieee_add(x, y, rounding_mode="rn")
T.ieee_sub(x, y, rounding_mode="rn")
T.ieee_mul(x, y, rounding_mode="rn")
T.ieee_fdiv(x, y, rounding_mode="rn")
T.ieee_fmaf(x, y, z, rounding_mode="rn")  # x*y + z
T.ieee_frcp(x, rounding_mode="rn")         # 1/x
T.ieee_fsqrt(x, rounding_mode="rn")        # sqrt(x)
T.ieee_frsqrt(x)                           # 1/sqrt(x) (仅 rn)
```

### 9.4 类型转换与特殊值

```python
# 类型转换
result = T.Cast(T.int32, float_value)
result = T.Cast(T.float16, int_value)

# 位重解释
uint_bits = T.reinterpret(T.uint32, float_value)

# 特殊值
T.infinity(T.float32)     # 正无穷
-T.infinity(T.float32)    # 负无穷
T.max_value(T.float32)    # 最大值
T.min_value(T.float32)    # 最小值
```

---

## 10. 高级特性

### 10.1 Layout 优化

```python
# Swizzled Layout - 减少 bank conflict
T.annotate_layout({
    A_shared: tilelang.layout.make_swizzled_layout(A_shared),
    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
})

# 自定义 Layout
from tilelang.layout import Layout, Fragment
custom_layout = Layout(shape, lambda i, j: ...)
T.annotate_layout({buffer: custom_layout})

# Fragment Layout (用于 Tensor Core)
frag_layout = Fragment(...)
T.annotate_layout({C_local: frag_layout})

# L2 Cache Swizzle
T.use_swizzle(panel_size=10, order="row", enable=True)

# L2 Hit Ratio 提示
T.annotate_l2_hit_ratio({global_buffer: 0.8})

# Safe Value 注解
T.annotate_safe_value({buffer: 0.0})

# Restrict 缓冲区别名标记
T.annotate_restrict_buffers(buffer_a, buffer_b)
```

### 10.2 Warp Specialization

```python
# 分配不同 warp 组执行不同代码
with T.ws(0):  # Warp group 0 (threads 0-127)
    # Producer 代码
    ...

with T.ws(1):  # Warp group 1 (threads 128-255)
    # Consumer 代码
    ...

# 多 warp 组
with T.ws(0, 1):  # Warp group 0 和 1
    ...
```

### 10.3 随机数生成

```python
# 初始化 RNG
T.rng_init(
    seed=42,
    seq=None,              # 默认使用 thread_id
    off=0,
    generator="curandStatePhilox4_32_10_t"  # 或 "curandStateMRG32k3a_t", "curandStateXORWOW_t"
)

# 生成随机数
rand_uint = T.rng_rand()                        # uint32
rand_float = T.rng_rand_float(bit=32, dist="uniform")   # float32 uniform
rand_normal = T.rng_rand_float(bit=64, dist="normal")   # float64 normal
```

### 10.4 逻辑操作

```python
# 检查是否存在 true 元素
has_any = T.any_of(buffer)
has_any = T.any_of(buffer[start:end])

# 检查是否全为 true
all_true = T.all_of(buffer)
all_true = T.all_of(buffer[start:end])
```

### 10.5 Macro 定义

```python
@T.macro
def my_reusable_pattern(input_buf: T.Buffer, output_buf: T.Buffer):
    """可复用的宏"""
    T.copy(input_buf, output_buf)
    for i in T.Parallel(input_buf.shape[0]):
        output_buf[i] = output_buf[i] * 2
```

### 10.6 动态形状

```python
@tilelang.jit
def dynamic_kernel():
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    
    @T.prim_func
    def kernel(
        input: T.Tensor((batch, seq_len), T.float32),
        output: T.Tensor((batch, seq_len), T.float32),
    ):
        ...
```

### 10.7 DP4A 指令

```python
# 4-element dot product with accumulation
T.dp4a(A, B, C)  # int8 dot product
```

---

## 11. 调试工具

### 11.1 打印

```python
# 打印变量
T.print(value, msg="debug info")

# 打印 buffer
T.print(buffer, msg="buffer contents", warp_group_id=0, warp_id=0)

# 支持的 buffer scope: local, local.fragment, shared, shared.dyn, global
```

### 11.2 设备断言

```python
# 基本断言
T.device_assert(condition, "Error message")

# 无堆栈信息 (更快)
T.device_assert(condition, "Error", no_stack_info=True)
```

---

## 12. 完整示例

### 12.1 GEMM + ReLU + Swizzle

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def matmul_relu(M, N, K, block_M=128, block_N=128, block_K=32, dtype=T.float16):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            
            # Layout 优化
            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })
            T.use_swizzle(10)
            
            T.clear(C_local)
            
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            # ReLU
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return kernel
```

### 12.2 Online Softmax (完整)

```python
@tilelang.jit(out_idx=[1])
def softmax_kernel(M: int, N: int, dtype=T.float32):
    BN = min(tilelang.next_power_of_2(N), 8192)
    NN = tilelang.cdiv(N, BN)
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(X: T.Tensor([M, N], dtype), Y: T.Tensor([M, N], dtype)):
        with T.Kernel(M, threads=128) as i_m:
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], T.float32)
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], T.float32)
            sum_exp = T.alloc_fragment([1], T.float32)
            
            T.fill(lse, -T.infinity(T.float32))

            # Pass 1: log-sum-exp
            for i_n in T.Pipelined(NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                T.reduce_max(x, max_x, dim=0, clear=True)
                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)
                T.reduce_sum(exp_x, sum_exp, dim=0, clear=True)
                lse[0] = max_x[0] * scale + T.log2(
                    T.exp2(lse[0] - max_x[0] * scale) + sum_exp[0]
                )

            # Pass 2: softmax
            for i_n in T.Pipelined(NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])
                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main
```

### 12.3 带 Warp Specialization 的 Producer-Consumer

```python
@tilelang.jit
def producer_consumer_kernel(N, block_size=128):
    @T.prim_func
    def kernel(
        Input: T.Tensor((N,), T.float32),
        Output: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=256) as bx:
            shared_buf = T.alloc_shared((block_size,), T.float32)
            
            with T.ws(0):  # Producer: Warp group 0
                T.copy(Input[bx * block_size], shared_buf)
            
            T.sync_threads()
            
            with T.ws(1):  # Consumer: Warp group 1
                for i in T.Parallel(block_size):
                    shared_buf[i] = shared_buf[i] * 2
            
            T.sync_threads()
            T.copy(shared_buf, Output[bx * block_size])
    
    return kernel
```

---

## 13. API 速查表

### 内存分配

| API | Scope | 说明 |
|-----|-------|------|
| `T.alloc_shared(shape, dtype)` | `shared.dyn` | Shared memory |
| `T.alloc_fragment(shape, dtype)` | `local.fragment` | Tensor Core friendly |
| `T.alloc_local(shape, dtype)` | `local` | Local memory |
| `T.alloc_var(dtype, init)` | `local.var` | 标量变量 |
| `T.alloc_barrier(count)` | `shared.barrier` | 屏障 |
| `T.alloc_tmem(shape, dtype)` | `shared.tmem` | Tensor Memory |
| `T.alloc_reducer(shape, dtype, op)` | `local.fragment` | 并行归约 |

### 数据移动

| API | 说明 |
|-----|------|
| `T.copy(src, dst, ...)` | 通用拷贝 |
| `T.c2d_im2col(...)` | Im2Col 变换 |

### 计算

| API | 说明 |
|-----|------|
| `T.gemm(A, B, C, ...)` | 矩阵乘法 |
| `T.gemm_sp(A, B, C, ...)` | 稀疏 GEMM |
| `T.reduce_sum/max/min(...)` | 归约 |
| `T.warp_reduce_sum/max/min(...)` | Warp 级归约 |
| `T.cumsum(src, dim, ...)` | 累积求和 |
| `T.clear(buf)` / `T.fill(buf, val)` | 初始化 |

### 循环

| API | 说明 |
|-----|------|
| `T.Parallel(...)` | 并行循环 |
| `T.serial(...)` / `T.Serial(...)` | 串行循环 |
| `T.unroll(...)` / `T.Unroll(...)` | 展开循环 |
| `T.Pipelined(n, num_stages)` | 流水线循环 |
| `T.Persistent(...)` | 持久化循环 |
| `T.vectorized(...)` | 向量化循环 |

### 同步与原子

| API | 说明 |
|-----|------|
| `T.sync_threads()` | 线程同步 |
| `T.atomic_add/max/min(...)` | 原子操作 |
| `T.atomic_load/store(...)` | 原子加载/存储 |

### 高级

| API | 说明 |
|-----|------|
| `T.ws(warp_group_id)` | Warp Specialization |
| `T.annotate_layout({...})` | Layout 注解 |
| `T.use_swizzle(...)` | L2 Cache Swizzle |
| `T.rng_init/rand/rand_float(...)` | 随机数 |
| `T.any_of/all_of(buffer)` | 逻辑操作 |
| `T.print(...)` / `T.device_assert(...)` | 调试 |

---

## 参考资料

- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [TileLang 官方文档](https://tilelang.com)
- [TileLang DeepWiki](https://deepwiki.com/tile-ai/tilelang)
- [TVM 项目](https://tvm.apache.org/)
