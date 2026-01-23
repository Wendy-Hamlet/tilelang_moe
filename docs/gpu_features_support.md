# TileLang GPU特性支持分析

## 🎯 支持的GPU架构总览

TileLang基于TVM构建，支持广泛的GPU架构，从旧的Tesla系列到最新的Hopper和Blackwell架构。

---

## 📊 NVIDIA GPU架构支持

### 完整支持列表

| Compute Capability | 架构 | 代表GPU | Hopper特性支持 |
|-------------------|------|---------|--------------|
| **sm_90a** | **Hopper** | **H100, H800** | **✅ 完整支持** |
| **sm_100a** | **Blackwell** | **B100, B200** | **✅ 完整支持** |
| sm_89 | Ada Lovelace | RTX 4090, RTX 4080 | ✗ |
| sm_87 | Ampere | Jetson AGX Orin | ✗ |
| sm_86 | Ampere | A40, A10, RTX 3090 | ✗ |
| sm_80 | Ampere | A100, A30 | ✗ |
| sm_75 | Turing | T4, RTX 2080, Quadro RTX 8000 | ✗ |
| sm_72 | Xavier | Jetson AGX Xavier | ✗ |
| sm_70 | Volta | V100, Quadro GV100 | ✗ |
| sm_61 | Pascal | P4, P6000, GTX 1080 | ✗ |
| sm_60 | Pascal | P100 | ✗ |
| sm_52 | Maxwell | M40, GTX 980 | ✗ |
| ... | 更老架构 | ... | ✗ |

### 官方测试设备

根据README，TileLang已在以下设备上测试验证：

**NVIDIA GPUs**:
- ✅ **H100** (完整Hopper特性：Auto TMA/WGMMA)
- ✅ A100
- ✅ V100
- ✅ RTX 4090
- ✅ RTX 3090
- ✅ RTX A6000

**AMD GPUs**:
- ✅ MI250 (Auto MatrixCore support)
- ✅ MI300X (Async Copy support)

---

## 🚀 Hopper (sm_90) 特性支持详解

### 1. TMA (Tensor Memory Accelerator)

**支持状态**: ✅ **完整支持**

TMA是Hopper架构的核心特性，用于高效的全局内存↔共享内存数据传输。

#### 配置选项

```python
# 默认启用TMA
@tilelang.jit
def kernel(...):
    pass

# 禁用TMA（回退到传统copy）
@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def kernel(...):
    pass

# 或使用PassConfigKey
from tilelang.transform import PassConfigKey

@tilelang.jit(
    pass_configs={
        PassConfigKey.TL_DISABLE_TMA_LOWER: True
    }
)
def kernel(...):
    pass
```

#### TMA支持的操作

1. **Bulk Load (TMA Load)**
   - Global Memory → Shared Memory
   - 支持多维张量传输
   - 自动处理swizzle布局
   - 硬件级别的barrier同步

2. **Bulk Store (TMA Store)**
   - Shared Memory → Global Memory
   - 支持1D/2D传输
   - 限制：某些swizzle模式不支持

3. **TMA Descriptor**
   - 自动生成TMA描述符
   - Prologue中初始化
   - 支持预取优化

#### 实现细节（源码）

```cpp
// src/op/copy.cc
// Lowers copy to a bulk TMA (Tensor Memory Accelerator) transfer.
// 自动检测是否可以使用TMA：
// - 需要Hopper+ GPU (sm_90a)
// - Global ↔ Shared scope
// - 匹配的数据类型
// - 满足对齐要求
```

#### 使用示例

```python
import tilelang as tl

@tl.jit
def tma_copy_example(M, N, dtype):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), dtype),  # Global
        B: T.Tensor((M, N), dtype),  # Shared
    ):
        with T.Kernel(...):
            # 自动检测并使用TMA
            T.copy(A[...], B[...])  # 如果满足条件，自动lower为TMA
    
    return kernel
```

### 2. WGMMA (Warp Group Matrix Multiply-Accumulate)

**支持状态**: ✅ **完整支持**

WGMMA是Hopper的新型矩阵乘法指令，性能显著优于Ampere的MMA。

#### 配置选项

```python
# 默认启用WGMMA
@tilelang.jit(target="cuda -arch=sm_90a")
def kernel(...):
    pass

# 禁用WGMMA（回退到传统MMA）
@tilelang.jit(pass_configs={"tl.disable_wgmma": True})
def kernel(...):
    pass
```

#### WGMMA特性

1. **Warp Group级别操作**
   - 4个warp为一组 (128 threads)
   - `num_warp_m` 必须是4的倍数

2. **支持的数据类型**
   - FP16/BF16
   - FP32 (通过TF32)
   - FP8 (E4M3, E5M2)
   - INT8

3. **布局支持**
   - 自动选择最优shared memory布局
   - 支持转置
   - 不支持自定义stride（限制）

#### 实现（源码）

```cpp
// src/tl_templates/cuda/gemm_sm90.h
template <int M, int N, int K, int num_warp_m, int num_warp_n, 
          bool trans_A, bool trans_B, ...>
class GemmTensorOp {
    // 使用CUTLASS的WGMMA builder
    using SmemLayoutAtomA = decltype(ss_smem_selector<...>());
    
    static_assert(num_warp_m % 4 == 0,
                  "num_warp_m must be a multiple of 4 for hopper wgmma");
    // ...
}
```

#### 自动调度条件

```cpp
// src/op/gemm.cc
// WGMMA自动启用条件：
bool allow_wgmma = TargetIsHopper(target) && 
                   (M >= 64) &&           // 最小矩阵大小
                   (K % 16 == 0) &&       // K对齐要求
                   !disable_wgmma;        // 未手动禁用
```

### 3. Warp Specialization

**支持状态**: ✅ **支持**

Warp specialization允许不同的warp执行不同的任务（如producer/consumer模式）。

#### 配置选项

```python
# 默认启用
@tilelang.jit
def kernel(...):
    pass

# 禁用warp specialization
@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def kernel(...):
    pass
```

#### 应用场景

1. **Producer-Consumer模式**
   - 部分warp负责数据加载（TMA）
   - 部分warp负责计算（WGMMA）

2. **Pipeline优化**
   - 隐藏数据传输延迟
   - 提高GPU利用率

### 4. Named Barriers

**支持状态**: ✅ **支持**

Hopper提供16个硬件命名barrier，用于细粒度同步。

```cpp
// src/transform/common/thread_sync_types.h
// There are 16 Named Barriers provided by Hardware starting in Hopper
```

### 5. Async Copy 增强

**支持状态**: ✅ **支持**

除了TMA，Hopper还增强了cp.async指令。

```python
# 配置
pass_configs = {
    "tir.use_async_copy": True  # 默认启用
}
```

---

## 🔧 在DeepSeek MoE例子中的使用

### 为什么禁用了Hopper特性？

```python
@tilelang.jit(pass_configs={
    "tl.disable_tma_lower": True,          # 禁用TMA
    "tl.disable_warp_specialized": True    # 禁用warp specialization
})
def moe_forward_tilelang_shared(...):
    pass
```

**原因分析**：

1. **通用性考虑**
   - 代码需要在多种GPU上运行（A100, V100等）
   - TMA/WGMMA仅在Hopper+上可用
   - 禁用后回退到更通用的实现

2. **调试和开发**
   - TMA/WGMMA的调试相对复杂
   - 先实现基础版本，后续优化

3. **性能权衡**
   - 对于MoE这种内存带宽受限的workload
   - TMA的收益可能不如计算密集型任务明显

### 如何启用Hopper优化？

```python
# 方法1：移除disable标志
@tilelang.jit(target="cuda -arch=sm_90a")
def moe_forward_tilelang_shared(...):
    # TMA和WGMMA会自动启用
    pass

# 方法2：显式配置
@tilelang.jit(
    target="cuda -arch=sm_90a",
    pass_configs={
        "tl.disable_tma_lower": False,
        "tl.disable_warp_specialized": False,
        "tl.disable_wgmma": False,
    }
)
def moe_forward_hopper_optimized(...):
    pass
```

---

## 📋 其他重要特性支持

### 1. FP8 支持 (Hopper+)

```python
# Hopper支持FP8类型
import torch

@tilelang.jit(target="cuda -arch=sm_90a")
def fp8_gemm(M, N, K):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), torch.float8_e4m3fn),  # FP8 E4M3
        B: T.Tensor((K, N), torch.float8_e5m2),    # FP8 E5M2
        C: T.Tensor((M, N), torch.float16),
    ):
        # WGMMA原生支持FP8
        pass
```

### 2. 2:4 结构化稀疏 (Ampere+)

```python
# Ampere及以上支持2:4稀疏
@tilelang.jit
def sparse_gemm(...):
    T.gemm_sp(...)  # 2:4 sparse tensor core
```

### 3. Swizzle优化

```python
# 自动swizzle优化（所有架构）
with T.Kernel(...):
    T.use_swizzle(10)  # 减少bank conflict
```

---

## 🎯 性能优化建议

### 针对H100优化

```python
@tilelang.jit(
    target="cuda -arch=sm_90a",
    pass_configs={
        # 启用所有Hopper特性
        "tl.disable_tma_lower": False,
        "tl.disable_wgmma": False,
        "tl.disable_warp_specialized": False,
        
        # 其他优化
        "tl.enable_fast_math": True,
        "tir.use_async_copy": True,
        "tir.merge_static_smem": True,
    }
)
def optimized_kernel(...):
    @T.prim_func
    def kernel(...):
        with T.Kernel(...) as (bx, by):
            # 使用shared memory swizzle
            T.use_swizzle(10)
            
            # Pipeline提高overlap
            for k in T.Pipelined(..., num_stages=3):
                T.copy(...)  # 自动TMA
                T.gemm(...)  # 自动WGMMA
```

### 多架构兼容

```python
from tilelang.utils.target import determine_target

# 运行时检测
target = determine_target("auto")  # 自动选择CUDA/HIP/Metal

# 或者明确指定fallback
if "sm_90" in str(target):
    # Hopper优化路径
    config = {"tl.disable_tma_lower": False}
else:
    # 通用路径
    config = {"tl.disable_tma_lower": True}

@tilelang.jit(target=target, pass_configs=config)
def kernel(...):
    pass
```

---

## 📖 参考文档

### 官方文档位置

- **GPU架构支持**: `/home/chjin/CUDA/tilelang/docs/get_started/targets.md`
- **Pass配置**: `/home/chjin/CUDA/tilelang/tilelang/transform/pass_config.py`
- **Hopper实现**: `/home/chjin/CUDA/tilelang/src/transform/lower_hopper_intrin.cc`
- **WGMMA实现**: `/home/chjin/CUDA/tilelang/src/tl_templates/cuda/gemm_sm90.h`
- **TMA实现**: `/home/chjin/CUDA/tilelang/src/op/copy.cc`

### 关键源码文件

```bash
# Hopper特性实现
tilelang/src/transform/lower_hopper_intrin.cc
tilelang/src/tl_templates/cuda/gemm_sm90.h
tilelang/src/op/copy.cc  # TMA实现
tilelang/src/layout/gemm_layouts.cc  # Hopper布局

# 配置和Pass
tilelang/tilelang/transform/pass_config.py
tilelang/tilelang/engine/phase.py
```

---

## ✅ 总结

### TileLang对Hopper的支持情况

| 特性 | 支持状态 | 配置项 | 默认 |
|-----|---------|--------|-----|
| **TMA** | ✅ 完整支持 | `tl.disable_tma_lower` | 启用 |
| **WGMMA** | ✅ 完整支持 | `tl.disable_wgmma` | 启用 |
| **Warp Specialization** | ✅ 支持 | `tl.disable_warp_specialized` | 启用 |
| **Named Barriers** | ✅ 支持 | 自动 | - |
| **FP8** | ✅ 支持 | 自动检测 | - |
| **Async Copy** | ✅ 增强支持 | `tir.use_async_copy` | 启用 |

### 关键要点

1. **自动检测**：TileLang会根据target自动启用Hopper特性
2. **可配置**：所有特性都可以通过pass_configs控制
3. **向后兼容**：禁用Hopper特性后可在旧GPU上运行
4. **性能**：启用Hopper特性可显著提升H100性能

### DeepSeek MoE优化建议

对于你的项目，如果目标是H100：

```python
# 启用完整Hopper优化
@tilelang.jit(
    target="cuda -arch=sm_90a",
    pass_configs={
        "tl.disable_tma_lower": False,      # 启用TMA
        "tl.disable_wgmma": False,          # 启用WGMMA
        "tl.enable_fast_math": True,        # 快速数学
    }
)
def moe_forward_h100_optimized(...):
    pass
```

预期性能提升：**20-40%**（相比禁用Hopper特性的版本）
