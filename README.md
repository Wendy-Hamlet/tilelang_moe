# TileLang MoE Kernels

基于TileLang的高性能MoE (Mixture-of-Experts) 算子优化实现

## 项目简介

本项目对DeepSeek MoE模型中的关键算子进行优化，包括：
1. **TopK+Softmax**: MoE路由层融合优化
2. **Grouped GEMM**: 专家计算批量矩阵乘法
3. **mHC (Manifold-Constrained Hyper-Connections)**: 流形约束超连接预处理/后处理

每个算子都实现了智能选择策略，根据输入参数自动选择最优实现。

## 官方配置性能总结 ⚡

### TopK+Softmax

**官方配置来源**: `tilelang/examples/`

#### 延迟对比 (ms)

| 配置 | Params | PyTorch | torch.compile | TL-Official | Ours |
|------|--------|---------|---------------|-------------|------|
| DeepSeek-V2 | M=320, N=128, K=6 | 0.0144 | 0.0144 | 0.0266 | **0.0082** |
| DeepSeek-V3 | M=64, N=32768, K=2048 | 0.1046 | 0.1758 | 0.0614 | **0.0594** |

*TL-Official = TileLang Softmax + TileLang Radix TopK (全TileLang实现)*

#### Ours vs 最强其他方案

| 配置 | 最强其他 | Ours | 加速比 | 状态 |
|------|----------|------|--------|------|
| DeepSeek-V2 | torch.compile (0.0144 ms) | 0.0082 ms | **1.75x** | ✅ |
| DeepSeek-V3 | TL-Official (0.0614 ms) | 0.0594 ms | 1.03x | ≈ |

**平均加速**: **1.39x** (vs 最强其他方案)

#### H100 延迟对比 (ms)

| 配置 | Params | PyTorch | torch.compile | TL-Official | Ours |
|------|--------|---------|---------------|-------------|------|
| DeepSeek-V2 | M=320, N=128, K=6 | 0.0213 | 0.0212 | 0.0348 | **0.0103** |
| DeepSeek-V3 | M=64, N=32768, K=2048 | 0.1269 | 0.1368 | 0.0618 | **0.0598** |

#### H100 Ours vs 最强其他方案

| 配置 | 最强其他 | Ours | 加速比 | 状态 |
|------|----------|------|--------|------|
| DeepSeek-V2 | torch.compile (0.0212 ms) | 0.0103 ms | **2.06x** | ✅ |
| DeepSeek-V3 | TL-Official (0.0618 ms) | 0.0598 ms | 1.03x | ≈ |

**H100 平均加速**: **1.54x** (vs 最强其他方案)

**自动选择策略**:
```
N <= 64:           Bitonic Sort (高并行度)
N in [65, 512]:    Selection Sort (低padding开销)
N > 512:           Radix TopK + Softmax (大规模高效)
```

### Grouped GEMM

**官方配置来源**: `tilelang/examples/`

#### 延迟对比 (ms)

| 配置 | Params | PyTorch | torch.compile | TL-Official | Ours |
|------|--------|---------|---------------|-------------|------|
| FusedMoE-Test | H=1024, I=256, E=8 | 0.1769 | 0.1857 | 0.1397 | **0.1393** |
| DeepSeek-V3 | H=7168, I=2048, E=256 | **5.9249** | 5.9443 | 38.9448* | 5.9290 |

*PyTorch = 高效实现(排序+连续访问，对齐TileLang官方baseline)*
*TL-Official在E=256时有精度问题(max_diff=0.25)*

#### Ours vs 最强其他方案

| 配置 | 最强其他 | Ours | 加速比 | 状态 |
|------|----------|------|--------|------|
| FusedMoE-Test | TL-Official (0.1397 ms) | 0.1393 ms | **1.00x** | ≈ |
| DeepSeek-V3 | PyTorch (5.9249 ms) | 5.9290 ms | 1.00x | ≈ |

**平均加速**: **1.00x** (自动选择最优实现)

#### H100 延迟对比 (ms)

| 配置 | Params | PyTorch | torch.compile | TL-Official | Ours |
|------|--------|---------|---------------|-------------|------|
| FusedMoE-Test | H=1024, I=256, E=8 | 0.4729 | 0.5842 | 0.2675 | **0.2671** |
| DeepSeek-V3 | H=7168, I=2048, E=256 | 7.7162 | 7.9759 | 31.3483 | **3.9945** |

#### H100 Ours vs 最强其他方案

| 配置 | 最强其他 | Ours | 加速比 | 状态 |
|------|----------|------|--------|------|
| FusedMoE-Test | TL-Official (0.2675 ms) | 0.2671 ms | 1.00x | ≈ |
| DeepSeek-V3 | PyTorch (7.7162 ms) | 3.9945 ms | **1.93x** | ✅ |

**H100 平均加速**: **1.47x** (vs 最强其他方案)

**TileLang优化点**:
- 双swizzled layout减少bank conflict
- num_stages=3增加流水线深度
- 调整block_N=64适应小I场景

**自动选择策略**:
```
H <= 2560 且 E <= 64:  TileLang grouped_gemm (1.27x vs PyTorch)
其他情况:              PyTorch循环 (cuBLAS对大矩阵更优)
```

**注意**: 大E(>64)时TileLang可能有精度问题，自动fallback到PyTorch。

### mHC (Manifold-Constrained Hyper-Connections)

**官方配置来源**: `tilelang/examples/deepseek_mhc/`

#### mHC Pre 延迟对比 (ms)

| Config | Params | PyTorch | torch.compile | TL-Official | Ours |
|--------|--------|---------|---------------|-------------|------|
| Config-1 | n=512, h=1280 | 0.2028 | 0.0717 | 0.0352 | **0.0221** |
| Config-2 | n=1024, h=2560 | 0.2969 | 0.1334 | 0.0676 | **0.0405** |
| Config-3 | n=2048, h=4096 | 0.8509 | 0.3075 | 0.1188 | **0.0983** |
| Config-4 | n=8192, h=4096 | 3.1334 | 1.1203 | 0.4039 | **0.3936** |

#### mHC Post 延迟对比 (ms)

| Config | Params | PyTorch | torch.compile | TL-Official | Ours |
|--------|--------|---------|---------------|-------------|------|
| Config-1 | n=512, h=1280 | 0.0800 | 0.0537 | **0.0102** | 0.0102 |
| Config-2 | n=1024, h=2560 | 0.2888 | 0.2134 | 0.0379 | **0.0369** |
| Config-3 | n=2048, h=4096 | 1.1298 | 0.7557 | **0.1065** | 0.1065 |
| Config-4 | n=8192, h=4096 | 4.6341 | 3.0111 | **0.4035** | 0.4035 |

#### Ours vs 最强其他方案

| Config | 最强其他 | Ours | 加速比 | 状态 |
|--------|----------|------|--------|------|
| Config-1 (Pre) | TL-Official (0.0352 ms) | 0.0221 ms | **1.59x** | ✅ |
| Config-2 (Pre) | TL-Official (0.0676 ms) | 0.0405 ms | **1.67x** | ✅ |
| Config-3 (Pre) | TL-Official (0.1188 ms) | 0.0983 ms | **1.21x** | ✅ |
| Config-4 (Pre) | TL-Official (0.4039 ms) | 0.3936 ms | 1.03x | ≈ |

**平均加速**: **1.19x** (Ours vs Best Other，含Pre+Post)

#### H100 mHC Pre 延迟对比 (ms)

| Config | Params | PyTorch | torch.compile | TL-Official | Ours |
|--------|--------|---------|---------------|-------------|------|
| Config-1 | n=512, h=1280 | 0.6453 | 0.2356 | 0.0360 | **0.0227** |
| Config-2 | n=1024, h=2560 | 0.4128 | 0.1228 | 0.0613 | **0.0313** |
| Config-3 | n=2048, h=4096 | 0.7679 | 0.2425 | 0.1113 | **0.0764** |
| Config-4 | n=8192, h=4096 | 2.2217 | 0.7252 | 0.2423 | 0.2640 |

#### H100 mHC Post 延迟对比 (ms)

| Config | Params | PyTorch | torch.compile | TL-Official | Ours |
|--------|--------|---------|---------------|-------------|------|
| Config-1 | n=512, h=1280 | 0.0779 | 0.0507 | **0.0107** | 0.0107 |
| Config-2 | n=1024, h=2560 | 0.2634 | 0.1934 | 0.0222 | **0.0221** |
| Config-3 | n=2048, h=4096 | 0.8081 | 0.5787 | **0.0572** | 0.0575 |
| Config-4 | n=8192, h=4096 | 3.1505 | 2.2867 | **0.2072** | 0.2071 |

#### H100 Ours vs 最强其他方案

| Config | 最强其他 | Ours | 加速比 | 状态 |
|--------|----------|------|--------|------|
| Config-1 (Pre) | TL-Official (0.0360 ms) | 0.0227 ms | **1.59x** | ✅ |
| Config-2 (Pre) | TL-Official (0.0613 ms) | 0.0313 ms | **1.96x** | ✅ |
| Config-3 (Pre) | TL-Official (0.1113 ms) | 0.0764 ms | **1.46x** | ✅ |
| Config-4 (Pre) | TL-Official (0.2423 ms) | 0.2640 ms | 0.92x | ❌ |
| Config-1 (Post) | TL-Official (0.0107 ms) | 0.0107 ms | 1.00x | ≈ |
| Config-2 (Post) | TL-Official (0.0222 ms) | 0.0221 ms | 1.00x | ≈ |
| Config-3 (Post) | TL-Official (0.0572 ms) | 0.0575 ms | 0.99x | ≈ |
| Config-4 (Post) | TL-Official (0.2072 ms) | 0.2071 ms | 1.00x | ≈ |

**H100 平均加速**: **1.24x** (Ours vs Best Other)

**自动选择策略**:
```
mHC Pre:
  Split-K条件: hc_hidden_size % 1024 == 0 且 
               (num_tokens <= 2048 或 (num_tokens >= 8192 且 hc_hidden_size >= 10240))
  否则使用Optimized版本

mHC Post: 使用TileLang官方实现（已高度优化，7-11x vs PyTorch）
```

## 快速开始

### 安装依赖

```bash
pip install torch tilelang tabulate
```

### 运行Benchmark

```bash
# TopK+Softmax
python benchmarks/benchmark_topk_softmax.py

# Grouped GEMM  
python benchmarks/benchmark_grouped_gemm.py

# mHC
python benchmarks/benchmark_mhc.py
```

### 使用API

```python
from kernels import (
    topk_softmax,      # TopK+Softmax (auto-select)
    grouped_gemm,      # Grouped GEMM (auto-select)
    mhc_pre, mhc_post  # mHC (auto-select)
)

# TopK+Softmax
topk_scores, topk_indices = topk_softmax(logits, k=8)

# Grouped GEMM
output = grouped_gemm(X, W, expert_indices)

# mHC Pre
mixes, sq, pre_mix = mhc_pre(x, residual, mhc_weight, hc_mult)
```

## 项目结构

```
tilelang_moe/
├── kernels/
│   ├── topk_softmax_baseline.py    # PyTorch baseline
│   ├── topk_softmax_tilelang.py    # TileLang实现 + auto-select
│   ├── grouped_gemm_baseline.py    # PyTorch baseline
│   ├── grouped_gemm_tilelang.py    # TileLang实现 + auto-select
│   ├── mhc_baseline.py             # PyTorch baseline
│   ├── mhc_tilelang.py             # TileLang实现 + auto-select
│   └── __init__.py
├── benchmarks/
│   ├── benchmark_topk_softmax.py
│   ├── benchmark_grouped_gemm.py
│   └── benchmark_mhc.py
├── third_party/
│   └── tilelang/                   # TileLang子模块
└── README.md
```

## 技术亮点

### TopK+Softmax 融合优化

**传统方法** (Method A):
```python
scores = softmax(logits)              # (B, E) - 对所有E个expert
topk_scores, indices = topk(scores, k)  # (B, k)
```

**Sonic优化** (Method B):
```python
topk_logits, indices = topk(logits, k)   # 先选k个
topk_scores = softmax(topk_logits)        # 只对k个softmax
```

**优势**:
- 内存访问减少 **49%** (E=256, k=8)
- 计算量减少 **97%** (只计算k个exp，而非E个)
- 数值稳定性更好（top-k值之间差距小）

### Grouped GEMM 智能调度

根据矩阵大小自动选择：
- **小H**: TileLang grouped kernel (减少kernel launch开销)
- **大H**: PyTorch循环调用cuBLAS (大矩阵GEMM更优化)

### mHC (Manifold-Constrained Hyper-Connections) 优化

**mHC Pre优化**:
- Split-K分解：将K维度分割并行计算，使用原子操作累加
- 去除pass_configs限制启用TMA/Warp Specialization
- 小配置实现 **1.59-1.67x** 加速

**mHC Post优化**:
- TileLang官方实现已高度优化（7-11x vs PyTorch）
- 流水线化内存访问减少延迟

## 测试环境

- GPU: NVIDIA RTX 5090 / H100
- CUDA: 12.8+
- PyTorch: 2.7.0+
- TileLang: latest

## 参考资料

- [TileLang](https://github.com/microsoft/TileLang): GPU kernel DSL
- [Sonic论文](https://arxiv.org/abs/2501.18655): Fast and Scalable Mixture-of-Experts
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437): MoE架构设计

## License

MIT License
