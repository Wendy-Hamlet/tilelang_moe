# TileLang MoE Kernels

基于TileLang的高性能MoE (Mixture-of-Experts) 算子优化实现

## 项目简介

本项目对DeepSeek MoE模型中的关键算子进行优化，包括：
1. **TopK+Softmax**: MoE路由层融合优化
2. **Grouped GEMM**: 专家计算批量矩阵乘法
3. **mHC (Multi-Head Cache)**: 多头缓存预处理/后处理

每个算子都实现了智能选择策略，根据输入参数自动选择最优实现。

## 官方配置性能总结 ⚡

### TopK+Softmax

**官方配置来源**: `tilelang/examples/`

#### 延迟对比 (ms)

| 配置 | Params | PyTorch | torch.compile | TL-Official | Ours |
|------|--------|---------|---------------|-------------|------|
| DeepSeek-V2 | M=320, N=128, K=6 | 0.0143 | 0.0144 | 0.0266 | **0.0078** |
| DeepSeek-V3 | M=64, N=32768, K=2048 | 0.1042 | 0.1757 | 0.0594 | **0.0591** |

#### Ours vs 最强其他方案

| 配置 | 最强其他 | Ours | 加速比 | 状态 |
|------|----------|------|--------|------|
| DeepSeek-V2 | PyTorch (0.0143 ms) | 0.0078 ms | **1.84x** | ✅ |
| DeepSeek-V3 | TL-Official (0.0594 ms) | 0.0591 ms | 1.01x | ≈ |

**平均加速**: **1.42x** (vs 最强其他方案)

**自动选择策略**:
```
N <= 64:           Bitonic Sort (高并行度)
N in [65, 512]:    Selection Sort (低padding开销)
N > 512:           Radix TopK + Softmax (大规模高效)
```

### Grouped GEMM

**官方配置参数来源**: `tilelang/examples/`

| 配置来源 | H | I | E | 策略 | vs PyTorch | 状态 |
|----------|---|---|---|------|------------|------|
| example_fusedmoe.py (DeepSeek) | 7168 | 2048 | 8 | PyTorch | 1.00x | ✅ |
| test_example_fusedmoe.py | 1024 | 256 | 8 | TileLang | **3.70x** | ✅ |
| example_grouped_gemm.py | 8192 | 8192 | 2 | PyTorch | 1.00x | ✅ |

**自动选择策略**:
```
H <= 2560:  TileLang grouped GEMM (1.06x - 3.7x 加速)
H > 2560:   PyTorch循环 (cuBLAS对大矩阵更优)
```

**注意**: 大H配置下的"正确性问题"实际是FP16累加精度差异，TileLang结果比PyTorch FP16更接近FP32参考值。

### mHC (Multi-Head Cache)

**官方配置参数来源**: `tilelang/examples/deepseek_mhc/`

| num_tokens | hidden_size | hc_mult | 策略 | vs 官方 | 状态 |
|------------|-------------|---------|------|---------|------|
| 512 | 1280 | 4 | Optimized | **1.20x** | ✅ |
| 1024 | 2560 | 4 | Optimized | **1.14x** | ✅ |
| 2048 | 4096 | 4 | Optimized | **1.12x** | ✅ |
| 8192 | 4096 | 4 | Split-K | **2.45x** | ✅ |

**平均加速**: **1.46x** (官方配置)

**自动选择策略**:
```
Split-K条件: hc_hidden_size % 1024 == 0 且 (num_tokens <= 2048 或 (num_tokens >= 8192 且 hc_hidden_size >= 10240))
否则使用Optimized版本
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

### mHC Split-K 优化

对于"瘦高"矩阵 (大K, 小N):
- 将K维度分割为多个部分并行计算
- 使用原子操作累加结果
- 在特定配置下实现 **2.45x** 加速

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
