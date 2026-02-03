# TileLang MoE Kernels

基于TileLang的高性能MoE (Mixture-of-Experts) 算子优化实现

## 项目简介

本项目对DeepSeek MoE模型中的关键算子进行优化，包括：
1. **TopK+Softmax**: MoE路由层融合优化
2. **Grouped GEMM**: 专家计算批量矩阵乘法
3. **mHC (Manifold-Constrained Hyper-Connections)**: 流形约束超连接预处理/后处理
4. **FusedMoE**: 完整MoE FFN计算 (含Shared + Routed Experts)

每个算子都实现了智能选择策略，根据输入参数自动选择最优实现。

---

## 📊 性能总结 (H100 SXM)

### 1. TopK+Softmax

| 配置 | Params | PyTorch | torch.compile | TL-Official | **Ours** | 加速比 |
|------|--------|---------|---------------|-------------|----------|--------|
| DeepSeek-V2 | M=320, N=128, K=6 | 0.0213 | 0.0212 | 0.0348 | **0.0103** | **2.06x** |
| DeepSeek-V3 | M=64, N=32768, K=2048 | 0.1269 | 0.1368 | 0.0618 | **0.0598** | 1.03x |

**平均加速: 1.54x** | 自动选择策略: N≤64用Bitonic, N∈[65,512]用Selection, N>512用Radix

### 2. Grouped GEMM

| 配置 | Params | PyTorch | torch.compile | TL-Official | **Ours** | 加速比 |
|------|--------|---------|---------------|-------------|----------|--------|
| FusedMoE-Test | H=1024, I=256, E=8 | 0.4729 | 0.5842 | 0.2675 | **0.2671** | 1.00x |
| DeepSeek-V3 | H=7168, I=2048, E=256 | 7.7162 | 7.9759 | 31.3483* | **3.9945** | **1.93x** |

**平均加速: 1.47x** | *TL-Official在E=256时有精度问题

### 3. mHC Pre

| 配置 | Params | PyTorch | torch.compile | TL-Official | **Ours** | 加速比 |
|------|--------|---------|---------------|-------------|----------|--------|
| Config-1 | n=512, h=1280 | 0.6453 | 0.2356 | 0.0360 | **0.0227** | **1.59x** |
| Config-2 | n=1024, h=2560 | 0.4128 | 0.1228 | 0.0613 | **0.0313** | **1.96x** |
| Config-3 | n=2048, h=4096 | 0.7679 | 0.2425 | 0.1113 | **0.0764** | **1.46x** |
| Config-4 | n=8192, h=4096 | 2.2217 | 0.7252 | 0.2423 | 0.2640 | 0.92x |

### 4. mHC Post

| 配置 | Params | PyTorch | torch.compile | TL-Official | **Ours** | 加速比 |
|------|--------|---------|---------------|-------------|----------|--------|
| Config-1 | n=512, h=1280 | 0.0779 | 0.0507 | **0.0107** | 0.0107 | 1.00x |
| Config-2 | n=1024, h=2560 | 0.2634 | 0.1934 | 0.0222 | **0.0221** | 1.00x |
| Config-3 | n=2048, h=4096 | 0.8081 | 0.5787 | **0.0572** | 0.0575 | 0.99x |
| Config-4 | n=8192, h=4096 | 3.1505 | 2.2867 | **0.2072** | 0.2071 | 1.00x |

**mHC平均加速: 1.24x**

### 5. FusedMoE (Shared + Routed Experts)

| 配置 | Params | TL-Official | 备注 |
|------|--------|-------------|------|
| Small-Test | BS=1×1024, H=2048, I=512, E=8 | **0.049 ms** | 无pass_configs (自动TMA+WS) |
| Medium-Test | BS=1×2048, H=2048, I=1024, E=8 | 待测试 | |
| Large-Batch | BS=4×1024, H=2048, I=512, E=8 | 待测试 | |

**注**: TileLang自动优化(无pass_configs)已达到最优，进一步优化需要算法改变

---

## 📊 性能总结 (RTX 5090)

<details>
<summary>点击展开 RTX 5090 详细数据</summary>

### TopK+Softmax

| 配置 | Params | PyTorch | torch.compile | TL-Official | **Ours** | 加速比 |
|------|--------|---------|---------------|-------------|----------|--------|
| DeepSeek-V2 | M=320, N=128, K=6 | 0.0144 | 0.0144 | 0.0266 | **0.0082** | **1.75x** |
| DeepSeek-V3 | M=64, N=32768, K=2048 | 0.1046 | 0.1758 | 0.0614 | **0.0594** | 1.03x |

**平均加速: 1.39x**

### Grouped GEMM

| 配置 | Params | PyTorch | torch.compile | TL-Official | **Ours** | 加速比 |
|------|--------|---------|---------------|-------------|----------|--------|
| FusedMoE-Test | H=1024, I=256, E=8 | 0.1769 | 0.1857 | 0.1397 | **0.1393** | 1.00x |
| DeepSeek-V3 | H=7168, I=2048, E=256 | **5.9249** | 5.9443 | 38.9448* | 5.9290 | 1.00x |

**平均加速: 1.00x**

### mHC Pre

| 配置 | Params | PyTorch | torch.compile | TL-Official | **Ours** | 加速比 |
|------|--------|---------|---------------|-------------|----------|--------|
| Config-1 | n=512, h=1280 | 0.2028 | 0.0717 | 0.0352 | **0.0221** | **1.59x** |
| Config-2 | n=1024, h=2560 | 0.2969 | 0.1334 | 0.0676 | **0.0405** | **1.67x** |
| Config-3 | n=2048, h=4096 | 0.8509 | 0.3075 | 0.1188 | **0.0983** | **1.21x** |
| Config-4 | n=8192, h=4096 | 3.1334 | 1.1203 | 0.4039 | **0.3936** | 1.03x |

### mHC Post

| 配置 | Params | PyTorch | torch.compile | TL-Official | **Ours** | 加速比 |
|------|--------|---------|---------------|-------------|----------|--------|
| Config-1 | n=512, h=1280 | 0.0800 | 0.0537 | **0.0102** | 0.0102 | 1.00x |
| Config-2 | n=1024, h=2560 | 0.2888 | 0.2134 | 0.0379 | **0.0369** | 1.03x |
| Config-3 | n=2048, h=4096 | 1.1298 | 0.7557 | **0.1065** | 0.1065 | 1.00x |
| Config-4 | n=8192, h=4096 | 4.6341 | 3.0111 | **0.4035** | 0.4035 | 1.00x |

**mHC平均加速: 1.19x**

</details>

---

## 自动选择策略

### TopK+Softmax
```
N <= 64:           Bitonic Sort (高并行度)
N in [65, 512]:    Selection Sort (低padding开销)
N > 512:           Radix TopK + Softmax (大规模高效)
```

### Grouped GEMM
```
H <= 2560 且 E <= 64:  TileLang grouped_gemm (减少kernel launch开销)
其他情况:              PyTorch循环 (cuBLAS对大矩阵更优)
```

### mHC
```
mHC Pre:
  Split-K: hc_hidden_size % 1024 == 0 且 (n <= 2048 或 (n >= 8192 且 h >= 10240))
  否则: Optimized版本

mHC Post: TileLang官方实现 (已高度优化，7-15x vs PyTorch)
```

### FusedMoE
```
推荐: @tilelang.jit 无pass_configs (TileLang自动优化TMA+WS)
  - Shared Expert Kernel: gate+up GEMM (fused SiLU) → down GEMM
  - Routed Expert Kernel: 分组GEMM + weight scaling + scatter_reduce
  - 进一步优化: 需要算法改变 (Kernel Fusion, Stream-K等)
```

---

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

---

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
│   ├── fusedmoe_baseline.py        # PyTorch baseline + torch.compile
│   ├── fusedmoe_tilelang.py        # TileLang实现 (自动优化TMA+WS)
│   └── __init__.py
├── benchmarks/
│   ├── benchmark_topk_softmax.py
│   ├── benchmark_grouped_gemm.py
│   ├── benchmark_mhc.py
│   └── benchmark_fusedmoe.py       # FusedMoE公平benchmark
├── third_party/
│   └── tilelang/                   # TileLang子模块
└── README.md
```

---

## 技术亮点

### TopK+Softmax 融合优化

**传统方法** (先Softmax后TopK):
```python
scores = softmax(logits)              # (B, E) - 对所有E个expert
topk_scores, indices = topk(scores, k)  # (B, k)
```

**Sonic优化** (先TopK后Softmax):
```python
topk_logits, indices = topk(logits, k)   # 先选k个
topk_scores = softmax(topk_logits)        # 只对k个softmax
```

**优势**: 内存访问减少49%，计算量减少97% (E=256, k=8)

### mHC 优化

- **Split-K分解**: 将K维度分割并行计算，使用原子操作累加
- **去除pass_configs**: 启用TMA/Warp Specialization
- **小配置加速**: 1.59-1.96x vs TL-Official

### FusedMoE 优化

**TL-Official** (推荐):
```python
@tilelang.jit  # 无pass_configs，TileLang自动优化
def kernel(...):
    ...
```

**关键发现**:
- 不写`pass_configs`时，TileLang自动启用TMA和Warp Specialization
- 自动优化已达到接近最优性能
- 进一步优化需要真正改变算法（如Kernel Fusion, Stream-K）

**待探索的算法优化方向**:
1. Kernel Fusion: 合并gate/up和down GEMM为单kernel
2. Stream-K: 动态工作分配，提高SM利用率
3. 专家并行: 多个expert同时计算

---

## 测试环境

- GPU: NVIDIA RTX 5090 / H100 SXM
- CUDA: 12.8+
- PyTorch: 2.7.0+
- TileLang: latest

## 参考资料

- [TileLang](https://github.com/microsoft/TileLang): GPU kernel DSL
- [Sonic论文](https://arxiv.org/abs/2501.18655): Fast and Scalable Mixture-of-Experts
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437): MoE架构设计

## License

MIT License
