# TileLang MoE - TopK+Softmax Fusion Kernels

高性能MoE路由融合实现：基于Sonic算法的TopK+Softmax融合优化

## 项目简介

在Mixture-of-Experts (MoE)模型中，路由层需要对每个token选择top-k个专家。传统方法先对所有E个专家做softmax，再选top-k；Sonic优化先选top-k，再对k个做softmax，减少49%内存访问和97%计算量。

本项目实现并对比了4种TopK+Softmax方案：
1. **PyTorch Baseline**: 标准实现 (softmax→topk)
2. **torch.compile**: JIT编译优化
3. **TileLang Separate**: TileLang TopK + PyTorch Softmax
4. **TileLang Sonic**: 完整融合版本 (topk→softmax，对标Sonic论文)

## 性能结果 ⚡

**测试环境**: NVIDIA H100, CUDA 12.9, PyTorch 2.8.0

### 延迟对比 (ms)

| 配置 | PyTorch | compile | TileLang-Sep | **Ours (Sonic)** | 加速比 |
|------|---------|---------|--------------|------------------|--------|
| B64_E160_K6 | 0.0214 | 0.0203 | 0.0247 | **0.0099** | **2.18x** ⭐ |
| B64_E256_K8 | 0.0197 | 0.0213 | 0.0261 | **0.0112** | **1.77x** |
| B128_E512_K16 | 0.0248 | 0.0232 | 0.0286 | **0.0186** | **1.33x** |
| B64_E1024_K8 | 0.0266 | 0.0262 | 0.0282 | **0.0134** | **1.99x** |
| B64_E2048_K16 | 0.0352 | 0.0332 | 0.0325 | **0.0314** | **1.12x** |

**平均加速**: **1.68x** (相对PyTorch baseline)  
**最佳配置**: B64_E160_K6 (DeepSeek-V2场景) - **2.18x加速**

## 快速开始

### 依赖安装

```bash
pip install torch tilelang
```

### 运行Benchmark

```bash
python benchmarks/benchmark_topk_softmax.py
```

## 技术亮点

### 1. Sonic融合算法

**传统MoE路由** (Method A):
```python
scores = softmax(logits)           # (B, E) - 对所有E个expert
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

### 2. TileLang Hopper优化

- ✅ **TMA (Tensor Memory Accelerator)**: 硬件加速内存访问
- ✅ **Manual Loop Unrolling**: 手动展开循环确保正确性
- ✅ **do_bench**: TileLang稳定性测量工具（median模式）

### 3. torch.compile修复

修复前（工厂模式）:
```python
@torch.compile(mode='max-autotune', fullgraph=True)
def compiled_topk_softmax(...):
    return _topk_softmax_separate_impl(...)
```
❌ 性能差: 0.7-1.3ms

修复后（模块级编译）:
```python
topk_softmax_separate_compiled = torch.compile(_topk_softmax_separate_impl)
```
✅ 性能好: 0.02-0.03ms

## 项目结构

```
kernels/
  topk_softmax_baseline.py      # PyTorch baseline + torch.compile
  topk_softmax_sonic.py          # TileLang Sonic融合版本 (ours)
  topk_softmax_tilelang_separate.py  # TileLang TopK + PyTorch softmax
  topk_radix.py                  # TileLang Radix TopK (基础组件)

benchmarks/
  benchmark_topk_softmax.py      # 4版本性能对比

docs/
  sonic_topk_softmax_analysis.md # Sonic算法深度分析
  triton_topk_softmax_research.md  # Triton调研结果
```

## 实现细节

### 版本对比

| 版本 | TopK算法 | Softmax | 融合方式 | 适用场景 |
|------|---------|---------|---------|---------|
| PyTorch Baseline | CUB RadixSelect | PyTorch | 未融合 | 通用基准 |
| torch.compile | JIT优化 | PyTorch | 自动融合 | 对比测试 |
| TileLang Separate | Radix Selection | PyTorch | 未融合 | Method A验证 |
| **TileLang Sonic** | Shared Mem TopK | Register Softmax | **完全融合** | **生产环境** |

### Sonic融合实现

**Step 1: Shared Memory TopK**
```python
# 每个block处理一行，线程协作做TopK
smem_topk = shared_memory(block_size)
for expert_id in range(num_experts):
    val = load_global(logits[token_id, expert_id])
    # 插入排序维护top-k
    insert_topk(smem_topk, val, expert_id, k)
```

**Step 2: Register Softmax**
```python
# 手动展开循环（8步，block_size=256）
max_val = -inf
for i in range(k):  # k很小，寄存器内完成
    max_val = max(max_val, topk_vals[i])

exp_sum = 0.0
for i in range(k):
    topk_vals[i] = exp(topk_vals[i] - max_val)
    exp_sum += topk_vals[i]

for i in range(k):
    topk_vals[i] /= exp_sum  # sum=1保证
```

### 正确性验证

**Method A** (baseline/compile/tilelang-separate):
- softmax(E) → topk(k)
- ✅ 检查: sum(topk_scores) < 1.0 (正常)

**Method B** (ours):
- topk(k) → softmax(k)  
- ✅ 检查: sum(topk_scores) == 1.0 (Sonic要求)
- ✅ 检查: indices overlap > 95% with baseline

## DeepSeek配置优化

| 模型 | E | k | 配置 | 加速比 |
|------|---|---|------|--------|
| DeepSeek-V2 | 160 | 6 | B64_E160_K6 | **2.18x** ⭐ |
| DeepSeek-V3 | 256 | 8 | B64_E256_K8 | **1.77x** |

## Triton调研结果

**结论**: Triton **没有** TopK+Softmax融合实现

- ✅ Triton有softmax (flash-attention)
- ❌ Triton没有高效TopK (需要复杂归约)
- ✅ Sonic-MOE使用CUTE (CUTLASS DSL)

详见: [docs/triton_topk_softmax_research.md](docs/triton_topk_softmax_research.md)

## 下一步计划

- [ ] **Grouped GEMM**: 实现专家计算部分，完成端到端MoE
- [ ] **端到端Benchmark**: 路由+计算完整流程
- [ ] **内存优化**: 减少中间结果显存占用

## 参考资料

- [Sonic论文](https://arxiv.org/abs/2501.18655): Fast and Scalable Mixture-of-Experts
- [TileLang文档](https://github.com/microsoft/TileLang): GPU kernel DSL
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437): MoE架构设计

## 许可证

MIT License


**2. 共享内存填充**
```python
# 避免bank conflict
s_histogram = T.alloc_shared([256 + 1], "int32")  # +1 padding
s_cached_uint32 = T.alloc_shared([SMEM_INPUT_SIZE + 1], "uint32")  # +1 padding
```

**性能提升分析**：
- ✅ 减少float→uint32转换次数（Stage 2复用）
- ✅ 避免shared memory bank conflict
- ✅ 相比官方版提升3-8%
- ⚠️  固定参数（BLOCK_SIZE=1024, SMEM_INPUT_SIZE=4096）

**未实现的优化**（TileLang编译器限制）：
- ❌ Warp级直方图归约（编译失败）
- ❌ AutoTune参数搜索（编译失败）

---

## 四个版本

1. **PyTorch Baseline** - `torch.topk`
2. **PyTorch Compiled** - `torch.compile(torch.topk, mode='max-autotune')`
3. **TileLang Official** - 官方Radix Selection算法
4. **TileLang Optimized** - 优化版（uint32缓存 + 共享内存填充）

## 项目结构

```
tilelang_moe/
├── kernels/
│   ├── topk_radix.py           # TileLang官方实现
│   └── topk_radix_optimized.py # TileLang优化版
├── benchmarks/
│   └── benchmark_optimizations.py  # 四版本性能对比
├── third_party/
│   └── tilelang/               # TileLang子模块
└── requirements.txt
```

## 运行测试

```bash
# 在H100服务器上运行
python benchmarks/benchmark_optimizations.py
```

## 测试场景

- 小规模 (DeepSeek-V2): N=160, k=6
- Small: N=1024, k=32
- Medium: N=4096, k=128
- Large: N=8192, k=256
- XLarge: N=16384, k=512
- 大规模 (DeepSeek-V3): N=32768, k=2048

## 环境要求

- CUDA 12.9+
- PyTorch 2.8.0+
- H100 GPU
- TileLang (git submodule)

## 使用说明

### 克隆项目
```bash
git clone --recursive <repo_url>
cd tilelang_moe
```

### 运行测试
```bash
# 在H100服务器上运行完整对比测试
python benchmarks/benchmark_optimizations.py
```

### 预期输出
测试将输出两个表格：
1. **性能对比总结**：各版本的绝对执行时间（ms）
2. **加速比总结**：相对PyTorch baseline的加速倍数

## 参考资源

- TileLang官方文档：[github.com/microsoft/TileLang](https://github.com/microsoft/TileLang)
- DeepSeek-V3 Radix TopK：`third_party/tilelang/examples/deepseek_v32/topk_selector.py`
