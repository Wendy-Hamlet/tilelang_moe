# TopK 实现版本说明

## 三个版本对比

### Baseline 1: PyTorch topk (工业级参考)
- **文件**: N/A (PyTorch 内置)
- **来源**: PyTorch 官方实现
- **特点**: 
  - 高度优化的 C++/CUDA 实现
  - 针对所有硬件和场景调优
  - 数千工程师多年优化
- **性能**: 
  - 小规模 (N=160): ~0.015ms
  - 大规模 (N=32768): ~0.100ms
- **用途**: 作为性能上限参考

---

### Baseline 2: 官方 TileLang Radix (学术实现)
- **文件**: `kernels/topk_radix.py`
- **来源**: TileLang 官方示例 (DeepSeek-V3)
- **算法**: Radix Selection (O(N) 理论复杂度)
- **特点**:
  - 基于 TileLang DSL 的高层实现
  - 算法清晰，易于理解
  - 未做底层优化
- **性能**:
  - 小规模 (N=160): ~0.045ms (慢 3x)
  - 大规模 (N=32768): ~0.042ms (快 2.4x)
- **瓶颈**:
  - 原子操作冲突严重
  - 重复读取全局内存
  - 固定参数未调优
- **用途**: 
  - 验证官方性能结果
  - 作为优化基准

---

### 优化版: 我们的改进实现
- **文件**: `kernels/topk_radix_optimized.py`
- **来源**: 基于官方算法的优化版本
- **优化技术**:
  1. **Warp-level 原子操作归约** (减少冲突 32x)
  2. **AutoTune 参数调优** (9 组配置自动选择)
  3. **缓存 uint32 值** (避免重复计算)
  4. **Padding 共享内存** (避免 bank conflict)
- **预期性能**:
  - vs Baseline 2: 1.5-2x 提升
  - vs Baseline 1: 
    - 小规模可能仍慢 (硬件优化难超越)
    - 大规模目标 3-5x 加速
- **用途**: 
  - 展示优化效果
  - 实际部署使用

---

## 性能对比表

| 场景 | N | k | PyTorch (Baseline 1) | 官方 TileLang (Baseline 2) | 优化版 (Ours) |
|------|---|---|---------------------|--------------------------|---------------|
| 小规模 | 160 | 6 | 0.015ms (1x) | 0.045ms (0.33x) | ~0.025ms (0.6x) |
| 中等 | 4096 | 128 | 0.033ms (1x) | 0.024ms (1.4x) | ~0.016ms (2.1x) |
| 大规模 | 32768 | 2048 | 0.100ms (1x) | 0.042ms (2.4x) | ~0.025ms (4x) |

*注: 括号内为相对 PyTorch 的 speedup*

---

## 性能目标

### 短期目标 (当前优化)
- [ ] 大规模场景达到 3-4x speedup (vs PyTorch)
- [ ] 优化版比官方快 1.5-2x
- [ ] 所有场景正确性 100%

### 中期目标 (深度优化)
- [ ] 大规模场景达到 5-6x speedup
- [ ] 小规模接近或超过 PyTorch
- [ ] 支持 TopK + Softmax 融合

### 长期目标 (研究级)
- [ ] 混合算法 (小规模 Bitonic, 大规模 Radix)
- [ ] 端到端 MoE 加速 2-3x
- [ ] 发布优化版本供社区使用

---

## Benchmark 脚本

### 1. 基础性能测试
```bash
# 测试官方实现 (Baseline 2 vs PyTorch)
python benchmarks/benchmark_radix_scale.py
```

### 2. 优化效果对比
```bash
# 对比三个版本
python benchmarks/benchmark_optimizations.py
```

### 3. Batch size 影响
```bash
# 分析 kernel overhead
python benchmarks/benchmark_batch_size.py
```

---

## 版本选择建议

### 什么时候用 PyTorch topk？
- 小规模场景 (N < 1000)
- 延迟敏感的应用
- 不需要与其他操作融合

### 什么时候用官方 TileLang？
- 学习 Radix Selection 算法
- 验证 TileLang 性能
- 不需要最优性能的研究

### 什么时候用优化版？
- 大规模场景 (N > 8192)
- 吞吐量优先的应用
- 需要与 Softmax 等融合 (未来)
- MoE 模型推理/训练

---

## 文件组织

```
kernels/
  ├── topk_radix.py              # Baseline 2: 官方实现
  ├── topk_radix_optimized.py    # 优化版本
  └── topk_gating.py             # Selection Sort 版本 (legacy)

benchmarks/
  ├── benchmark_radix_scale.py        # 规模测试 (Baseline 2)
  ├── benchmark_optimizations.py      # 优化对比 (3 版本)
  └── benchmark_batch_size.py         # Batch 影响分析

docs/
  ├── radix_topk_optimization.md     # 优化分析文档
  └── topk_versions.md               # 本文档
```
