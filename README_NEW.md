# TileLang MoE 算子测试项目

本项目旨在测试和优化基于 TileLang 的 Mixture-of-Experts (MoE) 算子，特别针对 DeepSeek-V2 配置。

## 项目目标

1. ~~优化 TopK Gating 操作~~ ❌ **已验证非瓶颈**
2. 实现 MoE GEMM 融合 ⏭️ **下一步重点**
3. 端到端性能优化

## 快速开始

### 环境要求

- CUDA 12.9+
- Python 3.8+
- PyTorch 2.8.0+
- NVIDIA H100 (推荐)

### 安装

```bash
# 克隆仓库（包含 submodules）
git clone --recursive git@github.com:YOUR_USERNAME/tilelang_moe.git
cd tilelang_moe

# 安装依赖
pip install -r requirements.txt
```

## 项目结构

```
tilelang_moe/
├── kernels/              # TileLang 内核实现
│   ├── topk_gating.py    # TopK 选择排序
│   └── topk_radix.py     # TopK 基数选择
├── baselines/            # PyTorch 基线实现
├── benchmarks/           # 性能测试脚本
│   ├── benchmark_topk.py          # 选择排序测试
│   └── benchmark_topk_radix.py    # 基数选择测试
├── tests/                # 正确性测试
├── docs/                 # 文档
│   ├── topk_optimization_summary.md  # TopK 优化总结
│   └── sonicmoe_analysis.md          # SonicMoE 分析
└── third_party/          # 第三方库（git submodules）
    ├── tilelang/         # TileLang 源码
    └── sonic-moe/        # SonicMoE 参考实现
```

## TopK 优化总结

### 测试配置

- **num_experts**: 160 (DeepSeek-V2)
- **topk**: 6
- **batch_size**: 1-256

### 性能结果

| 方法 | Batch=1 | Batch=64 | Batch=256 | 结论 |
|------|---------|----------|-----------|------|
| PyTorch | 0.011ms ⚡ | 0.010ms ⚡ | 0.010ms ⚡ | **最快** |
| Selection Sort | 0.039ms | 0.043ms | 0.043ms | 慢 3-4x |
| Radix Selection | 0.050ms | 0.043ms | 0.043ms | 慢 4-5x |

**结论**: TopK 不是 MoE 的性能瓶颈，PyTorch 已足够优化。

详见：[docs/topk_optimization_summary.md](docs/topk_optimization_summary.md)

## 参考实现

### TileLang
- **仓库**: https://github.com/tile-ai/tilelang
- **用途**: GPU kernel DSL
- **状态**: 作为 submodule 集成

### SonicMoE
- **仓库**: https://github.com/Dao-AILab/sonic-moe
- **论文**: arXiv:2512.14080
- **技术**: CuTeDSL + Triton, Bitonic TopK
- **状态**: 作为 submodule，用于学习参考
- **分析**: [docs/sonicmoe_analysis.md](docs/sonicmoe_analysis.md)

## 运行测试

```bash
# TopK 选择排序 (batch 256-65536)
python benchmarks/benchmark_topk.py

# TopK 基数选择 (batch 1-256)
python benchmarks/benchmark_topk_radix.py

# 快速正确性验证
python tests/test_topk_radix.py
```

## 关键发现

1. **TopK 优化困难**:
   - PyTorch topk 高度优化 (0.01ms)
   - TileLang 实现更慢 (0.04-0.05ms)
   - 问题规模太小 (N=160, k=6)

2. **算法选择重要**:
   - Selection Sort: O(k×N) - 适合小 k
   - Radix Selection: O(N) - 适合大 topk
   - Bitonic TopK: O(N log²k) - SonicMoE 使用

3. **融合操作关键**:
   - 单独优化 TopK 收益有限
   - Softmax+TopK 融合更有价值
   - 端到端 MoE 融合是方向

## 下一步计划

- [ ] 研究 SonicMoE 的 Bitonic TopK 实现
- [ ] 测试 Softmax+TopK 融合的收益
- [ ] 实现 MoE GEMM 融合（真正瓶颈）
- [ ] 端到端性能优化

## License

本项目用于研究和学习目的。

## 致谢

- TileLang team (tile-ai)
- SonicMoE authors (Dao-AILab)
- DeepSeek team
