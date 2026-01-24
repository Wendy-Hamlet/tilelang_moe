# 项目文件说明

## 目录结构

### third_party/tilelang/ - TileLang源码 (git submodule)
完整的TileLang源码仓库，用于：
- 学习TileLang的实现细节
- 查看更多官方示例
- 理解编译器和优化器原理

**获取方式**：
```bash
git submodule update --init --recursive
```

### reference_impl/ - 官方参考实现
从TileLang官方仓库复制的完整DeepSeek MoE实现，包含：
- 生产级的Grouped GEMM优化
- Shared/Routed Expert分离
- 双流并行执行
- 完整的TileLang kernel实现

**用途**：作为学习参考和性能对比基准

### kernels/ - 我们的TileLang实现
我们自己实现的TileLang kernels，从简单到复杂：
1. 基础版本（先跑通）
2. 添加算子融合
3. 实现Grouped GEMM
4. 优化内存访问

### baselines/ - 基准实现
- `pytorch/` - PyTorch原生实现，用于正确性验证

### benchmarks/ - 性能测试
- 与PyTorch baseline对比
- 与TileLang官方实现对比
- 性能分析和可视化

### docs/ - 文档
- `implementation_analysis.md` - 实现流程和优化空间分析
- `kernel_details.md` - Kernel详细说明（4个kernel）
- `topk_implementation.md` - TopK实现方式分析
- `gpu_features_support.md` - GPU特性支持（Hopper TMA/WGMMA）

## 开发流程建议

1. **学习阶段**
   - 阅读 `reference_impl/example_fusedmoe_torch.py` 理解PyTorch实现
   - 阅读 `reference_impl/example_fusedmoe_tilelang.py` 理解TileLang优化
   - 查看 `docs/` 中的分析文档

2. **实现阶段**
   - 先完善 `baselines/pytorch/moe_layer.py`
   - 实现简单的TileLang kernel在 `kernels/`
   - 逐步添加优化

3. **测试阶段**
   - 运行 `tests/test_correctness.py` 验证正确性
   - 运行 `benchmarks/` 对比性能
   - 参考官方实现调优

4. **优化阶段**
   - 参考官方实现的优化技术
   - 实现Grouped GEMM
   - 启用Hopper特性（如果在H100上）

## 快速命令

```bash
# 查看官方实现
cat reference_impl/example_fusedmoe_tilelang.py | head -200

# 运行官方测试
cd reference_impl && python test_example_fusedmoe.py

# 运行我们的实现
python examples/demo_moe.py

# 性能对比
python benchmarks/benchmark_moe.py
```

## 重要提示

- `reference_impl/` 的代码不要修改，保持原样作为参考
- 我们的实现在 `kernels/` 和 `baselines/`
- 遇到问题时可以对照官方实现学习
