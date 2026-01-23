# TileLang FusedMoE 参考实现

这个目录包含从TileLang官方仓库复制的DeepSeek MoE实现，作为参考和对比基准。

## 文件说明

- `example_fusedmoe_torch.py` - PyTorch参考实现
- `example_fusedmoe_tilelang.py` - TileLang优化实现
- `test_example_fusedmoe.py` - 测试脚本
- `regression_example_fusedmoe.py` - 回归测试脚本

## 运行示例

```bash
# 激活环境
conda activate tilelang

# 运行测试
python test_example_fusedmoe.py

# 或直接运行
python example_fusedmoe_tilelang.py
```

## 与项目其他部分的关系

- `reference_impl/` - 官方实现参考
- `baselines/` - 我们自己的baseline实现
- `kernels/` - 我们要实现的TileLang kernels

## 主要差异

参考实现是完整的、生产级的实现，包含：
- Grouped GEMM优化
- Shared/Routed Expert分离
- 双流并行执行
- 完整的性能优化

我们的项目会从简单版本开始，逐步实现这些优化。
