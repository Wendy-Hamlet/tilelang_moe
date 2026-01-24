# TileLang MoE Benchmark 快速入门

## 目标

测试TileLang实现的MoE算子是否比PyTorch更快

## 工作流程

### Step 1: 下载TileLang源码（可选但推荐）

```bash
# 下载tilelang源码到 third_party/tilelang/
git submodule update --init --recursive
```

这样你可以：
- 📖 阅读tilelang的源码实现
- 🔍 查看完整的示例代码
- 🎯 理解底层优化技术

### Step 2: 安装TileLang

```bash
pip install tilelang
```

### Step 3: 学习TileLang语法

查看官方参考实现：
```bash
# 项目内的参考实现
cat reference_impl/example_fusedmoe_tilelang.py

# 或者查看tilelang源码中的更多示例
ls third_party/tilelang/examples/
```

关键概念：
- `@T.prim_func` - 定义kernel函数
- `T.Buffer` - 定义tensor
- `T.grid()` - 定义并行循环
- `T.block()` - 定义计算块

### Step 3: 实现你的算子

在 `kernels/` 目录下实现两个核心算子：

1. **TopK Gating** (`topk_gating.py`)
   - 输入：[batch, num_experts] 的门控分数
   - 输出：选中的top-k专家索引和权重

2. **MoE GEMM** (`moe_gemm.py`)
   - 输入：tokens和对应的expert权重
   - 输出：经过专家网络的结果

### Step 4: 运行Benchmark

```bash
# 正确性测试
python tests/test_correctness.py

# 性能对比
python benchmarks/benchmark_moe.py
```

## 性能优化提示

TileLang优化技巧（参考官方实现）：
1. 使用Tensor Core加速矩阵乘法
2. Shared Memory优化数据重用
3. Grouped GEMM合并多个小GEMM
4. 算子融合减少内存访问

## 文件说明

- `reference_impl/` - **不要修改**，仅供学习参考
- `kernels/` - 你的TileLang实现
- `baselines/` - PyTorch baseline实现
- `benchmarks/` - 性能对比脚本
- `docs/` - 详细文档和分析

## 常见问题

**Q: 如何调试TileLang代码？**
A: 参考 `reference_impl/test_example_fusedmoe.py` 的测试方式

**Q: 性能不如预期？**
A: 对照 `reference_impl/example_fusedmoe_tilelang.py` 的优化技术

**Q: 如何理解TileLang语法？**
A: 查看 `docs/` 目录下的分析文档
