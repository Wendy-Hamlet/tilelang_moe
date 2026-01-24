# TileLang MoE Benchmark

使用TileLang实现MoE算子并与PyTorch进行性能对比

## 项目目标

1. ✅ 学习TileLang语法和优化技术
2. 🔧 实现高性能MoE算子（Top-K Gating + Expert GEMM）
3. 📊 对比TileLang vs PyTorch性能

## 项目结构

```
tilelang_moe/
├── reference_impl/      # TileLang官方FusedMoE实现（参考）
│   ├── example_fusedmoe_tilelang.py  # TileLang完整实现
│   ├── example_fusedmoe_torch.py     # PyTorch参考实现
│   ├── test_example_fusedmoe.py      # 测试脚本
│   └── README.md                      # 参考实现说明
├── kernels/              # TileLang算子实现（我们的实现）
│   ├── moe_gemm.py      # MoE GEMM核心算子
│   ├── topk_gating.py   # Top-K门控选择
│   └── __init__.py
├── baselines/           # Baseline实现
│   └── pytorch/         # PyTorch原生实现
│       └── moe_layer.py
├── benchmarks/          # 性能测试
│   ├── benchmark_moe.py # 完整MoE benchmark
│   ├── benchmark_kernels.py # 单个算子benchmark
│   └── results/         # 测试结果
├── tests/               # 单元测试
│   └── test_correctness.py # 正确性测试
├── configs/             # 配置文件
│   └── moe_config.yaml  # MoE模型配置
├── scripts/             # 运行脚本
│   ├── run_benchmark.sh
│   └── run_tests.sh
├── examples/            # 使用示例
│   └── demo_moe.py
├── utils/               # 工具函数
│   ├── profiler.py      # 性能分析工具
│   └── visualizer.py    # 结果可视化
├── docs/                # 文档
│   ├── implementation_analysis.md  # 实现流程分析
│   ├── kernel_details.md           # Kernel详细说明
│   ├── topk_implementation.md      # TopK实现分析
│   └── gpu_features_support.md     # GPU特性支持
├── requirements.txt     # 依赖列表
└── README.md
```

## 快速开始

### 方法1: 一键安装（推荐）

**Windows:**
```powershell
.\scripts\setup_tilelang.bat
```

**Linux/Mac:**
```bash
bash scripts/setup_tilelang.sh
```

### 方法2: 手动安装

```bash
# 1. 克隆项目（含TileLang源码）
git clone --recursive https://github.com/your-username/tilelang_moe.git
cd tilelang_moe

# 或者如果已经克隆，单独下载tilelang源码
git submodule update --init --recursive

# 2. 安装依赖
pip install tilelang
pip install -r requirements.txt
```

### 验证安装

```bash
python -c "import tilelang; print(f'TileLang version: {tilelang.__version__}')"
```

## 学习资源

| 资源 | 位置 | 说明 |
|------|------|------|
| 🎯 语法入门 | `examples/tilelang_syntax_examples.py` | TileLang基础语法示例 |
| 📚 官方示例 | `third_party/tilelang/examples/` | 完整的官方示例代码 |
| 🔍 参考实现 | `reference_impl/` | DeepSeek MoE的完整实现 |
| 📖 快速指南 | `QUICKSTART.md` | 详细的入门教程 |

## 开发流程

### 1. 运行参考实现（学习TileLang语法）

```bash
cd reference_impl
python test_example_fusedmoe.py
```

### 2. 实现你的算子

编辑 `kernels/moe_gemm.py` 和 `kernels/topk_gating.py`

### 3. 性能对比

```bash
python benchmarks/benchmark_moe.py
```

## 项目亮点

✅ **Git Submodule集成** - 一键下载完整TileLang源码  
✅ **官方参考实现** - DeepSeek MoE的生产级实现  
✅ **详细文档** - 从语法入门到性能优化  
✅ **自动化脚本** - 一键环境配置

## TileLang快速参考

```python
import tilelang.language as T

# 定义kernel
@T.prim_func
def my_kernel(A: T.Buffer, B: T.Buffer):
    # TileLang代码...
    pass
```

参考 `reference_impl/example_fusedmoe_tilelang.py` 学习完整实现。

## 快速开始

### 运行官方参考实现

```bash
# 激活环境
conda activate tilelang

# 运行TileLang官方实现（参考）
cd reference_impl
python test_example_fusedmoe.py

# 查看完整实现
python example_fusedmoe_tilelang.py
```

### 运行我们的实现

```bash
# 运行PyTorch baseline
python examples/demo_moe.py

# 运行测试
bash scripts/run_tests.sh

# 运行benchmark
bash scripts/run_benchmark.sh
```

## DeepSeek MoE关键算子

1. **Top-K Gating**: 选择top-k个专家
2. **Expert Dispatch**: 将tokens分发到选中的专家
3. **Expert Computation**: 专家前向计算 (GEMM密集)
4. **Expert Combine**: 合并专家输出并加权

## 性能对比

- TileLang vs PyTorch
- TileLang vs Triton (可选)
- 不同batch size/序列长度/专家数的对比
