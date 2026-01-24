# TileLang MoE Benchmark

MoE算子性能测试：TileLang vs PyTorch

## 项目目标

优化MoE Top-K Gating算子，在H100上测试性能提升

### 当前工作

- ✅ 基于官方TileLang TopK实现
- ✅ 启用Hopper特性 (H100优化)
- ✅ 自动调优支持
- 🔧 服务器性能测试

## 快速使用

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

## 快速使用

### 安装依赖

```bash
pip install tilelang torch pandas
```

### 运行Benchmark

```bash
# 在H100服务器上运行
python benchmarks/benchmark_topk.py
```

### 查看结果

```bash
cat benchmarks/results/topk_benchmark_results.csv
```

## 核心实现

### Top-K Gating算子

**PyTorch Baseline**: `torch.topk`  
**TileLang优化**: 基于官方实现 + Hopper特性 + AutoTune

文件位置：
- 实现：`kernels/topk_gating.py`
- 测试：`benchmarks/benchmark_topk.py`

### 配置 (DeepSeek-V2)

- Batch: 256-8192
- Num Experts: 160
- K: 6

## 项目结构

```
tilelang_moe/
├── kernels/
│   ├── topk_gating.py          # Top-K算子 (PyTorch + TileLang)
│   └── moe_gemm.py             # MoE GEMM (待实现)
├── benchmarks/
│   └── benchmark_topk.py       # Top-K性能测试
├── reference_impl/             # 官方fusedmoe参考
├── third_party/tilelang/       # TileLang源码 (git submodule)
└── docs/
    └── topk_implementation_plan.md
```

## 参考资源

- TileLang官方TopK: `third_party/tilelang/examples/topk/`
- DeepSeek MoE实现: `third_party/tilelang/examples/fusedmoe/`

