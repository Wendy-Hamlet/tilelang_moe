# TileLang MoE Implementation

DeepSeek MoE算子的TileLang实现及性能对比

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

## 环境配置

```bash
conda activate tilelang
pip install -r requirements.txt
```

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
