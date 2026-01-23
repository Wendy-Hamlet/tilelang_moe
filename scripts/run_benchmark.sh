#!/bin/bash

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tilelang

# 创建结果目录
mkdir -p benchmarks/results

# 运行benchmark
echo "Running MoE benchmarks..."
python benchmarks/benchmark_moe.py --config configs/moe_config.yaml

echo "Running kernel benchmarks..."
python benchmarks/benchmark_kernels.py --config configs/moe_config.yaml

echo "Benchmarks completed! Results saved in benchmarks/results/"
