#!/bin/bash

# 激活conda环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tilelang

# 运行测试
echo "Running correctness tests..."
python -m pytest tests/test_correctness.py -v

echo "Running numerical precision tests..."
python -m pytest tests/test_numerical.py -v

echo "All tests completed!"
