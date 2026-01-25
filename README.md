# TileLang TopK 性能对比

四版本TopK实现性能对比项目。

## 四个版本

1. **PyTorch Baseline** - `torch.topk`
2. **PyTorch Compiled** - `torch.compile(torch.topk, mode='max-autotune')`
3. **TileLang Official** - 官方Radix Selection算法
4. **TileLang Optimized** - 优化版（uint32缓存 + 共享内存填充）

## 项目结构

```
tilelang_moe/
├── kernels/
│   ├── topk_radix.py           # TileLang官方实现
│   └── topk_radix_optimized.py # TileLang优化版
├── benchmarks/
│   └── benchmark_optimizations.py  # 四版本性能对比
├── third_party/
│   └── tilelang/               # TileLang子模块
└── requirements.txt
```

## 运行测试

```bash
# 在H100服务器上运行
python benchmarks/benchmark_optimizations.py
```

## 测试场景

- 小规模 (DeepSeek-V2): N=160, k=6
- Small: N=1024, k=32
- Medium: N=4096, k=128
- Large: N=8192, k=256
- XLarge: N=16384, k=512
- 大规模 (DeepSeek-V3): N=32768, k=2048

## 环境要求

- CUDA 12.9+
- PyTorch 2.8.0+
- H100 GPU
- TileLang (git submodule)

│   ├── topk_implementation.md      # TopK实现分析
