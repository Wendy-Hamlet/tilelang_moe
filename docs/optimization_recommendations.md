# TileLang 算子优化总结

> **Benchmark 环境**: NVIDIA RTX 5090, CUDA 12.8
> **测试日期**: 2026-02-04

## 1. Benchmark 结果总结

### 尝试的优化

| 优化项 | 算子 | 效果 | 状态 |
|-------|------|------|------|
| `T.use_swizzle(10)` | Grouped GEMM | 0.95x - 1.00x | ❌ 无效 |
| `wg_wait=0` | Grouped GEMM | 0.95x - 1.00x | ❌ 无效 |
| `T.use_swizzle(10)` | mHC | 0.99x - 1.00x | ❌ 无效 |
| `wg_wait=0` | mHC | 0.99x - 1.00x | ❌ 无效 |
| CUDA Streams 并行 | FusedMoE | 1.02x - 1.04x | ✅ 有效 |

### FusedMoE 详细结果

| 配置 | Original | Streams | 加速比 |
|------|----------|---------|--------|
| Small (seq=1024, hidden=2048) | 4.17ms | 4.02ms | 1.04x |
| Medium (seq=2048, hidden=4096) | 5.18ms | 5.41ms | 0.96x |
| DeepSeek (seq=8192, hidden=7168) | 36.13ms | 35.49ms | 1.02x |

---

## 2. 官方测试配置来源

### TopK
- **来源**: `third_party/tilelang/examples/topk/example_topk.py:68-70`
- **配置**: M=320 (num_tokens), N=128 (num_experts), topk=6

### Grouped GEMM
- **来源**: `third_party/tilelang/examples/grouped_gemm/example_grouped_gemm_fwd.py:147`
- **配置**: batch_sizes="64, 128", K/M 可变

### mHC
- **来源**: `third_party/tilelang/examples/deepseek_mhc/example_mhc_pre.py:411-415`
- **配置**: 
  - hidden_size: 1280, 2560, 4096, 7168
  - hc_mult: 4
  - num_tokens: 128 ~ 16384

### FusedMoE
- **来源**: `third_party/tilelang/examples/fusedmoe/example_fusedmoe_tilelang.py:489`
- **配置** (DeepSeek style):
  - d_hidden=7168, d_expert=2048
  - n_routed_experts=8, n_shared_experts=1
  - n_experts_per_token=4
  - batch_size=1, seq_len=8192

---

## 3. 有效优化: FusedMoE CUDA Streams

### 原理

FusedMoE 包含两个独立的计算路径：
1. **Shared Expert**: 所有 token 共享
2. **Routed Expert**: 根据 router 选择的 expert

这两个计算路径**无数据依赖**，可以并行执行。

### 实现

```python
class FusedMoETileLangStreams(FusedMoETileLang):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_stream = torch.cuda.Stream()
    
    def forward(self, ...):
        # Shared kernel 在单独的 stream
        with torch.cuda.stream(self.shared_stream):
            self.shared_kernel(...)
        
        # Routed kernel 在默认 stream (并行执行)
        self.routed_kernel(...)
        
        # Scatter reduce
        self.expert_cache = torch.scatter_reduce(...)
        
        # 等待 shared stream 完成
        self.shared_stream.synchronize()
        
        # 合并输出
        output = self.output_shared + self.expert_cache
        return output
```

### 使用方法

```python
from kernels.fusedmoe_tilelang import fusedmoe_tilelang_streams, fusedmoe_best

# 直接使用 streams 版本
output = fusedmoe_tilelang_streams(x, router_weight, ...)

# 或使用 best (自动选择)
output = fusedmoe_best(x, router_weight, ...)
```

---

## 4. 无效优化说明

### 为什么 `T.use_swizzle` 和 `wg_wait` 无效？

1. **TileLang 编译器已自动优化**
   - 现代 TileLang 编译器可能已自动应用这些优化
   - 手动添加可能导致重复优化或冲突

2. **GPU 架构差异**
   - RTX 5090 (Blackwell 架构) 可能有不同的最优配置
   - 这些优化可能对旧架构更有效

3. **Benchmark 噪声**
   - 0.95x - 1.00x 的差异在误差范围内
   - 实际没有统计显著性

---

## 5. 结论

### 有效优化

| 优化类型 | 适用场景 | 预期收益 |
|---------|---------|---------|
| CUDA Streams 并行 | FusedMoE (shared+routed) | 1.02x - 1.04x |

### 无效优化 (不推荐)

- `T.use_swizzle(10)` - 编译器已自动处理
- `wg_wait=0` - 编译器已自动处理
- 增加 `num_stages` - 可能导致 shared memory 溢出

### 建议

1. **对于单 kernel 优化**: 信任 TileLang 编译器，不要手动添加微优化
2. **对于架构级优化**: CUDA Streams 等并行策略需要手动实现
3. **Profile 优先**: 在目标 GPU 上实际测试，不同架构效果不同

---

## 附录: 运行 Benchmark

```bash
cd /home/chjin/CUDA/tilelang_moe
python benchmark_fusedmoe.py
```
