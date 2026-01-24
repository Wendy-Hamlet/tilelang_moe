# TopK Radix Selection Implementation

## Overview

基于 TileLang 示例 `examples/deepseek_v32/topk_selector.py` 实现的 Radix Selection 算法。这是一种高效的 Top-K 选择算法，特别适合大规模并行 GPU 计算。

## Algorithm

### Radix Selection vs. Selection Sort

我们之前的实现使用**选择排序 (Selection Sort)**：
- 时间复杂度: O(k × M × N)，其中 k=6, M=batch_size, N=160
- 适合小 k 值的场景
- 在小 batch 时表现较差，大 batch 时性能提升

新的 **Radix Selection** 算法：
- 时间复杂度: O(M × N)，与 k 值无关！
- 使用直方图 (histogram) 快速定位 top-k 分界点
- 分两阶段处理：
  1. Stage 1: 8-bit 快速筛选
  2. Stage 2: 逐字节细化 (最多 4 轮)

### How It Works

1. **8-bit Fast Pass**:
   - 将 float32 转换为可排序的 uint16 (取高 8 位)
   - 构建 256 个桶的直方图
   - 通过累加和找到包含第 k 大元素的桶
   - 直接选出高于阈值桶的所有元素

2. **Iterative Refinement**:
   - 对阈值桶内的候选元素，使用下一个 8 位进行分桶
   - 最多迭代 4 轮（32 位 ÷ 8 位/轮）
   - 每轮缩小候选集，直到选出所有 top-k

3. **Key Optimizations**:
   - 使用共享内存存储直方图，减少全局内存访问
   - 原子操作保证线程安全
   - 双缓冲技术 (ping-pong buffer) 存储候选索引
   - 并行累加和计算

## Implementation Details

### Float to Sortable Integer

```python
def convert_to_uint16(x):
    # Convert to float16 then reinterpret as uint16
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    # Flip bits for negative, set sign bit for positive
    # This makes larger values have larger integer representation
    bits_uint = T.if_then_else(x < 0, ~bits_uint & 0xFFFF, bits_uint | 0x8000)
    return bits_uint >> 8  # Take high 8 bits
```

关键点：通过位操作，使得浮点数的大小关系在整数域中保持一致。

### Shared Memory Layout

```
s_histogram[257]      : 256 buckets + 1 for prefix sum
s_input_idx[2][4096]  : Ping-pong buffers for candidate indices
s_num_input[2]        : Candidate count for each buffer
s_threshold_bin_id[1] : Current threshold bucket
```

### Thread Configuration

- **BLOCK_SIZE**: 1024 threads per block
- **RADIX**: 256 buckets (2^8)
- **SMEM_INPUT_SIZE**: 4096 max candidates after first pass

## Performance Expectations

根据原始 TileLang 示例，在 DeepSeek-V3 场景下 (batch=64, seq_len=32K, topk=2048):
- Radix Selection 明显快于 torch.topk
- 特别是在大 topk 值时优势明显

对于我们的场景 (num_experts=160, topk=6):
- 预期在所有 batch size 下都优于选择排序
- 小 batch (1-32) 时可能与 PyTorch 持平或略快
- 大 batch (64-256) 时预期有显著加速

## Benchmark

运行测试：

```bash
python benchmarks/benchmark_topk_radix.py
```

测试配置：
- Batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256
- num_experts: 160
- topk: 6
- Warmup: 10 iterations
- Benchmark: 50 iterations

## Benchmark Results

### Actual Performance (H100, 2026-01-24)

| Batch | PyTorch(ms) | Radix(ms) | Speedup | Correct |
|-------|-------------|-----------|---------|----------|
| 1     | 0.0111      | 0.0501    | **0.22x** | ✅ |
| 2     | 0.0105      | 0.0432    | **0.24x** | ✅ |
| 4     | 0.0104      | 0.0426    | **0.24x** | ✅ |
| 8     | 0.0100      | 0.0424    | **0.24x** | ✅ |
| 16    | 0.0101      | 0.0425    | **0.24x** | ✅ |
| 32    | 0.0102      | 0.0418    | **0.24x** | ✅ |
| 64    | 0.0104      | 0.0432    | **0.24x** | ✅ |
| 128   | 0.0102      | 0.0430    | **0.24x** | ✅ |
| 256   | 0.0103      | 0.0428    | **0.24x** | ✅ |

### Analysis

**结果：Radix Selection 在所有 batch size 下都比 PyTorch 慢 4-5 倍！**

可能原因：
1. **问题规模太小**: num_experts=160, topk=6 对 Radix Selection 不够友好
   - 原始实现针对 seq_len=32K, topk=2048 设计
   - 我们的场景只有 160 个候选，topk=6
   - Radix 的多阶段开销无法摊销

2. **PyTorch 高度优化**: torch.topk 在小规模问题上极其高效
   - 可能使用了特殊的 warp-level primitives
   - 对小 N 有专门优化路径

3. **共享内存开销**: 
   - SMEM_INPUT_SIZE=4096 对我们的场景过大
   - 直方图 256 个桶可能太多（N=160 只需要 160 个桶）

4. **算法不匹配**: Radix Selection 优势在大 topk
   - topk=6 时，简单算法更高效
   - 多阶段细化的开销无法被节省的计算抵消

### Conclusion

**TopK 优化在 DeepSeek-V2 MoE 场景下不是瓶颈！**

- PyTorch topk 已经足够快（0.01ms）
- TileLang 的两种实现（Selection Sort, Radix Selection）都更慢
- 应该将优化重点放在 **MoE GEMM 融合** 上

关键指标：
- **Correctness**: 100% 正确 ✅
- **Speedup**: 0.22-0.24x（比 PyTorch 慢）❌
- **Scaling**: 各 batch size 性能稳定，无改善

## Code Structure

```
kernels/
  topk_radix.py              # Radix selection implementation
    - convert_to_uint16()    # Float to sortable int conversion
    - convert_to_uint32()    # 32-bit version
    - tl_topk_radix_impl()   # JIT kernel generator
    - topk_radix_tilelang()  # User-facing API

benchmarks/
  benchmark_topk_radix.py    # Performance benchmark (batch 1-256)
    - Correctness verification
    - Performance comparison vs PyTorch
    - CSV output to results/
```

## Next Steps

1. ✅ 实现 Radix Selection 核心算法
2. ✅ 创建 batch 1-256 测试脚本
3. ✅ 在 H100 上运行性能测试
4. ✅ 分析性能数据 - **结果：比 PyTorch 慢 4-5 倍**
5. ❌ TopK 优化放弃 - 不是瓶颈
6. ⏭️ 研究 **sonic-moe** (Dao-AILab) 的 TopK 加速方法
7. ⏭️ 转向 **MoE GEMM 融合** - 真正的性能瓶颈

## References

- Original: `tilelang/examples/deepseek_v32/topk_selector.py`
- Paper: DeepSeek-V3 Technical Report
- Algorithm: Radix Selection (基于基数排序的选择算法)
