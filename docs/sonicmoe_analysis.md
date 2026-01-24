# SonicMoE 研究笔记

## 项目概述

- **仓库**: https://github.com/Dao-AILab/sonic-moe
- **论文**: [arXiv:2512.14080](https://arxiv.org/abs/2512.14080)
- **作者**: Dao-AILab (UC Berkeley)
- **架构**: NVIDIA Hopper (H100, H200)
- **技术栈**: CuTeDSL + Triton

## TopK 实现分析

### 位置

`sonicmoe/functional/topk_softmax.py`

### 核心算法

**Bitonic TopK** - 基于双调排序网络

关键代码：
```python
from quack.sort.bitonic_sort import bitonic_topk

# 在 kernel 中:
topk_vals = bitonic_topk(tXrX_f32, self.next_power_of_2_K, warp_width=threads_per_row)
```

### 实现特点

1. **索引编码 (Index Encoding)**:
   ```python
   # 将索引编码到浮点数的低位
   log_N = int(math.log2(self.next_power_of_2_N))
   idx_mask = (1 << log_N) - 1
   encoded_idx = ~col_idx if tXrX_f32[i] >= 0 else col_idx
   tXrX_u32[i] = (tXrX_u32[i] & ~idx_mask) | encoded_idx
   ```
   - 利用浮点数的低位存储索引
   - 正数时反转索引位，使得相同值时选择较早的索引
   - 负数时保持原索引

2. **双调排序 (Bitonic Sort)**:
   - 来自 QuACK 库: `quack.sort.bitonic_sort`
   - 适合固定大小的并行排序
   - Warp 级别协作

3. **向量化加载**:
   ```python
   input_vecsize = 128 // input_dtype.width  # 128-bit loads
   ```
   - 充分利用内存带宽
   - 对齐访问优化

4. **线程分配策略**:
   ```python
   def _calculate_threads_per_row(self):
       N = self.next_power_of_2_N
       num_threads_per_row = max(min(N // self.k, 32, N // 64), 1)
       return num_threads_per_row
   ```
   - 每行至少 1 个线程，最多 32 个
   - 每个线程处理 ≥ k 个元素
   - 每个线程最多处理 64 个元素

5. **限制条件**:
   ```python
   assert k <= 128 and k <= N
   assert N <= 4096 and N % 8 == 0
   ```
   - k ≤ 128 (我们的场景: k=6 ✅)
   - N ≤ 4096 (我们的场景: N=160 ✅)
   - N 必须是 8 的倍数 (160 ✅)

### 与我们尝试的对比

| 特性 | SonicMoE | 我们的 Selection Sort | 我们的 Radix Selection |
|------|----------|----------------------|----------------------|
| 算法 | Bitonic TopK | 迭代选择 | 基数选择 |
| 复杂度 | O(N log²k) | O(k×N) | O(N) |
| 适用场景 | k≤128, N≤4096 | 小 k | 大 topk |
| 索引编码 | ✅ 位级技巧 | ❌ | ❌ |
| Warp 协作 | ✅ | 部分 | ✅ |
| Softmax 融合 | ✅ | ❌ | ❌ |

### 关键优势

1. **Softmax+TopK 融合**:
   ```python
   require_softmax_fusion: bool = True
   ```
   - 减少内存往返
   - 在同一个 kernel 完成 softmax 和 topk

2. **来自 QuACK**:
   - 高度优化的排序原语
   - CuTeDSL 实现，充分利用 Hopper 架构

3. **端到端集成**:
   - 不只是 topk，而是整个 MoE 流程优化
   - Router + Up projection + Down projection 全融合

## 为什么 SonicMoE 可能更快

1. **Bitonic Sort 固有优势**:
   - 网络固定，利于 GPU 并行
   - Warp shuffle 指令高效
   - 对小 k 值特别友好

2. **索引编码技巧**:
   - 避免额外的索引存储
   - 排序时同时处理值和索引

3. **CuTeDSL vs TileLang**:
   - CuTeDSL 更底层，更接近硬件
   - 可能有更好的指令选择

4. **Softmax 融合**:
   - 我们测试的是纯 topk
   - 实际 MoE 需要先 softmax 再 topk
   - 融合可以省去中间结果存储

## 下一步行动

### 1. 研究 QuACK 的 Bitonic TopK 实现

查看依赖：
```bash
cd third_party/sonic-moe
grep -r "quack" requirements.txt
```

### 2. 对比测试

创建 benchmark 对比：
- PyTorch topk
- TileLang Selection Sort
- TileLang Radix Selection
- SonicMoE Bitonic TopK

### 3. 考虑 Softmax+TopK 融合

如果单独的 topk 优化不够，测试融合版本：
```python
# 当前: logits -> softmax -> topk
# 优化: logits -> fused_softmax_topk
```

### 4. 学习 CuTeDSL

如果 TileLang 确实不够快，考虑：
- 学习 CuTeDSL 实现
- 或者直接使用 SonicMoE 的实现

## 关键发现

**为什么 torch.topk 这么快？**

可能使用了类似 Bitonic Sort 的优化：
- Warp-level primitives
- 高度优化的汇编代码
- 针对不同 (N, k) 的特化版本

**TileLang 的劣势**：
- 较高的抽象层次
- 缺少底层排序原语
- 没有 warp shuffle 支持？

## 参考资料

- QuACK: https://github.com/Dao-AILab/quack
- CuTeDSL: https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html
- Bitonic Sort: 经典并行排序算法
- SonicMoE Paper: 详细性能分析

---

**结论**: SonicMoE 的成功说明 TopK 优化在正确的实现下是可能的，关键在于：
1. 选择合适的算法（Bitonic）
2. 利用硬件特性（Warp shuffle）
3. 融合操作（Softmax+TopK）
4. 使用合适的工具（CuTeDSL）
