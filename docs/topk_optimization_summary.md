# TopK 优化总结

## 问题背景

DeepSeek-V2 MoE 配置：
- num_experts = 160
- topk = 6 (每个 token 选择 6 个专家)
- batch_size: 1-256

目标：优化 TopK 选择操作，超越 PyTorch baseline

## 尝试的方法

### 1. Selection Sort (基于 TileLang)

**实现**: [kernels/topk_gating.py](../kernels/topk_gating.py)
- 迭代 k 次，每次找到最大值
- 使用共享内存存储中间结果
- AutoTune 搜索最优线程配置

**结果** (2026-01-24):
- 小 batch (256-8192): 0.29x 慢于 PyTorch
- 大 batch (16384): 1.14x 快于 PyTorch ✅
- 大 batch (65536): 1.21x 快于 PyTorch ✅
- **结论**: 存在 crossover point，大 batch 有优势

**问题**:
- 初始实现有 race condition (已修复)
- 小 batch 性能差
- 不适合实际推理场景 (batch 通常较小)

### 2. Radix Selection (基于 DeepSeek-V3 示例)

**实现**: [kernels/topk_radix.py](../kernels/topk_radix.py)
- 两阶段：8-bit 快速筛选 + 迭代细化
- 时间复杂度 O(M×N)，与 k 无关
- 使用直方图和原子操作

**结果** (2026-01-24):

| Batch | PyTorch(ms) | Radix(ms) | Speedup |
|-------|-------------|-----------|---------|
| 1     | 0.0111      | 0.0501    | 0.22x   |
| 16    | 0.0101      | 0.0425    | 0.24x   |
| 64    | 0.0104      | 0.0432    | 0.24x   |
| 256   | 0.0103      | 0.0428    | 0.24x   |

**结论**: 所有 batch size 都比 PyTorch **慢 4-5 倍** ❌

## 失败原因分析

### 为什么 TileLang TopK 优化失败？

1. **问题规模太小**:
   - DeepSeek-V3 原始场景: seq_len=32K, topk=2048
   - 我们的场景: num_experts=160, topk=6
   - Radix 算法的多阶段开销无法摊销

2. **PyTorch 高度优化**:
   - torch.topk 在小规模问题上极其高效 (0.01ms)
   - 可能使用了特殊的 warp-level primitives
   - 对小 N 有专门优化路径

3. **算法不匹配**:
   - Selection Sort: O(k×M×N) - k 小时还可以
   - Radix Selection: O(M×N) - 但常数项大，适合大 topk

4. **TileLang 开销**:
   - Kernel launch overhead
   - 共享内存分配开销
   - 原子操作同步开销

## 性能对比总结

| 方法 | 小 Batch | 大 Batch | 适用场景 |
|------|----------|----------|----------|
| PyTorch topk | **最快** (0.01ms) | 中等 (0.5ms) | 所有场景 |
| Selection Sort | 慢 (0.04ms) | 快 (0.4ms) | batch > 16384 |
| Radix Selection | 很慢 (0.05ms) | 很慢 (0.04ms) | 大 topk (>100) |

## 结论

**TopK 操作不是 MoE 的性能瓶颈！**

关键发现：
- PyTorch topk 已经足够快 (0.01-0.5ms)
- TileLang 优化无法带来收益
- 应该专注于 **MoE GEMM 融合** (真正的瓶颈)

## 下一步

1. ✅ 验证 TopK 不是瓶颈
2. ⏭️ 研究 **sonic-moe** (Dao-AILab) 的优化策略
3. ⏭️ 实现 MoE GEMM 融合
4. ⏭️ 端到端性能测试

## 参考实现

- Selection Sort: [kernels/topk_gating.py](../kernels/topk_gating.py)
- Radix Selection: [kernels/topk_radix.py](../kernels/topk_radix.py)
- Benchmarks: [benchmarks/](../benchmarks/)
- Analysis: [docs/topk_analysis.md](topk_analysis.md), [docs/topk_radix.md](topk_radix.md)

---

**教训**: 优化前先用 profiler 确认瓶颈，不要盲目优化非关键路径。
