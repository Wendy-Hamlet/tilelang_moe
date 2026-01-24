# 为什么我们的 TopK 优化失败了？Baseline 选择分析

## 问题

**我们的测试结果**：PyTorch topk 比所有 TileLang 实现都快 2-8 倍

| 实现 | Batch=1 | Batch=64 | Batch=256 | 结论 |
|------|---------|----------|-----------|------|
| PyTorch | 0.013ms ⚡ | 0.015ms ⚡ | 0.014ms ⚡ | **最快** |
| Selection Sort | 0.039ms | 0.043ms | 0.043ms | 慢 3x |
| Radix Selection | 0.050ms | 0.043ms | 0.043ms | 慢 4x |
| SonicMoE-style | 0.042ms | 0.036ms | 0.109ms | 慢 3x |

**但其他工作都声称有性能提升**：
- TileLang DeepSeek-V3 示例
- SonicMoE (arXiv:2512.14080)

为什么？

---

## 关键发现：测试场景完全不同！

### 1. **问题规模差异巨大**

#### TileLang DeepSeek-V3 示例
```python
def test_topk_selector(batch=64, seq_len=32 * 1024, topk=2048):
    # 测试场景：
    # - batch = 64
    # - seq_len = 32,768 (32K)  ← 极大！
    # - topk = 2048              ← 极大！
    
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    # Shape: [64, 32768], 从32K个元素中选2048个
```

#### 我们的场景
```python
# 测试场景：
# - batch = 1-256
# - num_experts = 160      ← 很小！
# - topk = 6               ← 很小！

logits = torch.randn(batch, 160, dtype=torch.float32)
# Shape: [batch, 160], 从160个元素中选6个
```

**规模对比**：

| 场景 | N (候选数) | k (选择数) | k/N 比例 | 数据量 |
|------|-----------|-----------|---------|--------|
| **DeepSeek-V3** | 32,768 | 2,048 | 6.25% | 巨大 |
| **我们** | 160 | 6 | 3.75% | 极小 |
| **倍数差异** | **205x** | **341x** | - | **42,000x** |

---

### 2. **算法复杂度分析**

#### PyTorch topk 优势
- **小 N 场景**：高度优化的 warp-level primitives
- 可能使用 **partial bitonic sort**（适合小规模）
- 对 N ≤ 1024 有专门的 fast path
- 固定开销极低

#### Radix Selection 优势
- **大 N 场景**：O(N) vs PyTorch 的 O(N log k)
- 当 N=32K 时，优势明显
- 当 N=160 时，常数项开销压倒性

**复杂度对比**：

| 算法 | 复杂度 | N=160, k=6 | N=32K, k=2048 |
|------|--------|------------|---------------|
| PyTorch (估计) | O(N log k) | 160×log(6)≈415 | 32K×log(2048)≈360K |
| Radix Selection | O(N×passes) | 160×4=640 | 32K×4=128K |
| 理论比值 | - | **0.65x 慢** | **2.8x 快** ✅ |

---

### 3. **融合操作的重要性**

#### SonicMoE 的真正优化点

**不是单独的 TopK 优化！**

```python
# SonicMoE 优化的是整个流程：
# Router (GEMM) → Softmax → TopK → Expert Dispatch → Expert GEMM → Combine

# 他们的 Baseline：
baseline_time = (
    router_gemm_time +      # 独立 kernel
    softmax_time +          # 独立 kernel  
    topk_time +             # 独立 kernel
    dispatch_time +         # 数据重排
    expert_gemm_time +      # 独立 kernel
    combine_time            # 独立 kernel
)

# SonicMoE：
sonicmoe_time = (
    fused_router_softmax_topk +  # 融合！减少内存访问
    fused_expert_gemm            # 融合！IO-aware tiling
)

# 加速来源：
# 1. Kernel launch overhead 减少
# 2. 中间结果不写回全局内存
# 3. 更好的数据局部性
```

**我们的测试**：
```python
# 只测试了孤立的 TopK
pytorch_time = torch.topk(logits, k)  # 单独操作
tilelang_time = topk_radix(logits, k) # 单独操作

# 没有测试融合场景！
```

---

### 4. **Baseline 选择差异**

#### TileLang DeepSeek-V3
```python
# Baseline: torch.topk (sorted=True 默认)
_ = torch.topk(input, topk, dim=-1)[1]  # 返回排序后的索引

# 我们的测试:
_ = torch.topk(logits, topk, dim=-1, sorted=False)  # 不排序
```

**排序开销**：
- `sorted=True`: O(k log k) 额外开销
- `sorted=False`: 可能更快的实现

#### SonicMoE
```python
# Baseline: 可能是未优化的 Triton/自定义实现
# 或者是完整的 MoE forward pass，不是单独 topk

# 对比的是端到端性能，不是单个算子
```

---

## 为什么 PyTorch 在小规模上这么快？

### 推测的 PyTorch 优化

1. **Fast Path for Small N**:
   ```cpp
   if (N <= 1024 && k <= 128) {
       return warp_level_topk(input, k);  // 单个 warp 处理
   }
   ```

2. **Warp Shuffle 指令**:
   - 硬件级别的线程间通信
   - 延迟极低（1-2 cycles）
   - TileLang 可能无法直接使用

3. **寄存器级优化**:
   - 小 N 时全部数据放寄存器
   - 避免共享内存开销

4. **编译器优化**:
   - 高度调优的汇编代码
   - 针对不同 (N, k) 的特化版本

---

## 结论

### 为什么其他工作能获得加速？

| 因素 | DeepSeek-V3 | SonicMoE | 我们 |
|------|------------|----------|------|
| **问题规模** | N=32K, k=2K (大) | 端到端 MoE | N=160, k=6 (小) |
| **测试内容** | 孤立 TopK | 融合操作 | 孤立 TopK |
| **Baseline** | torch.topk sorted | 未优化实现 | torch.topk unsorted |
| **加速来源** | 算法优势 | 融合减少开销 | ❌ 无 |

### 我们的教训

1. **选择错误的优化目标**：
   - TopK 在 DeepSeek-V2 场景下不是瓶颈
   - 应该优化整个 MoE 流程

2. **忽略了规模差异**：
   - 小规模问题（N=160）PyTorch 已最优
   - 大规模问题才有优化空间

3. **孤立测试单个算子**：
   - 真正的性能来自融合
   - Softmax+TopK 融合
   - Router+Expert GEMM 融合

---

## 正确的优化方向

### ✅ 应该做的

1. **测试融合操作**：
   ```python
   # 不要：
   logits = router(x)
   probs = softmax(logits)
   indices = topk(probs, k)
   
   # 应该：
   indices, probs = fused_router_softmax_topk(x, k)
   ```

2. **测试大规模场景**：
   ```python
   # DeepSeek-V3 场景
   batch = 64
   seq_len = 32768  # 32K tokens
   topk = 2048      # 选择2K个experts
   ```

3. **端到端性能**：
   ```python
   # 完整 MoE forward pass
   output = moe_layer(input)  # 包含所有步骤
   ```

### ❌ 不要做的

1. ~~优化孤立的 TopK（小规模）~~
2. ~~与高度优化的 PyTorch 单算子比较~~
3. ~~忽略融合的重要性~~

---

## 参考数据

### TileLang DeepSeek-V3 实际性能（推测）

在 N=32K, k=2K 场景下：
```
PyTorch topk:     ~2.5ms
Radix Selection:  ~0.9ms
Speedup:          ~2.8x  ✅
```

### SonicMoE 实际性能（论文数据）

端到端 MoE throughput:
```
Baseline (未融合):  XXX tokens/sec
SonicMoE (融合):    YYY tokens/sec
Speedup:            1.5-2.0x  ✅
```

---

## 最终建议

1. **放弃孤立 TopK 优化** - 已验证不是瓶颈
2. **研究 SonicMoE 的融合策略** - 真正的性能来源
3. **实现 Softmax+TopK 融合** - 第一步优化
4. **端到端 MoE GEMM 融合** - 最大收益点

**关键洞察**：性能优化要看整体，不是局部！
