# Radix TopK 优化分析

## 当前实现对比

### 核心差异
**我们的实现** vs **官方实现**：
- ✅ 算法逻辑：**完全一致**（逐行对比确认）
- ✅ 数据类型：一致（float32 输入，int32 输出）
- ✅ Pass configs：一致（`TL_DISABLE_THREAD_STORAGE_SYNC`）
- ✅ 共享内存布局：一致
- ⚠️ 额外代码：我们加了类型转换保护（在 wrapper 函数中）

**结论**：我们的实现是官方算法的**精确复制**，没有算法层面的差异。

---

## 性能瓶颈分析

### 1. 原子操作瓶颈 🔴
```python
# Stage 1: 构建直方图
T.atomic_add(s_histogram[inval_int16], 1)

# Stage 1: 收集元素
pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
pos = T.atomic_add(s_num_input[0], 1, return_prev=True)

# Stage 2: 多轮迭代中也有原子操作
```

**问题**：
- 1024 个线程同时对少数几个桶做原子操作
- 高冲突率 → 线程等待 → 性能下降
- 小规模时（N=160），原子操作开销 >> 计算时间

**优化方向**：
- 使用线程私有计数器 + 最后归约（减少冲突）
- Warp-level 归约后再原子操作（32→1 减少冲突 32 倍）

---

### 2. 共享内存 Bank Conflict 🟡
```python
s_histogram = T.alloc_shared([RADIX + 1], T.int32)  # 257 个元素
```

**问题**：
- RADIX=256，累积和时线程访问相邻地址
- 可能存在 bank conflict（取决于访问模式）

**优化方向**：
- Padding 共享内存数组避免 bank conflict
- 使用 shuffle 指令替代共享内存通信

---

### 3. 累积和算法效率 🟡
```python
# 当前：简单的并行前缀和
for i in T.serial(8):  # log2(256) = 8
    offset = 1 << i
    T.sync_threads(3, RADIX)
    if tx < RADIX - offset:
        l_val = s_histogram[tx] + s_histogram[tx + offset]
    T.sync_threads(3, RADIX)
    if tx < RADIX - offset:
        s_histogram[tx] = l_val
```

**问题**：
- 8 轮同步（每轮两次 `sync_threads`）= 16 次同步
- 只有 256 个线程参与，其他 768 个线程空闲

**优化方向**：
- 使用 CUB 的高效 scan 原语
- Warp shuffle 减少同步开销

---

### 4. 二次遍历输入数据 🟠
```python
# 第一次遍历：构建直方图
for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
    # ... 读取 input[bx, input_idx]
    T.atomic_add(s_histogram[inval_int16], 1)

# 第二次遍历：收集元素
for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
    # ... 再次读取 input[bx, input_idx]
    if l_bin_id32 > l_threshold_bin_id:
        # ...
```

**问题**：
- 两次从全局内存读取相同数据
- 内存带宽浪费
- L1/L2 cache 可能不够大

**优化方向**：
- 一次遍历同时构建直方图和缓存值
- 使用共享内存缓存输入数据（如果足够大）
- Tiling 策略分块处理

---

### 5. Stage 2 的重复计算 🟡
```python
# Stage 2 中每轮都要重新计算 bin_id
for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
    l_bin_id32 = T.Cast(
        T.int32, 
        ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) 
          >> (24 - round * 8)) & 0xFF)
    )
```

**问题**：
- 每轮都从全局内存读取相同的 `input` 值
- 重复调用 `convert_to_uint32`（位操作）

**优化方向**：
- 在 Stage 1 收集时就计算并缓存 uint32 值
- 避免重复的全局内存访问

---

### 6. 固定的 SMEM_INPUT_SIZE 限制 🟠
```python
SMEM_INPUT_SIZE = 4096  # 固定假设阈值桶 < 4K
s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)
```

**问题**：
- 如果阈值桶 > 4096，会溢出（未处理）
- 浪费共享内存（阈值桶很小时）
- 共享内存是稀缺资源（限制 occupancy）

**优化方向**：
- 动态分配或使用全局内存作为 fallback
- 根据 topk 大小自适应调整
- 使用 AutoTune 测试不同大小

---

## 优化建议优先级

### 🔥 高优先级（立即可做）

#### 1. 减少原子操作冲突
```python
# 当前：所有线程直接原子操作
T.atomic_add(s_histogram[bin_id], 1)

# 优化：Warp-level 归约
# 每个 warp 内部先用 shuffle 归约，然后只有 lane 0 做原子操作
local_count = warp_reduce_sum(1 if my_bin == bin_id else 0)
if lane_id == 0:
    T.atomic_add(s_histogram[bin_id], local_count)
```
**预期提升**：小规模 1.5-2x，大规模 1.2x

#### 2. 使用 AutoTune 优化参数
```python
@tilelang.autotune(configs=[
    dict(BLOCK_SIZE=512, SMEM_SIZE=2048),
    dict(BLOCK_SIZE=1024, SMEM_SIZE=4096),
    dict(BLOCK_SIZE=256, SMEM_SIZE=8192),
])
```
**预期提升**：不同规模 1.1-1.3x

#### 3. 缓存输入数据避免重复读取
```python
# 第一次遍历时缓存 uint32 值
s_cached_values = T.alloc_shared([BLOCK_SIZE], T.uint32)
s_cached_values[tx] = convert_to_uint32(input[bx, input_idx])
# Stage 2 直接使用缓存值
```
**预期提升**：1.1-1.2x

---

### 🟡 中优先级（需要更多工程）

#### 4. 使用 Warp Shuffle 替代共享内存
- 累积和算法用 warp shuffle
- 减少同步开销

**预期提升**：1.1-1.15x

#### 5. 优化 Stage 2 的迭代策略
- 早停：如果候选集足够小，直接排序
- 自适应轮数：根据收敛速度调整

**预期提升**：特定场景 1.2x

---

### 🔵 低优先级（投入产出比低）

#### 6. Bank Conflict 优化
- Padding 数组
- 影响有限（现代 GPU bank conflict 处罚较小）

**预期提升**：< 1.05x

#### 7. 使用 Tensor Core（如果支持）
- 不适用（TopK 是选择操作，不是矩阵运算）

---

## 实际优化策略

### 阶段 1：快速优化（1-2 天）
1. ✅ AutoTune 添加多组配置
2. ✅ Warp-level 原子操作归约
3. ✅ 缓存输入数据

**目标**：小规模 2x，大规模 3-4x（接近官方性能）

### 阶段 2：深度优化（1 周）
4. Warp shuffle 重写累积和
5. Stage 2 自适应策略
6. 内存访问模式优化

**目标**：小规模 2.5x，大规模 5-6x（超越官方）

### 阶段 3：算法改进（研究级）
7. 混合算法：小规模用 Bitonic，大规模用 Radix
8. 与 Softmax 融合（真正的 MoE 场景）

**目标**：端到端 MoE 加速 2-3x

---

## 为什么官方性能更好？

**可能原因**：
1. **硬件差异**：官方可能在 A100 测试，我们在 H100（架构不同）
2. **编译器版本**：TileLang 版本不同，编译优化不同
3. **测试方法**：官方用 `do_bench`（更准确），我们用 `time.perf_counter`
4. **Batch size**：官方 batch=64 摊销开销，我们早期测试 batch 太小
5. **PyTorch 版本**：PyTorch 的 topk 实现可能在新版本中变快了

**验证方法**：
- 使用完全相同的测试脚本（官方的 test_topk_selector）
- 使用 nsys profile 对比详细性能
- 在相同硬件上测试

---

## 下一步

### 立即行动
1. **添加 AutoTune**：测试多组 BLOCK_SIZE 和 SMEM_SIZE
2. **使用官方测试脚本**：排除测试方法差异
3. **Profile 分析**：`nsys profile python benchmark.py`

### 未来方向
4. **实现 Warp-level 优化**
5. **与 Softmax 融合**（MoE 真正场景）
6. **发布优化版本**
