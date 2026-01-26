# Sonic Grouped GEMM 实现分析

基于 `sonic-moe/sonicmoe/functional/grouped_gemm.py` 的源码分析  
参考论文: https://arxiv.org/abs/2501.18655

## 1. 什么是Grouped GEMM

### MoE中的专家计算

在完成TopK+Softmax路由后，MoE需要对每个token计算被选中的k个专家：

```python
# 假设选中了expert_2, expert_5, expert_7
for expert_id in [2, 5, 7]:
    output += weight[expert_id] @ token * score[expert_id]
```

**问题**: 不同token选中的专家不同，无法用单一的GEMM批量计算

### 传统方案 vs Grouped GEMM

**方案A - For循环** (低效):
```python
for expert_id in range(E):
    tokens = gather_tokens_for_expert(expert_id)  # 不同长度
    outputs = W[expert_id] @ tokens
```
❌ GPU利用率低（每个expert的token数量不同）  
❌ Kernel启动开销大（E次kernel调用）

**方案B - Grouped GEMM** (Sonic方案):
```python
# 一次kernel处理所有expert计算
# 每个CTA (block)处理某个expert的一部分token
grouped_gemm(all_tokens, all_weights, expert_offsets)
```
✅ 单kernel完成所有计算  
✅ 动态任务调度，负载均衡  
✅ Hopper TMA加速内存访问

---

## 2. Sonic Grouped GEMM架构

### 2.1 核心类: `HopperWgmma_MoE_kernel`

**职责**: 
- Up Projection: `X @ W1 → Z` (计算激活)
- Down Projection: `Y1 @ W2 → Y2` (聚合回token)

**关键参数**:
```python
HopperWgmma_MoE_kernel(
    E=256,                           # 专家数量
    acc_dtype=Float32,               # 累加器类型
    tile_shape_mnk=(128, 256, 64),   # CTA tile大小
    cluster_shape_mnk=(2, 1, 1),     # Cluster配置
    is_A_gather=True,                # A矩阵是否需要gather
    epi_tile_size=32,                # Epilogue tile大小
    is_persistent=True,              # 持久化kernel
    pingpong=False,                  # Pingpong调度
)
```

### 2.2 两种Projection模式

#### Up Projection (Gather模式)

**计算**: `Z = X @ W1` 
- X: (T_active, H) - 被选中的token（gather）
- W1: (E, H, I) - 所有专家权重
- Z: (T_active, I) - 输出

**特点**:
- `is_A_gather=True`: 需要根据索引gather token
- 激活函数融合: SwiGLU/GELU/ReLU等
- Y1输出: 保存中间结果用于backward

#### Down Projection (Standard模式)

**计算**: `Y2 = Y1 @ W2`
- Y1: (T_active, I) - Up projection输出
- W2: (E, I, H) - 所有专家权重
- Y2: (T_active, H) - 最终输出

**特点**:
- `is_A_gather=False`: Y1已经是gather后的
- 最后需要scatter_reduce回原token位置

---

## 3. 关键技术详解

### 3.1 Tile配置自适应

根据I（intermediate size）动态选择tile配置：

**Up Projection配置**:
```python
if I >= 128 (GLU) or I >= 256 (non-GLU):
    tile_shape_mnk = (128, 256, 64)
    cluster_shape_mnk = (2, 1)
    epi_tile_size = 32
    is_pingpong = False
    
elif I == 64 (GLU) or I == 128 (non-GLU):
    tile_shape_mnk = (192, 128, 64)
    cluster_shape_mnk = (1, 1)
    epi_tile_size = 32
    is_pingpong = True
```

**Down Projection配置**:
```python
if I >= 1024:
    tile_shape_mnk = (128, 256, 64)
    cluster_shape_mnk = (2, 1)
    
elif I >= 256:
    tile_shape_mnk = (128, 192, 64)
    cluster_shape_mnk = (2, 1)
    is_pingpong = True
    
elif I >= 64:
    tile_shape_mnk = (128, 192, 64)
    cluster_shape_mnk = (1, 2)
    is_pingpong = True
```

**DeepSeek-V3配置** (H=7168, I=5376):
- Up: `(128, 256, 64)`, cluster `(2, 1)`, non-pingpong
- Down: `(128, 256, 64)`, cluster `(2, 1)`, non-pingpong

### 3.2 Pingpong调度

**目标**: 重叠计算和内存访问，提高SM利用率

**非Pingpong** (标准流水线):
```
WG0: Load → Compute → Store
WG1: Load → Compute → Store
     ↑              ↑
     等待Load      等待Compute
```

**Pingpong** (交替执行):
```
时间片1: WG0 Load  | WG1 Compute
时间片2: WG0 Compute | WG1 Load
时间片3: WG0 Load  | WG1 Compute
```

**适用场景**:
- ✓ 小I (I=64/128/256) - 计算少，访存瓶颈明显
- ✗ 大I (I≥1024) - 计算密集，流水线已饱和

### 3.3 Hopper TMA (Tensor Memory Accelerator)

**传统内存访问** (Ampere):
```python
# 每个线程手动load
for i in range(tile_size):
    shared_mem[i] = global_mem[i]
__syncthreads()
```

**TMA硬件加速** (Hopper):
```python
# TMA单元异步DMA，线程只需等待
tma.copy_async(global_mem → shared_mem, tile_descriptor)
tma_barrier.wait()  # 硬件信号量
```

**优势**:
- ✓ 硬件预取，减少latency
- ✓ 压缩传输（支持128-bit向量化）
- ✓ 释放线程做计算
- ✓ 自动处理越界（通过TensorMap）

### 3.4 Warp Group GEMM (WGMMA)

Hopper的硬件GEMM指令，一次计算 `64x64x16` (FP16):

```python
# 传统Tensor Core (1 warp = 32 threads)
mma.sync.aligned.m16n8k16.f32.f16.f16.f32 {...}

# Hopper WGMMA (1 warp group = 4 warps = 128 threads)
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16.f32 {...}
```

**优势**:
- 4x吞吐量（相对单warp）
- 减少指令调度开销
- 支持异步流水线（与TMA配合）

### 3.5 动态Tile调度器

**问题**: 不同expert的token数量差异大

**Sonic调度策略**:
```python
class SonicMoETileScheduler:
    def get_next_work_tile(self, cta_id):
        # 每个CTA动态获取下一个tile
        # 根据expert_offset计算当前expert的tile数量
        expert_id = binary_search(expert_offset, cta_id)
        tile_in_expert = cta_id - expert_offset[expert_id]
        
        if tile_in_expert >= expert_tile_count[expert_id]:
            # 当前expert完成，跳到下一个
            return get_next_work_tile(cta_id + 1)
        
        return (expert_id, tile_in_expert)
```

**负载均衡**:
- CTA空闲时自动领取新任务
- 避免某些expert计算完后SM闲置
- 支持动态expert频率（某些expert更热门）

---

## 4. 关键代码结构

### 4.1 初始化配置

```python
def __init__(
    self,
    E: int,                    # 专家数量
    tile_shape_mnk: Tuple,     # CTA tile (M, N, K)
    cluster_shape_mnk: Tuple,  # Cluster (nCTA_M, nCTA_N, nCTA_K)
    is_A_gather: bool,         # A矩阵gather模式
    epi_tile_size: int,        # Epilogue tile
    is_persistent: bool,       # 持久化kernel
    pingpong: bool,            # Pingpong调度
):
    # 1. Tile配置验证
    if not pingpong:
        assert tile_M in [64, 128, 192, 256, 320]
        assert tile_N % 16 == 0 and tile_N <= 512
    
    # 2. Warp Group配置
    if tile_M == 320:
        atom_layout_m, atom_layout_n = 1, 2  # N方向分割
    elif tile_M == 192:
        atom_layout_m, atom_layout_n = 3, 1  # M方向分割
    else:
        atom_layout_m = tile_M // 64
        atom_layout_n = 1
    
    self.mma_warp_groups = atom_layout_m * atom_layout_n * (2 if pingpong else 1)
    
    # 3. 共享内存布局
    self.a_smem_layout_staged = make_smem_layout_A(...)
    self.b_smem_layout_staged = make_smem_layout_B(...)
```

### 4.2 主计算Kernel

```python
@cute.jit
def __call__(
    self,
    mA: cute.Tensor,           # 输入矩阵A (token embeddings)
    mB: cute.Tensor,           # 权重矩阵B (expert weights)
    mD: cute.Tensor,           # 输出矩阵D
    mMoffset: cute.Tensor,     # Expert offset (前缀和)
    mAIdx: cute.Tensor,        # Gather索引 (if is_A_gather)
    ...
):
    # Step 1: 动态调度
    tile_scheduler = SonicMoETileScheduler(expert_offset, ...)
    work_tile = tile_scheduler.get_next_work_tile(blockIdx.x)
    
    # Step 2: TMA异步load
    tma_load_a_barrier.arrive()
    tma.copy_2d_async(mA_global → sA_smem, tma_descriptor_A)
    tma.copy_2d_async(mB_global → sB_smem, tma_descriptor_B)
    
    # Step 3: Warp Group GEMM
    for k_tile in range(K // tile_K):
        tma_load_a_barrier.wait()
        wgmma.mma_async(acc, sA_smem, sB_smem)
        wgmma_commit_group()
        
        # Pingpong: 下一个tile预取
        if pingpong and k_tile < K // tile_K - 1:
            tma.copy_2d_async(next_tile)
    
    wgmma_wait_group(0)  # 等待所有GEMM完成
    
    # Step 4: Epilogue (激活函数 + store)
    if compute_swiglu:
        z = silu(acc[:I//2]) * acc[I//2:]
    
    store_to_global(mD, z)
```

### 4.3 Epilogue优化

**问题**: 输出写回需要transpose + scatter

**Sonic方案**: 
1. **Shared Memory Epilogue**: 先写到共享内存，transpose后再写全局内存
2. **向量化Store**: 128-bit对齐写入
3. **Coalesced Access**: 相邻线程写相邻地址

```python
# Step 1: Accumulator → Shared Memory (transpose)
for i in range(epi_tile_size):
    smem_epi[threadIdx.x][i] = acc[i]
__syncthreads()

# Step 2: Shared Memory → Global (coalesced)
for i in range(epi_tile_size // 4):
    # 每个线程写4个float (128-bit)
    global_mem[base + i*4] = *(float4*)&smem_epi[i*4][threadIdx.x]
```

---

## 5. 性能优化策略

### 5.1 Cluster Multicast

**目标**: 减少重复数据传输

**场景**: Down Projection中，多个CTA需要相同的Y1数据

```python
# 非Multicast: 每个CTA独立load
CTA0: Load Y1[0:128] from GMEM
CTA1: Load Y1[0:128] from GMEM  # 重复传输！

# Multicast: 硬件广播
CTA0, CTA1: Multicast Load Y1[0:128]  # 只传输一次
```

**配置**:
```python
if is_A_gather:
    cluster_shape_mnk = (2, 1)  # M方向cluster
    # A需要scatter，不能multicast
else:
    cluster_shape_mnk = (2, 1) or (1, 2)
    # 根据raster_order选择multicast方向
```

### 5.2 Raster Order优化

**AlongM** (Up Projection):
```
CTA分配: expert_0 expert_0 expert_1 expert_1 ...
         tile_0   tile_1   tile_0   tile_1
```
优势: 相同expert的tile连续执行，权重复用

**AlongN** (Down Projection):
```
CTA分配: expert_0 expert_1 expert_2 expert_0 ...
         tile_0   tile_0   tile_0   tile_1
```
优势: 不同expert的相同tile位置一起执行，输入复用

### 5.3 寄存器压力管理

**问题**: GEMM累加器占用大量寄存器

```python
regs_per_thread = tile_M * tile_N / num_threads
# 例如: 128*256 / 256 = 128个float32 = 512 bytes

if regs_per_thread >= 208:  # 重度寄存器压力
    num_regs_load = 24
    num_regs_mma = 240
else:
    num_regs_load = 40
    num_regs_mma = 232
```

**策略**:
- 减少Load warp的寄存器分配
- 增加MMA warp的寄存器分配
- Pingpong时减少总寄存器需求（WG交替）

---

## 6. 实际应用示例

### DeepSeek-V3配置

**模型参数**:
- E = 256 (专家数量)
- H = 7168 (隐藏层维度)
- I = 5376 (中间层维度，H * 0.75)
- K = 8 (top-k)

**Grouped GEMM配置**:

**Up Projection** (X @ W1):
```python
HopperGEMMConfig(
    tile_shape_mnk=(128, 256, 64),
    cluster_shape_mnk=(2, 1),
    epi_tile_size=32,
    is_pingpong=False,
    raster_order=AlongM,
)
```

**Down Projection** (Y1 @ W2):
```python
HopperGEMMConfig(
    tile_shape_mnk=(128, 256, 64),
    cluster_shape_mnk=(2, 1),
    epi_tile_size=32,
    is_pingpong=False,
    raster_order=AlongN,
)
```

**性能数据** (论文Figure 5):
- 端到端MoE加速: **1.35x** (相对PyTorch)
- Grouped GEMM贡献: ~30%加速
- TopK+Softmax融合贡献: ~5%加速

---

## 7. TileLang实现策略

### 策略A - 简化版（推荐起步）

**目标**: 基本功能，降低复杂度

**特点**:
- 固定tile配置 (128, 256, 64)
- 非persistent kernel (简单调度)
- 不使用Pingpong
- 手动实现gather/scatter（PyTorch ops）

**实现步骤**:
1. 实现基本GEMM kernel（TileLang GEMM API）
2. 添加expert offset索引
3. 集成激活函数（SwiGLU）
4. Benchmark与PyTorch对比

### 策略B - 进阶版（接近Sonic）

**目标**: 性能优化，对标论文

**特点**:
- 动态tile配置
- Persistent kernel + 动态调度器
- Pingpong支持
- TMA异步内存访问
- Cluster Multicast

**挑战**:
- TileLang是否支持Hopper TMA？
- 如何实现persistent调度？
- Warp Group划分和同步

---

## 8. 关键实现要点

### 8.1 Expert Offset计算

**输入**:
```python
expert_indices = [2, 5, 7, 2, 5, ...]  # (T*K,)
expert_frequency = [0, 0, 2, 0, 0, 2, 0, 1, ...]  # (E,)
```

**Cumsum**:
```python
expert_offset = cumsum(expert_frequency)
# [0, 0, 0, 2, 2, 2, 4, 4, 5, ...]
```

**使用**:
```python
expert_id = 5
start_idx = expert_offset[expert_id]  # 2
end_idx = expert_offset[expert_id + 1]  # 4
num_tokens_for_expert5 = end_idx - start_idx  # 2
```

### 8.2 Gather实现

**目标**: 从X中选出被选中的token

```python
# 假设token_0选了expert_2和5，token_1选了expert_7
gather_idx = [0, 0, 1]  # token索引
expert_idx = [2, 5, 7]  # expert索引

X_gathered = X[gather_idx]  # (3, H)
# X_gathered[0] = X[0] (for expert_2)
# X_gathered[1] = X[0] (for expert_5)
# X_gathered[2] = X[1] (for expert_7)
```

**TileLang实现**:
```python
@tl.jit
def gather_kernel(X, gather_idx, X_gathered):
    token_id = blockIdx.x
    src_token_id = gather_idx[token_id]
    
    for i in range(H):
        X_gathered[token_id, i] = X[src_token_id, i]
```

### 8.3 Scatter-Reduce实现

**目标**: 将expert输出加回原token

```python
# Down Projection输出
Y2 = [y2_0_exp2, y2_0_exp5, y2_1_exp7]  # (3, H)
gather_idx = [0, 0, 1]
scores = [0.6, 0.4, 1.0]

# Scatter-reduce
output[0] += Y2[0] * 0.6  # expert_2
output[0] += Y2[1] * 0.4  # expert_5
output[1] += Y2[2] * 1.0  # expert_7
```

**TileLang实现**:
```python
@tl.jit
def scatter_reduce_kernel(Y2, gather_idx, scores, output):
    expert_token_id = blockIdx.x
    dst_token_id = gather_idx[expert_token_id]
    score = scores[expert_token_id]
    
    for i in range(H):
        atomic_add(&output[dst_token_id, i], Y2[expert_token_id, i] * score)
```

---

## 9. Benchmark设计

### 测试配置

**场景1: DeepSeek-V2** (E=160, H=5120, I=2560):
```python
batch_size = 64
seq_len = 2048
K = 6
```

**场景2: DeepSeek-V3** (E=256, H=7168, I=5376):
```python
batch_size = 64
seq_len = 2048
K = 8
```

### 性能指标

1. **延迟**: Up Projection + Down Projection总时间
2. **吞吐量**: Tokens/second
3. **加速比**: vs PyTorch baseline
4. **正确性**: L2误差 < 1e-3

### 预期结果

- PyTorch baseline: 1.0x
- TileLang (简化版): 1.1-1.2x
- TileLang (进阶版): 1.3-1.35x (对标Sonic)

---

## 总结

### Sonic Grouped GEMM的核心价值

**算法层面**:
- 单kernel完成所有expert计算
- 动态调度实现负载均衡
- Gather/Scatter高效索引

**实现层面**:
- Hopper TMA异步内存访问
- Warp Group GEMM硬件加速
- Pingpong重叠计算和访存
- Cluster Multicast减少重复传输

**性能层面**:
- 端到端MoE加速 1.35x
- Grouped GEMM贡献 ~30%
- 支持DeepSeek-V3规模 (E=256)

**适用场景**:
- ✓ MoE推理和训练
- ✓ Sparse Attention（类似gather pattern）
- ✗ 标准Dense GEMM（用cuBLAS更快）
