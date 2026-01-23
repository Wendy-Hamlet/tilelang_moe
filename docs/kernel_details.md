# TileLang DeepSeek MoE Kernels 详细分析

## 🎯 Kernel 总览

例子中包含 **2个主要的JIT编译函数**，每个函数内部包含 **2个子kernel**，总共 **4个实际执行的CUDA kernel**。

---

## 1️⃣ moe_forward_tilelang_shared (Shared Expert路径)

### 功能：处理所有token的Shared Expert计算

### 包含的子kernel：

#### Kernel 1.1: Gate & Up GEMM + SiLU Fusion
```python
with T.Kernel(
    T.ceildiv(num_tokens, block_token),      # Grid X: token维度
    T.ceildiv(dexpert, block_dexpert),       # Grid Y: expert维度
    threads=threads                           # 256 threads per block
):
```

**功能**：
- 计算 `gate_logits = input @ W_gate^T`
- 计算 `up_logits = input @ W_up^T`  
- 融合 SiLU 激活：`gate = gate / (1 + exp2(-gate * scale))`
- 融合 element-wise 乘法：`output = up * gate`

**输入**：
- `input`: [num_tokens, hidden_dim]
- `W_gate`: [expert_dim * n_shared, hidden_dim]
- `W_up`: [expert_dim * n_shared, hidden_dim]

**输出**：
- `up_logits`: [num_tokens, expert_dim * n_shared] (中间结果)

**关键优化**：
- 两个GEMM在同一kernel中完成，复用input的shared memory
- Pipeline加载：`T.Pipelined`隐藏访存延迟
- SiLU和element-wise乘法融合，避免额外的global memory访问

#### Kernel 1.2: Down GEMM
```python
with T.Kernel(
    T.ceildiv(num_tokens, block_token),      # Grid X: token维度
    T.ceildiv(dhidden, block_dhidden),       # Grid Y: hidden维度
    threads=threads
):
```

**功能**：
- 计算 `output = up_logits @ W_down^T`

**输入**：
- `up_logits`: [num_tokens, expert_dim * n_shared]
- `W_down`: [hidden_dim, expert_dim * n_shared]

**输出**：
- `output`: [num_tokens, hidden_dim] (shared expert最终输出)

---

## 2️⃣ moe_forward_tilelang_routed (Routed Expert路径)

### 功能：使用Grouped GEMM处理路由到各个专家的tokens

### 包含的子kernel：

#### Kernel 2.1: Routed Gate & Up Grouped GEMM + SiLU Fusion
```python
with T.Kernel(
    M,                                        # Grid X: 动态计算的总block数
    T.ceildiv(dexpert, block_dexpert),       # Grid Y: expert维度
    threads=threads
):
```

**功能**：
- **Grouped GEMM**：每个block处理一个专家的一批tokens
- 动态确定当前block处理哪个专家：`cur_group_idx = group_idx_for_bx[bx]`
- 计算该专家的 gate 和 up logits
- 融合 SiLU 和 element-wise 乘法
- 边界处理：`if i < actual_rows` (处理不同专家token数不同)

**输入**：
- `input`: [group_sum, hidden_dim] - 堆叠所有专家的tokens
- `routed_expert_gate`: [n_experts, expert_dim, hidden_dim]
- `routed_expert_up`: [n_experts, expert_dim, hidden_dim]
- `group_sizes`: [n_experts] - 每个专家的token数
- `group_offsets`: [n_experts] - 每个专家的起始offset
- `group_padded_offsets`: [n_experts] - padding后的offset
- `group_idx_for_bx`: [M] - 每个block对应的专家ID

**输出**：
- `up_logits`: [group_sum, expert_dim] (中间结果)

**关键优化**：
- **Grouped GEMM**：单次kernel处理所有专家，避免O(N)次kernel launch
- 动态索引：根据group_idx动态选择专家权重
- 边界保护：处理专家负载不均的情况
- Padding对齐：保证内存访问效率

#### Kernel 2.2: Routed Down Grouped GEMM + Weight Multiply
```python
with T.Kernel(
    M,                                        # Grid X: 动态计算的总block数
    T.ceildiv(dhidden, block_dhidden),       # Grid Y: hidden维度
    threads=threads
):
```

**功能**：
- Grouped GEMM：计算 `output = up_logits @ W_down^T`
- 融合专家权重乘法：`output *= routed_expert_weights`

**输入**：
- `up_logits`: [group_sum, expert_dim]
- `routed_expert_down`: [n_experts, hidden_dim, expert_dim]
- `routed_expert_weights`: [group_sum] - 每个token的专家权重

**输出**：
- `output`: [group_sum, hidden_dim] (加权后的输出)

**关键优化**：
- 融合权重乘法，避免额外的kernel
- 动态边界处理

---

## ⚖️ Kernel架构对比

### PyTorch Baseline (Naive实现)

```
For each expert (循环n_routed_experts次):
    Kernel 1: expert_gate(tokens)       ← O(N) launches
    Kernel 2: expert_up(tokens)         ← O(N) launches  
    Kernel 3: SiLU(gate)                ← O(N) launches
    Kernel 4: element_wise_mul(gate, up)← O(N) launches
    Kernel 5: expert_down(intermediate) ← O(N) launches
    Kernel 6: scatter_reduce(output)    ← O(N) launches

Total: 6N kernel launches for routed experts
```

### TileLang优化实现

```
Shared Expert:
    Kernel 1: gate & up GEMM + SiLU fusion
    Kernel 2: down GEMM

Routed Experts (并行):
    Kernel 3: grouped (gate & up) GEMM + SiLU fusion  ← 1次launch!
    Kernel 4: grouped down GEMM + weight multiply     ← 1次launch!

Total: 4 kernel launches (与专家数量无关)
```

### Kernel Launch 优化效果

| 实现方式 | Shared Kernels | Routed Kernels | 总Kernels |
|---------|---------------|----------------|----------|
| PyTorch Naive | 6 | 6N | 6N+6 |
| TileLang优化 | 2 | 2 | 4 |
| **加速比** | 3x | **3N×** | **~N×** |

对于8个专家：`48 → 4` kernels，减少 **91.7%** 的kernel launches！

---

## 🔄 Kernel执行流程图

```
Input [B, S, H]
    ↓
    ├─────────────────────┬──────────────────────┐
    ↓                     ↓                      ↓
Gate Network (PyTorch)   │                      │
    ↓                     │                      │
TopK Selection           │                      │
    ↓                     │                      │
Token Sorting/Grouping   │                      │
(CPU: argsort, bincount) │                      │
    ↓                     ↓                      ↓
    │         ┌───────────────────┐  ┌──────────────────┐
    │         │  Shared Stream    │  │  Routed Stream   │
    │         │                   │  │                  │
    │         │  Kernel 1.1:      │  │  Kernel 2.1:     │
    │         │  Gate&Up+SiLU     │  │  Grouped         │
    │         │  GEMM (fused)     │  │  Gate&Up+SiLU    │
    │         │       ↓           │  │       ↓          │
    │         │  Kernel 1.2:      │  │  Kernel 2.2:     │
    │         │  Down GEMM        │  │  Grouped Down    │
    │         │       ↓           │  │  + Weight Mul    │
    │         │  shared_output    │  │       ↓          │
    │         └───────────────────┘  │  Scatter-Reduce  │
    │                 ↓              │  (PyTorch)       │
    │                 │              │       ↓          │
    │                 │              │  routed_output   │
    │                 │              └──────────────────┘
    ↓                 ↓                      ↓
    └─────────────────┴──────────────────────┘
                      ↓
            Output = shared + routed
```

---

## 🔧 关键技术特性总结

| 特性 | Shared Kernels | Routed Kernels |
|-----|---------------|----------------|
| **Grouped GEMM** | ✗ | ✅ (核心) |
| **算子融合** | ✅ Gate+Up+SiLU | ✅ Gate+Up+SiLU+Weight |
| **Memory Pipeline** | ✅ Pipelined | ✅ Pipelined |
| **Swizzle优化** | ✅ use_swizzle(10) | ✅ use_swizzle(10) |
| **动态索引** | ✗ | ✅ group_idx |
| **边界处理** | ✗ | ✅ actual_rows |
| **并行执行** | 与Routed并行 | 与Shared并行 |

---

## 💡 性能关键点

### 1. 为什么分成两个kernel？

**Gate&Up Kernel → Down Kernel**

原因：
- Gate&Up输出的中间结果（`up_logits`）需要写回global memory
- Down kernel需要读取完整的`up_logits`
- 无法在一个kernel中完成（受shared memory限制）

### 2. Grouped GEMM的核心挑战

- **动态负载**：不同专家的token数差异大
- **内存不连续**：不同专家的权重需要动态索引
- **Padding开销**：需要对齐以提高效率

### 3. 未来可能的优化

- 将两个kernel融合（使用更大的shared memory或global memory staging）
- 更智能的padding策略
- 更好的load balancing

---

## 📊 详细的Kernel参数

### 默认配置

```python
block_token = 128       # 每个block处理的token数
block_dhidden = 128     # hidden维度的block大小
block_dexpert = 128     # expert维度的block大小
threads = 256           # 每个block的线程数
num_stages = 1          # Pipeline stage数
```

### Grid/Block配置

**Shared Kernel 1.1 (Gate&Up)**:
- Grid: `(ceildiv(num_tokens, 128), ceildiv(expert_dim*n_shared, 128))`
- Block: `256 threads`
- Shared Memory: `W_gate[128,128] + W_up[128,128]`

**Shared Kernel 1.2 (Down)**:
- Grid: `(ceildiv(num_tokens, 128), ceildiv(hidden_dim, 128))`
- Block: `256 threads`

**Routed Kernel 2.1 (Grouped Gate&Up)**:
- Grid: `(M, ceildiv(expert_dim, 128))`
  - M = 动态计算，考虑padding
- Block: `256 threads`
- 每个block动态确定处理哪个专家

**Routed Kernel 2.2 (Grouped Down)**:
- Grid: `(M, ceildiv(hidden_dim, 128))`
- Block: `256 threads`
