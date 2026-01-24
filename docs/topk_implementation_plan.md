# Top-K Gating Implementation Guide

## Overview

Based on your analysis of the official fusedmoe implementation, the Top-K gating is currently using PyTorch's `torch.topk`, which has significant optimization potential for TileLang.

## Current Status (Official Implementation)

```python
# From third_party/tilelang/examples/fusedmoe/example_fusedmoe_tilelang.py
topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
```

**Key characteristics:**
- Uses PyTorch's native implementation
- Not optimized for GPU memory hierarchy
- Potential bottleneck in MoE forward pass

## DeepSeek-V2 Configuration

- **Num Experts**: 160 (routed) + 2 (shared)
- **Experts per Token (k)**: 6
- **Typical Batch Size**: 256-4096 tokens
- **Input Shape**: `[batch, 160]` → `[batch, 6]`

## Optimization Strategies

### 1. **Radix Select** (Recommended for small k)
Best for k < 10 (our case: k=6)

**Advantages:**
- O(n) complexity for small k
- No sorting required
- Good memory access patterns

**Implementation approach:**
```python
@T.prim_func
def radix_select_topk(...):
    # Parallel partition by each batch
    for i in T.grid(batch):
        with T.block():
            # Quick select for k-th largest
            # Then collect top-k values
```

**Reference**: Check `third_party/tilelang/examples/topk/` for examples

### 2. **Bitonic Sort + Select**
Best for medium k (10 < k < 32)

**Advantages:**
- Efficient on GPU
- Predictable performance
- Easy to parallelize

### 3. **Heap-based Selection**
Best for large k (k > 32)

**Advantages:**
- O(n log k) complexity
- Memory efficient

## Implementation Plan

### Phase 1: PyTorch Baseline ✅
File: `kernels/topk_gating.py`

```python
def topk_gating_pytorch(scores, k):
    topk_weights, topk_indices = torch.topk(scores, k=k, dim=-1, sorted=False)
    return topk_indices, topk_weights
```

### Phase 2: TileLang Naive Implementation
Implement basic Top-K using TileLang primitives:
- Per-batch parallel processing
- Simple selection algorithm
- Verify correctness

### Phase 3: Optimization
1. **Memory optimization**: Use shared memory for scores
2. **Warp-level primitives**: Leverage warp shuffle for reduction
3. **Vectorization**: Process multiple elements per thread
4. **Register blocking**: Maximize register usage

### Phase 4: Integration with Grouped GEMM
Once Top-K is optimized, integrate with the full MoE pipeline to measure end-to-end impact.

## Benchmark Setup

File: `benchmarks/benchmark_topk.py`

**Test configurations:**
```python
batch_sizes = [256, 512, 1024, 2048, 4096]
num_experts = [160]  # DeepSeek-V2
k_values = [6]       # DeepSeek-V2
```

**Metrics:**
- Latency (ms)
- Throughput (tokens/sec)
- Speedup over PyTorch
- Correctness verification

## Expected Performance Gains

Based on similar TileLang optimizations:
- **Target**: 2-5x speedup over PyTorch
- **Memory bandwidth**: Better utilization through shared memory
- **Reduced overhead**: Direct kernel launch vs PyTorch dispatch

## Potential Challenges

1. **Handling ties**: When multiple experts have same score
2. **Numerical stability**: Ensure consistent results with PyTorch
3. **Edge cases**: k >= num_experts, batch=1, etc.
4. **Memory alignment**: Ensure coalesced memory access

## Reference Implementations

### Official TileLang Examples
```bash
# Check these for inspiration
ls third_party/tilelang/examples/topk/
cat third_party/tilelang/examples/topk/*.py
```

### Key TileLang Patterns
```python
# Shared memory allocation
scores_shared = T.alloc_buffer([BLOCK_SIZE], scope="shared")

# Warp-level reduction
max_val = T.warp_reduce_max(local_max)

# Thread synchronization
T.sync_threads()

# Atomic operations (if needed)
T.atomic_add(output[idx], value)
```

## Next Steps

1. ✅ **Setup benchmark framework** - `benchmark_topk.py`
2. ⬜ **Study TileLang topk examples** - Check official implementations
3. ⬜ **Implement naive version** - Get it working first
4. ⬜ **Profile and optimize** - Identify bottlenecks
5. ⬜ **Integrate with MoE** - Full pipeline testing

## Resources

- TileLang Examples: `third_party/tilelang/examples/topk/`
- TileLang Docs: `third_party/tilelang/docs/`
- Reference Paper: "Top-k Selection on GPUs" (if available)
- CUDA Radix Select: Various online implementations

## Running the Benchmark

```bash
# Test topk implementation
python kernels/topk_gating.py

# Run full benchmark
python benchmarks/benchmark_topk.py

# Check results
cat benchmarks/results/topk_benchmark_results.csv
```
