"""
分析官方版本与优化版本的差异
聚焦于N=32768, k=2048场景
"""

import torch
import time
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels.topk_radix import topk_radix_tilelang
from kernels.topk_radix_optimized import topk_radix_optimized

print("="*120)
print("官方版本 vs 优化版本差异分析")
print("="*120)

print("\n【官方版本（topk_radix.py）】")
print("  基础：官方DeepSeek V3实现精确复制")
print("  Stage 1:")
print("    - 8-bit histogram统计 (256个bin)")
print("    - 找threshold bin")
print("  Stage 2 (4轮迭代):")
print("    - 对每个候选元素重新计算 convert_to_uint32() → bit shift")
print("    - 没有缓存任何中间值")
print("")

print("【优化版本（topk_radix_optimized.py）】")
print("  改进1：uint32缓存")
print("    - 在Stage 1中缓存threshold bin内的元素的uint32值")
print("    - Stage 2中复用缓存，避免重复计算")
print("    - 代价：+4K共享内存")
print("")
print("  改进2：Bank conflict padding")
print("    - s_histogram[RADIX + 1] 而不是 [RADIX]")
print("    - 避免访问冲突（但在k=2048时可能无效）")
print("")

print("【关键问题：为什么大k场景优化无效？】")
print("")
print("当k=2048, N=32768时（k/N=6.25%）:")
print("  - threshold bin非常大，可能包含1000+个候选元素")
print("  - Optimized版本的缓存上限SMEM_INPUT_SIZE=4096")
print("  - 如果候选元素 > 4096，超出部分没有缓存优势")
print("  - Stage 2仍需重新计算这些元素的uint32值")
print("")
print("假设候选集分布：")
print("  - 小k (k=32): 候选集~100个 → 缓存很有效 ✅")
print("  - 大k (k=2048): 候选集~2000个 → 缓存还可以 (50%覆盖)")
print("  - 但2000个元素的4轮迭代中，bank conflict可能反而增加延迟")
print("")

N, k, batch = 32768, 2048, 64

# 固定种子以获得稳定数据
torch.manual_seed(42)
torch.cuda.manual_seed(42)

x = torch.randn(batch, N, device='cuda', dtype=torch.float32)

print("="*120)
print(f"性能对比测试: N={N}, k={k}, batch={batch}")
print("="*120)

# 官方版本
print("\n【Official】运行中...", end=" ", flush=True)
for _ in range(50):
    _ = topk_radix_tilelang(x, k)
torch.cuda.synchronize()

official_times = []
for _ in range(3):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        _ = topk_radix_tilelang(x, k)
    torch.cuda.synchronize()
    official_times.append((time.perf_counter() - start) / 1000 * 1000)

# 优化版本
print("Optimized", end=" ", flush=True)
for _ in range(50):
    _ = topk_radix_optimized(x, k)
torch.cuda.synchronize()

optimized_times = []
for _ in range(3):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(1000):
        _ = topk_radix_optimized(x, k)
    torch.cuda.synchronize()
    optimized_times.append((time.perf_counter() - start) / 1000 * 1000)

print("完成")

official_median = statistics.median(official_times)
optimized_median = statistics.median(optimized_times)

print(f"\nOfficial (中位数):  {official_median:.4f} ms")
print(f"Optimized (中位数): {optimized_median:.4f} ms")
print(f"Speedup: {official_median/optimized_median:.4f}x")

improvement = (official_median - optimized_median) / official_median * 100
if improvement > 1:
    print(f"✅ Optimized快 {improvement:.2f}%")
elif improvement < -1:
    print(f"❌ Optimized慢 {abs(improvement):.2f}%")
else:
    print(f"⚖️ 基本持平 (±{abs(improvement):.2f}%)")

print("\n" + "="*120)
print("现状分析")
print("="*120)
print(f"数据集大小: N={N}, k={k}, k/N比例={k/N*100:.2f}%")
print(f"特征: 大k场景（k占比高于1%阈值）")
print(f"")
print(f"结论: Optimized在该场景 {'✅ 有效' if improvement > 1 else '❌ 无效'}")
print(f"")
print("可能的优化方向：")
print("  1. 增大SMEM_INPUT_SIZE → 更好地缓存候选集")
print("     但需考虑H100 232KB共享内存限制")
print("  2. 针对大k优化算法 → large_k版本的目标")
print("  3. AutoTune选择 → 自动选择Official还是Optimized")
print("="*120)
