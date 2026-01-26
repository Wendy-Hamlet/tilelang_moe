python benchmarks/benchmark_grouped_gemm.py
================================================================================
Grouped GEMM Performance Benchmark
4 Versions: pytorch-loop, pytorch-compiled, tilelang-separate, ours (sonic)
================================================================================
Device: NVIDIA H100 80GB HBM3
PyTorch version: 2.8.0+cu129

================================================================================
Config: B128_H512_I384_E8 - Small Test
  total_tokens=128, H=512, I=384, E=8
================================================================================
Expert distribution: min=3, max=52, mean=16.0

[1] Correctness Verification:
2026-01-26 21:33:05  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
❌ Error in ours (sonic): unknown storage scope expert_id
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   0.7772 ms
  pytorch-compiled         :   0.7740 ms
  tilelang-separate        :   0.9187 ms
2026-01-26 21:33:07  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (unknown storage scope expert_id)

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-26 21:33:07  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
❌ Error in ours (sonic): unknown storage scope expert_id
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.5136 ms
  pytorch-compiled         :   1.5160 ms
  tilelang-separate        :   2.9938 ms
2026-01-26 21:33:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (unknown storage scope expert_id)

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
2026-01-26 21:33:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
❌ Error in ours (sonic): unknown storage scope expert_id
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   4.8666 ms
  pytorch-compiled         :   4.4309 ms
  tilelang-separate        :   4.2536 ms
2026-01-26 21:33:12  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (unknown storage scope expert_id)

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-26 21:33:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
❌ Error in ours (sonic): unknown storage scope expert_id
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   6.9417 ms
  pytorch-compiled         :   9.6713 ms
  tilelang-separate        :  11.6923 ms
2026-01-26 21:33:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
  ours (sonic)             : FAILED (unknown storage scope expert_id)

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep | Ours (Sonic)   |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    0.7772 |     0.774  |         0.9187 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    1.5136 |     1.516  |         2.9938 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    4.8666 |     4.4309 |         4.2536 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |    6.9417 |     9.6713 |        11.6923 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 1.00x      | 0.85x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 1.00x      | 0.51x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 1.10x      | 1.14x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 0.72x      | 0.59x          | N/A            |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 0.95x
  tilelang-separate        : 0.77x