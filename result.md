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
2026-01-27 12:53:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 12:53:55  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ✅ ours (sonic)             : max_diff = 0.000000

[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.3991 ms
  pytorch-compiled         :   1.3730 ms
  tilelang-separate        :   1.1442 ms
  ours (sonic)             :   0.1757 ms

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-27 12:53:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 12:54:03  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ✅ ours (sonic)             : max_diff = 0.000000

[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.7513 ms
  pytorch-compiled         :   2.7519 ms
  tilelang-separate        :   3.0900 ms
  ours (sonic)             :   0.2730 ms

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
2026-01-27 12:54:06  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 12:54:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ✅ ours (sonic)             : max_diff = 0.000000

[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   3.0564 ms
  pytorch-compiled         :   3.0368 ms
  tilelang-separate        :   4.1457 ms
  ours (sonic)             :   1.0033 ms

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-27 12:54:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 12:54:17  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = 0.250000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   6.7906 ms
  pytorch-compiled         :   6.7467 ms
  tilelang-separate        :   9.9415 ms
  ours (sonic)             :   6.9452 ms

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep |   Ours (Sonic) |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.3991 |     1.373  |         1.1442 |         0.1757 |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.7513 |     2.7519 |         3.09   |         0.273  |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    3.0564 |     3.0368 |         4.1457 |         1.0033 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |    6.7906 |     6.7467 |         9.9415 |         6.9452 |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 1.02x      | 1.22x          | 7.96x          |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 1.00x      | 0.89x          | 10.08x         |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 1.01x      | 0.74x          | 3.05x          |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 1.01x      | 0.68x          | 0.98x          |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 1.01x
  tilelang-separate        : 0.88x
  ours (sonic)             : 5.52x