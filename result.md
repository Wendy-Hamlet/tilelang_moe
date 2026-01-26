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
❌ Error in ours (sonic): 'BufferLoad' object does not support item assignment
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.1222 ms
  pytorch-compiled         :   1.2919 ms
  tilelang-separate        :   1.5196 ms
  ours (sonic)             : FAILED ('BufferLoad' object does not support item assignment)

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
❌ Error in ours (sonic): 'BufferLoad' object does not support item assignment
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.6410 ms
  pytorch-compiled         :   2.3400 ms
  tilelang-separate        :   3.0599 ms
  ours (sonic)             : FAILED ('BufferLoad' object does not support item assignment)

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
❌ Error in ours (sonic): 'BufferLoad' object does not support item assignment
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   5.2651 ms
  pytorch-compiled         :   5.2954 ms
  tilelang-separate        :   8.8585 ms
  ours (sonic)             : FAILED ('BufferLoad' object does not support item assignment)

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
❌ Error in ours (sonic): 'BufferLoad' object does not support item assignment
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  10.8073 ms
  pytorch-compiled         :  12.4619 ms
  tilelang-separate        :  17.8758 ms
  ours (sonic)             : FAILED ('BufferLoad' object does not support item assignment)

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep | Ours (Sonic)   |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.1222 |     1.2919 |         1.5196 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.641  |     2.34   |         3.0599 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    5.2651 |     5.2954 |         8.8585 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |   10.8073 |    12.4619 |        17.8758 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 0.87x      | 0.74x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 1.13x      | 0.86x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 0.99x      | 0.59x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 0.87x      | 0.60x          | N/A            |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 0.96x
  tilelang-separate        : 0.70x