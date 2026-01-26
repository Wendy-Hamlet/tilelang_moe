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
2026-01-26 21:58:30  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:58:37  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = 97.687500

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.2720 ms
  pytorch-compiled         :   1.2876 ms
  tilelang-separate        :   1.4528 ms
  ours (sonic)             :   0.4515 ms

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-26 21:58:41  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[4]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:58:46  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = 208.375000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.4761 ms
  pytorch-compiled         :   2.4827 ms
  tilelang-separate        :   2.8018 ms
  ours (sonic)             :   0.6305 ms

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
2026-01-26 21:58:50  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[4]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:58:55  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = 260.000000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   6.2353 ms
  pytorch-compiled         :   4.6648 ms
  tilelang-separate        :   5.6926 ms
  ours (sonic)             :   1.5445 ms

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-26 21:58:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[4]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:59:07  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = 436.250000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  12.3672 ms
  pytorch-compiled         :  10.3934 ms
  tilelang-separate        :  14.0631 ms
  ours (sonic)             :   7.6069 ms

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep |   Ours (Sonic) |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.272  |     1.2876 |         1.4528 |         0.4515 |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.4761 |     2.4827 |         2.8018 |         0.6305 |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    6.2353 |     4.6648 |         5.6926 |         1.5445 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |   12.3672 |    10.3934 |        14.0631 |         7.6069 |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 0.99x      | 0.88x          | 2.82x          |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 1.00x      | 0.88x          | 3.93x          |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 1.34x      | 1.10x          | 4.04x          |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 1.19x      | 0.88x          | 1.63x          |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 1.13x
  tilelang-separate        : 0.93x
  ours (sonic)             : 3.10x