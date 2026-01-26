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
2026-01-26 21:39:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:39:35  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = 126.500000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.4457 ms
  pytorch-compiled         :   1.4588 ms
  tilelang-separate        :   1.3146 ms
  ours (sonic)             :   0.2246 ms

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-26 21:39:39  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[3]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:39:43  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = 190.250000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.6673 ms
  pytorch-compiled         :   2.6774 ms
  tilelang-separate        :   2.0449 ms
  ours (sonic)             :   0.2366 ms

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
2026-01-26 21:39:46  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[3]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:39:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = 290.500000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   5.3321 ms
  pytorch-compiled         :   5.3325 ms
  tilelang-separate        :   6.4534 ms
  ours (sonic)             :   0.3162 ms

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-26 21:39:54  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[3]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:40:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = 419.000000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  10.8541 ms
  pytorch-compiled         :  11.8871 ms
  tilelang-separate        :  15.0598 ms
  ours (sonic)             :   0.5463 ms

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep |   Ours (Sonic) |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.4457 |     1.4588 |         1.3146 |         0.2246 |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.6673 |     2.6774 |         2.0449 |         0.2366 |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    5.3321 |     5.3325 |         6.4534 |         0.3162 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |   10.8541 |    11.8871 |        15.0598 |         0.5463 |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 0.99x      | 1.10x          | 6.44x          |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 1.00x      | 1.30x          | 11.27x         |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 1.00x      | 0.83x          | 16.86x         |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 0.91x      | 0.72x          | 19.87x         |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 0.98x
  tilelang-separate        : 0.99x
  ours (sonic)             : 13.61x