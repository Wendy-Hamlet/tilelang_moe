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
2026-01-26 21:19:35  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[-1]`
[INFO] IR saved to gemm.ir
2026-01-26 21:19:39  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:19:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:19:45  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:19:46  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:19:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = 126.500000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.3516 ms
  pytorch-compiled         :   1.3560 ms
  tilelang-separate        :   1.5431 ms
  ours (sonic)             :   0.2643 ms

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-26 21:19:55  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
❌ Error in tilelang-separate: warp_row_tiles must be divisible by 16, got 20
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = inf
  ❌ ours (sonic)             : max_diff = 190.250000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.6973 ms
  pytorch-compiled         :   2.7109 ms
2026-01-26 21:19:56  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
  tilelang-separate        : FAILED (warp_row_tiles must be divisible by 16, got 20)
  ours (sonic)             :   0.3185 ms

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
2026-01-26 21:19:57  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:20:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:20:03  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:20:06  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:20:07  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:20:12  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:20:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:20:18  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:20:19  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:20:24  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = 290.500000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   5.3023 ms
  pytorch-compiled         :   5.0391 ms
  tilelang-separate        :   6.5242 ms
  ours (sonic)             :   0.3936 ms

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-26 21:20:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:20:32  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:20:33  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
❌ Error in tilelang-separate: warp_row_tiles must be divisible by 16, got 20
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = inf
  ❌ ours (sonic)             : max_diff = 438.500000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  10.7881 ms
  pytorch-compiled         :  10.8287 ms
2026-01-26 21:20:35  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
  tilelang-separate        : FAILED (warp_row_tiles must be divisible by 16, got 20)
  ours (sonic)             :   0.5615 ms

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled | TileLang-Sep   |   Ours (Sonic) |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.3516 |     1.356  | 1.5431         |         0.2643 |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.6973 |     2.7109 | FAILED         |         0.3185 |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    5.3023 |     5.0391 | 6.5242         |         0.3936 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |   10.7881 |    10.8287 | FAILED         |         0.5615 |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 1.00x      | 0.88x          | 5.11x          |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 1.00x      | N/A            | 8.47x          |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 1.05x      | 0.81x          | 13.47x         |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 1.00x      | N/A            | 19.21x         |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 1.01x
  tilelang-separate        : 0.84x
  ours (sonic)             : 11.57x