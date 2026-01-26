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
2026-01-26 21:14:41  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[-1]`
[INFO] IR saved to gemm.ir
2026-01-26 21:14:45  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
❌ Error in tilelang-separate: Kernel expected 2 inputs, but 3 are provided.
2026-01-26 21:14:46  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[-1]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:14:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
❌ Error in ours (sonic): Kernel expected 4 inputs, but 5 are provided.
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = inf
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.3612 ms
  pytorch-compiled         :   1.3713 ms
  tilelang-separate        : FAILED (Kernel expected 2 inputs, but 3 are provided.)
  ours (sonic)             : FAILED (Kernel expected 4 inputs, but 5 are provided.)

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-26 21:14:52  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
❌ Error in tilelang-separate: Check failed: (M % kMPerWarp == 0) is false: M must be divisible by 16, but got 73
2026-01-26 21:14:52  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[4]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:14:57  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
❌ Error in ours (sonic): Kernel expected 4 inputs, but 5 are provided.
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = inf
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.7041 ms
  pytorch-compiled         :   2.7457 ms
2026-01-26 21:14:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
  tilelang-separate        : FAILED (Check failed: (M % kMPerWarp == 0) is false: M must be divisible by 16, but got 73)
  ours (sonic)             : FAILED (Kernel expected 4 inputs, but 5 are provided.)

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
2026-01-26 21:14:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
❌ Error in tilelang-separate: Check failed: (M % kMPerWarp == 0) is false: M must be divisible by 16, but got 127
2026-01-26 21:14:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[4]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:15:03  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
❌ Error in ours (sonic): Kernel expected 4 inputs, but 5 are provided.
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = inf
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   3.1930 ms
  pytorch-compiled         :   5.5421 ms
2026-01-26 21:15:04  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
  tilelang-separate        : FAILED (Check failed: (M % kMPerWarp == 0) is false: M must be divisible by 16, but got 127)
  ours (sonic)             : FAILED (Kernel expected 4 inputs, but 5 are provided.)

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-26 21:15:04  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:15:09  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
❌ Error in tilelang-separate: Kernel expected 2 inputs, but 3 are provided.
2026-01-26 21:15:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `grouped_gemm_sonic` with `out_idx=[4]`
[INFO] IR saved to grouped_gemm_sonic.ir
2026-01-26 21:15:15  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `grouped_gemm_sonic`
❌ Error in ours (sonic): Kernel expected 4 inputs, but 5 are provided.
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = inf
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  11.3036 ms
  pytorch-compiled         :  12.2778 ms
  tilelang-separate        : FAILED (Kernel expected 2 inputs, but 3 are provided.)
  ours (sonic)             : FAILED (Kernel expected 4 inputs, but 5 are provided.)

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled | TileLang-Sep   | Ours (Sonic)   |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.3612 |     1.3713 | FAILED         | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.7041 |     2.7457 | FAILED         | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    3.193  |     5.5421 | FAILED         | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |   11.3036 |    12.2778 | FAILED         | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 0.99x      | N/A            | N/A            |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 0.98x      | N/A            | N/A            |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 0.58x      | N/A            | N/A            |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 0.92x      | N/A            | N/A            |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 0.87x