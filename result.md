python benchmarks/benchmark_grouped_gemm.py 
================================================================================
Grouped GEMM Performance Benchmark
5 Versions: pytorch-loop, pytorch-compiled, tilelang-separate, tilelang-official, ours (sonic)
================================================================================
Device: NVIDIA H100 80GB HBM3
PyTorch version: 2.8.0+cu129

================================================================================
Config: B128_H512_I384_E8 - Small Test
  total_tokens=128, H=512, I=384, E=8
================================================================================
Expert distribution: min=3, max=52, mean=16.0

[1] Correctness Verification:
2026-01-27 13:30:45  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:30:49  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ✅ tilelang-official        : max_diff = 0.000000
  ✅ ours (sonic)             : max_diff = 0.000000

[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.1725 ms
  pytorch-compiled         :   1.3172 ms
  tilelang-separate        :   1.7358 ms
  tilelang-official        :   0.2267 ms
  ours (sonic)             :   0.2219 ms

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-27 13:30:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:30:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ✅ tilelang-official        : max_diff = 0.000000
  ✅ ours (sonic)             : max_diff = 0.000000

[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.6237 ms
  pytorch-compiled         :   1.6319 ms
  tilelang-separate        :   1.9716 ms
  tilelang-official        :   0.2684 ms
  ours (sonic)             :   0.2655 ms

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
2026-01-27 13:31:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:31:07  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ✅ tilelang-official        : max_diff = 0.000000
  ✅ ours (sonic)             : max_diff = 0.000000

[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   3.2912 ms
  pytorch-compiled         :   3.2686 ms
  tilelang-separate        :   4.3922 ms
  tilelang-official        :   1.0058 ms
  ours (sonic)             :   1.0071 ms

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-27 13:31:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:31:15  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ tilelang-official        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = 0.250000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   9.6263 ms
  pytorch-compiled         :   9.4196 ms
  tilelang-separate        :  10.9276 ms
  tilelang-official        :   7.0134 ms
  ours (sonic)             :   7.0483 ms

================================================================================
Config: B1024_H4096_I3072_E8 - Large tokens, few experts
  total_tokens=1024, H=4096, I=3072, E=8
================================================================================
Expert distribution: min=33, max=361, mean=128.0

[1] Correctness Verification:
2026-01-27 13:31:19  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:31:23  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.125000
  ❌ tilelang-official        : max_diff = 0.125000
  ❌ ours (sonic)             : max_diff = 0.125000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   1.4005 ms
  pytorch-compiled         :   1.5716 ms
  tilelang-separate        :   3.2064 ms
  tilelang-official        :   1.0799 ms
  ours (sonic)             :   1.0859 ms

================================================================================
Config: B1024_H4096_I3072_E16 - Large tokens, medium experts
  total_tokens=1024, H=4096, I=3072, E=16
================================================================================
Expert distribution: min=18, max=303, mean=64.0

[1] Correctness Verification:
2026-01-27 13:31:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:31:32  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.125000
  ❌ tilelang-official        : max_diff = 0.125000
  ❌ ours (sonic)             : max_diff = 0.125000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.7644 ms
  pytorch-compiled         :   2.4365 ms
  tilelang-separate        :   3.0856 ms
  tilelang-official        :   1.9303 ms
  ours (sonic)             :   1.9262 ms

================================================================================
Config: B1024_H4096_I3072_E32 - Large tokens, many experts
  total_tokens=1024, H=4096, I=3072, E=32
================================================================================
Expert distribution: min=4, max=276, mean=32.0

[1] Correctness Verification:
2026-01-27 13:31:36  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:31:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.125000
  ❌ tilelang-official        : max_diff = 0.125000
  ❌ ours (sonic)             : max_diff = 0.125000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   4.7775 ms
  pytorch-compiled         :   4.7821 ms
  tilelang-separate        :   7.0690 ms
  tilelang-official        :   3.6173 ms
  ours (sonic)             :   3.6493 ms

================================================================================
Config: B1024_H2048_I1536_E64 - Large E, smaller H/I
  total_tokens=1024, H=2048, I=1536, E=64
================================================================================
Expert distribution: min=1, max=196, mean=16.0

[1] Correctness Verification:
2026-01-27 13:31:45  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:31:49  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ✅ tilelang-official        : max_diff = 0.000000
  ✅ ours (sonic)             : max_diff = 0.000000

[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   9.3147 ms
  pytorch-compiled         :   9.6581 ms
  tilelang-separate        :  11.3114 ms
  tilelang-official        :   1.8790 ms
  ours (sonic)             :   1.8749 ms

================================================================================
Config: B1024_H1024_I768_E64 - Large E, small H/I
  total_tokens=1024, H=1024, I=768, E=64
================================================================================
Expert distribution: min=0, max=209, mean=16.0

[1] Correctness Verification:
2026-01-27 13:31:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:31:57  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ✅ tilelang-official        : max_diff = 0.000000
  ✅ ours (sonic)             : max_diff = 0.000000

[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   6.4908 ms
  pytorch-compiled         :   6.3428 ms
  tilelang-separate        :   7.7205 ms
  tilelang-official        :   0.5974 ms
  ours (sonic)             :   0.6016 ms

================================================================================
Config: B512_H4096_I3072_E64 - Fewer tokens, E=64
  total_tokens=512, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=99, mean=8.0

[1] Correctness Verification:
2026-01-27 13:32:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:32:06  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.125000
  ❌ tilelang-official        : max_diff = 0.125000
  ❌ ours (sonic)             : max_diff = 0.125000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  11.0837 ms
  pytorch-compiled         :  12.0844 ms
  tilelang-separate        :   9.4344 ms
  tilelang-official        :   6.9435 ms
  ours (sonic)             :   7.0278 ms

================================================================================
Config: B2048_H4096_I3072_E64 - More tokens, E=64
  total_tokens=2048, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=4, max=451, mean=32.0

[1] Correctness Verification:
2026-01-27 13:32:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:32:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.125000
  ❌ tilelang-official        : max_diff = 0.125000
  ❌ ours (sonic)             : max_diff = 0.125000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   7.2295 ms
  pytorch-compiled         :   7.2996 ms
  tilelang-separate        :  10.8092 ms
  tilelang-official        :   7.0835 ms
  ours (sonic)             :   7.1393 ms

================================================================================
Config: B1000_H4096_I3072_E64 - Non-pow2 tokens, E=64
  total_tokens=1000, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=204, mean=15.6

[1] Correctness Verification:
2026-01-27 13:32:19  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:32:22  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ tilelang-official        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = 0.250000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   9.6045 ms
  pytorch-compiled         :   9.6678 ms
  tilelang-separate        :  10.5215 ms
  tilelang-official        :   7.0208 ms
  ours (sonic)             :   7.0504 ms

================================================================================
Config: B1024_H4000_I3000_E64 - Non-pow2 H/I, E=64
  total_tokens=1024, H=4000, I=3000, E=64
================================================================================
Expert distribution: min=0, max=220, mean=16.0

[1] Correctness Verification:
2026-01-27 13:32:27  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `kernel` with `out_idx=[2]`
[INFO] IR saved to kernel.ir
2026-01-27 13:32:30  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `kernel`
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.125000
  ❌ tilelang-official        : max_diff = 0.125000
  ❌ ours (sonic)             : max_diff = 0.125000

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   9.7484 ms
  pytorch-compiled         :   7.1509 ms
  tilelang-separate        :  10.0992 ms
  tilelang-official        :   6.9243 ms
  ours (sonic)             :   6.9482 ms

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep |   Ours (Sonic) |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    1.1725 |     1.3172 |         1.7358 |         0.2219 |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    1.6237 |     1.6319 |         1.9716 |         0.2655 |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    3.2912 |     3.2686 |         4.3922 |         1.0071 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |    9.6263 |     9.4196 |        10.9276 |         7.0483 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E8  |    1.4005 |     1.5716 |         3.2064 |         1.0859 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E16 |    2.7644 |     2.4365 |         3.0856 |         1.9262 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E32 |    4.7775 |     4.7821 |         7.069  |         3.6493 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H2048_I1536_E64 |    9.3147 |     9.6581 |        11.3114 |         1.8749 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H1024_I768_E64  |    6.4908 |     6.3428 |         7.7205 |         0.6016 |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H4096_I3072_E64  |   11.0837 |    12.0844 |         9.4344 |         7.0278 |
+-----------------------+-----------+------------+----------------+----------------+
| B2048_H4096_I3072_E64 |    7.2295 |     7.2996 |        10.8092 |         7.1393 |
+-----------------------+-----------+------------+----------------+----------------+
| B1000_H4096_I3072_E64 |    9.6045 |     9.6678 |        10.5215 |         7.0504 |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4000_I3000_E64 |    9.7484 |     7.1509 |        10.0992 |         6.9482 |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 0.89x      | 0.68x          | 5.28x          |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 0.99x      | 0.82x          | 6.12x          |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 1.01x      | 0.75x          | 3.27x          |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 1.02x      | 0.88x          | 1.37x          |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E8  | 0.89x      | 0.44x          | 1.29x          |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E16 | 1.13x      | 0.90x          | 1.44x          |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E32 | 1.00x      | 0.68x          | 1.31x          |
+-----------------------+------------+----------------+----------------+
| B1024_H2048_I1536_E64 | 0.96x      | 0.82x          | 4.97x          |
+-----------------------+------------+----------------+----------------+
| B1024_H1024_I768_E64  | 1.02x      | 0.84x          | 10.79x         |
+-----------------------+------------+----------------+----------------+
| B512_H4096_I3072_E64  | 0.92x      | 1.17x          | 1.58x          |
+-----------------------+------------+----------------+----------------+
| B2048_H4096_I3072_E64 | 0.99x      | 0.67x          | 1.01x          |
+-----------------------+------------+----------------+----------------+
| B1000_H4096_I3072_E64 | 0.99x      | 0.91x          | 1.36x          |
+-----------------------+------------+----------------+----------------+
| B1024_H4000_I3000_E64 | 1.36x      | 0.97x          | 1.40x          |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 1.01x
  tilelang-separate        : 0.81x
  ours (sonic)             : 3.17x