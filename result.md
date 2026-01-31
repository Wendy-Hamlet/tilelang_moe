python benchmarks/benchmark_mhc.py
Device: NVIDIA GeForce RTX 5090
TileLang available: True
================================================================================
Running mHC Correctness Tests
================================================================================

Config: num_tokens=1024, hc_mult=4, hidden_size=2560

--- mHC Pre Correctness ---
/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:321: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/torch/_inductor/lowering.py:7627: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
2026-01-31 11:02:56  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_pre_gemm_sqrsum_kernel` with `out_idx=None`
2026-01-31 11:02:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_pre_gemm_sqrsum_kernel`
2026-01-31 11:02:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_pre_big_fuse_kernel` with `out_idx=None`
2026-01-31 11:03:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_pre_big_fuse_kernel`
  torch.compile_post: max_diff=0.000000 ✅
  torch.compile_comb: max_diff=0.000000 ✅
  torch.compile_layer: max_diff=0.007812 ✅
  tilelang_post: max_diff=0.000000 ✅
  tilelang_comb: max_diff=0.000000 ✅
  tilelang_layer: max_diff=0.015625 ✅

--- mHC Post Correctness ---
2026-01-31 11:03:01  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_post_kernel` with `out_idx=None`
2026-01-31 11:03:03  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_post_kernel`
  torch.compile: max_diff=0.031250 ✅
  tilelang: max_diff=0.031250 ✅

================================================================================
Running mHC Performance Benchmarks
================================================================================

Benchmarking config: (256, 4, 2560)
/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/torch/_inductor/lowering.py:7627: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
  Pre:  pytorch=0.264ms, torch.compile=0.111ms, tilelang=0.050ms
  Post: pytorch=0.055ms, torch.compile=0.037ms, tilelang=0.007ms

Benchmarking config: (512, 4, 2560)
  Pre:  pytorch=0.268ms, torch.compile=0.110ms, tilelang=0.050ms
  Post: pytorch=0.093ms, torch.compile=0.066ms, tilelang=0.007ms

Benchmarking config: (1024, 4, 2560)
  Pre:  pytorch=0.295ms, torch.compile=0.110ms, tilelang=0.052ms
  Post: pytorch=0.232ms, torch.compile=0.139ms, tilelang=0.010ms

Benchmarking config: (2048, 4, 2560)
  Pre:  pytorch=0.585ms, torch.compile=0.189ms, tilelang=0.058ms
  Post: pytorch=0.554ms, torch.compile=0.319ms, tilelang=0.017ms

Benchmarking config: (4096, 4, 2560)
  Pre:  pytorch=1.071ms, torch.compile=0.411ms, tilelang=0.073ms
  Post: pytorch=1.208ms, torch.compile=0.710ms, tilelang=0.113ms

Benchmarking config: (8192, 4, 2560)
  Pre:  pytorch=2.070ms, torch.compile=0.817ms, tilelang=0.249ms
  Post: pytorch=2.454ms, torch.compile=1.441ms, tilelang=0.236ms

Benchmarking config: (1024, 4, 5120)
/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/torch/_inductor/lowering.py:7627: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
2026-01-31 11:03:08  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_pre_gemm_sqrsum_kernel` with `out_idx=None`
2026-01-31 11:03:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_pre_gemm_sqrsum_kernel`
2026-01-31 11:03:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_pre_big_fuse_kernel` with `out_idx=None`
2026-01-31 11:03:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_pre_big_fuse_kernel`
2026-01-31 11:03:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_post_kernel` with `out_idx=None`
2026-01-31 11:03:15  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_post_kernel`
  Pre:  pytorch=0.551ms, torch.compile=0.206ms, tilelang=0.099ms
  Post: pytorch=0.544ms, torch.compile=0.349ms, tilelang=0.036ms

Benchmarking config: (2048, 4, 5120)
  Pre:  pytorch=1.079ms, torch.compile=0.408ms, tilelang=0.107ms
  Post: pytorch=1.194ms, torch.compile=0.723ms, tilelang=0.122ms

Benchmarking config: (1024, 8, 2560)
/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/torch/_inductor/lowering.py:7627: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
  Error: Assertion failed

Benchmarking config: (2048, 8, 2560)
  Error: Assertion failed

================================================================================
mHC Pre Benchmark Summary (Latency in ms)
================================================================================
Config                    |    PyTorch |  torch.compile |   TileLang |    Speedup
--------------------------------------------------------------------------------
(256, 4, 2560)            |      0.264 |          0.111 |      0.050 |      5.33x
(512, 4, 2560)            |      0.268 |          0.110 |      0.050 |      5.38x
(1024, 4, 2560)           |      0.295 |          0.110 |      0.052 |      5.65x
(2048, 4, 2560)           |      0.585 |          0.189 |      0.058 |     10.18x
(4096, 4, 2560)           |      1.071 |          0.411 |      0.073 |     14.59x
(8192, 4, 2560)           |      2.070 |          0.817 |      0.249 |      8.30x
(1024, 4, 5120)           |      0.551 |          0.206 |      0.099 |      5.59x
(2048, 4, 5120)           |      1.079 |          0.408 |      0.107 |     10.08x

================================================================================
mHC Post Benchmark Summary (Latency in ms)
================================================================================
Config                    |    PyTorch |  torch.compile |   TileLang |    Speedup
--------------------------------------------------------------------------------
(256, 4, 2560)            |      0.055 |          0.037 |      0.007 |      7.64x
(512, 4, 2560)            |      0.093 |          0.066 |      0.007 |     13.11x
(1024, 4, 2560)           |      0.232 |          0.139 |      0.010 |     22.41x
(2048, 4, 2560)           |      0.554 |          0.319 |      0.017 |     32.68x
(4096, 4, 2560)           |      1.208 |          0.710 |      0.113 |     10.73x
(8192, 4, 2560)           |      2.454 |          1.441 |      0.236 |     10.41x
(1024, 4, 5120)           |      0.544 |          0.349 |      0.036 |     15.06x
(2048, 4, 5120)           |      1.194 |          0.723 |      0.122 |      9.75x

mHC Pre Average Speedup (TileLang vs PyTorch): 8.14x