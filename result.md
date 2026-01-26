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
❌ Error in ours (sonic): Use immutable variable `expert_id` outside its defining region, did you forget **alloc_var**?
variable `expert_id` is defined in frame: script.ir_builder.tir.IfFrame(0x2b3c0158), current frames: [script.ir_builder.tir.PrimFuncFrame(0x28cfbe18), tl.KernelLaunchFrame(0x293770e8), script.ir_builder.tir.LetFrame(0x294a6638), script.ir_builder.tir.LetFrame(0x294ed8e8), script.ir_builder.tir.LetFrame(0x294edd38), script.ir_builder.tir.LetFrame(0x2b3bece8), script.ir_builder.tir.IfFrame(0x2b3bef08), script.ir_builder.tir.ThenFrame(0x2b3bed48), script.ir_builder.tir.ForFrame(0x13cf5bc8), script.ir_builder.tir.ForFrame(0x2b3c2208), script.ir_builder.tir.LetFrame(0x2b3c1588), script.ir_builder.tir.IfFrame(0x2b3c15e8), script.ir_builder.tir.ThenFrame(0x2b3c0b48)].
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   0.7736 ms
  pytorch-compiled         :   0.7734 ms
  tilelang-separate        :   0.9149 ms
  ours (sonic)             : FAILED (Use immutable variable `expert_id` outside its defining region, did you forget **alloc_var**?
variable `expert_id` is defined in frame: script.ir_builder.tir.IfFrame(0x2b3c64e8), current frames: [script.ir_builder.tir.PrimFuncFrame(0x2b3c7ed8), tl.KernelLaunchFrame(0x2b3c1648), script.ir_builder.tir.LetFrame(0x2b3ca468), script.ir_builder.tir.LetFrame(0x293258c8), script.ir_builder.tir.LetFrame(0x2b3c83e8), script.ir_builder.tir.LetFrame(0x29ab5868), script.ir_builder.tir.IfFrame(0x2947ae08), script.ir_builder.tir.ThenFrame(0x2948ec88), script.ir_builder.tir.ForFrame(0x21b44268), script.ir_builder.tir.ForFrame(0x2b3c0fe8), script.ir_builder.tir.LetFrame(0x2b3c1408), script.ir_builder.tir.IfFrame(0x2b423c68), script.ir_builder.tir.ThenFrame(0x2b3c7778)].)

================================================================================
Config: B256_H1024_I768_E16 - Medium Test
  total_tokens=256, H=1024, I=768, E=16
================================================================================
Expert distribution: min=3, max=73, mean=16.0

[1] Correctness Verification:
2026-01-26 21:23:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:23:59  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:24:00  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:24:05  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:24:06  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:24:11  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:24:13  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:24:18  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
❌ Error in ours (sonic): Use immutable variable `expert_id` outside its defining region, did you forget **alloc_var**?
variable `expert_id` is defined in frame: script.ir_builder.tir.IfFrame(0x2b470ff8), current frames: [script.ir_builder.tir.PrimFuncFrame(0x2b507c38), tl.KernelLaunchFrame(0x2b671538), script.ir_builder.tir.LetFrame(0x2b5f14c8), script.ir_builder.tir.LetFrame(0x2b3d9808), script.ir_builder.tir.LetFrame(0x2b3e6598), script.ir_builder.tir.LetFrame(0x2b3dd4e8), script.ir_builder.tir.IfFrame(0x2b433368), script.ir_builder.tir.ThenFrame(0x2b509038), script.ir_builder.tir.ForFrame(0x2b599c58), script.ir_builder.tir.ForFrame(0x2b597188), script.ir_builder.tir.LetFrame(0x2b457308), script.ir_builder.tir.IfFrame(0x2b498028), script.ir_builder.tir.ThenFrame(0x19e36d88)].
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   2.7284 ms
  pytorch-compiled         :   2.7431 ms
  tilelang-separate        :   3.0952 ms
  ours (sonic)             : FAILED (Use immutable variable `expert_id` outside its defining region, did you forget **alloc_var**?
variable `expert_id` is defined in frame: script.ir_builder.tir.IfFrame(0x2b546118), current frames: [script.ir_builder.tir.PrimFuncFrame(0x2b599b48), tl.KernelLaunchFrame(0x2b3bfb28), script.ir_builder.tir.LetFrame(0x2b50a448), script.ir_builder.tir.LetFrame(0x2b3ede08), script.ir_builder.tir.LetFrame(0x2b598fe8), script.ir_builder.tir.LetFrame(0x2b599c58), script.ir_builder.tir.IfFrame(0x2b431528), script.ir_builder.tir.ThenFrame(0x2b3ed858), script.ir_builder.tir.ForFrame(0x2b433368), script.ir_builder.tir.ForFrame(0x2b3eead8), script.ir_builder.tir.LetFrame(0x2b470ff8), script.ir_builder.tir.IfFrame(0x2b635978), script.ir_builder.tir.ThenFrame(0x2b5ef0b8)].)

================================================================================
Config: B512_H2048_I1536_E32 - DeepSeek-V2-like
  total_tokens=512, H=2048, I=1536, E=32
================================================================================
Expert distribution: min=1, max=127, mean=16.0

[1] Correctness Verification:
❌ Error in ours (sonic): Use immutable variable `expert_id` outside its defining region, did you forget **alloc_var**?
variable `expert_id` is defined in frame: script.ir_builder.tir.IfFrame(0x2b8dda58), current frames: [script.ir_builder.tir.PrimFuncFrame(0x2b74d188), tl.KernelLaunchFrame(0x2b884ab8), script.ir_builder.tir.LetFrame(0x2b882828), script.ir_builder.tir.LetFrame(0x2b8b4158), script.ir_builder.tir.LetFrame(0x2b498318), script.ir_builder.tir.LetFrame(0x2b498458), script.ir_builder.tir.IfFrame(0x2b498618), script.ir_builder.tir.ThenFrame(0x2b4984b8), script.ir_builder.tir.ForFrame(0x2b8ddb78), script.ir_builder.tir.ForFrame(0x2b90e448), script.ir_builder.tir.LetFrame(0x2b8de818), script.ir_builder.tir.IfFrame(0x2b90d8d8), script.ir_builder.tir.ThenFrame(0x2b8ddd98)].
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ✅ tilelang-separate        : max_diff = 0.000000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :   5.3969 ms
  pytorch-compiled         :   5.4302 ms
  tilelang-separate        :   6.6048 ms
  ours (sonic)             : FAILED (Use immutable variable `expert_id` outside its defining region, did you forget **alloc_var**?
variable `expert_id` is defined in frame: script.ir_builder.tir.IfFrame(0x2b999ea8), current frames: [script.ir_builder.tir.PrimFuncFrame(0x2b3eb3c8), tl.KernelLaunchFrame(0x2b90d938), script.ir_builder.tir.LetFrame(0x2b888be8), script.ir_builder.tir.LetFrame(0x2b884908), script.ir_builder.tir.LetFrame(0x294090d8), script.ir_builder.tir.LetFrame(0x2b74d188), script.ir_builder.tir.IfFrame(0x2b99aca8), script.ir_builder.tir.ThenFrame(0x2b882628), script.ir_builder.tir.ForFrame(0x2b8e2fa8), script.ir_builder.tir.ForFrame(0x2b8de658), script.ir_builder.tir.LetFrame(0x2b9970d8), script.ir_builder.tir.IfFrame(0x2b90deb8), script.ir_builder.tir.ThenFrame(0x2b99a208)].)

================================================================================
Config: B1024_H4096_I3072_E64 - DeepSeek-V3-like (scaled)
  total_tokens=1024, H=4096, I=3072, E=64
================================================================================
Expert distribution: min=0, max=208, mean=16.0

[1] Correctness Verification:
2026-01-26 21:24:24  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:24:28  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:24:29  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:24:34  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:24:35  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:24:40  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
2026-01-26 21:24:41  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `gemm` with `out_idx=[2]`
[INFO] IR saved to gemm.ir
2026-01-26 21:24:45  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `gemm`
❌ Error in ours (sonic): Use immutable variable `expert_id` outside its defining region, did you forget **alloc_var**?
variable `expert_id` is defined in frame: script.ir_builder.tir.IfFrame(0x2caef5b8), current frames: [script.ir_builder.tir.PrimFuncFrame(0x2cb544a8), tl.KernelLaunchFrame(0x2caa2b18), script.ir_builder.tir.LetFrame(0x2ca9e308), script.ir_builder.tir.LetFrame(0x2cb41cf8), script.ir_builder.tir.LetFrame(0x2cac2068), script.ir_builder.tir.LetFrame(0x2ca61898), script.ir_builder.tir.IfFrame(0x2cb90878), script.ir_builder.tir.ThenFrame(0x2ca87b98), script.ir_builder.tir.ForFrame(0x2cbbfd88), script.ir_builder.tir.ForFrame(0x2c969328), script.ir_builder.tir.LetFrame(0x2ca0d6e8), script.ir_builder.tir.IfFrame(0x2cbbfaf8), script.ir_builder.tir.ThenFrame(0x2cb151e8)].
  ✅ pytorch-loop             : max_diff = 0.000000
  ✅ pytorch-compiled         : max_diff = 0.000000
  ❌ tilelang-separate        : max_diff = 0.250000
  ❌ ours (sonic)             : max_diff = inf

⚠️  Warning: Some versions have correctness issues!
Continuing with benchmark, but results may be invalid.


[2] Performance Benchmark (warmup=50, rep=500):
  pytorch-loop             :  11.0327 ms
  pytorch-compiled         :  11.0127 ms
  tilelang-separate        :  10.3501 ms
  ours (sonic)             : FAILED (Use immutable variable `expert_id` outside its defining region, did you forget **alloc_var**?
variable `expert_id` is defined in frame: script.ir_builder.tir.IfFrame(0x2ca070d8), current frames: [script.ir_builder.tir.PrimFuncFrame(0x2b44a608), tl.KernelLaunchFrame(0x2ca0b318), script.ir_builder.tir.LetFrame(0x2cb54b38), script.ir_builder.tir.LetFrame(0x2cb986e8), script.ir_builder.tir.LetFrame(0x2ca87b08), script.ir_builder.tir.LetFrame(0x2cb56ee8), script.ir_builder.tir.IfFrame(0x2cbf3ab8), script.ir_builder.tir.ThenFrame(0x2c985588), script.ir_builder.tir.ForFrame(0x2cad8828), script.ir_builder.tir.ForFrame(0x2cb54ea8), script.ir_builder.tir.LetFrame(0x2caec278), script.ir_builder.tir.IfFrame(0x2c93bc68), script.ir_builder.tir.ThenFrame(0x2c93cdd8)].)

================================================================================
SUMMARY: Latency Comparison (ms)
================================================================================
+-----------------------+-----------+------------+----------------+----------------+
| Config                |   PyTorch |   Compiled |   TileLang-Sep | Ours (Sonic)   |
+=======================+===========+============+================+================+
| B128_H512_I384_E8     |    0.7736 |     0.7734 |         0.9149 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B256_H1024_I768_E16   |    2.7284 |     2.7431 |         3.0952 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B512_H2048_I1536_E32  |    5.3969 |     5.4302 |         6.6048 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 |   11.0327 |    11.0127 |        10.3501 | FAILED         |
+-----------------------+-----------+------------+----------------+----------------+

================================================================================
SUMMARY: Speedup vs PyTorch Baseline
================================================================================
+-----------------------+------------+----------------+----------------+
| Config                | Compiled   | TileLang-Sep   | Ours (Sonic)   |
+=======================+============+================+================+
| B128_H512_I384_E8     | 1.00x      | 0.85x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B256_H1024_I768_E16   | 0.99x      | 0.88x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B512_H2048_I1536_E32  | 0.99x      | 0.82x          | N/A            |
+-----------------------+------------+----------------+----------------+
| B1024_H4096_I3072_E64 | 1.00x      | 1.07x          | N/A            |
+-----------------------+------------+----------------+----------------+

================================================================================
Average Speedup:
================================================================================
  pytorch-compiled         : 1.00x
  tilelang-separate        : 0.90x