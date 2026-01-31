python benchmarks/benchmark_mhc_opt.py
Device: NVIDIA GeForce RTX 5090
================================================================================
Running mHC Correctness Tests
================================================================================

Config: num_tokens=1024, hc_mult=4, hidden_size=2560

--- mHC Pre Correctness ---
2026-01-31 11:35:14  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_pre_gemm_sqrsum_official_kernel` with `out_idx=None`
2026-01-31 11:35:16  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_pre_gemm_sqrsum_official_kernel`
2026-01-31 11:35:17  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_pre_big_fuse_official_kernel` with `out_idx=None`
2026-01-31 11:35:19  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_pre_big_fuse_official_kernel`
2026-01-31 11:35:20  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `mhc_pre_gemm_sqrsum_optimized_kernel` with `out_idx=None`
2026-01-31 11:35:22  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `mhc_pre_gemm_sqrsum_optimized_kernel`
Traceback (most recent call last):
  File "/home/chjin/CUDA/tilelang_moe/benchmarks/benchmark_mhc_opt.py", line 516, in <module>
    correctness_pass = run_correctness_tests()
  File "/home/chjin/CUDA/tilelang_moe/benchmarks/benchmark_mhc_opt.py", line 442, in run_correctness_tests
    pre_results = test_mhc_pre_correctness(num_tokens, hc_mult, hidden_size)
  File "/home/chjin/CUDA/tilelang_moe/benchmarks/benchmark_mhc_opt.py", line 99, in test_mhc_pre_correctness
    post_opt, comb_opt, layer_opt = mhc_pre_optimized(residual, fn, hc_scale, hc_base, **params)
                                    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chjin/CUDA/tilelang_moe/kernels/mhc_optimized.py", line 628, in mhc_pre_optimized
    mhc_pre_gemm_sqrsum_optimized_kernel(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        residual.view(num_tokens, hc_hidden_size),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
        hc_hidden_size,
        ^^^^^^^^^^^^^^^
    )
    ^
  File "/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/tilelang/jit/__init__.py", line 441, in __call__
    return kernel(*kernel_args.values())
  File "/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/tilelang/jit/kernel.py", line 207, in __call__
    return self.torch_function(*args, **kwds)
           ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
  File "/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/tilelang/jit/adapter/tvm_ffi.py", line 255, in func
    executable(*tensor_list)
    ~~~~~~~~~~^^^^^^^^^^^^^^
  File "/home/chjin/miniconda3/envs/tilelang/lib/python3.13/site-packages/tilelang/3rdparty/tvm/python/tvm/runtime/executable.py", line 42, in __call__
    return self.jit().main(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "python/tvm_ffi/cython/function.pxi", line 923, in tvm_ffi.core.Function.__call__
  File "<unknown>", line 0, in tvm::runtime::detail::LogFatal::Entry::Finalize()
tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 196608