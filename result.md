python benchmarks/benchmark_mhc_opt.py
Device: NVIDIA GeForce RTX 5090
================================================================================
Running mHC Post Correctness Tests
================================================================================

Config: num_tokens=1024, hc_mult=4, hidden_size=2560

--- Correctness Results ---
  torch.compile vs pytorch: max_diff=0.031250 ✅
  official vs pytorch: max_diff=0.031250 ✅
  optimized vs pytorch: max_diff=0.031250 ✅
  optimized vs official: max_diff=0.000000 ✅

================================================================================
Running mHC Post Performance Benchmarks
================================================================================

Benchmarking config: (256, 4, 2560)
  pytorch=0.055ms, torch.compile=0.037ms, official=0.007ms, optimized=0.007ms

Benchmarking config: (512, 4, 2560)
  pytorch=0.093ms, torch.compile=0.066ms, official=0.007ms, optimized=0.007ms

Benchmarking config: (1024, 4, 2560)
  pytorch=0.222ms, torch.compile=0.129ms, official=0.010ms, optimized=0.010ms

Benchmarking config: (2048, 4, 2560)
  pytorch=0.554ms, torch.compile=0.319ms, official=0.017ms, optimized=0.019ms

Benchmarking config: (4096, 4, 2560)
  pytorch=1.200ms, torch.compile=0.709ms, official=0.113ms, optimized=0.114ms

Benchmarking config: (8192, 4, 2560)
  pytorch=2.448ms, torch.compile=1.441ms, official=0.236ms, optimized=0.238ms

Benchmarking config: (1024, 4, 5120)
  pytorch=0.544ms, torch.compile=0.348ms, official=0.033ms, optimized=0.025ms

Benchmarking config: (2048, 4, 5120)
  pytorch=1.209ms, torch.compile=0.722ms, official=0.122ms, optimized=0.125ms

====================================================================================================
mHC Post Benchmark Summary (Latency in ms)
====================================================================================================

--- All Implementations vs PyTorch ---
Config               |    PyTorch |    Compile |   Official |  Optimized |  Speedup |  Opt/Off
----------------------------------------------------------------------------------------------------
(256, 4, 2560)       |      0.055 |      0.037 |      0.007 |      0.007 |    7.55x |    0.98x
(512, 4, 2560)       |      0.093 |      0.066 |      0.007 |      0.007 |   12.95x |    0.99x
(1024, 4, 2560)      |      0.222 |      0.129 |      0.010 |      0.010 |   21.39x |    1.00x
(2048, 4, 2560)      |      0.554 |      0.319 |      0.017 |      0.019 |   29.87x |    0.89x
(4096, 4, 2560)      |      1.200 |      0.709 |      0.113 |      0.114 |   10.51x |    0.99x
(8192, 4, 2560)      |      2.448 |      1.441 |      0.236 |      0.238 |   10.27x |    0.99x
(1024, 4, 5120)      |      0.544 |      0.348 |      0.033 |      0.025 |   21.42x |    1.29x
(2048, 4, 5120)      |      1.209 |      0.722 |      0.122 |      0.125 |    9.64x |    0.97x
----------------------------------------------------------------------------------------------------
Average              |          - |          - |          - |          - |   15.45x |    1.01x

--- Optimized vs Official Detailed Comparison ---
Config               |   Official |  Optimized |  Diff (ms) |      Ratio |     Winner
-------------------------------------------------------------------------------------
(256, 4, 2560)       |      0.007 |      0.007 |     -0.000 |      0.98x |        Tie
(512, 4, 2560)       |      0.007 |      0.007 |     -0.000 |      0.99x |        Tie
(1024, 4, 2560)      |      0.010 |      0.010 |     -0.000 |      1.00x |        Tie
(2048, 4, 2560)      |      0.017 |      0.019 |     -0.002 |      0.89x |   Official
(4096, 4, 2560)      |      0.113 |      0.114 |     -0.002 |      0.99x |        Tie
(8192, 4, 2560)      |      0.236 |      0.238 |     -0.003 |      0.99x |        Tie
(1024, 4, 5120)      |      0.033 |      0.025 |     +0.007 |      1.29x |  Optimized
(2048, 4, 5120)      |      0.122 |      0.125 |     -0.003 |      0.97x |        Tie
-------------------------------------------------------------------------------------

Summary: Optimized wins 1, Official wins 1, Ties 6

================================================================================
⚠️ MARGINAL DIFFERENCE: Optimized version is 1.01x vs official.
   Recommendation: Consider other optimizations.
================================================================================