python benchmarks/benchmark_topk_softmax.py
====================================================================================================
TopK + Softmax 性能对比
====================================================================================================
版本:
- pytorch-baseline: 全局softmax→topk
- torch.compile: 编译优化的baseline
- tilelang-separate: 官方TopK + PyTorch对k softmax
- ours: 单kernel融合 + sonic算法

计时: CUDA Events，100 warmup + 1000 runs 平均
注：torch.compile需要足够的warmup次数达到稳定性能
====================================================================================================

—— Running: B64_E160_K6 (B=64, E=160, K=6)
  ✓ pytorch-baseline: 0.0204 ms
  ✓ torch.compile: 0.0265 ms
  ✓ tilelang-separate: 0.0255 ms
  ✓ ours: 0.0100 ms
  DEBUG: pytorch-baseline scores.sum() = 0.213240 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.213240) max_diff=1.49e-08
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.213240) max_diff=1.76e-02
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.58e-01
  整体: OK

—— Running: B64_E256_K8 (B=64, E=256, K=8)
  ✓ pytorch-baseline: 0.0216 ms
  ✓ torch.compile: 0.0250 ms
  ✓ tilelang-separate: 0.0262 ms
  ✓ ours: 0.0112 ms
  DEBUG: pytorch-baseline scores.sum() = 0.188462 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.188462) max_diff=7.45e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.188462) max_diff=1.81e-02
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.60e-01
  整体: OK

—— Running: B128_E512_K16 (B=128, E=512, K=16)
  ✓ pytorch-baseline: 0.0242 ms
  ✓ torch.compile: 0.0352 ms
  ✓ tilelang-separate: 0.0293 ms
  ✓ ours: 0.0192 ms
  DEBUG: pytorch-baseline scores.sum() = 0.193001 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.193001) max_diff=7.45e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.193001) max_diff=1.30e-02
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=2.49e-01
  整体: OK

—— Running: B64_E1024_K8 (B=64, E=1024, K=8)
  ✓ pytorch-baseline: 0.0274 ms
  ✓ torch.compile: 0.0372 ms
  ✓ tilelang-separate: 0.0286 ms
  ✓ ours: 0.0132 ms
  DEBUG: pytorch-baseline scores.sum() = 0.077987 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.077987) max_diff=3.73e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.077987) max_diff=6.71e-03
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.78e-01
  整体: OK

—— Running: B64_E2048_K16 (B=64, E=2048, K=16)
  ✓ pytorch-baseline: 0.0362 ms
  ✓ torch.compile: 0.0447 ms
  ✓ tilelang-separate: 0.0343 ms
  ✓ ours: 0.0319 ms
  DEBUG: pytorch-baseline scores.sum() = 0.078677 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.078677) max_diff=3.73e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.078677) max_diff=3.75e-03
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=2.24e-01
  整体: OK

====================================================================================================
Latency (ms)
====================================================================================================
配置                 pytorch-baseline    torch.compile   tilelang-separate         ours
-------------------------------------------------------------------------------------
B64_E160_K6                  0.0204           0.0265              0.0255       0.0100
B64_E256_K8                  0.0216           0.0250              0.0262       0.0112
B128_E512_K16                0.0242           0.0352              0.0293       0.0192
B64_E1024_K8                 0.0274           0.0372              0.0286       0.0132
B64_E2048_K16                0.0362           0.0447              0.0343       0.0319

====================================================================================================
Speedup vs pytorch-baseline
====================================================================================================
配置                 pytorch-baseline    torch.compile   tilelang-separate         ours
-------------------------------------------------------------------------------------
B64_E160_K6                   1.00x            0.77x               0.80x        2.05x
B64_E256_K8                   1.00x            0.86x               0.82x        1.93x
B128_E512_K16                 1.00x            0.69x               0.83x        1.26x
B64_E1024_K8                  1.00x            0.74x               0.96x        2.07x
B64_E2048_K16                 1.00x            0.81x               1.06x        1.14x