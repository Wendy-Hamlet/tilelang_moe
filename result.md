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
  ✓ pytorch-baseline: 0.0258 ms
  ✓ torch.compile: 1.2739 ms
  ✓ tilelang-separate: 0.0565 ms
  ✓ ours: 0.0153 ms
  DEBUG: pytorch-baseline scores.sum() = 0.213240 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.213240) max_diff=1.49e-08
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.58e-01
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.58e-01
  整体: OK

—— Running: B64_E256_K8 (B=64, E=256, K=8)
  ✓ pytorch-baseline: 0.0280 ms
  ✓ torch.compile: 1.0038 ms
  ✓ tilelang-separate: 0.0414 ms
  ✓ ours: 0.0093 ms
  DEBUG: pytorch-baseline scores.sum() = 0.188462 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.188462) max_diff=7.45e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.60e-01
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.60e-01
  整体: OK

—— Running: B128_E512_K16 (B=128, E=512, K=16)
  ✓ pytorch-baseline: 0.0242 ms
  ✓ torch.compile: 0.7219 ms
  ✓ tilelang-separate: 0.0390 ms
  ✓ ours: 0.0156 ms
  DEBUG: pytorch-baseline scores.sum() = 0.193001 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.193001) max_diff=7.45e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=2.49e-01
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=2.49e-01
  整体: OK

—— Running: B64_E1024_K8 (B=64, E=1024, K=8)
  ✓ pytorch-baseline: 0.0250 ms
  ✓ torch.compile: 0.7110 ms
  ✓ tilelang-separate: 0.0360 ms
  ✓ ours: 0.0099 ms
  DEBUG: pytorch-baseline scores.sum() = 0.077987 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.077987) max_diff=3.73e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.78e-01
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.78e-01
  整体: OK

—— Running: B64_E2048_K16 (B=64, E=2048, K=16)
  ✓ pytorch-baseline: 0.0311 ms
  ✓ torch.compile: 0.7241 ms
  ✓ tilelang-separate: 0.0378 ms
  ✓ ours: 0.0283 ms
  DEBUG: pytorch-baseline scores.sum() = 0.078677 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.078677) max_diff=3.73e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=2.24e-01
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=2.24e-01
  整体: OK

====================================================================================================
Latency (ms)
====================================================================================================
配置                 pytorch-baseline    torch.compile   tilelang-separate         ours
-------------------------------------------------------------------------------------
B64_E160_K6                  0.0258           1.2739              0.0565       0.0153
B64_E256_K8                  0.0280           1.0038              0.0414       0.0093
B128_E512_K16                0.0242           0.7219              0.0390       0.0156
B64_E1024_K8                 0.0250           0.7110              0.0360       0.0099
B64_E2048_K16                0.0311           0.7241              0.0378       0.0283

====================================================================================================
Speedup vs pytorch-baseline
====================================================================================================
配置                 pytorch-baseline    torch.compile   tilelang-separate         ours
-------------------------------------------------------------------------------------
B64_E160_K6                   1.00x            0.02x               0.46x        1.69x
B64_E256_K8                   1.00x            0.03x               0.68x        3.01x
B128_E512_K16                 1.00x            0.03x               0.62x        1.54x
B64_E1024_K8                  1.00x            0.04x               0.69x        2.52x
B64_E2048_K16                 1.00x            0.04x               0.82x        1.10x