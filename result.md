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
  ✓ pytorch-baseline: 0.0214 ms
  ✓ torch.compile: 0.0203 ms
  ✓ tilelang-separate: 0.0247 ms
  ✓ ours: 0.0099 ms
  DEBUG: pytorch-baseline scores.sum() = 0.213240 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.213240) max_diff=1.49e-08
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.213240) max_diff=1.75e-02
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.58e-01
  整体: OK

—— Running: B64_E256_K8 (B=64, E=256, K=8)
  ✓ pytorch-baseline: 0.0197 ms
  ✓ torch.compile: 0.0213 ms
  ✓ tilelang-separate: 0.0261 ms
  ✓ ours: 0.0112 ms
  DEBUG: pytorch-baseline scores.sum() = 0.188462 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.188462) max_diff=7.45e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.188462) max_diff=1.81e-02
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.60e-01
  整体: OK

—— Running: B128_E512_K16 (B=128, E=512, K=16)
  ✓ pytorch-baseline: 0.0248 ms
  ✓ torch.compile: 0.0232 ms
  ✓ tilelang-separate: 0.0286 ms
  ✓ ours: 0.0186 ms
  DEBUG: pytorch-baseline scores.sum() = 0.193001 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.193001) max_diff=7.45e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.193001) max_diff=1.30e-02
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=2.49e-01
  整体: OK

—— Running: B64_E1024_K8 (B=64, E=1024, K=8)
  ✓ pytorch-baseline: 0.0266 ms
  ✓ torch.compile: 0.0262 ms
  ✓ tilelang-separate: 0.0282 ms
  ✓ ours: 0.0134 ms
  DEBUG: pytorch-baseline scores.sum() = 0.077987 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.077987) max_diff=3.73e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.077987) max_diff=6.71e-03
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=3.78e-01
  整体: OK

—— Running: B64_E2048_K16 (B=64, E=2048, K=16)
  ✓ pytorch-baseline: 0.0352 ms
  ✓ torch.compile: 0.0332 ms
  ✓ tilelang-separate: 0.0325 ms
  ✓ ours: 0.0314 ms
  DEBUG: pytorch-baseline scores.sum() = 0.078677 (期望 < 1.0)
  正确性 (vs pytorch-baseline):
    ✓ torch.compile      overlap= 100.0% sum_valid= 100.0% (scores.sum=0.078677) max_diff=3.73e-09
    ✓ tilelang-separate  overlap= 100.0% sum_valid= 100.0% (scores.sum=0.078677) max_diff=3.41e-03
    ✓ ours               overlap= 100.0% sum_valid= 100.0% (scores.sum=1.000000) max_diff=2.24e-01
  整体: OK

====================================================================================================
Latency (ms)
====================================================================================================
配置                 pytorch-baseline    torch.compile   tilelang-separate         ours
-------------------------------------------------------------------------------------
B64_E160_K6                  0.0214           0.0203              0.0247       0.0099
B64_E256_K8                  0.0197           0.0213              0.0261       0.0112
B128_E512_K16                0.0248           0.0232              0.0286       0.0186
B64_E1024_K8                 0.0266           0.0262              0.0282       0.0134
B64_E2048_K16                0.0352           0.0332              0.0325       0.0314

====================================================================================================
Speedup vs pytorch-baseline
====================================================================================================
配置                 pytorch-baseline    torch.compile   tilelang-separate         ours
-------------------------------------------------------------------------------------
B64_E160_K6                   1.00x            1.06x               0.87x        2.18x
B64_E256_K8                   1.00x            0.92x               0.76x        1.77x
B128_E512_K16                 1.00x            1.07x               0.87x        1.33x
B64_E1024_K8                  1.00x            1.02x               0.95x        1.99x
B64_E2048_K16                 1.00x            1.06x               1.08x        1.12x