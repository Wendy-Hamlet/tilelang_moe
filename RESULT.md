python tests/test_topk_softmax_conventions.py
TopK+Softmax Convention Correctness Verification
GPU: NVIDIA H100 80GB HBM3
Batch size: 512

==========================================================================================
Convention Correctness Verification
==========================================================================================

Table 1: Index Equivalence  (all three conventions should select identical experts)
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| Model        | Config    | Conv   | Renorm   | A-norenorm==B   | A-renorm==B   | A-norenorm==A-renorm   |
+==============+===========+========+==========+=================+===============+========================+
| Mixtral-HF   | E=8 K=2   | A      | True     | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| DBRX         | E=16 K=4  | A      | True     | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| Qwen3-30B    | E=128 K=8 | A      | True     | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| DeepSeek-V2  | E=160 K=6 | A      | False    | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| OLMoE        | E=64 K=8  | A      | False    | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| Qwen2-57B    | E=64 K=8  | A      | False    | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| Mixtral-Mist | E=8 K=2   | B      | False    | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+

Table 2: Score Differences
  'A-renorm vs B max diff': expect ~1e-8 (fp32 rounding, algebraically exact)
  'delta': probability mass outside top-K;  'B/A-norenorm ratio': expect 1/(1-delta)
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| Model        | Config    |   sum(A-norenorm) |   sum(A-renorm) |   sum(B) |   A-renorm vs B |   A-renorm vs B |   delta | B/A-norenorm   |
|              |           |                   |                 |          |        max diff |       mean diff |         | ratio range    |
+==============+===========+===================+=================+==========+=================+=================+=========+================+
| Mixtral-HF   | E=8 K=2   |            0.5605 |               1 |        1 |        1.19e-07 |        1.8e-08  |   0.439 | 1.09-3.13      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| DBRX         | E=16 K=4  |            0.5963 |               1 |        1 |        1.19e-07 |        1.1e-08  |   0.404 | 1.13-2.69      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| Qwen3-30B    | E=128 K=8 |            0.2881 |               1 |        1 |        5.96e-08 |        6.04e-09 |   0.712 | 2.26-6.17      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| DeepSeek-V2  | E=160 K=6 |            0.21   |               1 |        1 |        1.19e-07 |        8.01e-09 |   0.79  | 2.40-7.92      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| OLMoE        | E=64 K=8  |            0.4223 |               1 |        1 |        5.96e-08 |        6.1e-09  |   0.578 | 1.58-3.31      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| Qwen2-57B    | E=64 K=8  |            0.4311 |               1 |        1 |        5.96e-08 |        6.17e-09 |   0.569 | 1.57-3.39      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| Mixtral-Mist | E=8 K=2   |            0.5689 |               1 |        1 |        1.19e-07 |        1.81e-08 |   0.431 | 1.15-2.93      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+

==========================================================================================
Detail: A-norenorm vs B  (systematic uniform scale, not random noise)
==========================================================================================

  Mixtral-HF  (E=8, K=2, renorm=True)  [mistralai/Mixtral-8x7B-v0.1 (HuggingFace)]
    delta (mass outside top-K) = 0.4354
    theoretical 1/(1-delta)    = 1.7712
    actual B/A-norenorm ratio  : min=1.0861  max=3.0800  mean=1.8366
    sum(A-norenorm) mean = 0.5646  sum(B) mean = 1.0000
    A-renorm vs B: max diff = 1.19e-07  (fp32 rounding only)

  DBRX  (E=16, K=4, renorm=True)  [databricks/dbrx]
    delta (mass outside top-K) = 0.4039
    theoretical 1/(1-delta)    = 1.6777
    actual B/A-norenorm ratio  : min=1.1040  max=2.7853  mean=1.7124
    sum(A-norenorm) mean = 0.5961  sum(B) mean = 1.0000
    A-renorm vs B: max diff = 5.96e-08  (fp32 rounding only)

  Qwen3-30B  (E=128, K=8, renorm=True)  [Qwen/Qwen3-30B-A3B]
    delta (mass outside top-K) = 0.7127
    theoretical 1/(1-delta)    = 3.4813
    actual B/A-norenorm ratio  : min=2.1559  max=5.1128  mean=3.5580
    sum(A-norenorm) mean = 0.2873  sum(B) mean = 1.0000
    A-renorm vs B: max diff = 5.96e-08  (fp32 rounding only)

  DeepSeek-V2  (E=160, K=6, renorm=False)  [deepseek-ai/DeepSeek-V2]
    delta (mass outside top-K) = 0.7909
    theoretical 1/(1-delta)    = 4.7828
    actual B/A-norenorm ratio  : min=2.3181  max=7.4053  mean=4.9224
    sum(A-norenorm) mean = 0.2091  sum(B) mean = 1.0000

  OLMoE  (E=64, K=8, renorm=False)  [allenai/OLMoE-1B-7B]
    delta (mass outside top-K) = 0.5732
    theoretical 1/(1-delta)    = 2.3430
    actual B/A-norenorm ratio  : min=1.4850  max=3.3312  mean=2.3834
    sum(A-norenorm) mean = 0.4268  sum(B) mean = 1.0000

  Qwen2-57B  (E=64, K=8, renorm=False)  [Qwen/Qwen2-57B-A14B]
    delta (mass outside top-K) = 0.5722
    theoretical 1/(1-delta)    = 2.3376
    actual B/A-norenorm ratio  : min=1.5360  max=3.3163  mean=2.3751
    sum(A-norenorm) mean = 0.4278  sum(B) mean = 1.0000

Done.

python tests/test_topk_softmax_conventions.py
TopK+Softmax Convention Correctness Verification
GPU: NVIDIA H100 80GB HBM3
Batch size: 512

==========================================================================================
Convention Correctness Verification
==========================================================================================

Table 1: Index Equivalence  (all three conventions should select identical experts)
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| Model        | Config    | Conv   | Renorm   | A-norenorm==B   | A-renorm==B   | A-norenorm==A-renorm   |
+==============+===========+========+==========+=================+===============+========================+
| Mixtral-HF   | E=8 K=2   | A      | True     | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| DBRX         | E=16 K=4  | A      | True     | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| Qwen3-30B    | E=128 K=8 | A      | True     | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| DeepSeek-V2  | E=160 K=6 | A      | False    | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| OLMoE        | E=64 K=8  | A      | False    | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| Qwen2-57B    | E=64 K=8  | A      | False    | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+
| Mixtral-Mist | E=8 K=2   | B      | False    | YES             | YES           | YES                    |
+--------------+-----------+--------+----------+-----------------+---------------+------------------------+

Table 2: Score Differences
  'A-renorm vs B max diff': expect ~1e-8 (fp32 rounding, algebraically exact)
  'delta': probability mass outside top-K;  'B/A-norenorm ratio': expect 1/(1-delta)
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| Model        | Config    |   sum(A-norenorm) |   sum(A-renorm) |   sum(B) |   A-renorm vs B |   A-renorm vs B |   delta | B/A-norenorm   |
|              |           |                   |                 |          |        max diff |       mean diff |         | ratio range    |
+==============+===========+===================+=================+==========+=================+=================+=========+================+
| Mixtral-HF   | E=8 K=2   |            0.5605 |               1 |        1 |        1.19e-07 |        1.8e-08  |   0.439 | 1.09-3.13      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| DBRX         | E=16 K=4  |            0.5963 |               1 |        1 |        1.19e-07 |        1.1e-08  |   0.404 | 1.13-2.69      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| Qwen3-30B    | E=128 K=8 |            0.2881 |               1 |        1 |        5.96e-08 |        6.04e-09 |   0.712 | 2.26-6.17      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| DeepSeek-V2  | E=160 K=6 |            0.21   |               1 |        1 |        1.19e-07 |        8.01e-09 |   0.79  | 2.40-7.92      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| OLMoE        | E=64 K=8  |            0.4223 |               1 |        1 |        5.96e-08 |        6.1e-09  |   0.578 | 1.58-3.31      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| Qwen2-57B    | E=64 K=8  |            0.4311 |               1 |        1 |        5.96e-08 |        6.17e-09 |   0.569 | 1.57-3.39      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+
| Mixtral-Mist | E=8 K=2   |            0.5689 |               1 |        1 |        1.19e-07 |        1.81e-08 |   0.431 | 1.15-2.93      |
+--------------+-----------+-------------------+-----------------+----------+-----------------+-----------------+---------+----------------+

==========================================================================================
Detail: A-norenorm vs B  (systematic uniform scale, not random noise)
==========================================================================================

  Mixtral-HF  (E=8, K=2, renorm=True)  [mistralai/Mixtral-8x7B-v0.1 (HuggingFace)]
    delta (mass outside top-K) = 0.4354
    theoretical 1/(1-delta)    = 1.7712
    actual B/A-norenorm ratio  : min=1.0861  max=3.0800  mean=1.8366
    sum(A-norenorm) mean = 0.5646  sum(B) mean = 1.0000
    A-renorm vs B: max diff = 1.19e-07  (fp32 rounding only)

  DBRX  (E=16, K=4, renorm=True)  [databricks/dbrx]
    delta (mass outside top-K) = 0.4039
    theoretical 1/(1-delta)    = 1.6777
    actual B/A-norenorm ratio  : min=1.1040  max=2.7853  mean=1.7124
    sum(A-norenorm) mean = 0.5961  sum(B) mean = 1.0000
    A-renorm vs B: max diff = 5.96e-08  (fp32 rounding only)

  Qwen3-30B  (E=128, K=8, renorm=True)  [Qwen/Qwen3-30B-A3B]
    delta (mass outside top-K) = 0.7127
    theoretical 1/(1-delta)    = 3.4813
    actual B/A-norenorm ratio  : min=2.1559  max=5.1128  mean=3.5580
    sum(A-norenorm) mean = 0.2873  sum(B) mean = 1.0000
    A-renorm vs B: max diff = 5.96e-08  (fp32 rounding only)

  DeepSeek-V2  (E=160, K=6, renorm=False)  [deepseek-ai/DeepSeek-V2]
    delta (mass outside top-K) = 0.7909
    theoretical 1/(1-delta)    = 4.7828
    actual B/A-norenorm ratio  : min=2.3181  max=7.4053  mean=4.9224
    sum(A-norenorm) mean = 0.2091  sum(B) mean = 1.0000

  OLMoE  (E=64, K=8, renorm=False)  [allenai/OLMoE-1B-7B]
    delta (mass outside top-K) = 0.5732
    theoretical 1/(1-delta)    = 2.3430
    actual B/A-norenorm ratio  : min=1.4850  max=3.3312  mean=2.3834
    sum(A-norenorm) mean = 0.4268  sum(B) mean = 1.0000

  Qwen2-57B  (E=64, K=8, renorm=False)  [Qwen/Qwen2-57B-A14B]
    delta (mass outside top-K) = 0.5722
    theoretical 1/(1-delta)    = 2.3376
    actual B/A-norenorm ratio  : min=1.5360  max=3.3163  mean=2.3751
    sum(A-norenorm) mean = 0.4278  sum(B) mean = 1.0000

Done.
(tilelang) root@xhluo0-0:/cephfs/xhluo/chjin/palette/3rdparty/tilelang_moe# python benchmarks/benchmark_topk_softmax.py
TopK+Softmax Experiment Benchmark
GPU       : NVIDIA H100 80GB HBM3
PyTorch   : 2.8.0+cu129
batch_size: 256  (tokens)

===============================================================================================
Experiment 1  |  Convention + Programming Model Chain  (algorithm = radix throughout)
  BL-1 (Conv A, PyTorch)  --[Q1: convention]-->  BL-4 (Conv B, PyTorch unfused)
  BL-4                    --[Q2: prog model]-->  BL-5 (Conv B, TileLang radix unfused)
  batch_size = 256
===============================================================================================
2026-02-21 16:51:17  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `tl_topk_radix_kernel` with `out_idx=None`
2026-02-21 16:51:22  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `tl_topk_radix_kernel`
2026-02-21 16:51:25  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `tl_topk_radix_kernel` with `out_idx=None`
2026-02-21 16:51:30  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `tl_topk_radix_kernel`
2026-02-21 16:51:36  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `tl_topk_radix_kernel` with `out_idx=None`
2026-02-21 16:51:43  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `tl_topk_radix_kernel`
+-------------+-----------+----------------------+----------------------+-----------------------+--------------+--------------+
| Model       | Config    |   BL-1 A-PyTorch(ms) |   BL-4 B-PyTorch(ms) |   BL-5 B-TL-radix(ms) | Q1 speedup   | Q2 speedup   |
+=============+===========+======================+======================+=======================+==============+==============+
| Mixtral     | E=8 K=2   |               0.0194 |               0.0193 |                0.0277 | 1.00x        | 0.70x        |
+-------------+-----------+----------------------+----------------------+-----------------------+--------------+--------------+
| DBRX        | E=16 K=4  |               0.0189 |               0.0188 |                0.0312 | 1.01x        | 0.60x        |
+-------------+-----------+----------------------+----------------------+-----------------------+--------------+--------------+
| OLMoE/Qwen2 | E=64 K=8  |               0.0213 |               0.022  |                0.0314 | 0.97x        | 0.70x        |
+-------------+-----------+----------------------+----------------------+-----------------------+--------------+--------------+
| Qwen3       | E=128 K=8 |               0.0218 |               0.0211 |                0.0308 | 1.03x        | 0.68x        |
+-------------+-----------+----------------------+----------------------+-----------------------+--------------+--------------+
| DeepSeek-V2 | E=160 K=6 |               0.0205 |               0.0201 |                0.0306 | 1.02x        | 0.66x        |
+-------------+-----------+----------------------+----------------------+-----------------------+--------------+--------------+
| DeepSeek-V3 | E=256 K=8 |               0.0217 |               0.0209 |                0.0322 | 1.04x        | 0.65x        |
+-------------+-----------+----------------------+----------------------+-----------------------+--------------+--------------+

===============================================================================================
Experiment 2  |  Algorithm Comparison  (Conv B + TileLang)
  radix [unfused = BL-5]  |  bitonic  |  selection
  Note: bitonic/selection are single-kernel because the algorithm never
        writes intermediate indices to global memory -- not classical fusion.
  batch_size = 256
===============================================================================================
2026-02-21 16:51:47  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=None`
2026-02-21 16:51:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:51:53  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[1, 2]`
2026-02-21 16:51:57  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:51:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=None`
2026-02-21 16:52:04  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:05  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[1, 2]`
2026-02-21 16:52:10  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:12  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=None`
2026-02-21 16:52:18  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:19  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[1, 2]`
2026-02-21 16:52:24  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:26  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=None`
2026-02-21 16:52:31  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:33  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[1, 2]`
2026-02-21 16:52:37  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:38  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=None`
2026-02-21 16:52:44  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:45  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[1, 2]`
2026-02-21 16:52:49  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:51  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=None`
2026-02-21 16:52:57  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
2026-02-21 16:52:58  [TileLang:tilelang.jit.kernel:INFO]: TileLang begins to compile kernel `main` with `out_idx=[1, 2]`
2026-02-21 16:53:02  [TileLang:tilelang.jit.kernel:INFO]: TileLang completes to compile kernel `main`
+-------------+-----------+---------------------+---------------+-----------------+----------------------+-----------+
| Model       | Config    |   radix-unfused(ms) |   bitonic(ms) |   selection(ms) | best/radix speedup   | winner    |
+=============+===========+=====================+===============+=================+======================+===========+
| Mixtral     | E=8 K=2   |              0.0278 |        0.0071 |          0.0069 | 4.04x                | selection |
+-------------+-----------+---------------------+---------------+-----------------+----------------------+-----------+
| DBRX        | E=16 K=4  |              0.0289 |        0.0086 |          0.0091 | 3.37x                | bitonic   |
+-------------+-----------+---------------------+---------------+-----------------+----------------------+-----------+
| OLMoE/Qwen2 | E=64 K=8  |              0.031  |        0.0104 |          0.0114 | 2.99x                | bitonic   |
+-------------+-----------+---------------------+---------------+-----------------+----------------------+-----------+
| Qwen3       | E=128 K=8 |              0.0309 |        0.0117 |          0.0126 | 2.63x                | bitonic   |
+-------------+-----------+---------------------+---------------+-----------------+----------------------+-----------+
| DeepSeek-V2 | E=160 K=6 |              0.031  |        0.0137 |          0.0101 | 3.06x                | selection |
+-------------+-----------+---------------------+---------------+-----------------+----------------------+-----------+
| DeepSeek-V3 | E=256 K=8 |              0.0322 |        0.0142 |          0.0115 | 2.80x                | selection |
+-------------+-----------+---------------------+---------------+-----------------+----------------------+-----------+

===============================================================================================
Q4  |  Total Speedup  --  BL-1 (Conv A, PyTorch) vs Ours-best (Conv B, TileLang)
  batch_size = 256
===============================================================================================
+-------------+-----------+--------------------------------------------+------------+-----------------+-----------------+
| Model       | Config    | Representative models                      |   BL-1(ms) |   Ours-best(ms) | Total speedup   |
+=============+===========+============================================+============+=================+=================+
| Mixtral     | E=8 K=2   | Mixtral-8x7B/8x22B, Grok-1                 |     0.0188 |          0.0069 | 2.73x           |
+-------------+-----------+--------------------------------------------+------------+-----------------+-----------------+
| DBRX        | E=16 K=4  | databricks/dbrx                            |     0.02   |          0.0079 | 2.54x           |
+-------------+-----------+--------------------------------------------+------------+-----------------+-----------------+
| OLMoE/Qwen2 | E=64 K=8  | allenai/OLMoE, Qwen/Qwen2-57B-A14B         |     0.0201 |          0.0102 | 1.97x           |
+-------------+-----------+--------------------------------------------+------------+-----------------+-----------------+
| Qwen3       | E=128 K=8 | Qwen/Qwen3-30B-A3B, Qwen3-235B-A22B        |     0.0204 |          0.0116 | 1.75x           |
+-------------+-----------+--------------------------------------------+------------+-----------------+-----------------+
| DeepSeek-V2 | E=160 K=6 | deepseek-ai/DeepSeek-V2/V2.5               |     0.0219 |          0.01   | 2.18x           |
+-------------+-----------+--------------------------------------------+------------+-----------------+-----------------+
| DeepSeek-V3 | E=256 K=8 | deepseek-ai/DeepSeek-V3 routing (softmax p |     0.0213 |          0.0114 | 1.87x           |
+-------------+-----------+--------------------------------------------+------------+-----------------+-----------------+

Correctness verification: python tests/test_topk_softmax_conventions.py