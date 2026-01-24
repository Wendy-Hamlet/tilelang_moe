# Top-K Gating Benchmark - Quick Guide

## Setup (H100 Server)

```bash
# 1. Install dependencies
pip install tilelang torch pandas

# 2. Clone with submodules
git clone --recursive <repo-url>
cd tilelang_moe
```

## Run Benchmark

```bash
# Clear TileLang kernel cache (IMPORTANT: do this after git pull)
rm -rf ~/.tilelang_cache

# Execute benchmark (auto-tunes and tests multiple batch sizes)
python benchmarks/benchmark_topk.py
```

## Check Results

```bash
# View CSV results
cat benchmarks/results/topk_benchmark_results.csv

# Or use pandas
python -c "import pandas as pd; df = pd.read_csv('benchmarks/results/topk_benchmark_results.csv'); print(df)"
```

## Implementation Details

**File**: `kernels/topk_gating.py`

**Features**:
- Based on official TileLang TopK implementation
- Hopper features enabled for H100
- Auto-tuning for optimal block sizes and thread counts
- DeepSeek-V2 configuration (160 experts, k=6)

**Configurations tested**:
- Batch sizes: 256, 512, 1024, 2048, 4096, 8192
- Num experts: 160
- K: 6

## Expected Output

```
================================================================================
Top-K Gating Benchmark: PyTorch vs TileLang (Hopper Features Enabled)
================================================================================

────────────────────────────────────────────────────────────────────────────────
Config: batch=1024, num_experts=160, k=6
────────────────────────────────────────────────────────────────────────────────
  [1/2] Benchmarking PyTorch...
  [2/2] Benchmarking TileLang (Hopper)...

  Results:
    PyTorch:  0.0234 ms
    TileLang: 0.0156 ms
    Speedup:  1.50x
    Correct:  ✓

...

================================================================================
Summary Statistics
================================================================================
   batch  num_experts  k  pytorch_ms  tilelang_ms  speedup  correct
     256          160  6      0.0112       0.0089     1.26     True
     512          160  6      0.0156       0.0121     1.29     True
    1024          160  6      0.0234       0.0156     1.50     True
    ...

Average speedup: 1.45x
Max speedup: 1.60x
Min speedup: 1.20x
All tests correct: ✓

Results saved to: benchmarks/results/topk_benchmark_results.csv
================================================================================
```

## Notes

- First run will trigger JIT compilation and auto-tuning
- Subsequent runs will use cached optimal configurations
- Correctness is verified for each configuration
- All times are averaged over 100 runs with 20 warmup iterations
