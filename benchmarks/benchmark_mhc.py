"""
DeepSeek mHC Performance Benchmark

四个版本对比:
1. pytorch-baseline: PyTorch实现
2. pytorch-compiled: torch.compile优化
3. tilelang-official: TileLang官方实现
4. ours: 优化实现 (Split-K + 自动选择)

运行方法:
python benchmarks/benchmark_mhc.py
"""

import sys
import warnings
from pathlib import Path

# 过滤torch.compile的警告（需要在torch导入前设置）
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Dict, List
from tilelang.profiler import do_bench

# 启用TF32以获得更好的float32矩阵乘法性能
torch.set_float32_matmul_precision('high')

from kernels.mhc_baseline import (
    mhc_pre_pytorch,
    mhc_pre_pytorch_compiled,
    mhc_post_pytorch,
    mhc_post_pytorch_compiled,
)
from kernels.mhc_tilelang import (
        mhc_pre_tilelang_official,
    mhc_pre,
        mhc_post_tilelang_official,
    mhc_post,
    )


# ============ Benchmark Configuration ============

BENCHMARK_CONFIGS = [
    # (num_tokens, hc_mult, hidden_size)
    (256, 4, 2560),
    (512, 4, 2560),
    (1024, 4, 2560),
    (2048, 4, 2560),
    (4096, 4, 2560),
    # 更大hidden_size
    (1024, 4, 5120),
    (2048, 4, 5120),
]

WARMUP = 100
REP = 500


# ============ Benchmark Functions ============

def benchmark_mhc_pre(num_tokens: int, hc_mult: int, hidden_size: int) -> Dict[str, float]:
    """Benchmark mHC Pre各实现"""
    device = "cuda"
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    
    torch.manual_seed(42)
    residual = (torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.float32)
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).bfloat16()
    fn = (torch.randn(hc_mult3, hc_mult, hidden_size, device=device, dtype=torch.float32) * 1e-4
          * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).flatten(1, 2)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    
    params = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0, sinkhorn_repeat=10)
    
    results = {}
    
    # PyTorch baseline
    results["pytorch"] = do_bench(
        lambda: mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params),
        warmup=WARMUP, rep=REP, return_mode='median'
    )
    
    # torch.compile
    results["torch.compile"] = do_bench(
        lambda: mhc_pre_pytorch_compiled(residual, fn, hc_scale, hc_base, **params),
        warmup=WARMUP, rep=REP, return_mode='median'
    )
    
    # TileLang official
    results["tilelang-official"] = do_bench(
        lambda: mhc_pre_tilelang_official(residual, fn, hc_scale, hc_base, **params),
        warmup=WARMUP, rep=REP, return_mode='median'
    )
    
    # Ours (auto-select best)
    results["ours"] = do_bench(
        lambda: mhc_pre(residual, fn, hc_scale, hc_base, **params),
        warmup=WARMUP, rep=REP, return_mode='median'
    )
    
    return results


def benchmark_mhc_post(num_tokens: int, hc_mult: int, hidden_size: int) -> Dict[str, float]:
    """Benchmark mHC Post各实现"""
    device = "cuda"
    
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    
    results = {}
    
    # PyTorch baseline
    results["pytorch"] = do_bench(
        lambda: mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix),
        warmup=WARMUP, rep=REP, return_mode='median'
    )
    
    # torch.compile
    results["torch.compile"] = do_bench(
        lambda: mhc_post_pytorch_compiled(x, residual, post_layer_mix, comb_res_mix),
        warmup=WARMUP, rep=REP, return_mode='median'
    )
    
    # TileLang official
    results["tilelang-official"] = do_bench(
        lambda: mhc_post_tilelang_official(x, residual, post_layer_mix, comb_res_mix),
        warmup=WARMUP, rep=REP, return_mode='median'
    )
    
    # Ours (same as official for Post)
    results["ours"] = do_bench(
        lambda: mhc_post(x, residual, post_layer_mix, comb_res_mix),
        warmup=WARMUP, rep=REP, return_mode='median'
    )
    
    return results


def test_correctness(num_tokens: int = 1024, hc_mult: int = 4, hidden_size: int = 2560):
    """测试正确性"""
    device = "cuda"
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    
    torch.manual_seed(42)
    residual = (torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.float32)
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).bfloat16()
    fn = (torch.randn(hc_mult3, hc_mult, hidden_size, device=device, dtype=torch.float32) * 1e-4
          * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).flatten(1, 2)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    
    params = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0, sinkhorn_repeat=10)
    
    print("=" * 60)
    print("Correctness Test")
    print("=" * 60)
    
    # mHC Pre
    post_ref, comb_ref, layer_ref = mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params)
    post_tl, comb_tl, layer_tl = mhc_pre_tilelang_official(residual, fn, hc_scale, hc_base, **params)
    post_ours, comb_ours, layer_ours = mhc_pre(residual, fn, hc_scale, hc_base, **params)
    
    print("\nmHC Pre:")
    tl_diff = max(
        (post_tl - post_ref).abs().max().item(),
        (comb_tl - comb_ref).abs().max().item(),
        (layer_tl.float() - layer_ref.float()).abs().max().item()
    )
    ours_diff = max(
        (post_ours - post_ref).abs().max().item(),
        (comb_ours - comb_ref).abs().max().item(),
        (layer_ours.float() - layer_ref.float()).abs().max().item()
    )
    ours_vs_tl_diff = max(
        (post_ours - post_tl).abs().max().item(),
        (comb_ours - comb_tl).abs().max().item(),
        (layer_ours.float() - layer_tl.float()).abs().max().item()
    )
    print(f"  TL-Official vs PyTorch: max_diff={tl_diff:.6f} {'✅' if tl_diff < 0.1 else '❌'}")
    print(f"  Ours vs PyTorch: max_diff={ours_diff:.6f} {'✅' if ours_diff < 0.1 else '❌'}")
    print(f"  Ours vs TL-Official: max_diff={ours_vs_tl_diff:.6f} {'✅' if ours_vs_tl_diff < 0.1 else '❌'}")
    
    # mHC Post
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    
    output_ref = mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)
    output_tl = mhc_post_tilelang_official(x, residual, post_layer_mix, comb_res_mix)
    output_ours = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    
    print("\nmHC Post:")
    tl_diff = (output_tl.float() - output_ref.float()).abs().max().item()
    ours_diff = (output_ours.float() - output_ref.float()).abs().max().item()
    print(f"  TL-Official vs PyTorch: max_diff={tl_diff:.6f} {'✅' if tl_diff < 0.1 else '❌'}")
    print(f"  Ours vs PyTorch: max_diff={ours_diff:.6f} {'✅' if ours_diff < 0.1 else '❌'}")


def run_benchmarks():
    """运行所有benchmark"""
    print("\n" + "=" * 100)
    print("mHC Pre Benchmark (Latency in ms)")
    print("=" * 100)
    print(f"{'Config':<22} | {'PyTorch':>10} | {'Compiled':>10} | {'TL-Offic':>10} | {'Ours':>10} | {'Speedup':>10}")
    print("-" * 100)
    
    pre_results = []
    for num_tokens, hc_mult, hidden_size in BENCHMARK_CONFIGS:
        config_str = f"({num_tokens}, {hc_mult}, {hidden_size})"
        try:
            r = benchmark_mhc_pre(num_tokens, hc_mult, hidden_size)
            speedup = r["tilelang-official"] / r["ours"]
            print(f"{config_str:<22} | {r['pytorch']:>10.4f} | {r['torch.compile']:>10.4f} | {r['tilelang-official']:>10.4f} | {r['ours']:>10.4f} | {speedup:>9.2f}x")
            pre_results.append({"config": config_str, **r, "speedup": speedup})
        except Exception as e:
            print(f"{config_str:<22} | Error: {e}")
    
    print("\n" + "=" * 100)
    print("mHC Post Benchmark (Latency in ms)")
    print("=" * 100)
    print(f"{'Config':<22} | {'PyTorch':>10} | {'Compiled':>10} | {'TL-Offic':>10} | {'Ours':>10} | {'Speedup':>10}")
    print("-" * 100)
    
    post_results = []
    for num_tokens, hc_mult, hidden_size in BENCHMARK_CONFIGS:
        config_str = f"({num_tokens}, {hc_mult}, {hidden_size})"
        try:
            r = benchmark_mhc_post(num_tokens, hc_mult, hidden_size)
            speedup = r["pytorch"] / r["ours"]
            print(f"{config_str:<22} | {r['pytorch']:>10.4f} | {r['torch.compile']:>10.4f} | {r['tilelang-official']:>10.4f} | {r['ours']:>10.4f} | {speedup:>9.2f}x")
            post_results.append({"config": config_str, **r, "speedup": speedup})
        except Exception as e:
            print(f"{config_str:<22} | Error: {e}")
    
    # Summary
    if pre_results:
        avg_pre_speedup = sum(r["speedup"] for r in pre_results) / len(pre_results)
        print(f"\nmHC Pre Average Speedup (Ours vs TileLang-Official): {avg_pre_speedup:.2f}x")
    if post_results:
        avg_post_speedup = sum(r["speedup"] for r in post_results) / len(post_results)
        print(f"mHC Post Average Speedup (Ours vs PyTorch): {avg_post_speedup:.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
    
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Correctness test
    test_correctness()
    
    # Performance benchmark
    run_benchmarks()
