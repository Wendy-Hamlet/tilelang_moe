"""
DeepSeek mHC (Manifold-Constrained Hyper-Connections) Performance Benchmark

对比4个实现：
1. PyTorch: 基础实现
2. torch.compile: 编译优化版本
3. TL-Official: TileLang官方实现
4. Ours: 优化实现 (Split-K + 自动选择)

官方配置来源：tilelang/examples/deepseek_mhc/
- num_tokens: 512, 1024, 2048, 8192
- hidden_size: 1280, 2560, 4096
- hc_mult: 4 (固定)
"""

import sys
import warnings
from pathlib import Path

# 过滤torch.compile的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from typing import Dict, List, Tuple
from tabulate import tabulate

# 启用TF32
torch.set_float32_matmul_precision('high')

# 导入所有实现
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

# TileLang profiler
try:
    from tilelang.profiler import do_bench
except ImportError:
    def do_bench(fn, warmup=100, rep=500, return_mode='median'):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        import time
        times = []
        for _ in range(rep):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        return times[len(times)//2]


# ============ 官方配置 ============
OFFICIAL_CONFIGS = [
    # (name, num_tokens, hidden_size, hc_mult)
    # 来源: tilelang/examples/deepseek_mhc/example_mhc_pre.py main()
    ("Config-1", 512, 1280, 4),
    ("Config-2", 1024, 2560, 4),
    ("Config-3", 2048, 4096, 4),
    ("Config-4", 8192, 4096, 4),
]

WARMUP = 50
REP = 500


def generate_mhc_pre_data(num_tokens: int, hidden_size: int, hc_mult: int):
    """生成mHC Pre测试数据"""
    device = "cuda"
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    
    torch.manual_seed(42)
    residual = (torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.float32)
                * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).bfloat16()
    fn = (torch.randn(hc_mult3, hc_mult, hidden_size, device=device, dtype=torch.float32) * 1e-4
          * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))).flatten(1, 2)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32) * 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    
    params = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, 
                  hc_post_mult_value=1.0, sinkhorn_repeat=10)
    
    return residual, fn, hc_scale, hc_base, params


def generate_mhc_post_data(num_tokens: int, hidden_size: int, hc_mult: int):
    """生成mHC Post测试数据"""
    device = "cuda"
    
    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(num_tokens, hc_mult, 1, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)
    
    return x, residual, post_layer_mix, comb_res_mix


def benchmark_mhc_pre_config(name: str, num_tokens: int, hidden_size: int, hc_mult: int):
    """运行单个mHC Pre配置的benchmark"""
    print(f"\n{'='*70}")
    print(f"mHC Pre - {name}: num_tokens={num_tokens}, hidden_size={hidden_size}, hc_mult={hc_mult}")
    print(f"{'='*70}")
    
    residual, fn, hc_scale, hc_base, params = generate_mhc_pre_data(num_tokens, hidden_size, hc_mult)
    
    versions = {
        'PyTorch': lambda: mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params),
        'torch.compile': lambda: mhc_pre_pytorch_compiled(residual, fn, hc_scale, hc_base, **params),
        'TL-Official': lambda: mhc_pre_tilelang_official(residual, fn, hc_scale, hc_base, **params),
        'Ours': lambda: mhc_pre(residual, fn, hc_scale, hc_base, **params),
    }
    
    # 正确性验证
    print("\n[Correctness]")
    ref = mhc_pre_pytorch(residual, fn, hc_scale, hc_base, **params)
    
    for vname, func in versions.items():
        try:
            out = func()
            max_diff = max(
                (out[0] - ref[0]).abs().max().item(),
                (out[1] - ref[1]).abs().max().item(),
                (out[2].float() - ref[2].float()).abs().max().item()
            )
            status = "✅" if max_diff < 0.1 else "⚠️"
            print(f"  {status} {vname}: max_diff = {max_diff:.6f}")
        except Exception as e:
            print(f"  ❌ {vname}: {e}")
    
    # 性能Benchmark
    print("\n[Performance]")
    latencies = {}
    
    for vname, func in versions.items():
        try:
            _ = func()
            torch.cuda.synchronize()
            latency = do_bench(func, warmup=WARMUP, rep=REP, return_mode='median')
            latencies[vname] = latency
            print(f"  {vname}: {latency:.4f} ms")
        except Exception as e:
            print(f"  {vname}: FAILED ({e})")
            latencies[vname] = float('inf')
    
    return latencies


def benchmark_mhc_post_config(name: str, num_tokens: int, hidden_size: int, hc_mult: int):
    """运行单个mHC Post配置的benchmark"""
    print(f"\n{'='*70}")
    print(f"mHC Post - {name}: num_tokens={num_tokens}, hidden_size={hidden_size}, hc_mult={hc_mult}")
    print(f"{'='*70}")
    
    x, residual, post_layer_mix, comb_res_mix = generate_mhc_post_data(num_tokens, hidden_size, hc_mult)
    
    versions = {
        'PyTorch': lambda: mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix),
        'torch.compile': lambda: mhc_post_pytorch_compiled(x, residual, post_layer_mix, comb_res_mix),
        'TL-Official': lambda: mhc_post_tilelang_official(x, residual, post_layer_mix, comb_res_mix),
        'Ours': lambda: mhc_post(x, residual, post_layer_mix, comb_res_mix),
    }
    
    # 正确性验证
    print("\n[Correctness]")
    ref = mhc_post_pytorch(x, residual, post_layer_mix, comb_res_mix)
    
    for vname, func in versions.items():
        try:
            out = func()
            max_diff = (out.float() - ref.float()).abs().max().item()
            status = "✅" if max_diff < 0.1 else "⚠️"
            print(f"  {status} {vname}: max_diff = {max_diff:.6f}")
        except Exception as e:
            print(f"  ❌ {vname}: {e}")
    
    # 性能Benchmark
    print("\n[Performance]")
    latencies = {}
    
    for vname, func in versions.items():
        try:
            _ = func()
            torch.cuda.synchronize()
            latency = do_bench(func, warmup=WARMUP, rep=REP, return_mode='median')
            latencies[vname] = latency
            print(f"  {vname}: {latency:.4f} ms")
        except Exception as e:
            print(f"  {vname}: FAILED ({e})")
            latencies[vname] = float('inf')
    
    return latencies


def main():
    """主Benchmark函数"""
    print("="*70)
    print("mHC (Manifold-Constrained Hyper-Connections) Benchmark")
    print("4 Versions: PyTorch, torch.compile, TL-Official, Ours")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # ============ mHC Pre Benchmark ============
    pre_results = []
    for name, num_tokens, hidden_size, hc_mult in OFFICIAL_CONFIGS:
        latencies = benchmark_mhc_pre_config(name, num_tokens, hidden_size, hc_mult)
        pre_results.append({
            "config": name,
            "params": f"n={num_tokens}, h={hidden_size}",
            "latencies": latencies
        })
    
    # ============ mHC Post Benchmark ============
    post_results = []
    for name, num_tokens, hidden_size, hc_mult in OFFICIAL_CONFIGS:
        latencies = benchmark_mhc_post_config(name, num_tokens, hidden_size, hc_mult)
        post_results.append({
            "config": name,
            "params": f"n={num_tokens}, h={hidden_size}",
            "latencies": latencies
        })
    
    # ============ 汇总表格: mHC Pre ============
    print("\n" + "="*70)
    print("SUMMARY: mHC Pre Latency Comparison (ms)")
    print("="*70)
    
    pre_table = []
    for result in pre_results:
        row = [result["config"], result["params"]]
        for v in ["PyTorch", "torch.compile", "TL-Official", "Ours"]:
            lat = result["latencies"].get(v, float('inf'))
            row.append(f"{lat:.4f}" if lat != float('inf') else "FAILED")
        pre_table.append(row)
    
    headers = ["Config", "Params", "PyTorch", "torch.compile", "TL-Official", "Ours"]
    print(tabulate(pre_table, headers=headers, tablefmt="pipe"))
    
    # ============ 汇总表格: mHC Post ============
    print("\n" + "="*70)
    print("SUMMARY: mHC Post Latency Comparison (ms)")
    print("="*70)
    
    post_table = []
    for result in post_results:
        row = [result["config"], result["params"]]
        for v in ["PyTorch", "torch.compile", "TL-Official", "Ours"]:
            lat = result["latencies"].get(v, float('inf'))
            row.append(f"{lat:.4f}" if lat != float('inf') else "FAILED")
        post_table.append(row)
    
    print(tabulate(post_table, headers=headers, tablefmt="pipe"))
    
    # ============ Ours vs Best Other ============
    print("\n" + "="*70)
    print("SUMMARY: Ours vs Best Other")
    print("="*70)
    
    compare_table = []
    total_speedup = []
    
    # mHC Pre
    for result in pre_results:
        config = result["config"] + " (Pre)"
        latencies = result["latencies"]
        ours = latencies.get("Ours", float('inf'))
        
        best_other_name = None
        best_other_lat = float('inf')
        for v in ["PyTorch", "torch.compile", "TL-Official"]:
            lat = latencies.get(v, float('inf'))
            if lat < best_other_lat:
                best_other_lat = lat
                best_other_name = v
        
        if ours != float('inf') and best_other_lat != float('inf'):
            speedup = best_other_lat / ours
            total_speedup.append(speedup)
            status = "✅" if speedup >= 1.05 else ("≈" if speedup >= 0.95 else "❌")
            compare_table.append([
                config,
                f"{best_other_name} ({best_other_lat:.4f} ms)",
                f"{ours:.4f} ms",
                f"{speedup:.2f}x",
                status
            ])
    
    # mHC Post
    for result in post_results:
        config = result["config"] + " (Post)"
        latencies = result["latencies"]
        ours = latencies.get("Ours", float('inf'))
        
        best_other_name = None
        best_other_lat = float('inf')
        for v in ["PyTorch", "torch.compile", "TL-Official"]:
            lat = latencies.get(v, float('inf'))
            if lat < best_other_lat:
                best_other_lat = lat
                best_other_name = v
        
        if ours != float('inf') and best_other_lat != float('inf'):
            speedup = best_other_lat / ours
            total_speedup.append(speedup)
            status = "✅" if speedup >= 1.05 else ("≈" if speedup >= 0.95 else "❌")
            compare_table.append([
                config,
                f"{best_other_name} ({best_other_lat:.4f} ms)",
                f"{ours:.4f} ms",
                f"{speedup:.2f}x",
                status
            ])
    
    headers = ["Config", "Best Other", "Ours", "Speedup", "Status"]
    print(tabulate(compare_table, headers=headers, tablefmt="pipe"))
    
    # 平均加速比
    if total_speedup:
        avg_speedup = sum(total_speedup) / len(total_speedup)
        print(f"\n**平均加速**: **{avg_speedup:.2f}x** (Ours vs Best Other)")
    
    # ============ 自动选择策略说明 ============
    print("\n" + "="*70)
    print("AUTO-SELECT STRATEGY (mHC Pre):")
    print("="*70)
    print("""
Split-K条件: hc_hidden_size % 1024 == 0 且 
             (num_tokens <= 2048 或 (num_tokens >= 8192 且 hc_hidden_size >= 10240))
否则使用Optimized版本

mHC Post: 使用TileLang官方实现（已高度优化）
""")


if __name__ == "__main__":
    main()
