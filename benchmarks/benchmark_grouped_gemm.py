"""
Grouped GEMM Performance Benchmark

对比4个实现：
1. PyTorch: 高效PyTorch实现 (排序+scatter_reduce，对齐TileLang官方baseline)
2. torch.compile: 编译优化版本
3. TL-Official: TileLang官方grouped_gemm kernel
4. Ours: 自动选择最佳实现

官方配置来源：
- FusedMoE Test: d_hidden=1024, d_expert=256, n_routed_experts=8
- DeepSeek-V3: dim=7168, moe_inter_dim=2048, n_routed_experts=256
"""

import sys
import os
import warnings
from pathlib import Path

# 过滤torch.compile的警告
warnings.filterwarnings('ignore', category=UserWarning, module='torch._inductor')

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import math
from typing import Dict, List, Tuple
from tabulate import tabulate

# 启用TF32以获得更好的float32矩阵乘法性能
torch.set_float32_matmul_precision('high')


# ============ 导入所有实现 ============
from kernels.grouped_gemm_tilelang import (
    grouped_gemm_pytorch,
    grouped_gemm_tilelang,
    grouped_gemm_best,
)


# ============ torch.compile优化版本 ============
_compiled_fn_cache = {}


def grouped_gemm_compiled(
    X: torch.Tensor,
    W: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """torch.compile优化版本"""
    E = W.shape[0]
    
    # 为每个E值预编译
    if E not in _compiled_fn_cache:
        _compiled_fn_cache[E] = torch.compile(grouped_gemm_pytorch)
    
    return _compiled_fn_cache[E](X, W, expert_indices)


# ============ TileLang do_bench集成 ============
try:
    from tilelang.profiler import do_bench
    
    def benchmark_function(func, *args, warmup=100, rep=1000):
        """使用TileLang do_bench进行稳定测量"""
        def wrapper():
            return func(*args)
        
        time_ms = do_bench(wrapper, warmup=warmup, rep=rep, return_mode='median')
        return time_ms
        
except ImportError:
    def benchmark_function(func, *args, warmup=100, rep=1000):
        """手动CUDA Events测量（fallback）"""
        for _ in range(warmup):
            func(*args)
        
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(rep):
            func(*args)
        end_event.record()
        
        torch.cuda.synchronize()
        
        time_ms = start_event.elapsed_time(end_event) / rep
        return time_ms


# ============ 官方Benchmark配置 ============
OFFICIAL_CONFIGS = [
    # (name, H, I, E, total_tokens, description)
    # FusedMoE Test配置 (tilelang/examples/fusedmoe/test_example_fusedmoe.py)
    ("FusedMoE-Test", 1024, 256, 8, 1024, "d_hidden=1024, d_expert=256, E=8"),
    # DeepSeek-V3配置 (tilelang/examples/deepseek_v32/inference/config_671B_v3.2.json)
    ("DeepSeek-V3", 7168, 2048, 256, 1024, "dim=7168, moe_inter_dim=2048, E=256"),
]


def generate_test_data(total_tokens: int, H: int, I: int, E: int, device="cuda"):
    """生成测试数据"""
    torch.manual_seed(42)
    
    X = torch.randn(total_tokens, H, device=device, dtype=torch.float16)
    W = torch.randn(E, I, H, device=device, dtype=torch.float16)
    
    # 模拟真实MoE分布 (zipf)
    expert_probs = torch.tensor([1.0 / (i + 1) for i in range(E)], device=device)
    expert_probs = expert_probs / expert_probs.sum()
    expert_indices = torch.multinomial(expert_probs, total_tokens, replacement=True)
    
    return X, W, expert_indices


def benchmark_config(H: int, I: int, E: int, total_tokens: int, config_name: str):
    """运行单个配置的benchmark"""
    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"  H={H}, I={I}, E={E}, total_tokens={total_tokens}")
    print(f"{'='*70}")
    
    # 生成数据
    X, W, expert_indices = generate_test_data(total_tokens, H, I, E)
    
    # 所有版本
    versions = {
        'PyTorch': lambda x, w, idx: grouped_gemm_pytorch(x, w, idx),
        'torch.compile': lambda x, w, idx: grouped_gemm_compiled(x, w, idx),
        'TL-Official': lambda x, w, idx: grouped_gemm_tilelang(x, w, idx),
        'Ours': lambda x, w, idx: grouped_gemm_best(x, w, idx),
    }
    
    # 正确性验证
    print("\n[Correctness]")
    baseline = grouped_gemm_pytorch(X, W, expert_indices)
    
    for name, func in versions.items():
        try:
            output = func(X, W, expert_indices)
            max_diff = (output - baseline).abs().max().item()
            # FP16容差
            status = "✅" if max_diff < 0.1 else "⚠️"
            print(f"  {status} {name}: max_diff = {max_diff:.6f}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    # 性能Benchmark
    print("\n[Performance]")
    latencies = {}
    
    for name, func in versions.items():
        try:
            # 预热
            _ = func(X, W, expert_indices)
            torch.cuda.synchronize()
            
            # Benchmark
            latency_ms = benchmark_function(func, X, W, expert_indices, warmup=50, rep=500)
            latencies[name] = latency_ms
            print(f"  {name}: {latency_ms:.4f} ms")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            latencies[name] = float('inf')
    
    return latencies


def main():
    """主Benchmark函数"""
    print("="*70)
    print("Grouped GEMM Performance Benchmark")
    print("4 Versions: PyTorch, torch.compile, TL-Official, Ours")
    print("="*70)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 运行所有官方配置
    all_results = []
    
    for name, H, I, E, total_tokens, desc in OFFICIAL_CONFIGS:
        latencies = benchmark_config(H, I, E, total_tokens, name)
        all_results.append({
            "config": name,
            "params": f"H={H}, I={I}, E={E}",
            "latencies": latencies
        })
    
    # ============ 汇总表格1: 延迟对比 ============
    print("\n" + "="*70)
    print("SUMMARY: Latency Comparison (ms)")
    print("="*70)
    
    latency_table = []
    for result in all_results:
        row = [result["config"], result["params"]]
        for version in ["PyTorch", "torch.compile", "TL-Official", "Ours"]:
            lat = result["latencies"].get(version, float('inf'))
            if lat == float('inf'):
                row.append("FAILED")
            else:
                row.append(f"{lat:.4f}")
        latency_table.append(row)
    
    headers = ["Config", "Params", "PyTorch", "torch.compile", "TL-Official", "Ours"]
    print(tabulate(latency_table, headers=headers, tablefmt="pipe"))
    
    # ============ 汇总表格2: Ours vs 最强其他 ============
    print("\n" + "="*70)
    print("SUMMARY: Ours vs Best Other")
    print("="*70)
    
    compare_table = []
    total_speedup = []
    
    for result in all_results:
        config = result["config"]
        latencies = result["latencies"]
        
        ours = latencies.get("Ours", float('inf'))
        
        # 找到最强其他
        other_versions = ["PyTorch", "torch.compile", "TL-Official"]
        best_other_name = None
        best_other_lat = float('inf')
        
        for v in other_versions:
            lat = latencies.get(v, float('inf'))
            if lat < best_other_lat:
                best_other_lat = lat
                best_other_name = v
        
        if ours == float('inf') or best_other_lat == float('inf'):
            continue
        
        speedup = best_other_lat / ours
        total_speedup.append(speedup)
        
        if speedup >= 1.05:
            status = "✅"
        elif speedup >= 0.95:
            status = "≈"
        else:
            status = "❌"
        
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
    print("AUTO-SELECT STRATEGY:")
    print("="*70)
    print("""
H <= 2560 且 E <= 64:  TileLang grouped_gemm (小矩阵时kernel调度开销更低)
其他情况:              PyTorch循环 (大矩阵/大E时cuBLAS更高效)

Note: TileLang在E > 64时可能有精度问题，因此需要fallback到PyTorch
""")


if __name__ == "__main__":
    main()
