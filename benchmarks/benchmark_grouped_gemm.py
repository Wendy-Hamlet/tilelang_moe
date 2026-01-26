"""
Grouped GEMM Performance Benchmark

4个版本对比：
1. pytorch-loop: PyTorch简单循环
2. pytorch-compiled: torch.compile优化循环
3. tilelang-separate: TileLang分expert调用
4. ours (sonic): Sonic风格融合kernel

配置：DeepSeek-V2/V3场景
"""

import torch
import time
from typing import Dict, List, Tuple
from tabulate import tabulate

# 导入所有实现
from kernels.grouped_gemm_baseline import (
    grouped_gemm_loop,
    grouped_gemm_loop_compiled
)
from kernels.grouped_gemm_tilelang_separate import grouped_gemm_tilelang_separate
from kernels.grouped_gemm_sonic import grouped_gemm_sonic


# ============ TileLang do_bench集成 ============
try:
    from tilelang.profiler import do_bench
    
    def benchmark_function(func, *args, warmup=100, rep=1000):
        """使用TileLang do_bench进行稳定测量"""
        def wrapper():
            return func(*args)
        
        # do_bench返回median时间（毫秒）
        time_ms = do_bench(wrapper, warmup=warmup, rep=rep, return_mode='median')
        return time_ms
        
except ImportError:
    print("Warning: TileLang do_bench not available, using manual CUDA Events")
    
    def benchmark_function(func, *args, warmup=100, rep=1000):
        """手动CUDA Events测量（fallback）"""
        # Warmup
        for _ in range(warmup):
            func(*args)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(rep):
            func(*args)
        end_event.record()
        
        torch.cuda.synchronize()
        
        time_ms = start_event.elapsed_time(end_event) / rep
        return time_ms


# ============ Benchmark配置 ============
BENCHMARK_CONFIGS = [
    # (name, total_tokens, H, I, E, description)
    ("B128_H512_I384_E8", 128, 512, 384, 8, "Small Test"),
    ("B256_H1024_I768_E16", 256, 1024, 768, 16, "Medium Test"),
    ("B512_H2048_I1536_E32", 512, 2048, 1536, 32, "DeepSeek-V2-like"),
    ("B1024_H4096_I3072_E64", 1024, 4096, 3072, 64, "DeepSeek-V3-like (scaled)"),
]


def generate_test_data(total_tokens: int, H: int, I: int, E: int, device="cuda"):
    """生成测试数据"""
    torch.manual_seed(42)
    
    X = torch.randn(total_tokens, H, device=device, dtype=torch.float16)
    W = torch.randn(E, I, H, device=device, dtype=torch.float16)
    
    # 生成expert indices（模拟真实MoE分布）
    # 使用zipf分布模拟热门expert
    expert_probs = torch.tensor([1.0 / (i + 1) for i in range(E)], device=device)
    expert_probs = expert_probs / expert_probs.sum()
    expert_indices = torch.multinomial(expert_probs, total_tokens, replacement=True)
    
    return X, W, expert_indices


def verify_correctness(
    X: torch.Tensor,
    W: torch.Tensor,
    expert_indices: torch.Tensor,
    versions: Dict[str, callable],
    baseline_name: str = "pytorch-loop"
) -> Dict[str, Tuple[bool, float]]:
    """
    验证所有版本的正确性
    
    Returns:
        {version_name: (is_correct, max_diff)}
    """
    results = {}
    
    # 计算baseline
    baseline_output = versions[baseline_name](X, W, expert_indices)
    
    for name, func in versions.items():
        try:
            output = func(X, W, expert_indices)
            
            # 计算差异
            max_diff = (output - baseline_output).abs().max().item()
            is_correct = max_diff < 1e-2  # FP16容差
            
            results[name] = (is_correct, max_diff)
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results[name] = (False, float('inf'))
    
    return results


def run_benchmark_config(
    config_name: str,
    total_tokens: int,
    H: int,
    I: int,
    E: int,
    description: str,
    warmup: int = 50,
    rep: int = 500
) -> Dict[str, float]:
    """运行单个配置的benchmark"""
    
    print(f"\n{'='*80}")
    print(f"Config: {config_name} - {description}")
    print(f"  total_tokens={total_tokens}, H={H}, I={I}, E={E}")
    print(f"{'='*80}")
    
    # 生成数据
    X, W, expert_indices = generate_test_data(total_tokens, H, I, E)
    
    # 统计expert分布
    expert_counts = torch.bincount(expert_indices, minlength=E)
    print(f"Expert distribution: min={expert_counts.min()}, max={expert_counts.max()}, "
          f"mean={expert_counts.float().mean():.1f}")
    
    # 所有版本
    versions = {
        "pytorch-loop": grouped_gemm_loop,
        "pytorch-compiled": grouped_gemm_loop_compiled,
        "tilelang-separate": grouped_gemm_tilelang_separate,
        "ours (sonic)": grouped_gemm_sonic,
    }
    
    # 1. 正确性验证
    print("\n[1] Correctness Verification:")
    correctness = verify_correctness(X, W, expert_indices, versions)
    
    all_correct = True
    for name, (is_correct, max_diff) in correctness.items():
        status = "✅" if is_correct else "❌"
        print(f"  {status} {name:25s}: max_diff = {max_diff:.6f}")
        if not is_correct:
            all_correct = False
    
    if not all_correct:
        print("\n⚠️  Warning: Some versions have correctness issues!")
        print("Continuing with benchmark, but results may be invalid.\n")
    
    # 2. 性能Benchmark
    print(f"\n[2] Performance Benchmark (warmup={warmup}, rep={rep}):")
    
    latencies = {}
    
    for name, func in versions.items():
        try:
            # 预编译（避免首次编译计入时间）
            _ = func(X, W, expert_indices)
            torch.cuda.synchronize()
            
            # Benchmark
            latency_ms = benchmark_function(func, X, W, expert_indices, warmup=warmup, rep=rep)
            latencies[name] = latency_ms
            
            print(f"  {name:25s}: {latency_ms:8.4f} ms")
            
        except Exception as e:
            print(f"  {name:25s}: FAILED ({e})")
            latencies[name] = float('inf')
    
    return latencies


def main():
    """主Benchmark函数"""
    print("="*80)
    print("Grouped GEMM Performance Benchmark")
    print("4 Versions: pytorch-loop, pytorch-compiled, tilelang-separate, ours (sonic)")
    print("="*80)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device_name = torch.cuda.get_device_name()
    print(f"Device: {device_name}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 运行所有配置
    all_results = []
    
    for config in BENCHMARK_CONFIGS:
        config_name, total_tokens, H, I, E, description = config
        
        latencies = run_benchmark_config(
            config_name, total_tokens, H, I, E, description,
            warmup=50, rep=500
        )
        
        # 保存结果
        all_results.append({
            "config": config_name,
            "latencies": latencies
        })
    
    # ============ 汇总表格 ============
    print("\n" + "="*80)
    print("SUMMARY: Latency Comparison (ms)")
    print("="*80)
    
    # 延迟表格
    latency_table = []
    for result in all_results:
        config = result["config"]
        latencies = result["latencies"]
        
        row = [config]
        for version in ["pytorch-loop", "pytorch-compiled", "tilelang-separate", "ours (sonic)"]:
            lat = latencies.get(version, float('inf'))
            if lat == float('inf'):
                row.append("FAILED")
            else:
                row.append(f"{lat:.4f}")
        
        latency_table.append(row)
    
    headers = ["Config", "PyTorch", "Compiled", "TileLang-Sep", "Ours (Sonic)"]
    print(tabulate(latency_table, headers=headers, tablefmt="grid"))
    
    # 加速比表格
    print("\n" + "="*80)
    print("SUMMARY: Speedup vs PyTorch Baseline")
    print("="*80)
    
    speedup_table = []
    for result in all_results:
        config = result["config"]
        latencies = result["latencies"]
        
        baseline = latencies.get("pytorch-loop", float('inf'))
        if baseline == float('inf'):
            continue
        
        row = [config]
        for version in ["pytorch-compiled", "tilelang-separate", "ours (sonic)"]:
            lat = latencies.get(version, float('inf'))
            if lat == float('inf') or lat == 0:
                row.append("N/A")
            else:
                speedup = baseline / lat
                row.append(f"{speedup:.2f}x")
        
        speedup_table.append(row)
    
    headers = ["Config", "Compiled", "TileLang-Sep", "Ours (Sonic)"]
    print(tabulate(speedup_table, headers=headers, tablefmt="grid"))
    
    # ============ 计算平均加速 ============
    print("\n" + "="*80)
    print("Average Speedup:")
    print("="*80)
    
    for version in ["pytorch-compiled", "tilelang-separate", "ours (sonic)"]:
        speedups = []
        for result in all_results:
            latencies = result["latencies"]
            baseline = latencies.get("pytorch-loop", float('inf'))
            lat = latencies.get(version, float('inf'))
            
            if baseline != float('inf') and lat != float('inf') and lat > 0:
                speedups.append(baseline / lat)
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"  {version:25s}: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
