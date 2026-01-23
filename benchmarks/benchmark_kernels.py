"""
单个算子性能测试
测试TopK、GEMM等单独算子的性能
"""
import torch
import yaml
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '/home/chjin/CUDA/tilelang_moe')

from kernels.topk_gating import topk_gating_tilelang
from utils.profiler import Profiler
import argparse


def benchmark_topk_gating(config_path: str):
    """测试TopK门控性能"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    bench_cfg = config['benchmark']
    device = f"cuda:{config['device']['cuda_device']}"
    
    profiler = Profiler(
        warmup_iters=bench_cfg['warmup_iters'],
        test_iters=bench_cfg['test_iters']
    )
    
    results = []
    
    print("=" * 80)
    print("TopK Gating Benchmark")
    print("=" * 80)
    
    for batch_size in bench_cfg['batch_sizes']:
        for seq_len in bench_cfg['seq_lengths']:
            num_tokens = batch_size * seq_len
            logits = torch.randn(
                num_tokens, model_cfg['num_experts'],
                device=device
            )
            
            # Benchmark PyTorch
            perf_pt = profiler.benchmark(
                lambda: torch.topk(logits, model_cfg['num_experts_per_tok'], dim=-1)
            )
            
            # Benchmark TileLang
            perf_tl = profiler.benchmark(
                topk_gating_tilelang, logits, model_cfg['num_experts_per_tok']
            )
            
            results.append({
                'num_tokens': num_tokens,
                'num_experts': model_cfg['num_experts'],
                'k': model_cfg['num_experts_per_tok'],
                'pytorch_ms': perf_pt['mean_ms'],
                'tilelang_ms': perf_tl['mean_ms'],
                'speedup': perf_pt['mean_ms'] / perf_tl['mean_ms']
            })
            
            print(f"Tokens={num_tokens}: PyTorch {perf_pt['mean_ms']:.3f}ms, "
                  f"TileLang {perf_tl['mean_ms']:.3f}ms, "
                  f"Speedup {perf_pt['mean_ms']/perf_tl['mean_ms']:.2f}x")
    
    # 保存结果
    df = pd.DataFrame(results)
    output_path = Path('benchmarks/results/topk_benchmark.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/moe_config.yaml')
    args = parser.parse_args()
    
    benchmark_topk_gating(args.config)
