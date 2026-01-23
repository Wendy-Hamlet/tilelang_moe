"""
MoE性能基准测试
对比TileLang实现与PyTorch baseline的性能
"""
import torch
import yaml
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, '/home/chjin/CUDA/tilelang_moe')

from baselines.pytorch.moe_layer import MoELayer
from utils.profiler import Profiler
import argparse


def benchmark_moe(config_path: str):
    """运行MoE完整benchmark"""
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    bench_cfg = config['benchmark']
    
    # 设置设备
    device = f"cuda:{config['device']['cuda_device']}"
    
    # 初始化profiler
    profiler = Profiler(
        warmup_iters=bench_cfg['warmup_iters'],
        test_iters=bench_cfg['test_iters']
    )
    
    results = []
    
    print("=" * 80)
    print("DeepSeek MoE Benchmark")
    print("=" * 80)
    
    for batch_size in bench_cfg['batch_sizes']:
        for seq_len in bench_cfg['seq_lengths']:
            print(f"\nTesting: batch_size={batch_size}, seq_len={seq_len}")
            
            # 创建模型
            moe_layer = MoELayer(
                hidden_size=model_cfg['hidden_size'],
                intermediate_size=model_cfg['intermediate_size'],
                num_experts=model_cfg['num_experts'],
                num_experts_per_tok=model_cfg['num_experts_per_tok']
            ).to(device)
            
            # 创建输入
            x = torch.randn(
                batch_size, seq_len, model_cfg['hidden_size'],
                device=device
            )
            
            # Benchmark PyTorch
            perf = profiler.benchmark(moe_layer, x)
            
            results.append({
                'implementation': 'PyTorch',
                'batch_size': batch_size,
                'seq_len': seq_len,
                'mean_ms': perf['mean_ms'],
                'std_ms': perf['std_ms'],
                'throughput_tokens_per_sec': (batch_size * seq_len * 1000) / perf['mean_ms']
            })
            
            print(f"  PyTorch: {perf['mean_ms']:.3f} ± {perf['std_ms']:.3f} ms")
            
            # TODO: Benchmark TileLang实现
            # perf_tl = profiler.benchmark(moe_layer_tilelang, x)
            
    # 保存结果
    df = pd.DataFrame(results)
    output_path = Path('benchmarks/results/moe_benchmark.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print(f"Results saved to {output_path}")
    print("=" * 80)
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/moe_config.yaml')
    args = parser.parse_args()
    
    benchmark_moe(args.config)
