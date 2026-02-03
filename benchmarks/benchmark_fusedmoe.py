"""
FusedMoE Benchmark

对比两种实现:
1. PyTorch: 基于排序+分组的高效实现
2. TL-Official: TileLang实现 (无pass_configs，自动启用TMA+WS)

公平性保证:
- 所有实现使用相同的输入数据
- 只计时kernel执行，不计时数据准备
- Router计算(softmax+topk)和数据分组在所有版本中统一处理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import torch
import torch.nn.functional as F
from tabulate import tabulate
from tilelang.profiler import do_bench

from kernels.fusedmoe_baseline import (
    generate_fusedmoe_inputs,
)
from kernels.fusedmoe_tilelang import FusedMoETileLang


# 官方配置 (来自 tilelang/examples/fusedmoe)
OFFICIAL_CONFIGS = [
    {
        "name": "Small-Test",
        "batch_size": 1,
        "seq_len": 1024,
        "d_hidden": 2048,
        "d_expert": 512,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_experts_per_token": 2,
    },
    {
        "name": "DeepSeek-V3",
        "batch_size": 1,
        "seq_len": 8192,
        "d_hidden": 7168,
        "d_expert": 2048,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_experts_per_token": 4,
    },
]


def prepare_routed_data(
    x_flat: torch.Tensor,
    router_weight: torch.Tensor,
    n_experts_per_token: int,
    n_routed_experts: int,
    block_token: int = 128,
    return_counts: bool = False,
):
    """
    预处理Router分配，返回分组后的数据
    这部分在所有实现中共享，不计入benchmark时间
    """
    num_tokens, d_hidden = x_flat.shape
    device = x_flat.device
    dtype = x_flat.dtype
    
    # Router: softmax + topk
    logits = x_flat @ router_weight.T
    scores = F.softmax(logits, dim=-1)
    topk_scores, topk_indices = torch.topk(
        scores, k=n_experts_per_token, dim=-1, sorted=False
    )
    
    # 分组信息
    flat_indices = topk_indices.view(-1)
    flat_weights = topk_scores.view(-1)
    
    sorted_idx = flat_indices.argsort()
    counts = flat_indices.bincount(minlength=n_routed_experts).cpu().numpy()
    offsets = counts.cumsum()
    
    num_per_tok = n_experts_per_token
    token_idxs = sorted_idx // num_per_tok
    
    # 预分配buffer
    group_sum = num_tokens * n_experts_per_token
    stacked_expert_tokens = torch.empty(group_sum, d_hidden, device=device, dtype=dtype)
    stacked_expert_weights = torch.empty(group_sum, device=device, dtype=dtype)
    stacked_expert_tokens_idxs = torch.empty(group_sum, dtype=torch.int64, device=device)
    
    # 填充buffer
    for expert_id in range(n_routed_experts):
        start_idx = 0 if expert_id == 0 else offsets[expert_id - 1]
        end_idx = offsets[expert_id]
        if start_idx == end_idx:
            continue
        
        exp_token_idxs = token_idxs[start_idx:end_idx]
        stacked_expert_tokens[start_idx:end_idx] = x_flat[exp_token_idxs]
        stacked_expert_tokens_idxs[start_idx:end_idx] = exp_token_idxs
        stacked_expert_weights[start_idx:end_idx] = flat_weights[sorted_idx[start_idx:end_idx]]
    
    # Group info tensors
    group_sizes = torch.tensor(counts, dtype=torch.int32, device=device)
    group_offsets_tensor = torch.tensor(offsets - counts, dtype=torch.int32, device=device)
    
    # Padded offsets
    group_padded_offsets = [0] * n_routed_experts
    for i in range(1, n_routed_experts):
        group_padded_offsets[i] = group_padded_offsets[i-1] + math.ceil((counts[i-1] + 1) / block_token) * block_token
    
    M = math.ceil(group_sum / block_token) + n_routed_experts
    group_idx_for_bx = [0] * M
    for bx in range(M):
        m_start_padded = bx * block_token
        for i in range(n_routed_experts):
            if m_start_padded >= group_padded_offsets[i]:
                group_idx_for_bx[bx] = i
    
    group_padded_offsets = torch.tensor(group_padded_offsets, dtype=torch.int32, device=device)
    group_idx_for_bx = torch.tensor(group_idx_for_bx, dtype=torch.int32, device=device)
    
    return {
        "stacked_expert_tokens": stacked_expert_tokens,
        "stacked_expert_weights": stacked_expert_weights,
        "stacked_expert_tokens_idxs": stacked_expert_tokens_idxs,
        "group_sizes": group_sizes,
        "group_offsets": group_offsets_tensor,
        "group_padded_offsets": group_padded_offsets,
        "group_idx_for_bx": group_idx_for_bx,
    }


def benchmark_pytorch_kernel(
    x_flat: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_up: torch.Tensor,
    routed_down: torch.Tensor,
    routed_data: dict,
    d_hidden: int,
):
    """
    PyTorch kernel only (不含router和数据准备)
    """
    num_tokens = x_flat.shape[0]
    device = x_flat.device
    dtype = x_flat.dtype
    n_routed_experts = routed_gate.shape[0]
    
    # Shared Expert FFN
    shared_gate_out = F.silu(x_flat @ shared_gate.T)
    shared_up_out = x_flat @ shared_up.T
    shared_output = (shared_gate_out * shared_up_out) @ shared_down.T
    
    # Routed Expert FFN (使用预处理好的分组数据)
    stacked_tokens = routed_data["stacked_expert_tokens"]
    stacked_weights = routed_data["stacked_expert_weights"]
    stacked_idxs = routed_data["stacked_expert_tokens_idxs"]
    group_sizes = routed_data["group_sizes"]
    
    expert_cache = torch.zeros(num_tokens, d_hidden, device=device, dtype=dtype)
    
    offsets = group_sizes.cumsum(dim=0).cpu().numpy()
    
    for expert_id in range(n_routed_experts):
        start_idx = 0 if expert_id == 0 else offsets[expert_id - 1]
        end_idx = offsets[expert_id]
        if start_idx == end_idx:
            continue
        
        expert_tokens = stacked_tokens[start_idx:end_idx]
        
        gate_out = F.silu(expert_tokens @ routed_gate[expert_id].T)
        up_out = expert_tokens @ routed_up[expert_id].T
        expert_out = (gate_out * up_out) @ routed_down[expert_id].T
        
        weights = stacked_weights[start_idx:end_idx].unsqueeze(1)
        expert_out = expert_out * weights
        
        exp_idxs = stacked_idxs[start_idx:end_idx]
        expert_cache.scatter_reduce_(
            0,
            exp_idxs.view(-1, 1).expand(-1, d_hidden),
            expert_out,
            reduce="sum"
        )
    
    output = shared_output + expert_cache
    return output


def benchmark_tilelang_kernel(
    x_flat: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_up: torch.Tensor,
    routed_down: torch.Tensor,
    routed_data: dict,
    moe_module: FusedMoETileLang,
    d_hidden: int,
):
    """
    TileLang kernel only (不含router和数据准备)
    """
    num_tokens = x_flat.shape[0]
    
    # 使用预分配的buffer
    moe_module.expert_cache.zero_()
    
    # Shared Expert Kernel
    moe_module.shared_kernel(
        x_flat,
        shared_gate,
        shared_up,
        shared_down,
        moe_module.up_logits_shared,
        moe_module.output_shared,
    )
    
    # Routed Expert Kernel
    moe_module.routed_kernel(
        routed_data["stacked_expert_tokens"],
        routed_gate,
        routed_up,
        routed_down,
        routed_data["stacked_expert_weights"],
        routed_data["group_sizes"],
        routed_data["group_offsets"],
        routed_data["group_padded_offsets"],
        routed_data["group_idx_for_bx"],
        moe_module.up_logits_routed,
        moe_module.output_routed,
    )
    
    # Scatter reduce
    expert_cache = torch.scatter_reduce(
        moe_module.expert_cache,
        0,
        routed_data["stacked_expert_tokens_idxs"].view(-1, 1).expand(-1, d_hidden),
        moe_module.output_routed,
        reduce="sum",
    )
    
    output = moe_module.output_shared + expert_cache
    return output


def benchmark_config(config: dict, warmup: int = 10) -> dict:
    """Benchmark单个配置"""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Generate data
    x, weights = generate_fusedmoe_inputs(
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        d_hidden=config["d_hidden"],
        d_expert=config["d_expert"],
        n_routed_experts=config["n_routed_experts"],
        n_shared_experts=config["n_shared_experts"],
        n_experts_per_token=config["n_experts_per_token"],
    )
    
    n_experts_per_token = config["n_experts_per_token"]
    n_routed_experts = config["n_routed_experts"]
    d_hidden = config["d_hidden"]
    
    x_flat = x.view(-1, d_hidden)
    
    print(f"Input shape: {x.shape}")
    print(f"Routed experts: {n_routed_experts}, Experts per token: {n_experts_per_token}")
    
    # 预处理分组数据 (不计入benchmark时间)
    print("\nPreparing routed data (not timed)...")
    routed_data = prepare_routed_data(
        x_flat, weights["router_weight"],
        n_experts_per_token, n_routed_experts
    )
    
    # 创建TileLang模块 (编译kernel，不计入benchmark)
    print("Compiling TileLang kernels (not timed)...")
    moe_module = FusedMoETileLang(
        d_hidden=config["d_hidden"],
        d_expert=config["d_expert"],
        n_routed_experts=n_routed_experts,
        n_shared_experts=config["n_shared_experts"],
        n_experts_per_token=n_experts_per_token,
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
    )
    
    results = {"name": name, "config": config}
    
    # ============ Benchmark PyTorch Kernel ============
    print("\nBenchmarking PyTorch kernel...")
    def run_pytorch():
        return benchmark_pytorch_kernel(
            x_flat,
            weights["shared_gate"], weights["shared_up"], weights["shared_down"],
            weights["routed_gate"], weights["routed_up"], weights["routed_down"],
            routed_data, d_hidden
        )
    
    # Warmup
    for _ in range(warmup):
        _ = run_pytorch()
    torch.cuda.synchronize()
    
    results["pytorch"] = do_bench(run_pytorch)
    print(f"  PyTorch: {results['pytorch']:.4f} ms")
    
    # ============ Benchmark TileLang Kernel ============
    print("Benchmarking TileLang kernel...")
    def run_tilelang():
        return benchmark_tilelang_kernel(
            x_flat,
            weights["shared_gate"], weights["shared_up"], weights["shared_down"],
            weights["routed_gate"], weights["routed_up"], weights["routed_down"],
            routed_data, moe_module, d_hidden
        )
    
    try:
        # Warmup
        for _ in range(warmup):
            _ = run_tilelang()
        torch.cuda.synchronize()
        
        results["tilelang"] = do_bench(run_tilelang)
        print(f"  TileLang: {results['tilelang']:.4f} ms")
    except Exception as e:
        print(f"  TileLang: Error - {e}")
        results["tilelang"] = float('inf')
    
    # 验证正确性
    print("\nVerifying correctness...")
    output_pt = run_pytorch()
    output_tl = run_tilelang()
    diff = (output_pt.float() - output_tl.float()).abs().max().item()
    status = "✅" if diff < 0.1 else "⚠️"
    print(f"  Max diff: {diff:.6f} {status}")
    
    return results


def print_summary(all_results: list):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY (Kernel Only, Fair Comparison)")
    print("="*80)
    
    # Table: Latency comparison
    print("\n### Kernel延迟对比 (ms)")
    print("(不含Router计算和数据分组)")
    headers = ["配置", "Params", "PyTorch", "TileLang", "加速比"]
    rows = []
    
    for r in all_results:
        cfg = r["config"]
        params = f"H={cfg['d_hidden']}, I={cfg['d_expert']}, E={cfg['n_routed_experts']}"
        
        pytorch_str = f"{r['pytorch']:.4f}"
        tilelang_str = f"{r['tilelang']:.4f}" if r['tilelang'] < float('inf') else "Error"
        
        if r['tilelang'] < float('inf'):
            speedup = r['pytorch'] / r['tilelang']
            speedup_str = f"{speedup:.2f}x"
            if speedup > 1.05:
                status = "✅"
            elif speedup > 0.95:
                status = "≈"
            else:
                status = "❌"
        else:
            speedup_str = "N/A"
            status = "❌"
        
        # Bold the faster one
        if r['tilelang'] < float('inf'):
            if r['pytorch'] < r['tilelang']:
                pytorch_str = f"**{pytorch_str}**"
            else:
                tilelang_str = f"**{tilelang_str}**"
        
        rows.append([r["name"], params, pytorch_str, tilelang_str, f"{speedup_str} {status}"])
    
    print(tabulate(rows, headers=headers, tablefmt="github"))
    
    # Average speedup
    speedups = []
    for r in all_results:
        if r['tilelang'] < float('inf'):
            speedups.append(r['pytorch'] / r['tilelang'])
    
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"\n**TileLang平均加速**: {avg_speedup:.2f}x vs PyTorch")


def main():
    print("="*80)
    print("FusedMoE Kernel Benchmark (Fair Comparison)")
    print("="*80)
    print("NOTE: Only timing kernel execution, NOT router/data preparation")
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Get GPU info
    device = torch.cuda.get_device_name(0)
    print(f"GPU: {device}")
    
    all_results = []
    
    for config in OFFICIAL_CONFIGS:
        try:
            results = benchmark_config(config)
            all_results.append(results)
        except Exception as e:
            print(f"Error benchmarking {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
