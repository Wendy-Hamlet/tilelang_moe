"""
FusedMoE TileLang Implementation

基于 tilelang/examples/fusedmoe/example_fusedmoe_tilelang.py

TL-Official: 无pass_configs，让TileLang自动优化 (TMA + Warp Specialization)
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

import tilelang
import tilelang.language as T


@tilelang.jit  # 无pass_configs，TileLang自动优化
def _moe_shared_kernel(
    d_hidden: int,
    d_expert: int,
    n_shared_experts: int,
    dtype,
    num_tokens: int,
    block_token: int = 128,
    block_dhidden: int = 128,
    block_dexpert: int = 128,
    threads: int = 256,
    num_stages: int = 1,
):
    """Shared Expert Kernel: 所有token共享"""
    scale = 1.44269504  # log2(e)
    
    dhidden = d_hidden
    dexpert = d_expert * n_shared_experts
    
    accum_type = T.float32

    @T.prim_func
    def kernel_shared(
        input: T.Tensor((num_tokens, dhidden), dtype),
        shared_W_gate: T.Tensor((dexpert, dhidden), dtype),
        shared_W_up: T.Tensor((dexpert, dhidden), dtype),
        shared_W_down: T.Tensor((dhidden, dexpert), dtype),
        up_logits: T.Tensor((num_tokens, dexpert), dtype),
        output: T.Tensor((num_tokens, dhidden), dtype),
    ):
        # Step 1: Compute gate and up logits
        with T.Kernel(T.ceildiv(num_tokens, block_token), T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
            input_shared = T.alloc_shared((block_token, block_dhidden), dtype=dtype)
            W_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            W_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)

            gate_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_type)
            up_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_type)

            T.annotate_layout({
                input_shared: tilelang.layout.make_swizzled_layout(input_shared),
                W_gate_shared: tilelang.layout.make_swizzled_layout(W_gate_shared),
                W_up_shared: tilelang.layout.make_swizzled_layout(W_up_shared),
            })
            T.use_swizzle(10)
            T.clear(gate_logits_local)
            T.clear(up_logits_local)

            for k in T.Pipelined(T.ceildiv(dhidden, block_dhidden), num_stages=num_stages):
                T.copy(input[bx * block_token, k * block_dhidden], input_shared)
                T.copy(shared_W_gate[by * block_dexpert, k * block_dhidden], W_gate_shared)
                T.copy(shared_W_up[by * block_dexpert, k * block_dhidden], W_up_shared)
                T.gemm(input_shared, W_gate_shared, gate_logits_local, transpose_B=True)
                T.gemm(input_shared, W_up_shared, up_logits_local, transpose_B=True)

            # SiLU and element-wise product
            for i, j in T.Parallel(block_token, block_dexpert):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            T.copy(up_logits_local, up_logits[bx * block_token, by * block_dexpert])

        # Step 2: Compute down logits
        with T.Kernel(T.ceildiv(num_tokens, block_token), T.ceildiv(dhidden, block_dhidden), threads=threads) as (bx, by):
            up_logits_shared = T.alloc_shared((block_token, block_dexpert), dtype=dtype)
            W_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_type)

            T.annotate_layout({
                up_logits_shared: tilelang.layout.make_swizzled_layout(up_logits_shared),
                W_down_shared: tilelang.layout.make_swizzled_layout(W_down_shared),
            })
            T.use_swizzle(10)
            T.clear(output_local)

            for k in T.Pipelined(T.ceildiv(dexpert, block_dexpert), num_stages=num_stages):
                T.copy(up_logits[bx * block_token, k * block_dexpert], up_logits_shared)
                T.copy(shared_W_down[by * block_dhidden, k * block_dexpert], W_down_shared)
                T.gemm(up_logits_shared, W_down_shared, output_local, transpose_B=True)

            T.copy(output_local, output[bx * block_token, by * block_dhidden])

    return kernel_shared


@tilelang.jit  # 无pass_configs，TileLang自动优化
def _moe_routed_kernel(
    d_hidden: int,
    d_expert: int,
    n_routed_experts: int,
    dtype,
    group_sum: int,
    group_count: int,
    block_token: int = 128,
    block_dhidden: int = 128,
    block_dexpert: int = 128,
    threads: int = 256,
    num_stages: int = 1,
):
    """Routed Expert Kernel: 分组GEMM"""
    scale = 1.44269504  # log2(e)
    
    dhidden = d_hidden
    dexpert = d_expert
    
    M = math.ceil(group_sum / block_token) + group_count
    accum_dtype = T.float32

    @T.prim_func
    def kernel(
        input: T.Tensor((group_sum, dhidden), dtype),
        routed_expert_gate: T.Tensor((n_routed_experts, dexpert, dhidden), dtype),
        routed_expert_up: T.Tensor((n_routed_experts, dexpert, dhidden), dtype),
        routed_expert_down: T.Tensor((n_routed_experts, dhidden, dexpert), dtype),
        routed_expert_weights: T.Tensor((group_sum,), dtype),
        group_sizes: T.Tensor((n_routed_experts,), T.int32),
        group_offsets: T.Tensor((n_routed_experts,), T.int32),
        group_padded_offsets: T.Tensor((n_routed_experts,), T.int32),
        group_idx_for_bx: T.Tensor((M,), T.int32),
        up_logits: T.Tensor((group_sum, dexpert), dtype),
        output: T.Tensor((group_sum, dhidden), dtype),
    ):
        # Step 1: Compute gate and up logits
        with T.Kernel(M, T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
            input_shared = T.alloc_shared((block_token, block_dhidden), dtype=dtype)
            routed_expert_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)
            routed_expert_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype=dtype)

            gate_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)
            up_logits_local = T.alloc_fragment((block_token, block_dexpert), dtype=accum_dtype)

            T.annotate_layout({
                input_shared: tilelang.layout.make_swizzled_layout(input_shared),
                routed_expert_gate_shared: tilelang.layout.make_swizzled_layout(routed_expert_gate_shared),
                routed_expert_up_shared: tilelang.layout.make_swizzled_layout(routed_expert_up_shared),
            })
            T.use_swizzle(10)

            m_start_padded = bx * block_token
            cur_group_idx = group_idx_for_bx[bx]
            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(gate_logits_local)
            T.clear(up_logits_local)

            for k in T.Pipelined(T.ceildiv(dhidden, block_dhidden), num_stages=num_stages):
                T.copy(input[m_start : m_start + block_token, k * block_dhidden : (k + 1) * block_dhidden], input_shared)
                T.copy(
                    routed_expert_gate[cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden],
                    routed_expert_gate_shared,
                )
                T.gemm(input_shared, routed_expert_gate_shared, gate_logits_local, transpose_B=True)
                T.copy(
                    routed_expert_up[cur_group_idx, by * block_dexpert : (by + 1) * block_dexpert, k * block_dhidden : (k + 1) * block_dhidden],
                    routed_expert_up_shared,
                )
                T.gemm(input_shared, routed_expert_up_shared, up_logits_local, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dexpert):
                gate_logits_local[i, j] = gate_logits_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_logits_local[i, j] * scale)))
                up_logits_local[i, j] = up_logits_local[i, j] * gate_logits_local[i, j]

            for i, j in T.Parallel(block_token, block_dexpert):
                if i < actual_rows:
                    up_logits[m_start + i, by * block_dexpert + j] = up_logits_local[i, j]

        # Step 2: Compute down logits
        with T.Kernel(M, T.ceildiv(dhidden, block_dhidden), threads=threads) as (bx, by):
            up_logits_shared = T.alloc_shared((block_token, block_dexpert), dtype=dtype)
            routed_expert_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype=dtype)
            output_local = T.alloc_fragment((block_token, block_dhidden), dtype=accum_dtype)

            T.annotate_layout({
                up_logits_shared: tilelang.layout.make_swizzled_layout(up_logits_shared),
                routed_expert_down_shared: tilelang.layout.make_swizzled_layout(routed_expert_down_shared),
            })
            T.use_swizzle(10)

            m_start_padded = bx * block_token
            cur_group_idx = group_idx_for_bx[bx]
            cur_group_size = group_sizes[cur_group_idx]
            m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
            actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))

            T.clear(output_local)

            for k in T.Pipelined(T.ceildiv(dexpert, block_dexpert), num_stages=num_stages):
                T.copy(up_logits[m_start : m_start + block_token, k * block_dexpert : (k + 1) * block_dexpert], up_logits_shared)
                T.copy(
                    routed_expert_down[cur_group_idx, by * block_dhidden : (by + 1) * block_dhidden, k * block_dexpert : (k + 1) * block_dexpert],
                    routed_expert_down_shared,
                )
                T.gemm(up_logits_shared, routed_expert_down_shared, output_local, transpose_B=True)

            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    output[m_start + i, by * block_dhidden + j] = output_local[i, j] * routed_expert_weights[m_start + i]

    return kernel


class FusedMoETileLang:
    """
    TileLang FusedMoE Wrapper
    
    管理kernel编译和执行状态
    """
    
    def __init__(
        self,
        d_hidden: int,
        d_expert: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_experts_per_token: int,
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype = torch.float16,
        block_token: int = 128,
    ):
        self.d_hidden = d_hidden
        self.d_expert = d_expert
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_experts_per_token = n_experts_per_token
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.block_token = block_token
        self.dtype = dtype
        self.device = torch.device("cuda")
        
        num_tokens = batch_size * seq_len
        group_sum = num_tokens * n_experts_per_token
        
        # Compile kernels
        dtype_str = T.float16 if dtype == torch.float16 else T.bfloat16
        
        self.shared_kernel = _moe_shared_kernel(
            d_hidden=d_hidden,
            d_expert=d_expert,
            n_shared_experts=n_shared_experts,
            dtype=dtype_str,
            num_tokens=num_tokens,
            block_token=block_token,
        )
        
        self.routed_kernel = _moe_routed_kernel(
            d_hidden=d_hidden,
            d_expert=d_expert,
            n_routed_experts=n_routed_experts,
            dtype=dtype_str,
            group_sum=group_sum,
            group_count=n_routed_experts,
            block_token=block_token,
        )
        
        # Pre-allocate buffers
        shared_dim = d_expert * n_shared_experts
        self.up_logits_shared = torch.empty(num_tokens, shared_dim, device=self.device, dtype=dtype)
        self.output_shared = torch.empty(num_tokens, d_hidden, device=self.device, dtype=dtype)
        self.up_logits_routed = torch.empty(group_sum, d_expert, device=self.device, dtype=dtype)
        self.output_routed = torch.empty(group_sum, d_hidden, device=self.device, dtype=dtype)
        
        # Token buffers for routed experts
        self.stacked_expert_tokens = torch.empty(group_sum, d_hidden, device=self.device, dtype=dtype)
        self.stacked_expert_weights = torch.empty(group_sum, device=self.device, dtype=dtype)
        self.stacked_expert_tokens_idxs = torch.empty(group_sum, dtype=torch.int64, device=self.device)
        self.expert_cache = torch.zeros(num_tokens, d_hidden, device=self.device, dtype=dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        router_weight: torch.Tensor,
        shared_gate: torch.Tensor,
        shared_up: torch.Tensor,
        shared_down: torch.Tensor,
        routed_gate: torch.Tensor,
        routed_up: torch.Tensor,
        routed_down: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using TileLang kernels
        """
        batch_size, seq_len, d_hidden = x.shape
        x_flat = x.view(-1, d_hidden)
        num_tokens = x_flat.shape[0]
        
        # 1. Router
        logits = x_flat @ router_weight.T
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(
            scores, k=self.n_experts_per_token, dim=-1, sorted=False
        )
        
        # Prepare for grouped GEMM
        flat_indices = topk_indices.view(-1)
        flat_weights = topk_scores.view(-1)
        
        sorted_idx = flat_indices.argsort()
        counts = flat_indices.bincount(minlength=self.n_routed_experts).cpu().numpy()
        offsets = counts.cumsum()
        
        num_per_tok = self.n_experts_per_token
        token_idxs = sorted_idx // num_per_tok
        
        # Fill buffers
        for expert_id in range(self.n_routed_experts):
            start_idx = 0 if expert_id == 0 else offsets[expert_id - 1]
            end_idx = offsets[expert_id]
            if start_idx == end_idx:
                continue
            
            exp_token_idxs = token_idxs[start_idx:end_idx]
            self.stacked_expert_tokens[start_idx:end_idx] = x_flat[exp_token_idxs]
            self.stacked_expert_tokens_idxs[start_idx:end_idx] = exp_token_idxs
            self.stacked_expert_weights[start_idx:end_idx] = flat_weights[sorted_idx[start_idx:end_idx]]
        
        # Prepare group info
        group_sizes = torch.tensor(counts, dtype=torch.int32, device=self.device)
        group_offsets = torch.tensor(offsets - counts, dtype=torch.int32, device=self.device)
        
        # Compute padded offsets
        group_padded_offsets = [0] * self.n_routed_experts
        for i in range(1, self.n_routed_experts):
            group_padded_offsets[i] = group_padded_offsets[i-1] + math.ceil((counts[i-1] + 1) / self.block_token) * self.block_token
        
        M = math.ceil(num_tokens * self.n_experts_per_token / self.block_token) + self.n_routed_experts
        group_idx_for_bx = [0] * M
        for bx in range(M):
            m_start_padded = bx * self.block_token
            for i in range(self.n_routed_experts):
                if m_start_padded >= group_padded_offsets[i]:
                    group_idx_for_bx[bx] = i
        
        group_padded_offsets = torch.tensor(group_padded_offsets, dtype=torch.int32, device=self.device)
        group_idx_for_bx = torch.tensor(group_idx_for_bx, dtype=torch.int32, device=self.device)
        
        # Reset expert cache
        self.expert_cache.zero_()
        
        # 2. Shared Expert Kernel
        # shared_down is (d_hidden, shared_dim), which is the expected shape
        self.shared_kernel(
            x_flat,
            shared_gate,
            shared_up,
            shared_down,  # (dhidden, dexpert) - no transpose needed
            self.up_logits_shared,
            self.output_shared,
        )
        
        # 3. Routed Expert Kernel
        self.routed_kernel(
            self.stacked_expert_tokens,
            routed_gate,
            routed_up,
            routed_down,
            self.stacked_expert_weights,
            group_sizes,
            group_offsets,
            group_padded_offsets,
            group_idx_for_bx,
            self.up_logits_routed,
            self.output_routed,
        )
        
        # 4. Scatter reduce
        self.expert_cache = torch.scatter_reduce(
            self.expert_cache,
            0,
            self.stacked_expert_tokens_idxs.view(-1, 1).expand(-1, d_hidden),
            self.output_routed,
            reduce="sum",
        )
        
        # 5. Combine outputs
        output = (self.output_shared + self.expert_cache).view(batch_size, seq_len, d_hidden)
        return output


# Global cache for compiled modules
_tilelang_moe_cache: Dict[Tuple, FusedMoETileLang] = {}


def fusedmoe_tilelang(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    shared_gate: torch.Tensor,
    shared_up: torch.Tensor,
    shared_down: torch.Tensor,
    routed_gate: torch.Tensor,
    routed_up: torch.Tensor,
    routed_down: torch.Tensor,
    n_experts_per_token: int,
) -> torch.Tensor:
    """
    TileLang FusedMoE functional API
    
    自动缓存编译好的kernel
    """
    batch_size, seq_len, d_hidden = x.shape
    n_routed_experts = routed_gate.shape[0]
    d_expert = routed_gate.shape[1]
    shared_dim = shared_gate.shape[0]
    n_shared_experts = shared_dim // d_expert
    
    # Cache key
    key = (batch_size, seq_len, d_hidden, d_expert, n_routed_experts, n_shared_experts, n_experts_per_token, x.dtype)
    
    if key not in _tilelang_moe_cache:
        _tilelang_moe_cache[key] = FusedMoETileLang(
            d_hidden=d_hidden,
            d_expert=d_expert,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            n_experts_per_token=n_experts_per_token,
            batch_size=batch_size,
            seq_len=seq_len,
            dtype=x.dtype,
        )
    
    moe = _tilelang_moe_cache[key]
    return moe.forward(
        x, router_weight,
        shared_gate, shared_up, shared_down,
        routed_gate, routed_up, routed_down
    )


if __name__ == "__main__":
    from fusedmoe_baseline import generate_fusedmoe_inputs, fusedmoe_pytorch
    
    torch.manual_seed(42)
    
    config = {
        "batch_size": 1,
        "seq_len": 1024,
        "d_hidden": 2048,
        "d_expert": 512,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_experts_per_token": 2,
    }
    
    x, weights = generate_fusedmoe_inputs(**config)
    
    # PyTorch reference
    output_pt = fusedmoe_pytorch(
        x, weights["router_weight"],
        weights["shared_gate"], weights["shared_up"], weights["shared_down"],
        weights["routed_gate"], weights["routed_up"], weights["routed_down"],
        config["n_experts_per_token"]
    )
    
    # TileLang
    output_tl = fusedmoe_tilelang(
        x, weights["router_weight"],
        weights["shared_gate"], weights["shared_up"], weights["shared_down"],
        weights["routed_gate"], weights["routed_up"], weights["routed_down"],
        config["n_experts_per_token"]
    )
    
    diff = (output_pt.float() - output_tl.float()).abs().max().item()
    print(f"Max diff: {diff}")
    if diff < 0.1:
        print("✅ TileLang FusedMoE matches PyTorch!")
    else:
        print("❌ Large difference detected")
