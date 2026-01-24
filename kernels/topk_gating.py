"""
TileLang implementation of Top-K Gating for MoE
Based on official implementation: third_party/tilelang/examples/topk/example_topk.py
"""
import torch
import tilelang
import tilelang.language as T
import itertools


def topk_gating_pytorch(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch baseline implementation of Top-K gating"""
    topk_weights, topk_indices = torch.topk(scores, k=k, dim=-1, sorted=False)
    return topk_indices, topk_weights


def get_topk_configs():
    """
    Auto-tune configurations for TopK kernel
    Note: threads must be divisible by N (num_experts) for reduce operations
    For DeepSeek-V2 (N=160), valid thread counts: 160, 320, 480, 640, 800, 960
    """
    iter_params = dict(
        blk_m=[64, 128, 256],
        threads=[160, 320, 480],  # Must divide 160 (num_experts for DeepSeek-V2)
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_topk_configs())
@tilelang.jit(out_idx=[1, 2], target="cuda")
def tl_topk(M, N, topk, blk_m, threads=128):
    """
    TileLang Top-K kernel with auto-tuning
    Based on official implementation
    """
    dtype = T.float32

    @T.prim_func
    def topk_kernel(
        logits: T.Tensor([M, N], dtype),
        topk_gates: T.Tensor([M, topk], dtype),
        topk_indices: T.Tensor([M, topk], T.int32),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            # Allocate fragments for computation
            logits_frag = T.alloc_fragment([blk_m, N], dtype=dtype)
            max_val = T.alloc_fragment([blk_m], dtype=dtype)
            expand_max_idx = T.alloc_fragment([blk_m, N], T.int32)
            max_idx = T.alloc_fragment([blk_m], T.int32)

            # Load input logits
            T.copy(logits[bx * blk_m, 0], logits_frag)

            # Iteratively select top-k elements
            for k in T.serial(topk):
                T.fill(expand_max_idx, -1)
                
                # Find maximum value
                T.reduce_max(logits_frag, max_val, dim=1, clear=True)

                # Find index of maximum value
                for i, j in T.Parallel(blk_m, N):
                    expand_max_idx[i, j] = T.if_then_else(
                        max_val[i] == logits_frag[i, j], j, expand_max_idx[i, j]
                    )

                T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)

                # Mask out selected value for next iteration
                for i, j in T.Parallel(blk_m, N):
                    logits_frag[i, j] = T.if_then_else(
                        max_val[i] == logits_frag[i, j], -10000.0, logits_frag[i, j]
                    )

                # Store results
                for i in T.Parallel(blk_m):
                    topk_gates[bx * blk_m + i, k] = max_val[i]
                    topk_indices[bx * blk_m + i, k] = max_idx[i]

    return topk_kernel


def topk_gating_tilelang(scores: torch.Tensor, k: int, blk_m: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang optimized Top-K gating
    
    Args:
        scores: [batch, num_experts] - Gate scores (after softmax)
        k: Number of experts to select per token
        blk_m: Block size for M dimension (auto-tuned by default)
        
    Returns:
        indices: [batch, k] - Selected expert indices
        weights: [batch, k] - Corresponding weights (scores)
    """
    M, N = scores.shape
    
    # Compile kernel with auto-tuning
    kernel = tl_topk(M=M, N=N, topk=k, blk_m=blk_m)
    
    # Execute kernel
    topk_weights, topk_indices = kernel(scores)
    
    return topk_indices, topk_weights


if __name__ == "__main__":
    # Quick test (for local verification)
    batch, num_experts, k = 320, 160, 6
    scores = torch.randn(batch, num_experts, device="cuda", dtype=torch.float32)
    
    print("Testing Top-K implementations...")
    print(f"Config: batch={batch}, num_experts={num_experts}, k={k}")
    
    # PyTorch baseline
    indices_pt, weights_pt = topk_gating_pytorch(scores, k)
    print(f"✓ PyTorch: indices={indices_pt.shape}, weights={weights_pt.shape}")
    
    # TileLang version
    indices_tl, weights_tl = topk_gating_tilelang(scores, k)
    print(f"✓ TileLang: indices={indices_tl.shape}, weights={weights_tl.shape}")
    
    # Verify correctness
    torch.testing.assert_close(weights_tl, weights_pt, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(indices_tl, indices_pt)
    print("✓ Correctness verified!")

