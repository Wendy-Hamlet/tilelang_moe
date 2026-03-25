"""
Triton Grouped GEMM kernel — adapted from facebookexperimental/triton tutorial 08-grouped-gemm.

Two versions:
  1. grouped_matmul_kernel: pointer-based persistent kernel (non-TMA)
  2. grouped_matmul_tma_kernel: TMA descriptor-based (Hopper+)

Adapted for BF16 and MoE-style contiguous data layout.

Reference: https://github.com/facebookexperimental/triton/blob/main/python/tutorials/08-grouped-gemm.py
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


def _num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def _supports_tma():
    return torch.cuda.get_device_capability()[0] >= 9


# ============================================================
# Non-TMA kernel (all GPU architectures)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 84}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': _num_sms()}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': _num_sms()}),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    group_a_ptrs, group_b_ptrs, group_c_ptrs,
    group_gemm_sizes, g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.bfloat16)
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            tl.store(c_ptrs, c)
            tile_idx += NUM_SM
        last_problem_end = last_problem_end + num_tiles


# ============================================================
# TMA kernel (Hopper+ only)
# ============================================================
_tma_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK},
                  num_stages=s, num_warps=w)
    for BM in [128]
    for BN in [128, 256]
    for BK in [64, 128]
    for s in [3, 4]
    for w in [4, 8]
]


@triton.autotune(
    _tma_configs,
    key=['group_a_ptrs', 'group_b_ptrs', 'group_c_ptrs', 'group_size'],
)
@triton.jit
def grouped_matmul_tma_kernel(
    group_a_ptrs, group_b_ptrs, group_c_ptrs,
    group_gemm_sizes, g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    dtype = tl.bfloat16
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))
            a_desc = tl.make_tensor_descriptor(
                a_ptr, shape=[gm, gk], strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
            b_desc = tl.make_tensor_descriptor(
                b_ptr, shape=[gn, gk], strides=[ldb, 1],
                block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
            c_desc = tl.make_tensor_descriptor(
                c_ptr, shape=[gm, gn], strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
                k = gk
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)
                tile_idx += NUM_SM
        last_problem_end = last_problem_end + num_tiles


# ============================================================
# Wrapper: prepare pointer metadata from MoE-style contiguous data
# ============================================================
def prepare_triton_metadata(X_sorted, W_HI, batch_sizes, device="cuda"):
    """Convert contiguous MoE data to Triton's pointer-based format.

    Args:
        X_sorted: (routed_tokens_local, H) BF16, sorted by expert
        W_HI: (E_local, H, I) BF16 weight tensor
        batch_sizes: list of int, tokens per expert

    Returns:
        Tuple of device tensors for kernel launch (all outside timing).
        Also returns Y (output tensor) and group_C (list of output slices).
    """
    E_local = len(batch_sizes)
    H = X_sorted.shape[1]
    I = W_HI.shape[2]

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []

    # Allocate output
    Y = torch.empty(X_sorted.shape[0], I, device=device, dtype=X_sorted.dtype)

    offset = 0
    for eid in range(E_local):
        M = batch_sizes[eid]
        if M == 0:
            # Triton kernel iterates over group_size, need a valid entry
            # Use dummy 0-tile problem
            A_addrs.append(X_sorted.data_ptr())
            B_addrs.append(W_HI[eid].data_ptr())
            C_addrs.append(Y.data_ptr())
            g_sizes += [0, I, H]
            g_lds += [H, I, I]
        else:
            # A = X_sorted[offset:offset+M] shape (M, H), stride(0)=H
            A_addrs.append(X_sorted[offset].data_ptr())
            # B = W_HI[eid] shape (H, I), stride(0)=I
            B_addrs.append(W_HI[eid].data_ptr())
            # C = Y[offset:offset+M] shape (M, I), stride(0)=I
            C_addrs.append(Y[offset].data_ptr())
            g_sizes += [M, I, H]  # M, N, K
            g_lds += [H, I, I]    # lda, ldb, ldc
        offset += M

    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)

    return d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, Y


def prepare_triton_tma_metadata(X_sorted, W_HI, batch_sizes, device="cuda"):
    """Prepare metadata for TMA kernel. B must be transposed to (N, K) = (I, H)."""
    E_local = len(batch_sizes)
    H = X_sorted.shape[1]
    I = W_HI.shape[2]

    # TMA kernel expects B in (N, K) layout, i.e. W transposed to (I, H)
    W_IH = W_HI.transpose(1, 2).contiguous()  # (E_local, I, H)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []

    Y = torch.empty(X_sorted.shape[0], I, device=device, dtype=X_sorted.dtype)

    offset = 0
    for eid in range(E_local):
        M = batch_sizes[eid]
        if M == 0:
            A_addrs.append(X_sorted.data_ptr())
            B_addrs.append(W_IH[eid].data_ptr())
            C_addrs.append(Y.data_ptr())
            g_sizes += [0, I, H]
            g_lds += [H, H, I]  # lda=H, ldb=H (B is (I,H), stride(0)=H), ldc=I
        else:
            A_addrs.append(X_sorted[offset].data_ptr())
            B_addrs.append(W_IH[eid].data_ptr())
            C_addrs.append(Y[offset].data_ptr())
            g_sizes += [M, I, H]
            g_lds += [H, H, I]
        offset += M

    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)

    return d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, Y


def triton_grouped_gemm(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size):
    """Launch non-TMA Triton grouped GEMM kernel (pure kernel call for benchmarking)."""
    grid = lambda META: (META['NUM_SM'],)
    grouped_matmul_kernel[grid](
        d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size)


def triton_grouped_gemm_tma(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size):
    """Launch TMA Triton grouped GEMM kernel (pure kernel call for benchmarking)."""
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    grid = lambda META: (META['NUM_SM'],)
    grouped_matmul_tma_kernel[grid](
        d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size,
        NUM_SM=_num_sms())
