# TileLang MoE Kernels

# TopK + Softmax kernels
from .topk_softmax_baseline import (
    topk_softmax_separate,
    topk_softmax_separate_compiled,
    topk_softmax_fused,
)
from .topk_softmax_tilelang import (
    # Radix TopK (official algorithm)
    topk_radix_tilelang,
    # Selection Sort fusion kernel
    topk_softmax_selection,
    topk_softmax_sonic,  # alias for backward compatibility
    # Bitonic Sort fusion kernel (Sonic-style, parallel sort)
    topk_softmax_bitonic,
    # Non-fused TileLang version
    topk_softmax_tilelang_separate,
    # Auto-select best implementation
    topk_softmax_best,
    topk_softmax,  # alias for topk_softmax_best
)

# Grouped GEMM kernels
from .grouped_gemm_baseline import (
    grouped_gemm_loop,
    grouped_gemm_loop_compiled,
    grouped_gemm_padded,
)
from .grouped_gemm_tilelang import (
    grouped_gemm_tilelang,
    grouped_gemm_pytorch,
    grouped_gemm_best,
    grouped_gemm,  # alias for grouped_gemm_best
)

# mHC (Multi-Head Cache) kernels
from .mhc_baseline import (
    mhc_pre_pytorch,
    mhc_pre_pytorch_compiled,
    mhc_post_pytorch,
    mhc_post_pytorch_compiled,
)
from .mhc_tilelang import (
    mhc_pre_tilelang_official,
    mhc_pre_tilelang_optimized,
    mhc_pre_tilelang_splitk,
    mhc_pre,
    mhc_post_tilelang_official,
    mhc_post,
)

__all__ = [
    # TopK + Softmax
    'topk_radix_tilelang',
    'topk_softmax_separate',
    'topk_softmax_separate_compiled',
    'topk_softmax_fused',
    'topk_softmax_selection',  # Selection Sort based
    'topk_softmax_sonic',      # alias for topk_softmax_selection
    'topk_softmax_bitonic',    # Bitonic Sort based (Sonic-style)
    'topk_softmax_tilelang_separate',
    'topk_softmax_best',
    'topk_softmax',  # recommended: auto-select best
    # Grouped GEMM
    'grouped_gemm_loop',
    'grouped_gemm_loop_compiled',
    'grouped_gemm_padded',
    'grouped_gemm_tilelang',
    'grouped_gemm_pytorch',
    'grouped_gemm_best',
    'grouped_gemm',  # recommended: auto-select best
    # mHC
    'mhc_pre_pytorch',
    'mhc_pre_pytorch_compiled',
    'mhc_post_pytorch',
    'mhc_post_pytorch_compiled',
    'mhc_pre_tilelang_official',
    'mhc_pre_tilelang_optimized',
    'mhc_pre_tilelang_splitk',
    'mhc_pre',  # recommended: auto-select best
    'mhc_post_tilelang_official',
    'mhc_post',  # recommended
]
