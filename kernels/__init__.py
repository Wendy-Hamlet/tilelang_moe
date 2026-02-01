# TileLang MoE Kernels

# TopK + Softmax kernels
from .topk_radix import topk_radix_tilelang
from .topk_softmax_baseline import (
    topk_softmax_separate,
    topk_softmax_separate_compiled,
    topk_softmax_fused,
)
from .topk_softmax_sonic import topk_softmax_sonic
from .topk_softmax_tilelang_separate import topk_softmax_tilelang_separate

# Grouped GEMM kernels
from .grouped_gemm_baseline import (
    grouped_gemm_loop,
    grouped_gemm_loop_compiled,
    grouped_gemm_padded,
)
from .grouped_gemm_tilelang_separate import grouped_gemm_tilelang_separate
from .grouped_gemm_tilelang_official import grouped_gemm_tilelang_official
from .grouped_gemm_sonic import grouped_gemm_sonic

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
    'topk_softmax_sonic',
    'topk_softmax_tilelang_separate',
    # Grouped GEMM
    'grouped_gemm_loop',
    'grouped_gemm_loop_compiled',
    'grouped_gemm_padded',
    'grouped_gemm_tilelang_separate',
    'grouped_gemm_tilelang_official',
    'grouped_gemm_sonic',
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
