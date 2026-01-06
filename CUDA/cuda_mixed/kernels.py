import os
from typing import Optional
from torch.utils.cpp_extension import load

_EXT_NAME = "mixed_precision_w2w4_kernels"
_ext = None


def _want_verbose(user_flag: Optional[bool] = None) -> bool:
    if user_flag:
        return True
    env = os.environ.get("CUDA_MIXED_VERBOSE", "")
    return env.lower() in {"1", "true", "yes", "y", "on"}


def load_mixed_precision_extension(verbose: Optional[bool] = None):
    global _ext
    if _ext is not None:
        return _ext

    this_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(this_dir, "csrc")
    sources = [
        os.path.join(csrc_dir, "pybind.cpp"),
        os.path.join(csrc_dir, "gemv_mixed.cu"),
    ]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-Xptxas",
        "-O3",
    ]

    _ext = load(
        name=_EXT_NAME,
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=_want_verbose(verbose),
    )
    return _ext


__all__ = ["load_mixed_precision_extension"]
