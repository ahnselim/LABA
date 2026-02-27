import math
import os
from typing import Dict
import torch
import torch.nn as nn

from .kernels import load_mixed_precision_extension


def _unpack_qzeros(packed: torch.Tensor, groups: int) -> torch.Tensor:
    """Unpack 4-bit zero-points (two per byte) into [O, groups] layout."""
    if packed.numel() == 0:
        return torch.empty(0, groups, dtype=torch.uint8, device=packed.device)
    packed = packed.contiguous()
    out_cols = packed.size(1) * 2
    zeros = torch.empty(packed.size(0), out_cols, dtype=torch.uint8, device=packed.device)
    zeros[:, 0::2] = packed & 0x0F
    zeros[:, 1::2] = packed >> 4
    return zeros[:, :groups].contiguous()


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Unpack two 4-bit values from each byte."""
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack((low, high), dim=-1).reshape(packed.size(0), -1)


def _unpack_2bit(packed: torch.Tensor) -> torch.Tensor:
    """Unpack four 2-bit values from each byte."""
    vals = torch.empty(
        packed.size(0),
        packed.size(1) * 4,
        dtype=torch.uint8,
        device=packed.device,
    )
    vals[:, 0::4] = packed & 0x03
    vals[:, 1::4] = (packed >> 2) & 0x03
    vals[:, 2::4] = (packed >> 4) & 0x03
    vals[:, 3::4] = (packed >> 6) & 0x03
    return vals


class CudaMixedBitLinear(nn.Module):
    """Cuda backend for mixed bit-width linear layers."""

    def __init__(self, in_features: int, out_features: int, entry: Dict, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit = int(entry["bit"])
        self.quant_type = entry["quant_type"]
        self.group_size = int(entry["group_size"])
        self.groups = math.ceil(self.in_features / self.group_size)
        gid = torch.arange(self.in_features, dtype=torch.long)
        gid = torch.div(gid, self.group_size, rounding_mode="floor")
        self.register_buffer("gid", gid, persistent=False)

        qweight = entry["qweight_packed"].to(torch.uint8).contiguous()
        self.register_buffer("qweight", qweight)

        scales = entry["scales"].contiguous()
        if scales.dtype != torch.float16:
            scales = scales.to(torch.float16)
        self.register_buffer("scales", scales)

        if self.quant_type == "asym":
            zeros_packed = entry.get("qzeros_packed")
            if zeros_packed is None:
                raise ValueError("Asymmetric quantization requires qzeros_packed.")
            zeros = _unpack_qzeros(zeros_packed.to(torch.uint8), self.groups)
            self.register_buffer("qzeros", zeros)
            self.register_buffer("qzeros_fp16", zeros.to(torch.float16), persistent=False)
        else:
            self.register_buffer(
                "qzeros",
                torch.empty(0, dtype=torch.uint8, device=qweight.device),
                persistent=False,
            )
            self.register_buffer(
                "qzeros_fp16",
                torch.empty(0, dtype=torch.float16, device=qweight.device),
                persistent=False,
            )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self._ext = None
        self._gemv_threshold = int(os.environ.get("CUDA_MIXED_GEMV_M_MAX", "128"))

    def __repr__(self) -> str:
        return (
            f"CudaMixedBitLinear(in={self.in_features}, out={self.out_features}, "
            f"bit={self.bit}, type={self.quant_type}, group_size={self.group_size})"
        )

    def _run_cuda_kernel(self, x2d: torch.Tensor) -> torch.Tensor:
        if self._ext is None:
            self._ext = load_mixed_precision_extension()

        if self.quant_type == "asym":
            return self._ext.gemv_wx_asym(
                x2d,
                self.qweight,
                self.scales,
                self.qzeros,
                int(self.group_size),
                int(self.bit),
            )
        if self.quant_type == "qtr_sym":
            return self._ext.gemv_w2_qtr(
                x2d,
                self.qweight,
                self.scales,
                int(self.group_size),
            )
        if self.quant_type == "lloyd_sym":
            return self._ext.gemv_w2_lloyd(
                x2d,
                self.qweight,
                self.scales,
                int(self.group_size),
            )
        raise ValueError(f"Unknown quant_type={self.quant_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            M = orig_shape[0] * orig_shape[1]
            K = orig_shape[2]
            x2d = x.reshape(M, K)
        else:
            x2d = x.reshape(-1, x.shape[-1])
            M, K = x2d.shape
        if K != self.in_features:
            raise ValueError(f"Input dim mismatch: got {K}, expected {self.in_features}")
        x2d = x2d.contiguous()
        if x2d.dtype != torch.float16:
            x2d = x2d.to(torch.float16)

        force_gemm = bool(int(os.environ.get("CUDA_MIXED_FORCE_GEMM", "0")))
        force_gemv = bool(int(os.environ.get("CUDA_MIXED_FORCE_GEMV", "0")))
        prefer_gemv = not force_gemm and (force_gemv or M <= self._gemv_threshold)

        if prefer_gemv:
            out = self._run_cuda_kernel(x2d)
        else:
            out = self._matmul_with_dequant(x2d)

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def _matmul_with_dequant(self, x2d: torch.Tensor) -> torch.Tensor:
        chunk = int(os.environ.get("CUDA_MIXED_DEQUANT_CHUNK", "512"))
        out = torch.empty(
            (x2d.size(0), self.out_features),
            device=x2d.device,
            dtype=torch.float32,
        )
        for start in range(0, self.out_features, chunk):
            end = min(start + chunk, self.out_features)
            w_chunk = self._dequant_chunk(start, end)
            w_t = w_chunk.transpose(0, 1).contiguous()
            tmp = torch.matmul(x2d, w_t)
            out[:, start:end] = tmp.to(out.dtype)
        return out

    def _dequant_chunk(self, start: int, end: int) -> torch.Tensor:
        if self.quant_type == "asym":
            return self._dequant_chunk_asym(start, end)
        if self.quant_type == "qtr_sym":
            return self._dequant_chunk_qtr(start, end)
        if self.quant_type == "lloyd_sym":
            return self._dequant_chunk_lloyd(start, end)
        raise ValueError(f"Unknown quant_type={self.quant_type}")

    def _dequant_chunk_asym(self, start: int, end: int) -> torch.Tensor:
        packed = self.qweight[start:end, :]
        vals = _unpack_nibbles(packed)[:, : self.in_features]
        vals = vals.to(torch.float16)
        zeros = self.qzeros_fp16[start:end, :]
        scales = self.scales[start:end, :].to(torch.float16)
        zeros_k = zeros.index_select(1, self.gid)
        scales_k = scales.index_select(1, self.gid)
        return (vals - zeros_k) * scales_k

    def _dequant_chunk_qtr(self, start: int, end: int) -> torch.Tensor:
        packed = self.qweight[start:end, :]
        codes = _unpack_2bit(packed)[:, : self.in_features]
        codes = codes.to(torch.int16)
        sign_bit = (codes >> 1) & 0x1
        mag_bit = codes & 0x1
        scales = self.scales[start:end, :].to(torch.float16)
        scales_k = scales.index_select(1, self.gid)
        mags = torch.where(mag_bit.bool(), scales_k, torch.zeros_like(scales_k))
        signs = torch.where(sign_bit.bool(), torch.full_like(mags, -1.0), torch.ones_like(mags))
        return (signs * mags).to(torch.float16)

    def _dequant_chunk_lloyd(self, start: int, end: int) -> torch.Tensor:
        packed = self.qweight[start:end, :]
        codes = _unpack_2bit(packed)[:, : self.in_features]
        codes = codes.to(torch.int16)
        sign_bit = (codes >> 1) & 0x1
        mag_bit = codes & 0x1
        scales = self.scales[start:end, :].to(torch.float16)  # shape [rows, groups, 2]
        alpha = scales[:, :, 0]
        beta = scales[:, :, 1]
        alpha_k = alpha.index_select(1, self.gid)
        beta_k = beta.index_select(1, self.gid)
        mags = torch.where(mag_bit.bool(), beta_k, alpha_k)
        signs = torch.where(sign_bit.bool(), torch.full_like(mags, -1.0), torch.ones_like(mags))
        return (signs * mags).to(torch.float16)
