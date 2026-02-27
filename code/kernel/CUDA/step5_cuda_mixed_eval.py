#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 5 (True Quant CUDA Mixed-Precision Version + Memory Check)
- ✨ [신규] 'main' 함수 시작 시 원본 FP16 모델을 로드하여 GPU 메모리 사용량 측정
- ✨ [신규] 퀀타이즈된 모델(CUDA + SVD) 로드 후 GPU 메모리 사용량 측정
- ✨ [비교] 두 모델의 'Allocated GiB' (실제 텐서 크기)를 비교하여 양자화 효과 확인
- ✨ [FIX 11/11 v5] clear_memory/log_memory_stats 함수가 device.index를 사용하도록 수정
- ✨ [FIX 11/11 v6] 첫 clear_memory 호출을 CUDA 초기화(모델 로드) 이후로 이동

사용 예시:
CUDA_VISIBLE_DEVICES=1 \
python CUDA/step5_cuda_mixed_eval.py \
    --model_name meta-llama/Llama-3.2-3B \
    --quant_path ./artifacts/bitmin/step4_budget_triton/quantized_model_triton.pt \
    --correction_path ./artifacts/bitmin/step4_budget_triton/correction_layerwise.pt \
    --bmap_path ./artifacts/bitmin/step4_budget_triton/b_ref_map_layerwise.json \
    --alpha_svd 1.0 \
    --device cuda:0 \
    --trust_remote_code \
    --use_cuda_kernels
"""
import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, math, gc, os, time, re, sys
from pprint import pformat
from contextlib import contextmanager
from typing import Optional, Tuple, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ----------------------------------
# [FIX v5] 메모리 측정 헬퍼 (device.index 사용)
# ----------------------------------
def _bytes_to_gib(value: int) -> float:
    return round(value / 1024**3, 3)


def clear_memory(device: torch.device):
    gc.collect()
    if torch.cuda.is_available() and device.type == "cuda":
        dev_idx = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev_idx)


def log_memory_stats(tag: str, device: torch.device):
    if not torch.cuda.is_available() or device.type != "cuda":
        print(f"[{tag}] CUDA not available or device is not CUDA.")
        return

    dev_idx = device.index if device.index is not None else torch.cuda.current_device()

    torch.cuda.synchronize(dev_idx)
    allocated = torch.cuda.memory_allocated(dev_idx)
    reserved = torch.cuda.memory_reserved(dev_idx)
    peak = torch.cuda.max_memory_allocated(dev_idx)

    print(
        f"\n--- Memory Stats ({tag}) ---\n"
        f"  Device   : {dev_idx}\n"
        f"  Allocated: {_bytes_to_gib(allocated):.3f} GiB  <-- (Sum of Real Tenser size)\n"
        f"  Reserved : {_bytes_to_gib(reserved):.3f} GiB\n"
        f"  Peak Alloc: {_bytes_to_gib(peak):.3f} GiB\n"
        f"-------------------------------\n"
    )


# ----------------------------------
# Triton 커널 (W4-Asym, W2-QTR, W2-Lloyd)
# ( ... 이전과 동일 ... )
# ----------------------------------
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton is not installed. Please install it with 'pip install triton'")
    sys.exit(1)


# --- Kernel 1: W4/W3 Asymmetric ---
@triton.jit
def triton_gemm_w4_asym_kernel(
    x_ptr,
    qweight_ptr,
    qzeros_ptr,
    scales_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_qwm,
    stride_qwk,
    stride_qzm,
    stride_qzk,
    stride_sm,
    stride_sk,
    stride_om,
    stride_on,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    qweight_ptrs = qweight_ptr + (
        offs_bn[None, :] * stride_qwm + (offs_k[:, None] // 2) * stride_qwk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_offs = k_start + offs_k

        x_mask = (offs_am[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)  # x is fp16

        q_mask = (k_offs[:, None] < K) & (offs_bn[None, :] < N)
        packed_weights = tl.load(qweight_ptrs, mask=q_mask, other=0)

        is_low_nibble = k_offs[:, None] % 2 == 0
        nibbles = tl.where(is_low_nibble, packed_weights & 0x0F, packed_weights >> 4)

        group_id = k_offs[:, None] // group_size
        scales_ptrs = scales_ptr + (offs_bn[None, :] * stride_sm + group_id * stride_sk)
        scales = tl.load(scales_ptrs, mask=q_mask, other=0.0)  # scales is fp16

        zeros_group_id = group_id // 2
        qzeros_ptrs = qzeros_ptr + (
            offs_bn[None, :] * stride_qzm + zeros_group_id * stride_qzk
        )
        packed_zeros = tl.load(qzeros_ptrs, mask=q_mask, other=0)

        is_low_zero_nibble = group_id % 2 == 0
        zeros = tl.where(is_low_zero_nibble, packed_zeros & 0x0F, packed_zeros >> 4)

        dequant_weights = (nibbles.to(tl.float32) - zeros.to(tl.float32)) * scales.to(
            tl.float32
        )

        accumulator += tl.dot(x.to(tl.float32), dequant_weights)  # FP32 @ FP32

        x_ptrs += BLOCK_SIZE_K * stride_xk
        qweight_ptrs += (BLOCK_SIZE_K // 2) * stride_qwk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        accumulator = accumulator + bias[None, :]

    c = accumulator.to(output_ptr.dtype.element_ty)  # Cast to output (FP32)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, c, mask=c_mask)


# --- Kernel 2: W2-QTR Symmetric ---
@triton.jit
def triton_gemm_w2_qtr_sym_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_qwm,
    stride_qwk,
    stride_sm,
    stride_sk,
    stride_om,
    stride_on,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    qweight_ptrs = qweight_ptr + (
        offs_bn[None, :] * stride_qwm + (offs_k[:, None] // 4) * stride_qwk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_offs = k_start + offs_k

        x_mask = (offs_am[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        q_mask = (k_offs[:, None] < K) & (offs_bn[None, :] < N)
        packed_weights = tl.load(qweight_ptrs, mask=q_mask, other=0)

        k_mod_4 = k_offs[:, None] % 4
        qvals = tl.where(
            k_mod_4 == 0,
            (packed_weights & 0x03),
            tl.where(
                k_mod_4 == 1,
                (packed_weights >> 2) & 0x03,
                tl.where(
                    k_mod_4 == 2,
                    (packed_weights >> 4) & 0x03,
                    (packed_weights >> 6) & 0x03,
                ),
            ),
        )

        sign_bit = (qvals >> 1) & 0x01
        mag_bit = qvals & 0x01

        signs = tl.where(sign_bit == 0, 1.0, -1.0)

        group_id = k_offs[:, None] // group_size
        scales_ptrs = scales_ptr + (offs_bn[None, :] * stride_sm + group_id * stride_sk)
        scales_alpha = tl.load(scales_ptrs, mask=q_mask, other=0.0)

        mags = tl.where(mag_bit == 0, 0.0, scales_alpha)

        dequant_weights = (signs * mags).to(tl.float32)

        accumulator += tl.dot(x.to(tl.float32), dequant_weights)  # FP32 @ FP32

        x_ptrs += BLOCK_SIZE_K * stride_xk
        qweight_ptrs += (BLOCK_SIZE_K // 4) * stride_qwk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        accumulator = accumulator + bias[None, :]

    c = accumulator.to(output_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, c, mask=c_mask)


# --- Kernel 3: W2-Lloyd Symmetric ---
@triton.jit
def triton_gemm_w2_lloyd_sym_kernel(
    x_ptr,
    qweight_ptr,
    scales_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_qwm,
    stride_qwk,
    stride_sm,
    stride_sk,
    stride_sl,  # scales stride for last dim
    stride_om,
    stride_on,
    group_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    qweight_ptrs = qweight_ptr + (
        offs_bn[None, :] * stride_qwm + (offs_k[:, None] // 4) * stride_qwk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        k_offs = k_start + offs_k

        x_mask = (offs_am[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        q_mask = (k_offs[:, None] < K) & (offs_bn[None, :] < N)
        packed_weights = tl.load(qweight_ptrs, mask=q_mask, other=0)

        k_mod_4 = k_offs[:, None] % 4
        qvals = tl.where(
            k_mod_4 == 0,
            (packed_weights & 0x03),
            tl.where(
                k_mod_4 == 1,
                (packed_weights >> 2) & 0x03,
                tl.where(
                    k_mod_4 == 2,
                    (packed_weights >> 4) & 0x03,
                    (packed_weights >> 6) & 0x03,
                ),
            ),
        )

        sign_bit = (qvals >> 1) & 0x01
        mag_bit = qvals & 0x01

        signs = tl.where(sign_bit == 0, 1.0, -1.0)

        group_id = k_offs[:, None] // group_size
        scales_base_ptrs = scales_ptr + (
            offs_bn[None, :] * stride_sm + group_id * stride_sk
        )
        scales_ptrs_alpha = scales_base_ptrs
        scales_ptrs_beta = scales_base_ptrs + stride_sl

        scales_alpha = tl.load(scales_ptrs_alpha, mask=q_mask, other=0.0)
        scales_beta = tl.load(scales_ptrs_beta, mask=q_mask, other=0.0)

        mags = tl.where(mag_bit == 0, scales_alpha, scales_beta)

        dequant_weights = (signs * mags).to(tl.float32)

        accumulator += tl.dot(x.to(tl.float32), dequant_weights)  # FP32 @ FP32

        x_ptrs += BLOCK_SIZE_K * stride_xk
        qweight_ptrs += (BLOCK_SIZE_K // 4) * stride_qwk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
        accumulator = accumulator + bias[None, :]

    c = accumulator.to(output_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_ptrs = (
        output_ptr + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, c, mask=c_mask)


# --- Triton Launcher Functions ---


def gemm_w4_asym(x, qweight, qzeros, scales, bias, group_size):
    original_shape = x.shape
    M, K = (
        (original_shape[0] * original_shape[1], original_shape[2])
        if x.dim() == 3
        else x.shape
    )
    N = scales.shape[0]
    x = x.reshape(M, K)

    output = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    triton_gemm_w4_asym_kernel[grid](
        x,
        qweight,
        qzeros,
        scales,
        bias,
        output,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        qzeros.stride(0),
        qzeros.stride(1),
        scales.stride(0),
        scales.stride(1),
        output.stride(0),
        output.stride(1),
        group_size=group_size,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        HAS_BIAS=(bias is not None),
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(*original_shape[:-1], N)


def gemm_w2_qtr_sym(x, qweight, scales, bias, group_size):
    original_shape = x.shape
    M, K = (
        (original_shape[0] * original_shape[1], original_shape[2])
        if x.dim() == 3
        else x.shape
    )
    N = scales.shape[0]
    x = x.reshape(M, K)

    output = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    triton_gemm_w2_qtr_sym_kernel[grid](
        x,
        qweight,
        scales,
        bias,
        output,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        output.stride(0),
        output.stride(1),
        group_size=group_size,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        HAS_BIAS=(bias is not None),
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(*original_shape[:-1], N)


def gemm_w2_lloyd_sym(x, qweight, scales, bias, group_size):
    original_shape = x.shape
    M, K = (
        (original_shape[0] * original_shape[1], original_shape[2])
        if x.dim() == 3
        else x.shape
    )
    N = scales.shape[0]
    x = x.reshape(M, K)

    output = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    triton_gemm_w2_lloyd_sym_kernel[grid](
        x,
        qweight,
        scales,
        bias,
        output,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        scales.stride(2),  # Pass all 3 strides
        output.stride(0),
        output.stride(1),
        group_size=group_size,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        HAS_BIAS=(bias is not None),
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(*original_shape[:-1], N)


# -----------------------------
# [NEW] Triton Mixed-Bit Module
# -----------------------------
class TritonMixedBitLinear(nn.Module):
    def __init__(self, in_features, out_features, entry: Dict, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bit = entry["bit"]
        self.quant_type = entry["quant_type"]
        self.group_size = entry["group_size"]

        self.register_buffer("qweight", entry["qweight_packed"])
        self.register_buffer("scales", entry["scales"])

        if self.quant_type == "asym":
            self.register_buffer("qzeros", entry["qzeros_packed"])

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x는 fp16
        if self.quant_type == "asym":
            # 런처는 fp32 반환
            return gemm_w4_asym(
                x, self.qweight, self.qzeros, self.scales, self.bias, self.group_size
            )
        elif self.quant_type == "qtr_sym":
            # 런처는 fp32 반환
            return gemm_w2_qtr_sym(
                x, self.qweight, self.scales, self.bias, self.group_size
            )
        elif self.quant_type == "lloyd_sym":
            # 런처는 fp32 반환
            return gemm_w2_lloyd_sym(
                x, self.qweight, self.scales, self.bias, self.group_size
            )
        else:
            raise ValueError(f"Unknown quant_type: {self.quant_type}")

    def __repr__(self):
        return f"TritonMixedBitLinear(in={self.in_features}, out={self.out_features}, bit={self.bit}, type={self.quant_type}, group_size={self.group_size})"


# -----------------------------
# Utils
# -----------------------------
def get_parent_module(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def infer_device_dtype(module: nn.Module):
    for t in list(module.buffers()):
        if t.device.type == "cuda":
            return t.device, torch.float16  # A,B는 fp16
    for t in list(module.parameters()):
        if t.device.type == "cuda":
            return t.device, torch.float16  # A,B는 fp16
    return (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
        torch.float16,
    )


# -----------------------------
# AB 래퍼
# -----------------------------
class AddABCorrection(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        A_q: torch.Tensor,
        B_q: torch.Tensor,
        alpha_svd: float = 1.0,
    ):
        super().__init__()
        self.inner = inner
        self.alpha_svd = alpha_svd
        dev, dt = infer_device_dtype(inner)
        self.register_buffer(
            "A_q", A_q.to(device=dev, dtype=dt, copy=True), persistent=False
        )
        self.register_buffer(
            "B_q", B_q.to(device=dev, dtype=dt, copy=True), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is fp16

        # 1. True Quant (W2/W3/W4) GEMM
        # inner(x)는 이제 FP32 텐서(z_fp32)를 반환
        z_fp32 = self.inner(x)

        # 2. SVD Correction
        if self.alpha_svd == 0.0:
            # Quant-Only path
            return z_fp32.to(x.dtype)  # 다음 레이어를 위해 fp16으로 캐스팅

        # Quant+AB path
        x_fp32 = x.to(torch.float32)
        A = self.A_q.to(torch.float32)
        B = self.B_q.to(torch.float32)

        rfeat = F.linear(x_fp32, B)
        corr = F.linear(rfeat, A)

        # FP32로 덧셈 수행 후, 원래 dtype (fp16)으로 복귀
        return (z_fp32 + (corr * self.alpha_svd)).to(x.dtype)


# -----------------------------
# AB 키 정규화 & 페어 수집
# ( ... 기존과 동일 ... )
# -----------------------------
def _normalize_key(k: str) -> str:
    if k.endswith(".weight.A"):
        return k[:-8] + ".A"
    if k.endswith(".weight.B"):
        return k[:-8] + ".B"
    return k


def collect_ab_pairs(
    correction_tensors: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    norm2orig = {}
    for k in correction_tensors.keys():
        norm2orig[_normalize_key(k)] = k
    bases = set()
    for nk in norm2orig.keys():
        if nk.endswith(".A") or nk.endswith(".B"):
            bases.add(nk[:-2])
    pairs = {}
    for base in bases:
        ak = base + ".A"
        bk = base + ".B"
        ak_orig = norm2orig.get(ak)
        bk_orig = norm2orig.get(bk)
        if ak_orig in correction_tensors and bk_orig in correction_tensors:
            pairs[base] = (correction_tensors[ak_orig], correction_tensors[bk_orig])
    return pairs


# -----------------------------
# AB 패치 (Triton 모듈을 래핑하도록 수정)
# ( ... 기존과 동일 ... )
# -----------------------------
def patch_ab_from_corrections(
    model: nn.Module,
    correction_tensors: Dict[str, torch.Tensor],
    bmap_keys_weight: set,
    alpha_svd: float = 1.0,
    allowed_inner_types: Optional[Tuple[type, ...]] = None,
) -> int:
    if allowed_inner_types is None:
        allowed_inner_types = (TritonMixedBitLinear,)

    pairs = collect_ab_pairs(correction_tensors)
    if not pairs:
        print("[AB] no (A,B) pairs found in correction file.")
        return 0

    target_modules = set(pairs.keys())
    if bmap_keys_weight:
        filter_modules = set(
            [k[:-7] for k in bmap_keys_weight if k.endswith(".weight")]
        )
        target_modules = target_modules & filter_modules
        print(
            f"[AB] filtering by bmap: {len(filter_modules)} keys → {len(target_modules)} targets"
        )

    patched, missing_module = 0, 0
    examples = []
    for base in sorted(target_modules):
        try:
            parent, attr = get_parent_module(model, base)
        except AttributeError:
            missing_module += 1
            continue

        inner = getattr(parent, attr, None)
        if inner is None:
            missing_module += 1
            continue

        if isinstance(inner, AddABCorrection):
            inner = inner.inner

        if not isinstance(inner, allowed_inner_types):
            missing_module += 1
            continue

        A, B = pairs[base]
        setattr(parent, attr, AddABCorrection(inner, A, B, alpha_svd=alpha_svd))
        patched += 1
        if len(examples) < 3:
            examples.append((base, tuple(A.shape), tuple(B.shape)))

    print(
        f"[AB] patched={patched}, missing_module={missing_module} (non-target layers skipped)"
    )
    if examples:
        print("[AB] examples:", examples)
    return patched


# -----------------------------
# PPL 평가
# ( ... 기존과 동일 ... )
# -----------------------------
@torch.no_grad()
def evaluate_ppl(
    model: nn.Module,
    tokenizer,
    device,
    tag="Eval",
    eval_seq_len=2048,
    dataset="wikitext",
    config="wikitext-2-raw-v1",
):
    print(f"\n--- PPL Eval: {tag} on {dataset}/{config} ---")
    model.eval()
    ds = load_dataset(dataset, config, split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)

    total_nll, total_tok = 0.0, 0
    for i in tqdm(range(0, input_ids.size(1), eval_seq_len), desc="Eval"):
        begin, end = i, min(i + eval_seq_len, input_ids.size(1))
        if end - begin <= 1:
            continue
        x = input_ids[:, begin:end]
        y = x
        with torch.no_grad():
            out = model(x)
        logits = out.logits

        shift_logits = logits[..., :-1, :].contiguous().to(torch.float32)
        shift_labels = y[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tok += shift_labels.numel()

    if total_tok == 0 or total_nll == 0 or not math.isfinite(total_nll):
        ppl = float("nan")
    else:
        ppl = math.exp(total_nll / total_tok)

    print(f"PPL({tag}) = {ppl:.4f}")
    return ppl


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        "Step 5 — True Quant Triton Mixed-Precision PPL Evaluation"
    )
    ap.add_argument("--model_name", required=True)
    ap.add_argument(
        "--quant_path",
        required=True,
        help="Path to 'quantized_model_triton.pt' from step4",
    )
    ap.add_argument(
        "--correction_path",
        required=True,
        help="Path to 'correction_layerwise.pt' (A,B matrices)",
    )
    ap.add_argument(
        "--bmap_path", default=None, help="optional; used only as a filter by keys"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--alpha_svd", type=float, default=1.0)
    ap.add_argument(
        "--tf32", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True
    )
    ap.add_argument("--eval_seq_len", type=int, default=2048)
    ap.add_argument("--eval_dataset", type=str, default="wikitext")
    ap.add_argument("--eval_config", type=str, default="wikitext-2-raw-v1")
    ap.add_argument(
        "--use_cuda_kernels",
        action="store_true",
        help="Use the custom CUDA mixed-bit kernels instead of Triton.",
    )
    ap.add_argument(
        "--cuda_kernel_verbose",
        action="store_true",
        help="Print verbose logs while building the CUDA extension.",
    )
    args = ap.parse_args()

    if args.cuda_kernel_verbose:
        os.environ["CUDA_MIXED_VERBOSE"] = "1"

    if not HAS_TRITON and not args.use_cuda_kernels:
        print(
            "Triton is required to run this script (install or use --use_cuda_kernels)."
        )
        return

    if args.tf32:
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)

    module_cls = TritonMixedBitLinear
    backend_label = "Triton"
    if args.use_cuda_kernels:
        try:
            from cuda_mixed.linear import CudaMixedBitLinear
        except Exception as exc:
            raise RuntimeError(
                "Failed to import CUDA mixed-bit backend. Ensure the module exists under cuda_mixed/."
            ) from exc
        module_cls = CudaMixedBitLinear
        backend_label = "CUDA"
        print("⚙️  Using custom CUDA mixed-bit kernels.")
    else:
        print("⚙️  Using Triton mixed-bit kernels.")
    quant_label = f"{backend_label} W2/W3/W4"

    # --- [NEW] Part 1: Measure FP16 Model ---
    print(f"\n[Memory Check] Loading ORIGINAL FP16 model to {device} for baseline...")
    model_fp16 = None
    try:
        # [FIX v6] CUDA가 초기화되기 전에 clear_memory를 호출하지 않도록 이동
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=args.trust_remote_code,
            device_map=device,  # Load directly to GPU
        )
        # [FIX v6] 모델 로드(CUDA 초기화) 이후에 메모리 클리어
        clear_memory(device)
        log_memory_stats("1. Baseline FP16 Model", device)
    except Exception as e:
        print(f"Failed to load FP16 model for memory check: {e}")
    finally:
        if model_fp16 is not None:
            del model_fp16
        clear_memory(device)
        print("[Memory Check] FP16 model cleared.\n")

    # --- [EXISTING] Part 2: Load and Measure Quantized Model ---

    # 1) 모델 로드 (CPU, fp16 구조만)
    print(f"[Load] Loading model structure: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10**9

    # 2) True Quant 파라미터 로드 및 주입
    print(f"[Load] Loading True Quant params from: {args.quant_path}")
    quant_data = torch.load(args.quant_path, map_location="cpu")

    module_name = module_cls.__name__
    print(f"🔄 Injecting {module_name} modules ({backend_label} backend)...")
    target_layers = set(quant_data.keys())
    injected_count = 0

    original_state_dict = model.state_dict()

    for name, module in model.named_modules():
        weight_name = f"{name}.weight"

        if weight_name in target_layers:
            entry = quant_data[weight_name]
            O, I = entry["shape"]

            bias_key = f"{name}.bias"
            has_bias = (
                bias_key in original_state_dict
                and original_state_dict[bias_key] is not None
            )

            new_module = module_cls(
                in_features=I, out_features=O, entry=entry, bias=has_bias
            )

            if has_bias:
                new_module.bias.data.copy_(original_state_dict[bias_key])

            parent, attr_name = get_parent_module(model, name)
            setattr(parent, attr_name, new_module)
            injected_count += 1

    print(f"✅ Injected {injected_count} {module_name} modules.")

    del original_state_dict
    gc.collect()

    # 3) 모델을 GPU로 이동
    print(f"Moving model to device: {device}")
    clear_memory(device)  # GPU로 보내기 전 메모리 초기화
    model = model.to(device)
    log_memory_stats("2a. Quantized Model (W_q only)", device)

    # 4) AB 로드 및 패치
    print(f"[Load] Loading SVD corrections: {args.correction_path}")
    correction = torch.load(args.correction_path, map_location=device)

    bmap_keys = set()
    if args.bmap_path and os.path.exists(args.bmap_path):
        with open(args.bmap_path, "r") as f:
            bmap = json.load(f)
        bmap_keys = set(bmap.keys())
        print(f"[bmap] keys loaded: {len(bmap_keys)}")

    allowed_types = (module_cls,)
    patched = patch_ab_from_corrections(
        model,
        correction,
        bmap_keys,
        alpha_svd=args.alpha_svd,
        allowed_inner_types=allowed_types,
    )
    if patched == 0 and len(correction) > 0:
        print("[AB][warn] patched=0. Retrying WITHOUT bmap filtering...")
        patched = patch_ab_from_corrections(
            model,
            correction,
            set(),
            alpha_svd=args.alpha_svd,
            allowed_inner_types=allowed_types,
        )

    del correction
    gc.collect()
    torch.cuda.empty_cache()

    log_memory_stats("2b. Quantized Model + SVD (A,B)", device)

    # 5) 평가
    # 5-1) Quant-only (SVD 끔)
    print(f"\n--- Evaluating Quant-Only ({quant_label}) ---")
    for m in model.modules():
        if isinstance(m, AddABCorrection):
            m.alpha_svd = 0.0

    ppl_q = evaluate_ppl(
        model,
        tokenizer,
        device,
        tag="Quant-only (α=0)",
        eval_seq_len=args.eval_seq_len,
        dataset=args.eval_dataset,
        config=args.eval_config,
    )

    # 5-2) Quant+AB (SVD 켬)
    print(f"\n--- Evaluating Quant + SVD ({quant_label} + A,B) ---")
    for m in model.modules():
        if isinstance(m, AddABCorrection):
            m.alpha_svd = args.alpha_svd

    ppl_ab = evaluate_ppl(
        model,
        tokenizer,
        device,
        tag=f"Quant+AB (α={args.alpha_svd})",
        eval_seq_len=args.eval_seq_len,
        dataset=args.eval_dataset,
        config=args.eval_config,
    )

    print("\n===== SUMMARY =====")
    print(f"Model          : {args.model_name}")
    print(f"Triton Params  : {args.quant_path}")
    print(f"SVD Params     : {args.correction_path}")
    print("-" * 20)
    print(f"Quant-only PPL : {ppl_q:.4f} ({quant_label}, α=0)")
    print(f"Quant+AB  PPL  : {ppl_ab:.4f} ({quant_label} + AB, α={args.alpha_svd})")


if __name__ == "__main__":
    main()
