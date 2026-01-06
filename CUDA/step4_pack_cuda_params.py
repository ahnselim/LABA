#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4 (CUDA Parameter Packing Version)
- [기존] 2/3/4bit Fake Quant (fp16) Wq를 계산하여 SVD용 오차(E) 계산 및 (A,B) 복원
- [신규] bit_assign.csv에 따라 2/3/4bit True Quant 파라미터를 패킹하여 별도 저장
- 2bit: Lloyd/QTR 대칭 (step4 원본 로직 유지)
- 3/4bit: Asymmetric (step1/3 CUDA 로직과 호환)

입력:
 • --model_id
 • --bit_assign_csv : Step3 결과 CSV (layer_name, R_int)
 • --reuse_calib 등 (기존과 동일)

출력:
 • out_dir/quantized_model_triton.pt  # [신규] CUDA 커널용 (W_packed, scales, zeros, quant_type)
 • out_dir/correction_layerwise.pt    # [기존] SVD 복원용 (A, B)
 • out_dir/quantized_weights.pt       # [기존] Fake Quant Wq (SVD 계산 검증용)
 • out_dir/b_ref_map_layerwise.json
 
CUDA_VISIBLE_DEVICES=1 python CUDA/step4_pack_cuda_params.py \
    --model_id meta-llama/Llama-3.2-3B \
    --bit_assign_csv ./artifacts/bitmin/step3b_budget/bit_assign.csv \
    --out_dir ./artifacts/bitmin/step4_budget_triton \
    --rank 64 \
    --group_size 128 \
    --dataset DKYoon/SlimPajama-6B \
    --nsamples 64 \
    --seqlen 2048 \
    --reuse_calib \
    --calib_cache_dir ./artifacts/bitmin \
    --trust_remote_code
"""

import os, re, gc, csv, json, argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

# Disable flash-attn auto-import; the bundled wheel often mismatches local CUDA.
if "FLASH_ATTENTION_FORCE_DISABLE" not in os.environ:
    os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import import_utils as _tf_import_utils
except Exception as e:
    raise RuntimeError(
        "transformers가 필요합니다: pip install transformers datasets accelerate"
    ) from e


def _disable_flash_attn_package():
    sentinel = "_flash_attn_guard_installed"
    if getattr(_tf_import_utils, sentinel, False):
        return
    orig = _tf_import_utils._is_package_available

    def _patched(pkg: str):
        if pkg == "flash_attn":
            return False
        return orig(pkg)

    _tf_import_utils._is_package_available = _patched
    setattr(_tf_import_utils, sentinel, True)


if os.environ.get("FLASH_ATTENTION_FORCE_DISABLE", "1") == "1":
    _disable_flash_attn_package()

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

# -------------------------------
# 대상 Linear 필터 (LLaMA/Mistral 호환)
# -------------------------------

TARGET_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
}


def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )


def module_name_from_weight(full_weight_name: str) -> str:
    return full_weight_name[: -len(".weight")]


def extract_layer_index(name: str) -> Optional[str]:
    m = re.search(r"layers\.(\d+)\.", name)
    return m.group(1) if m else None


# -------------------------------
# SlimPajama-friendly dataset loader (streaming 우선)
# -------------------------------

import re as _re


def _canonical_dataset_name(name: str) -> str:
    a = name.strip()
    low = a.lower()
    if low in {"dkyoon/slimpajama-6b", "slimpajama-6b", "dkyoon_slimpajama_6b"}:
        return "DKYoon/SlimPajama-6B"
    if low in {"slimpajama-627b", "cerebras/slimpajama-627b", "slimpajama627b"}:
        return "cerebras/SlimPajama-627B"
    return name


def open_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str = "train",
    streaming: bool = True,
):
    if not HAS_DATASETS:
        raise RuntimeError("datasets 라이브러리가 필요합니다: pip install datasets")
    dataset_name = _canonical_dataset_name(dataset_name)
    if streaming:
        try:
            ds = load_dataset(
                dataset_name, name=dataset_config, split=split, streaming=True
            )
            return ds, dataset_name, dataset_config, True
        except Exception as e:
            msg = str(e)
            if ("available configs" in msg) or ("Config name is missing" in msg):
                m = _re.search(r"\[(.*?)\]", msg, flags=_re.S)
                if m:
                    cands = [
                        c.strip().strip("'\"")
                        for c in m.group(1).split(",")
                        if c.strip()
                    ]
                    for cand in cands:
                        try:
                            ds = load_dataset(
                                dataset_name, name=cand, split=split, streaming=True
                            )
                            return ds, dataset_name, cand, True
                        except Exception:
                            pass
    # non-streaming
    try:
        ds = load_dataset(
            dataset_name, name=dataset_config, split=split, streaming=False
        )
        return ds, dataset_name, dataset_config, False
    except Exception as e:
        msg = str(e)
        if ("available configs" in msg) or ("Config name is missing" in msg):
            m = _re.search(r"\[(.*?)\]", msg, flags=_re.S)
            if m:
                cands = [
                    c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()
                ]
                for cand in cands:
                    try:
                        ds = load_dataset(
                            dataset_name, name=cand, split=split, streaming=False
                        )
                        return ds, dataset_name, cand, False
                    except Exception:
                        pass
        raise


def build_calibration_tokens(
    tokenizer,
    nsamples=64,
    seqlen=2048,
    dataset_name="DKYoon/SlimPajama-6B",
    dataset_config=None,
    split="train",
    use_streaming=True,
):
    ds, dataset_name, dataset_config, is_streaming = open_hf_dataset(
        dataset_name, dataset_config, split=split, streaming=use_streaming
    )
    print(
        f"[Step4] Using calibration dataset={dataset_name}, config={dataset_config}, streaming={is_streaming}"
    )
    sample_budget = max(nsamples * 5, nsamples)
    iterator = (
        ds.take(sample_budget)
        if hasattr(ds, "take")
        else ds.select(range(min(sample_budget, len(ds))))
    )
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_id is None and getattr(tokenizer, "eos_token", None):
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    buf, samples = [], []
    for row in iterator:
        text = None
        for key in ("text", "content", "raw_content"):
            if key in row and isinstance(row[key], str) and row[key].strip():
                text = row[key]
                break
        if text is None:
            for v in row.values():
                if isinstance(v, str) and v.strip():
                    text = v
                    break
        if not text:
            continue
        ids = (
            tokenizer(text, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .tolist()
        )
        if not ids:
            continue
        if eos_id is not None:
            ids.append(eos_id)
        buf.extend(ids)
        while len(buf) >= seqlen and len(samples) < nsamples:
            samples.append(torch.tensor(buf[:seqlen], dtype=torch.long))
            buf = buf[seqlen:]
            if len(samples) >= nsamples:
                break
        if len(samples) >= nsamples:
            break
    if len(samples) < nsamples:
        print(f"[Step4][warn] Collected only {len(samples)}/{nsamples} sequences.")
    return (
        torch.stack(samples, dim=0)
        if samples
        else torch.empty(0, seqlen, dtype=torch.long)
    )


# -------------------------------
# Σ_x 대각 OAS (모듈별)
# -------------------------------
@torch.no_grad()
def estimate_diag_cov_oas_per_module(
    model: nn.Module,
    tokenizer,
    device,
    nsamples=64,
    seqlen=2048,
    calib_dataset="DKYoon/SlimPajama-6B",
    calib_config=None,
    split="train",
    use_streaming=True,
    matmul_dtype=torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    model.eval()
    name_to_dim: Dict[str, int] = {}
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    def hook_factory(mod_name: str):
        def hook(module, inp, _out):
            x = (
                inp[0]
                .detach()
                .reshape(-1, inp[0].shape[-1])
                .to(device=device, dtype=matmul_dtype)
            )
            d = x.shape[-1]
            name_to_dim[mod_name] = d
            if mod_name not in stats:
                stats[mod_name] = {
                    "sum": torch.zeros(d, dtype=torch.float64, device="cpu"),
                    "sumsq": torch.zeros(d, dtype=torch.float64, device="cpu"),
                    "n": torch.tensor(0, dtype=torch.long),
                }
            x_cpu = x.to("cpu", dtype=torch.float32)
            stats[mod_name]["sum"] += x_cpu.sum(dim=0, dtype=torch.float64)
            stats[mod_name]["sumsq"] += (x_cpu.to(torch.float64).pow(2)).sum(dim=0)
            stats[mod_name]["n"] += x_cpu.shape[0]

        return hook

    for mn, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(hook_factory(mn)))

    tokens = build_calibration_tokens(
        tokenizer, nsamples, seqlen, calib_dataset, calib_config, split, use_streaming
    )
    if tokens.numel() == 0:
        for h in handles:
            h.remove()
        raise RuntimeError("Calibration tokens unavailable.")

    with torch.no_grad():
        for i in tqdm(range(tokens.shape[0]), desc="Calibration Forward"):
            model(tokens[i : i + 1].to(device))

    for h in handles:
        h.remove()

    ops: Dict[str, Dict[str, torch.Tensor]] = {}
    for mn, st in stats.items():
        n = int(st["n"].item())
        if n <= 1:
            continue
        d = name_to_dim[mn]
        sumv, sumsq = st["sum"], st["sumsq"]
        mean = sumv / n
        ex2 = sumsq / n
        var = torch.clamp(ex2 - mean.pow(2), min=1e-12)
        p = float(d)
        trS = var.sum().item()
        trS2 = (var.pow(2)).sum().item()
        num = (1.0 - 2.0 / p) * trS2 + (trS * trS)
        den = (n + 1.0 - 2.0 / p) * (trS2 - (trS * trS) / p)
        alpha = 1.0 if den <= 0 else max(0.0, min(1.0, num / den))
        mu = trS / p
        sigma_diag = (1.0 - alpha) * var + alpha * mu
        sqrt_diag = torch.sqrt(torch.clamp(sigma_diag, min=1e-12)).to(torch.float32)
        inv_sqrt_diag = (1.0 / torch.clamp(sqrt_diag, min=1e-12)).to(torch.float32)
        ops[mn] = {"sqrt": sqrt_diag.cpu(), "inv_sqrt": inv_sqrt_diag.cpu()}
    print(f"[Step4] Σ_x^{1/2} prepared for {len(ops)} linear modules.")
    return ops


# -------------------------------
# Group reshape helpers
# -------------------------------


def _to_groups(W: torch.Tensor, group_size: int):
    O, I = W.shape
    pad = (group_size - (I % group_size)) % group_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    O_, I_pad = W.shape
    G = I_pad // group_size
    return W.view(O_, G, group_size), O_, G, group_size, I


def _from_groups(Xg: torch.Tensor, orig_I: int) -> torch.Tensor:
    O_, G, S = Xg.shape
    return Xg.reshape(O_, G * S)[:, :orig_I]


@torch.no_grad()
def _percentile_clip_lastdim(
    X: torch.Tensor, lo_pct: float, hi_pct: float
) -> torch.Tensor:
    lo = torch.quantile(X, lo_pct / 100.0, dim=-1, keepdim=True)
    hi = torch.quantile(X, hi_pct / 100.0, dim=-1, keepdim=True)
    return torch.clamp(X, lo, hi)


# -------------------------------
# Quantization kernels (Fake Quant - SVD 계산용)
# -------------------------------
@torch.no_grad()
def dequant_2bit_lloyd2_sym(
    W: torch.Tensor, group_size: int, clip_pct: float = 99.9, steps: int = 4
):
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
    Y = X.abs()
    eps = 1e-12
    alpha = torch.quantile(Y, 0.5, dim=-1, keepdim=True).clamp_min(eps)
    beta = torch.quantile(Y, 0.9, dim=-1, keepdim=True)
    beta = torch.maximum(beta, alpha + 1e-6)
    for _ in range(max(1, steps)):
        t = 0.5 * (alpha + beta)
        mask_lo = Y < t
        cnt_lo = mask_lo.sum(dim=-1, keepdim=True).clamp_min(1)
        cnt_hi = (~mask_lo).sum(dim=-1, keepdim=True).clamp_min(1)
        new_alpha = (Y * mask_lo).sum(dim=-1, keepdim=True) / cnt_lo
        new_beta = (Y * (~mask_lo)).sum(dim=-1, keepdim=True) / cnt_hi
        alpha = torch.where(new_alpha > eps, new_alpha, alpha)
        beta = torch.where(new_beta > alpha + 1e-6, new_beta, beta)
    t = 0.5 * (alpha + beta)
    mag = torch.where(Y < t, alpha, beta)
    sgn = torch.sign(X)
    sgn[sgn == 0] = 1.0
    deq = sgn * mag
    return _from_groups(deq, orig_I).to(dtype=W.dtype, device=W.device)


@torch.no_grad()
def dequant_2bit_qtr_zero(
    W: torch.Tensor, group_size: int, clip_pct: float = 99.9, steps: int = 6
):
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
    Y = X.abs()
    eps = 1e-12
    alpha = Y.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    for _ in range(max(1, steps)):
        t = 0.5 * alpha
        mask = Y >= t
        cnt = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        new_alpha = (Y * mask).sum(dim=-1, keepdim=True) / cnt
        alpha = torch.where(new_alpha > eps, new_alpha, alpha)
    mask = (Y >= 0.5 * alpha).float()
    mag = alpha * mask
    sgn = torch.sign(X)
    sgn[sgn == 0] = 1.0
    deq = sgn * mag
    return _from_groups(deq, orig_I).to(dtype=W.dtype, device=W.device)


@torch.no_grad()
def dequant_uniform_asym(
    W: torch.Tensor,
    b: int,
    group_size: int,
    clip_pct: float = 0.0,
    rounding: str = "nearest",
):
    assert b in (3, 4)
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
    qmax = float(2**b - 1)
    eps = 1e-12
    minv = X.amin(dim=-1)
    maxv = X.amax(dim=-1)
    span = (maxv - minv).clamp_min(eps)
    scale = span / qmax
    zp = (-minv / scale).round()
    qf = X / scale.unsqueeze(-1) + zp.unsqueeze(-1)
    if rounding == "stochastic":
        frac = qf - torch.floor(qf)
        q = torch.floor(qf) + (torch.rand_like(frac) < frac).float()
    else:
        q = torch.round(qf)
    q = torch.clamp(q, 0.0, qmax)
    deq = (q - zp.unsqueeze(-1)) * scale.unsqueeze(-1)
    flat = span <= eps
    if flat.any():
        deq[flat] = minv[flat].unsqueeze(-1)
    return _from_groups(deq, orig_I).to(dtype=W.dtype, device=W.device)


# -------------------------------
# [NEW] True Quantization Packing Kernels (Triton용)
# -------------------------------


@torch.no_grad()
def pack_weights(
    q_values_unpacked: torch.Tensor, bit: int, group_size: int
) -> torch.Tensor:
    """
    비트별로 양자화된 정수(unpacked)를 uint8 텐서(packed)로 압축합니다.
    q_values_unpacked: (O, G, S) 형태, dtype=torch.uint8
    """
    O, G, S = q_values_unpacked.shape
    assert S == group_size, f"Unpacked shape mismatch. Expected S={group_size}, got {S}"

    if bit == 4:
        # (O, G, S) -> (O, G, S/2)
        low_w = q_values_unpacked[..., 0::2]
        high_w = q_values_unpacked[..., 1::2]
        packed = ((high_w << 4) | low_w).reshape(O, G * S // 2)

    elif bit == 2:
        # (O, G, S) -> (O, G, S/4)
        v0 = q_values_unpacked[..., 0::4]
        v1 = q_values_unpacked[..., 1::4]
        v2 = q_values_unpacked[..., 2::4]
        v3 = q_values_unpacked[..., 3::4]
        packed = ((v3 << 6) | (v2 << 4) | (v1 << 2) | v0).reshape(O, G * S // 4)

    elif bit == 3:
        # 3비트는 4비트 컨테이너에 0-7 값으로 저장 (Triton 커널에서 0-7로 디양자화)
        # (O, G, S) -> (O, G, S/2)
        low_w = q_values_unpacked[..., 0::2]
        high_w = q_values_unpacked[..., 1::2]
        packed = ((high_w << 4) | low_w).reshape(O, G * S // 2)

    else:
        raise ValueError(f"Unsupported bitwidth for packing: {bit}")

    return packed.to(torch.uint8)


@torch.no_grad()
def pack_zeros(zeros_unpacked: torch.Tensor, bit: int) -> torch.Tensor:
    """
    비대칭 Zero-point(unpacked)를 uint8 텐서(packed)로 압축합니다.
    zeros_unpacked: (O, G) 형태, dtype=torch.uint8
    """
    O, G = zeros_unpacked.shape

    if bit == 4:
        # (O, G) -> (O, G/2)
        # 패딩이 G % 2 != 0 일 경우를 대비
        if G % 2 != 0:
            zeros_unpacked = torch.nn.functional.pad(
                zeros_unpacked, (0, 1), "constant", 0
            )
            G += 1
        low_z = zeros_unpacked[..., 0::2]
        high_z = zeros_unpacked[..., 1::2]
        packed = (high_z << 4) | low_z

    elif bit == 2:
        # (O, G) -> (O, G/4)
        pad = (4 - G % 4) % 4
        if pad != 0:
            zeros_unpacked = torch.nn.functional.pad(
                zeros_unpacked, (0, pad), "constant", 0
            )
            G += pad
        z0 = zeros_unpacked[..., 0::4]
        z1 = zeros_unpacked[..., 1::4]
        z2 = zeros_unpacked[..., 2::4]
        z3 = zeros_unpacked[..., 3::4]
        packed = (z3 << 6) | (z2 << 4) | (z1 << 2) | z0

    elif bit == 3:
        # 3비트는 4비트 컨테이너 사용
        if G % 2 != 0:
            zeros_unpacked = torch.nn.functional.pad(
                zeros_unpacked, (0, 1), "constant", 0
            )
            G += 1
        low_z = zeros_unpacked[..., 0::2]
        high_z = zeros_unpacked[..., 1::2]
        packed = (high_z << 4) | low_z

    else:
        raise ValueError(f"Unsupported bitwidth for packing zeros: {bit}")

    return packed.to(torch.uint8)


@torch.no_grad()
def get_params_uniform_asym(
    W: torch.Tensor, b: int, group_size: int, rounding: str = "nearest"
):
    """
    3/4비트 비대칭 양자화 파라미터(q_packed, z_packed, scales)를 계산합니다.
    """
    assert b in (3, 4), "Asymmetric packing is for 3/4 bits"
    W32 = W.to(torch.float32)
    Wg, O, G, S, orig_I = _to_groups(W32, group_size)

    qmax = float(2**b - 1)
    eps = 1e-12
    minv = Wg.amin(dim=-1)
    maxv = Wg.amax(dim=-1)
    span = (maxv - minv).clamp_min(eps)
    scales = span / qmax
    # 스케일이 0인 경우 zp도 0이 되도록 (NaN 방지)
    zeros_float = (
        torch.where(scales > eps, (-minv / scales), 0.0).round().clamp(0.0, qmax)
    )

    scales = scales.clamp_min(eps)  # 0 스케일 방지

    qf = Wg / scales.unsqueeze(-1) + zeros_float.unsqueeze(-1)

    if rounding == "stochastic":
        frac = qf - torch.floor(qf)
        q_values = torch.floor(qf) + (torch.rand_like(frac) < frac).float()
    else:
        q_values = torch.round(qf)

    q_values = torch.clamp(q_values, 0.0, qmax).to(torch.uint8)
    zeros_uint8 = zeros_float.to(torch.uint8)

    # 패킹 수행
    qweight_packed = pack_weights(q_values, b, group_size)
    qzeros_packed = pack_zeros(zeros_uint8, b)

    # 원본 In-feature 크기로 qweight을 잘라냄 (Padding 제거)
    if orig_I % group_size != 0:
        bits_per_byte = 8 // b
        if b == 3:
            bits_per_byte = 2  # 3bit는 4bit 컨테이너(2개) 사용

        packed_I = (orig_I + bits_per_byte - 1) // bits_per_byte
        qweight_packed = qweight_packed[:, :packed_I].contiguous()

    return qweight_packed, qzeros_packed, scales.to(torch.float16)


@torch.no_grad()
def get_params_2bit_lloyd_sym(
    W: torch.Tensor, group_size: int, clip_pct: float, steps: int
):
    """2비트 Lloyd 파라미터(q_packed, scales)를 계산합니다."""
    W32 = W.to(torch.float32)
    Wg, O, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
    Y = X.abs()
    eps = 1e-12

    alpha = torch.quantile(Y, 0.5, dim=-1, keepdim=True).clamp_min(eps)
    beta = torch.quantile(Y, 0.9, dim=-1, keepdim=True)
    beta = torch.maximum(beta, alpha + 1e-6)
    for _ in range(max(1, steps)):
        t = 0.5 * (alpha + beta)
        mask_lo = Y < t
        cnt_lo = mask_lo.sum(dim=-1, keepdim=True).clamp_min(1)
        cnt_hi = (~mask_lo).sum(dim=-1, keepdim=True).clamp_min(1)
        new_alpha = (Y * mask_lo).sum(dim=-1, keepdim=True) / cnt_lo
        new_beta = (Y * (~mask_lo)).sum(dim=-1, keepdim=True) / cnt_hi
        alpha = torch.where(new_alpha > eps, new_alpha, alpha)
        beta = torch.where(new_beta > alpha + 1e-6, new_beta, beta)

    t = 0.5 * (alpha + beta)
    mag_indices = (Y >= t).to(torch.uint8)  # 1 if magnitude is beta, 0 if alpha
    sgn_indices = (X < 0).to(torch.uint8)  # 1 if negative, 0 if positive

    q_2bit = (sgn_indices << 1) | mag_indices  # (sign, mag) -> 2-bit value

    qweight_packed = pack_weights(q_2bit, 2, group_size)
    scales = (
        torch.stack([alpha, beta], dim=-1).squeeze(-2).to(torch.float16)
    )  # Shape [O, G, 2]

    if orig_I % group_size != 0:
        packed_I = (orig_I + 3) // 4
        qweight_packed = qweight_packed[:, :packed_I].contiguous()

    return qweight_packed, scales


@torch.no_grad()
def get_params_2bit_qtr_zero(
    W: torch.Tensor, group_size: int, clip_pct: float, steps: int
):
    """2비트 QTR-Zero 파라미터(q_packed, scales)를 계산합니다."""
    W32 = W.to(torch.float32)
    Wg, O, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
    Y = X.abs()
    eps = 1e-12

    alpha = Y.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    for _ in range(max(1, steps)):
        t = 0.5 * alpha
        mask_hi = Y >= t
        cnt_hi = mask_hi.sum(dim=-1, keepdim=True).clamp_min(1)
        new_alpha = (Y * mask_hi).sum(dim=-1, keepdim=True) / cnt_hi
        alpha = torch.where(new_alpha > eps, new_alpha, alpha)

    mag_indices = (Y >= 0.5 * alpha).to(
        torch.uint8
    )  # 1 if magnitude is alpha, 0 if zero
    sgn_indices = (X < 0).to(torch.uint8)  # 1 if negative, 0 if positive

    q_2bit = (sgn_indices << 1) | mag_indices  # (sign, mag) -> 2-bit value

    qweight_packed = pack_weights(q_2bit, 2, group_size)
    scales = alpha.squeeze(-1).to(torch.float16)  # Shape [O, G]

    if orig_I % group_size != 0:
        packed_I = (orig_I + 3) // 4
        qweight_packed = qweight_packed[:, :packed_I].contiguous()

    return qweight_packed, scales


# -------------------------------
# 선택 비트 로드 (Step3 산출물)
# -------------------------------
def load_selected_bits(csv_path: str) -> Dict[str, int]:
    sel: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = (
                row.get("layer_name") or row.get("module") or row.get("name") or ""
            ).strip()
            if not name:
                continue
            b = None
            for key in ("R_int", "selected_bit", "bit"):
                if key in row and str(row[key]).strip() != "":
                    try:
                        b = int(float(row[key]))
                        break
                    except Exception:
                        pass
            if b is None:
                continue
            b = max(2, min(4, b))  # 2~4 클램프
            sel[name] = b
    return sel


# -------------------------------
# 유틸: 파일 이름 정리
# -------------------------------
def _safe_name(s: str) -> str:
    s = str(s) if s is not None else "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


# -------------------------------
# 메인
# -------------------------------
def main():
    ap = argparse.ArgumentParser(
        "Step 4 — Apply Quantization, Restore (A,B), and Pack Triton Params"
    )

    # 모델/장치
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
    )
    ap.add_argument("--device", default="cuda", help='e.g., "cuda", "cpu", "cuda:1"')
    # 비트 할당
    ap.add_argument(
        "--bit_assign_csv",
        required=True,
        help="Step3 bit_assign.csv (layer_name,R_int)",
    )
    ap.add_argument("--group_size", type=int, default=128)
    # 복원(AB)
    ap.add_argument("--rank", type=int, default=64)
    # 캘리브레이션(Σ_x)
    ap.add_argument("--dataset", type=str, default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in ["1", "true", "yes"],
        default=True,
    )
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    # Calibration 캐시 (Step2와 동일 규칙)
    ap.add_argument(
        "--reuse_calib", action="store_true", help="기존 Σ_x^{1/2} 캐시(.pt) 재사용"
    )
    ap.add_argument("--calib_cache_dir", type=str, default="./artifacts/bitmin")
    # 2/3/4bit 세부
    ap.add_argument(
        "--clip_percentile", type=float, default=99.9, help="2bit 권장 99.9; 0이면 off"
    )
    ap.add_argument("--lloyd_steps", type=int, default=4)
    ap.add_argument("--qtr_steps", type=int, default=6)
    ap.add_argument(
        "--uniform_rounding",
        type=str,
        default="nearest",
        choices=["nearest", "stochastic"],
    )
    # 출력
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--save_patched_model",
        action="store_true",
        help="W_eff = Wq + A@B 를 fold하여 전체 state_dict를 저장",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.dtype == "bf16":
        load_dtype = torch.bfloat16
    elif args.dtype in ("fp16", "float16"):
        load_dtype = torch.float16
    elif args.dtype in ("fp32", "float32"):
        load_dtype = torch.float32
    else:  # auto
        load_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # 모델/토크나이저
    print(f"[Step4] Loading model: {args.model_id} ({args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(
            load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None
        ),
        trust_remote_code=args.trust_remote_code,
        device_map=None,  # 한 디바이스에 올림 (hook 수집/forward 편의)
    ).to(device)

    # Σ_x^{±1/2} (diag OAS) — 캐시 로드/생성
    safe_model = _safe_name(
        args.model_id if args.revision is None else f"{args.model_id}@{args.revision}"
    )
    safe_dataset = _safe_name(_canonical_dataset_name(args.dataset))
    safe_config = _safe_name(args.dataset_config)
    calib_basename = f"calib_oas_sqrtdiag_{safe_model}__{safe_dataset}__{safe_config}__{args.split}__ns{args.nsamples}_L{args.seqlen}.pt"
    calib_path = os.path.join(args.calib_cache_dir, calib_basename)

    cov_ops: Dict[str, Dict[str, torch.Tensor]]
    if args.reuse_calib and os.path.exists(calib_path):
        print(f"[Step4] Loading cached Σ_x from: {calib_path}")
        payload = torch.load(calib_path, map_location="cpu")
        cov_ops = payload.get("cov_ops", payload)
        changed = False
        for k, entry in cov_ops.items():
            if "inv_sqrt" not in entry and "sqrt" in entry:
                sqrt = entry["sqrt"].to(torch.float32)
                inv = (1.0 / torch.clamp(sqrt, min=1e-12)).to(torch.float32)
                entry["inv_sqrt"] = inv
                changed = True
        if changed:
            meta = payload.get(
                "meta",
                {
                    "model_id": args.model_id,
                    "revision": args.revision,
                    "dataset": _canonical_dataset_name(args.dataset),
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "nsamples": args.nsamples,
                    "seqlen": args.seqlen,
                },
            )
            try:
                torch.save({"cov_ops": cov_ops, "meta": meta}, calib_path)
            except Exception:
                pass
    else:
        cov_ops = estimate_diag_cov_oas_per_module(
            model,
            tokenizer,
            device,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            calib_dataset=args.dataset,
            calib_config=args.dataset_config,
            split=args.split,
            use_streaming=bool(args.use_streaming),
            matmul_dtype=torch.float32,
        )
        os.makedirs(args.calib_cache_dir, exist_ok=True)
        meta = {
            "model_id": args.model_id,
            "revision": args.revision,
            "dataset": _canonical_dataset_name(args.dataset),
            "dataset_config": args.dataset_config,
            "split": args.split,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
        }
        torch.save({"cov_ops": cov_ops, "meta": meta}, calib_path)
        print(f"[Step4] Saved Σ_x cache: {calib_path}")

    # bit 할당 로드
    sel_bits = load_selected_bits(args.bit_assign_csv)
    print(f"[Step4] Loaded bit assignments: {len(sel_bits)} entries.")

    # 원 state dict (CPU)로 백업
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}

    # [수정] 출력 컨테이너
    qweights: Dict[str, torch.Tensor] = {}  # [기존] Fake Quant (fp16 Wq)용
    quantized_model_data: Dict[str, Dict] = {}  # [신규] True Quant (Triton)용
    corrections: Dict[str, torch.Tensor] = {}  # [기존] SVD (A,B)용
    bmap: Dict[str, str] = {}
    patched_state = state.copy()  # fold 저장용

    # 본체 해제(메모리 여유)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # [수정] 작업 루프
    print("[Step4] Building Wq (for SVD) AND Packed Params (for Triton)...")

    for full_name, W_cpu in tqdm(state.items()):
        if full_name.endswith(".weight") is False:
            continue

        # 대상 레이어 + bit 존재 여부 확인
        mod_name = module_name_from_weight(full_name)

        if (mod_name not in sel_bits) and (
            full_name.replace(".weight", "") not in sel_bits
        ):
            continue
        if not is_target_weight(full_name, W_cpu):
            continue

        # 매칭 비트
        bit = sel_bits.get(
            mod_name, sel_bits.get(full_name.replace(".weight", ""), None)
        )
        if bit is None:
            continue
        if bit < 2 or bit > 4:
            print(f"[warn] Unsupported bit={bit} at {mod_name}; clamped to [2,4].")
            bit = max(2, min(4, bit))

        if mod_name not in cov_ops:
            print(f"[warn] Σ_x not found for {mod_name}; skipping.")
            continue

        sqrt_diag = cov_ops[mod_name]["sqrt"].to(dtype=torch.float32)
        inv_sqrt_diag = cov_ops[mod_name].get("inv_sqrt")
        if inv_sqrt_diag is None:
            inv_sqrt_diag = (1.0 / torch.clamp(sqrt_diag, min=1e-12)).to(torch.float32)
        inv_sqrt_diag = inv_sqrt_diag.to(dtype=torch.float32)

        # 디바이스로 이동
        device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        W = W_cpu.to(device=device0, dtype=torch.float32)
        m, n = W.shape
        sqrt_diag = sqrt_diag.to(device=device0)
        inv_sqrt_diag = inv_sqrt_diag.to(device=device0)

        # [신규] 이 레이어의 Triton 파라미터를 담을 딕셔너리
        layer_quant_params = {}

        # -------- (1) 양자화: 2bit 자동선택, 3/4bit uniform --------
        if bit == 2:
            # [기존 로직] Fake Quant Wq 계산 (SVD 오차 계산용)
            Wq_l = dequant_2bit_lloyd2_sym(
                W, args.group_size, args.clip_percentile, args.lloyd_steps
            )
            Wq_z = dequant_2bit_qtr_zero(
                W, args.group_size, args.clip_percentile, args.qtr_steps
            )
            # weighted Frobenius
            El = (W - Wq_l) * sqrt_diag.unsqueeze(0)
            Ez = (W - Wq_z) * sqrt_diag.unsqueeze(0)
            score_l = torch.linalg.norm(El)
            score_z = torch.linalg.norm(Ez)

            if score_z <= score_l:
                Wq = Wq_z  # [기존] SVD용 Wq 선택

                # [신규] Triton용 파라미터 패킹 (QTR)
                q_packed, scales = get_params_2bit_qtr_zero(
                    W, args.group_size, args.clip_percentile, args.qtr_steps
                )
                layer_quant_params = {
                    "quant_type": "qtr_sym",
                    "qweight_packed": q_packed.cpu(),
                    "scales": scales.cpu(),  # Shape [O, G]
                }
            else:
                Wq = Wq_l  # [기존] SVD용 Wq 선택

                # [신규] Triton용 파라미터 패킹 (Lloyd)
                q_packed, scales = get_params_2bit_lloyd_sym(
                    W, args.group_size, args.clip_percentile, args.lloyd_steps
                )
                layer_quant_params = {
                    "quant_type": "lloyd_sym",
                    "qweight_packed": q_packed.cpu(),
                    "scales": scales.cpu(),  # Shape [O, G, 2]
                }
        else:
            # [기존 로직] Fake Quant Wq 계산 (SVD 오차 계산용)
            Wq = dequant_uniform_asym(
                W,
                b=bit,
                group_size=args.group_size,
                clip_pct=0.0,
                rounding=args.uniform_rounding,
            )

            # [신규] Triton용 파라미터 패킹 (Asymmetric)
            q_packed, z_packed, scales = get_params_uniform_asym(
                W, bit, args.group_size, args.uniform_rounding
            )
            layer_quant_params = {
                "quant_type": "asym",
                "qweight_packed": q_packed.cpu(),
                "qzeros_packed": z_packed.cpu(),
                "scales": scales.cpu(),  # Shape [O, G]
            }

        # [기존] Fake Quant Wq 저장
        qweights[full_name] = Wq.detach().to(torch.float16).cpu()

        # [신규] True Quant 파라미터 저장
        layer_quant_params["bit"] = bit
        layer_quant_params["shape"] = (m, n)
        layer_quant_params["group_size"] = args.group_size
        quantized_model_data[full_name] = layer_quant_params

        # -------- (2) 복원(AB): SVD(E Σ^{1/2}) rank-r --------
        # [기존 로직과 완벽히 동일] (Wq는 위에서 계산된 Fake Quant fp16 텐서를 사용)
        r_eff = min(args.rank, min(m, n))
        E = W - Wq.to(dtype=W.dtype, device=W.device)
        E_tilde = E * sqrt_diag.unsqueeze(0)  # (m,n)

        try:
            U, S, Vh = torch.linalg.svd(E_tilde, full_matrices=False)
        except RuntimeError:
            U, S, Vh = torch.linalg.svd(E_tilde.to("cpu"), full_matrices=False)
            U = U.to(W.device)
            S = S.to(W.device)
            Vh = Vh.to(W.device)

        r_eff = min(r_eff, S.numel())
        if r_eff == 0:
            A = torch.zeros(m, 0, dtype=torch.float16)
            B = torch.zeros(0, n, dtype=torch.float16)
            W_eff = Wq
        else:
            U_r = U[:, :r_eff]
            S_r = S[:r_eff]
            V_r_T = Vh[:r_eff, :]
            S_sqrt = torch.sqrt(torch.clamp(S_r, min=1e-12))
            A = (
                (U_r * S_sqrt.unsqueeze(0)).to(dtype=torch.float16).detach().cpu()
            )  # [m,r]
            B = (
                (S_sqrt.unsqueeze(1) * V_r_T * inv_sqrt_diag.unsqueeze(0))
                .to(dtype=torch.float16)
                .detach()
                .cpu()
            )  # [r,n]
            W_eff = (
                Wq
                + (
                    A.to(W.device, dtype=torch.float32)
                    @ B.to(W.device, dtype=torch.float32)
                )
            ).to(torch.float32)

        corrections[f"{mod_name}.A"] = A
        corrections[f"{mod_name}.B"] = B
        bmap[full_name] = f"{mod_name}.B"

        if args.save_patched_model:
            patched_state[full_name] = W_eff.detach().to(torch.float16).cpu()

        del W, Wq, E, E_tilde, A, B, W_eff
        if "q_packed" in locals():
            del q_packed, scales
        if "z_packed" in locals():
            del z_packed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # [수정] 저장
    # [기존] 저장 로직
    torch.save(qweights, os.path.join(args.out_dir, "quantized_weights.pt"))
    torch.save(corrections, os.path.join(args.out_dir, "correction_layerwise.pt"))
    with open(os.path.join(args.out_dir, "b_ref_map_layerwise.json"), "w") as f:
        json.dump(bmap, f, indent=2)

    # [신규] Triton용 파라미터 저장
    triton_path = os.path.join(args.out_dir, "quantized_model_triton.pt")
    torch.save(quantized_model_data, triton_path)

    print(f"[Step4] Saved:")
    print(
        f"  • {os.path.join(args.out_dir,'quantized_weights.pt')}      ({len(qweights)} tensors) [For Fake Quant Step5]"
    )
    print(
        f"  • {triton_path} ({len(quantized_model_data)} layers) [For True Quant Triton Step]"
    )
    print(
        f"  • {os.path.join(args.out_dir,'correction_layerwise.pt')}   ({len([k for k in corrections if k.endswith('.A')])} layers) [For SVD Correction]"
    )
    print(f"  • {os.path.join(args.out_dir,'b_ref_map_layerwise.json')}")

    if args.save_patched_model:
        torch.save(patched_state, os.path.join(args.out_dir, "patched_state.pt"))
        print(f"  • {os.path.join(args.out_dir,'patched_state.pt')}  (folded W_eff)")


if __name__ == "__main__":
    main()
