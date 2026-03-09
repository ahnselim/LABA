#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixture joint evaluation module for Step2.5/Step3 outputs (`Wdq*`, `AB*`).

역할:
  - 저장된 `wdq_star` / `low_rank_ab*` 계열 아티팩트를 HF 모델에 주입
  - `Wdq*-only` vs `Wdq* + AB*` PPL 및 선택적 generation 성능 비교

용도:
  - `step0_3_BO_opt.py`의 optional eval 경로에서 호출되는 기본 평가 스크립트
  - 필요 시 단독 CLI 실행 가능 (`main()`)
"""

import os, re, math, gc, time, argparse, json
from time import perf_counter
from statistics import mean, median
from contextlib import contextmanager
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# -------------------------
# Utils: safe load_state_dict (partial)
# -------------------------
@torch.no_grad()
def safe_load_state_dict_partial(model: nn.Module, sd: Dict[str, torch.Tensor], verbose=True):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[LoadStateDict(strict=False)] missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) and len(missing) <= 20:
            print("  missing (first):", missing[:20])
        if len(unexpected) and len(unexpected) <= 20:
            print("  unexpected (first):", unexpected[:20])


def get_parent_module(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


# -------------------------
# Wrapper: Wdq already injected into weight
# forward: y = inner(x) + alpha * ( (x @ B^T) @ A^T )
# -------------------------
class AddLowRankCorrection(nn.Module):
    def __init__(self, inner: nn.Module, A: torch.Tensor, B: torch.Tensor, alpha: float = 1.0):
        super().__init__()
        self.inner = inner
        self.register_buffer("A", A.to(torch.float16), persistent=False)  # [O, r]
        self.register_buffer("B", B.to(torch.float16), persistent=False)  # [r, I]
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha == 0.0:
            return z
        r = F.linear(x, self.B)      # [*, r]
        corr = F.linear(r, self.A)   # [*, O]
        return z.add_(corr, alpha=self.alpha)


class AddLowRankCorrectionFP32(nn.Module):
    """
    Numerically safer AB correction:
      - Keep A/B in fp16 buffers (memory), but compute correction in fp32.
      - Useful when Wdq is very low-bit and AB magnitude is large.
    """

    def __init__(self, inner: nn.Module, A: torch.Tensor, B: torch.Tensor, alpha: float = 1.0):
        super().__init__()
        self.inner = inner
        self.register_buffer("A", A.to(torch.float16), persistent=False)  # [O, r]
        self.register_buffer("B", B.to(torch.float16), persistent=False)  # [r, I]
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha == 0.0:
            return z
        x32 = x.to(torch.float32)
        r = F.linear(x32, self.B.to(torch.float32))
        corr = F.linear(r, self.A.to(torch.float32))
        return (z.to(torch.float32) + corr * self.alpha).to(x.dtype)


def _unwrap_base_linear(module: nn.Module) -> nn.Module:
    while isinstance(module, (AddLowRankCorrection, AddLowRankCorrectionFP32)):
        module = module.inner
    return module


# -------------------------
# Inject Wdq* into model weights
# -------------------------
@torch.no_grad()
def apply_wdq_star(model: nn.Module, wdq: Dict[str, torch.Tensor]):
    injected, missing, mismatch = 0, 0, 0
    for wkey, Wdq in wdq.items():
        if not (isinstance(wkey, str) and wkey.endswith(".weight") and getattr(Wdq, "ndim", 0) == 2):
            continue
        module_name = wkey[:-7]  # drop ".weight"
        try:
            parent, attr = get_parent_module(model, module_name)
            current = getattr(parent, attr, None)
        except AttributeError:
            missing += 1
            continue
        if current is None:
            missing += 1
            continue

        inner = _unwrap_base_linear(current)
        if not hasattr(inner, "weight"):
            missing += 1
            continue
        if inner.weight.shape != Wdq.shape:
            mismatch += 1
            continue

        inner.weight.data.copy_(Wdq.to(device=inner.weight.device, dtype=inner.weight.dtype))
        injected += 1

    print(f"[Inject Wdq*] injected={injected}, missing={missing}, shape_mismatch={mismatch}")


# -------------------------
# Build B from Bbar using:
#  (1) diag:   B = Bbar * inv_s
#  (2) lowrank: B = Bbar @ (Lambda^{-1/2} U^T)
# -------------------------
@torch.no_grad()
def build_B_from_Bbar(Bbar: torch.Tensor, inv_s: torch.Tensor) -> torch.Tensor:
    # Bbar: [r, I], inv_s: [I]
    return (Bbar.to(torch.float32) * inv_s.to(torch.float32).unsqueeze(0)).to(Bbar.dtype)

@torch.no_grad()
def build_B_from_Bbar_lowrank(Bbar: torch.Tensor, U: torch.Tensor, sqrt_lambda: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Bbar: [r, k]
    U: [I, k]
    sqrt_lambda: [k]
    return B: [r, I]
    Formula: B = Bbar @ (Lambda^{-1/2} U^T)
    """
    inv_sqrt = 1.0 / sqrt_lambda.to(torch.float32).clamp_min(eps)           # [k]
    T = (inv_sqrt.unsqueeze(1) * U.to(torch.float32).T).contiguous()        # [k, I]
    B = (Bbar.to(torch.float32) @ T).to(Bbar.dtype)                         # [r, I]
    return B


# -------------------------
# Patch layerwise AB from Step2.5 outputs
# -------------------------
@torch.no_grad()
def patch_layerwise_ab_from_step2p5(
    model: nn.Module,
    low_rank_ab: Optional[Dict[str, Any]] = None,
    low_rank_abbar: Optional[Dict[str, Any]] = None,
    calib_s: Optional[Dict[str, Any]] = None,
    calib_h_lowrank: Optional[Dict[str, Any]] = None,
    alpha: float = 1.0,
    ab_compute: str = "fp16",
):
    assert (low_rank_ab is not None) or (low_rank_abbar is not None), "Need low_rank_ab or low_rank_abbar"

    if ab_compute not in {"fp16", "fp32"}:
        raise ValueError(f"Unknown ab_compute: {ab_compute} (expected: fp16|fp32)")

    patched, skipped = 0, 0
    uvab_folded = 0
    keys = set()
    if low_rank_ab is not None:
        keys |= set(low_rank_ab.keys())
    if low_rank_abbar is not None:
        keys |= set(low_rank_abbar.keys())
    keys = sorted([k for k in keys if isinstance(k, str) and k.endswith(".weight")])

    for wkey in tqdm(keys, desc="Patching AB*"):
        module_name = wkey[:-7]

        A = None
        B = None

        if low_rank_ab is not None and wkey in low_rank_ab:
            item = low_rank_ab[wkey]
            if isinstance(item, dict) and ("A" in item) and ("B" in item):
                A, B = item["A"], item["B"]
                # Step3 uv-ab format:
                #   Rw ~= diag(u) (A B) diag(v),   Rw = (W-Wdq) * s
                # so correction in weight domain is:
                #   W-Wdq ~= diag(u) (A B) diag(v * inv_s)
                if ("u" in item) or ("v" in item):
                    if ("u" not in item) or ("v" not in item):
                        skipped += 1
                        continue
                    if calib_s is None or wkey not in calib_s or "inv_s" not in calib_s[wkey]:
                        raise ValueError(
                            f"low_rank_ab[{wkey}] has u,v (Step3 uv-ab) but missing calib_s inv_s. "
                            "Pass --calib_s_path (Step2 calib_sqrtdiag.pt)."
                        )
                    u = item["u"]      # [O]
                    v = item["v"]      # [I]
                    inv_s = calib_s[wkey]["inv_s"]  # [I]
                    A = (u.to(torch.float32).unsqueeze(1) * A.to(torch.float32)).to(A.dtype)
                    B = (B.to(torch.float32) * (v.to(torch.float32) * inv_s.to(torch.float32)).unsqueeze(0)).to(B.dtype)
                    uvab_folded += 1

        if (A is None or B is None) and (low_rank_abbar is not None) and (wkey in low_rank_abbar):
            item = low_rank_abbar[wkey]
            if isinstance(item, dict) and ("A" in item) and ("Bbar" in item):
                A = item["A"]
                Bbar = item["Bbar"]
                # Decide diag vs lowrank by shape or meta
                meta = item.get("meta", {}) if isinstance(item, dict) else {}
                h_weighting = meta.get("h_weighting", None)

                # locate module first to know target I
                try:
                    parent0, attr0 = get_parent_module(model, module_name)
                    cur0 = getattr(parent0, attr0, None)
                except AttributeError:
                    skipped += 1
                    continue
                if cur0 is None:
                    skipped += 1
                    continue
                inner0 = _unwrap_base_linear(cur0)
                if not hasattr(inner0, "weight"):
                    skipped += 1
                    continue
                O0, I0 = inner0.weight.shape

                # Case 1) diag: Bbar [r, I]
                if (h_weighting == "diag") or (Bbar.ndim == 2 and Bbar.shape[1] == I0):
                    if calib_s is None or wkey not in calib_s or "inv_s" not in calib_s[wkey]:
                        skipped += 1
                        continue
                    inv_s = calib_s[wkey]["inv_s"]  # [I]
                    B = build_B_from_Bbar(Bbar, inv_s)
                else:
                    # Case 2) lowrank: Bbar [r, k] and need U,sqrt_lambda
                    if calib_h_lowrank is None or wkey not in calib_h_lowrank:
                        skipped += 1
                        continue
                    U = calib_h_lowrank[wkey]["U"]                 # [I, k]
                    sqrt_l = calib_h_lowrank[wkey]["sqrt_lambda"]  # [k]
                    if U.ndim != 2 or sqrt_l.ndim != 1:
                        skipped += 1
                        continue
                    if U.shape[0] != I0:
                        skipped += 1
                        continue
                    if Bbar.shape[1] != U.shape[1] or sqrt_l.numel() != U.shape[1]:
                        skipped += 1
                        continue
                    B = build_B_from_Bbar_lowrank(Bbar, U, sqrt_l)

        if A is None or B is None:
            skipped += 1
            continue

        # locate module
        try:
            parent, attr = get_parent_module(model, module_name)
            current = getattr(parent, attr, None)
        except AttributeError:
            skipped += 1
            continue
        if current is None:
            skipped += 1
            continue

        inner = _unwrap_base_linear(current)
        if not hasattr(inner, "weight"):
            skipped += 1
            continue

        O, I = inner.weight.shape
        if A.ndim != 2 or B.ndim != 2:
            skipped += 1
            continue
        if A.shape[0] != O or B.shape[1] != I or A.shape[1] != B.shape[0]:
            skipped += 1
            continue

        wrapper_cls = AddLowRankCorrectionFP32 if (ab_compute == "fp32") else AddLowRankCorrection
        wrapped = wrapper_cls(
            inner,
            A.to(device=inner.weight.device),
            B.to(device=inner.weight.device),
            alpha=alpha,
        )
        setattr(parent, attr, wrapped)
        patched += 1

    print(f"[Patch AB*] patched={patched}, skipped={skipped}, uvab_folded={uvab_folded}")
    return model


# ======================================================================
# ✅ Step1-matched quant (theta_star로 Wdq 재생성)
#   - Step2.5(내가 준 Step1-matched 버전)의 theta_star 포맷을 그대로 지원
# ======================================================================
def _to_groups(W2: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int, int, int, int]:
    assert W2.ndim == 2
    O, orig_I = W2.shape
    S = int(group_size)
    pad = (S - (orig_I % S)) % S
    if pad:
        W2 = F.pad(W2, (0, pad), value=0.0)
    I_pad = W2.shape[1]
    G = I_pad // S
    Wg = W2.view(O, G, S)
    return Wg, O, G, S, orig_I

def _from_groups(Wg: torch.Tensor, orig_I: int) -> torch.Tensor:
    W2 = Wg.reshape(Wg.shape[0], Wg.shape[1] * Wg.shape[2])
    return W2[:, :orig_I].contiguous()

def _percentile_clip_lastdim(Wg: torch.Tensor, upper_pct: float, lower_pct: float):
    """
    Step1과 동일:
      - Wg: [O,G,gs]
      - per-group(G) 기준으로 O*gs를 flatten해서 percentile threshold 계산
      - upper_pct/lower_pct: 0~100 (percent)
    """
    assert Wg.ndim == 3
    O, G, gs = Wg.shape
    flat = Wg.permute(1, 0, 2).reshape(G, -1)  # [G, O*gs]
    n = flat.shape[1]

    lo_k = max(1, int((lower_pct / 100.0) * n))
    hi_k = max(1, int((upper_pct / 100.0) * n))
    lo_k = min(n, lo_k)
    hi_k = min(n, hi_k)

    lo = torch.kthvalue(flat, lo_k, dim=1).values  # [G]
    hi = torch.kthvalue(flat, hi_k, dim=1).values  # [G]
    lo = lo.view(1, G, 1)
    hi = hi.view(1, G, 1)
    return Wg.clamp(min=lo, max=hi)

def _percentile_clip_lastdim_rowwise(Wg: torch.Tensor, upper_pct: float, lower_pct: float):
    lo = torch.quantile(Wg, lower_pct / 100.0, dim=-1, keepdim=True)
    hi = torch.quantile(Wg, upper_pct / 100.0, dim=-1, keepdim=True)
    return Wg.clamp(min=lo, max=hi)

def _gs_for_bit(bit: int, args) -> int:
    """bit별 group_size 선택 (없으면 global group_size fallback)"""
    if bit == 1 and getattr(args, "group_size_1", None) is not None:
        return int(args.group_size_1)
    if bit == 2 and getattr(args, "group_size_2", None) is not None:
        return int(args.group_size_2)
    if bit == 3 and getattr(args, "group_size_3", None) is not None:
        return int(args.group_size_3)
    if bit == 4 and getattr(args, "group_size_4", None) is not None:
        return int(args.group_size_4)
    return int(args.group_size)


def _stochastic_round(qf: torch.Tensor) -> torch.Tensor:
    flo = torch.floor(qf)
    frac = qf - flo
    rnd = (torch.rand_like(frac) < frac).to(qf.dtype)
    return flo + rnd


@torch.no_grad()
def dequant_1bit_mu_beta(W: torch.Tensor, group_size: int):
    W32 = W.to(torch.float32)
    Wg, O, G, gs, orig_I = _to_groups(W32, group_size)
    mu = Wg.mean(dim=-1, keepdim=True)                  # [O,G,1]
    centered = Wg - mu
    beta = centered.abs().mean(dim=-1, keepdim=True)   # [O,G,1]
    sgn = torch.where(Wg >= mu, 1.0, -1.0)
    deq_g = mu + beta * sgn
    deq = _from_groups(deq_g, orig_I)
    return deq.to(dtype=W.dtype)


@torch.no_grad()
def dequant_2bit_lloyd2_sym(
    W: torch.Tensor, group_size: int, clip_pct: float = 99.9, steps: int = 4
):
    W32 = W.to(torch.float32)
    Wg, O, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim_rowwise(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
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
    return _from_groups(deq, orig_I).to(dtype=W.dtype)


@torch.no_grad()
def dequant_2bit_qtr_zero(
    W: torch.Tensor, group_size: int, clip_pct: float = 99.9, steps: int = 6
):
    W32 = W.to(torch.float32)
    Wg, O, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim_rowwise(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
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
    return _from_groups(deq, orig_I).to(dtype=W.dtype)

@torch.no_grad()
def dequant_uniform_asym_from_theta_star(
    W: torch.Tensor,                     # [O,I] fp16/fp32 on device
    theta_star_item: Dict[str, Any],     # {"mode":..., params...}
    b: int,
    group_size: int,
    clip_pct: float,
    rounding: str,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Step2.5 Step1-matched quant 정의로 Wdq 재생성.
    rounding:
      - nearest: round
      - stochastic: stochastic round (eval에서는 stochastic가 흔들릴 수 있음 주의)
    """
    assert b in (1, 2, 3, 4)
    if b == 1:
        return dequant_1bit_mu_beta(W, group_size)
    if b == 2:
        Wq_l = dequant_2bit_lloyd2_sym(W, group_size, clip_pct=clip_pct, steps=4)
        Wq_z = dequant_2bit_qtr_zero(W, group_size, clip_pct=clip_pct, steps=6)
        El = (W - Wq_l).float()
        Ez = (W - Wq_z).float()
        return Wq_z if torch.linalg.norm(Ez) <= torch.linalg.norm(El) else Wq_l

    qmax = float((1 << b) - 1)

    W32 = W.to(torch.float32)
    Wg, O, G, S, orig_I = _to_groups(W32, group_size)
    if clip_pct > 0:
        lower = float(clip_pct)
        upper = 100.0 - float(clip_pct)
        X = _percentile_clip_lastdim(Wg, upper_pct=upper, lower_pct=lower)
    else:
        X = Wg

    base_min = X.amin(dim=-1)  # [O,G]
    base_max = X.amax(dim=-1)

    mode = theta_star_item.get("mode", None)
    if mode is None:
        raise ValueError("theta_star item missing 'mode'")

    if mode == "minmax":
        minv = theta_star_item["min"].to(device=W.device, dtype=torch.float32)
        maxv = theta_star_item["max"].to(device=W.device, dtype=torch.float32)
    elif mode == "gamma":
        gamma = theta_star_item["gamma"].to(device=W.device, dtype=torch.float32)
        c0 = 0.5 * (base_min + base_max)
        h0 = 0.5 * (base_max - base_min)
        minv = c0 - gamma * h0
        maxv = c0 + gamma * h0
    elif mode == "gamma_delta":
        gamma = theta_star_item["gamma"].to(device=W.device, dtype=torch.float32)
        delta = theta_star_item["delta"].to(device=W.device, dtype=torch.float32)
        c0 = 0.5 * (base_min + base_max) + delta
        h0 = 0.5 * (base_max - base_min)
        minv = c0 - gamma * h0
        maxv = c0 + gamma * h0
    else:
        raise ValueError(f"Unknown theta mode: {mode}")

    span = (maxv - minv).clamp_min(eps)
    scale = span / qmax
    zp_f = (-minv / scale)

    # Step1: zp는 deterministic round
    zp = torch.round(zp_f)
    zp = zp.clamp(0.0, qmax)

    qf = X / scale.unsqueeze(-1) + zp.unsqueeze(-1)
    if rounding == "stochastic":
        q = _stochastic_round(qf)
    else:
        q = torch.round(qf)

    q = q.clamp(0.0, qmax)
    deq = (q - zp.unsqueeze(-1)) * scale.unsqueeze(-1)

    flat = span <= eps
    if flat.any():
        deq[flat] = minv[flat].unsqueeze(-1)

    Wdq = _from_groups(deq, orig_I).to(dtype=W.dtype)
    return Wdq


@torch.no_grad()
def build_wdq_dict_from_theta_star(
    model: nn.Module,
    theta_star: Dict[str, Any],
    b: int,
    group_size: int,
    clip_pct: float,
    rounding: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    model의 현재 weight(W)를 사용해서, theta_star keys에 해당하는 Wdq를 재생성해서 dict로 반환.
    """
    out = {}
    injected_like = 0
    missing = 0
    mismatch = 0

    keys = sorted([k for k in theta_star.keys() if isinstance(k, str) and k.endswith(".weight")])
    for wkey in tqdm(keys, desc="Rebuild Wdq* from theta_star"):
        module_name = wkey[:-7]
        try:
            parent, attr = get_parent_module(model, module_name)
            cur = getattr(parent, attr, None)
        except AttributeError:
            missing += 1
            continue
        if cur is None:
            missing += 1
            continue

        inner = _unwrap_base_linear(cur)
        if not hasattr(inner, "weight"):
            missing += 1
            continue

        W = inner.weight.data
        item = theta_star[wkey]
        try:
            Wdq = dequant_uniform_asym_from_theta_star(
                W=W,
                theta_star_item=item,
                b=b,
                group_size=group_size,
                clip_pct=clip_pct,
                rounding=rounding,
            )
        except Exception as e:
            print(f"[theta->Wdq] failed at {wkey}: {e}")
            mismatch += 1
            continue

        if Wdq.shape != W.shape:
            mismatch += 1
            continue

        out[wkey] = Wdq.detach().to("cpu", dtype=torch.float16).contiguous()
        injected_like += 1

    print(f"[Build Wdq* from theta] built={injected_like}, missing={missing}, mismatch_or_fail={mismatch}")
    return out


# -------------------------
# Eval: PPL (WikiText-2)
# -------------------------
@torch.no_grad()
def evaluate_ppl_wikitext2(
    model,
    tokenizer,
    device,
    label: str,
    stride: int = 2048,
    max_tokens: int = 0,
):
    print(f"\n--- Evaluating PPL: {label} ---")
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    seq_len = input_ids.size(1)
    if max_tokens and max_tokens > 0 and seq_len > int(max_tokens):
        seq_len = int(max_tokens)
        input_ids = input_ids[:, :seq_len].contiguous()
    print(f"[PPL] tokenized seq_len={seq_len} (stride={stride})")

    total_loss, total_tokens = 0.0, 0
    start = time.time()

    pbar = tqdm(range(0, seq_len, stride), desc=f"PPL {label}")
    loss_fct = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for i in pbar:
            t0 = time.time()
            begin, end = i, min(i + stride, seq_len)
            if end - begin <= 1:
                continue
            x = input_ids[:, begin:end].to(device)
            out = model(x, use_cache=False)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = x[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ntok = shift_labels.numel()
            total_loss += loss.item() * ntok
            total_tokens += ntok
            pbar.set_description(f"PPL {label} (loss={loss.item():.4f}, {time.time() - t0:.2f}s/it)")

    ppl = math.exp(total_loss / max(1, total_tokens))
    elapsed = time.time() - start
    print(f"✅ PPL({label}) = {ppl:.4f} | time={elapsed:.2f}s")
    return ppl, elapsed


# -------------------------
# Generation metrics (optional)
# -------------------------
def _cuda_sync(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)

@contextmanager
def temp_generation_overrides(model, **overrides):
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        yield
        return
    old_vals = {k: getattr(gen_cfg, k, None) for k in overrides}
    for k, v in overrides.items():
        try:
            setattr(gen_cfg, k, v)
        except:
            pass
    try:
        yield
    finally:
        for k, v in old_vals.items():
            try:
                setattr(gen_cfg, k, v)
            except:
                pass

def _get_sequences_from_generate(output):
    return output.sequences if hasattr(output, "sequences") else output

@torch.no_grad()
def measure_generation_metrics(
    model,
    tokenizer,
    device,
    prompts,
    max_new_tokens=50,
    do_sample=False,
    num_beams=1,
    temperature=1.0,
    top_p=1.0,
    repeats=1,
):
    model.eval()
    pad_id = tokenizer.eos_token_id
    override_kwargs = {"temperature": 1.0, "top_p": 1.0} if not do_sample else {}

    with temp_generation_overrides(model, **override_kwargs):
        # warmup
        try:
            w = tokenizer(prompts[0], return_tensors="pt", truncation=True, max_length=512).to(device)
            if w["input_ids"].dim() == 1:
                w["input_ids"] = w["input_ids"].unsqueeze(0)
            if "attention_mask" in w and w["attention_mask"].dim() == 1:
                w["attention_mask"] = w["attention_mask"].unsqueeze(0)
            _ = model.generate(**w, max_new_tokens=1, use_cache=True)
            _cuda_sync(device)
        except Exception as e:
            print(f"⚠️ warmup failed: {e}")

        ttfb_ms, tokps, total_times, gen_ntoks = [], [], [], []
        gen_kwargs = dict(
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=pad_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

        for _ in range(repeats):
            for p in prompts:
                inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(device)
                if inp["input_ids"].dim() == 1:
                    inp["input_ids"] = inp["input_ids"].unsqueeze(0)
                if "attention_mask" in inp and inp["attention_mask"].dim() == 1:
                    inp["attention_mask"] = inp["attention_mask"].unsqueeze(0)

                _cuda_sync(device)
                t0 = perf_counter()
                model.generate(**inp, max_new_tokens=1, **gen_kwargs)
                _cuda_sync(device)
                ttfb_ms.append((perf_counter() - t0) * 1000.0)

                _cuda_sync(device)
                t1 = perf_counter()
                out = model.generate(**inp, max_new_tokens=max_new_tokens, **gen_kwargs)
                _cuda_sync(device)
                tt = perf_counter() - t1

                newt = _get_sequences_from_generate(out).shape[1] - inp["input_ids"].shape[1]
                tokps.append((newt / tt) if tt > 0 else 0.0)
                total_times.append(tt)
                gen_ntoks.append(newt)

    return {
        "ttfb_ms_mean": mean(ttfb_ms) if ttfb_ms else 0.0,
        "ttfb_ms_median": median(ttfb_ms) if ttfb_ms else 0.0,
        "tok_s_mean": mean(tokps) if tokps else 0.0,
        "tok_s_median": median(tokps) if tokps else 0.0,
        "avg_total_time_s": mean(total_times) if total_times else 0.0,
        "avg_new_tokens": mean(gen_ntoks) if gen_ntoks else 0.0,
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Step4 Eval (Step2.5/Step3 outputs: Wdq* or theta* + AB*)")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trust_remote_code", action="store_true")

    # optional partial original load
    ap.add_argument("--original_weights_path", default=None, help="optional original_weights(.pt), partial ok")

    # Step2.5 artifacts: choose one of (wdq_star) or (theta_star)
    ap.add_argument("--wdq_star_path", default=None, help="wdq_star.pt {wkey: Wdq*}")
    ap.add_argument("--theta_star_path", default=None, help="theta_star.pt {wkey: {mode, ...}} (rebuild Wdq*)")

    # quant cfg for theta->Wdq rebuild
    ap.add_argument("--bits", type=int, default=1, choices=[1])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--group_size_1", type=int, default=None, help="1-bit group size (override)")
    ap.add_argument("--group_size_2", type=int, default=None, help="2-bit group size (override)")
    ap.add_argument("--group_size_3", type=int, default=None, help="3-bit group size (override)")
    ap.add_argument("--group_size_4", type=int, default=None, help="4-bit group size (override)")
    ap.add_argument("--clip_pct", type=float, default=0.0)
    ap.add_argument("--rounding", type=str, default="nearest", choices=["nearest", "stochastic"])

    # AB artifacts
    ap.add_argument("--low_rank_ab_path", default=None, help="low_rank_ab.pt {wkey:{A,B}}")
    ap.add_argument("--low_rank_abbar_path", default=None, help="low_rank_abbar.pt {wkey:{A,Bbar}}")
    ap.add_argument("--calib_s_path", default=None, help="Step2 calib_sqrtdiag.pt (needed for abbar-diag or Step3 uv-ab)")
    ap.add_argument("--calib_h_lowrank_path", default=None, help="Step2 low-rank H cache (needed if using abbar from lowrank H)")
    ap.add_argument("--ab_compute", type=str, default="fp16", choices=["fp16", "fp32"],
                    help="AB application compute dtype: fp16 (faster) vs fp32 (more stable).")
    ap.add_argument("--ab_alpha", type=float, default=1.0, help="Alpha for AB correction in eval-2 (Wdq*+AB*).")

    # eval
    ap.add_argument("--ppl_stride", type=int, default=2048, help="Chunk length for PPL eval (smaller = faster).")
    ap.add_argument("--ppl_max_tokens", type=int, default=0, help="If >0, evaluate on first N tokens only.")
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--gen_max_new_tokens", type=int, default=50)
    ap.add_argument("--gen_repeats", type=int, default=1)
    ap.add_argument("--gen_do_sample", action="store_true")
    ap.add_argument("--gen_num_beams", type=int, default=1)
    ap.add_argument("--gen_temperature", type=float, default=1.0)
    ap.add_argument("--gen_top_p", type=float, default=1.0)

    args = ap.parse_args()
    group_size = _gs_for_bit(args.bits, args)

    if args.low_rank_ab_path is None and args.low_rank_abbar_path is None:
        raise ValueError("Need --low_rank_ab_path or --low_rank_abbar_path")
    # NOTE:
    #   - if low_rank_abbar comes from diag, need calib_s_path
    #   - if low_rank_abbar comes from lowrank H, need calib_h_lowrank_path
    #   - if low_rank_ab_path is Step3 uv-ab (has u,v), need calib_s_path

    if args.wdq_star_path is None and args.theta_star_path is None:
        raise ValueError("Need --wdq_star_path or --theta_star_path")

    device = torch.device(args.device)

    print(f"📥 Loading tokenizer: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"📥 Loading base model (fp16): {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # optional partial original load (권장: theta로 Wdq 재생성할 때 W를 확정하고 싶으면)
    if args.original_weights_path is not None:
        print(f"📦 Loading original weights (partial ok): {args.original_weights_path}")
        orig_sd = torch.load(args.original_weights_path, map_location="cpu")
        safe_load_state_dict_partial(model, orig_sd, verbose=True)
        del orig_sd

    # load AB artifacts
    low_rank_ab = None
    low_rank_abbar = None
    calib_s = None
    calib_h_lowrank = None

    if args.low_rank_ab_path is not None:
        print(f"📦 Loading AB*: {args.low_rank_ab_path}")
        low_rank_ab = torch.load(args.low_rank_ab_path, map_location="cpu")

    if args.calib_s_path is not None:
        print(f"📦 Loading calib_s (for inv_s): {args.calib_s_path}")
        calib_s = torch.load(args.calib_s_path, map_location="cpu")

    if args.low_rank_abbar_path is not None:
        print(f"📦 Loading ABbar*: {args.low_rank_abbar_path}")
        low_rank_abbar = torch.load(args.low_rank_abbar_path, map_location="cpu")
        if args.calib_h_lowrank_path is not None:
            print(f"📦 Loading calib_h_lowrank (for U,sqrt_lambda): {args.calib_h_lowrank_path}")
            calib_h_lowrank = torch.load(args.calib_h_lowrank_path, map_location="cpu")

    # Detect Step3 uv-ab in low_rank_ab and enforce calib_s availability.
    if low_rank_ab is not None:
        has_uvab = any(
            isinstance(item, dict) and (("u" in item) or ("v" in item))
            for item in low_rank_ab.values()
        )
        if has_uvab and calib_s is None:
            raise ValueError(
                "Detected Step3 uv-ab format in --low_rank_ab_path (u/v found), "
                "but --calib_s_path is missing. "
                "Please pass Step2 calib_sqrtdiag.pt so eval can map weighted residual to weight correction."
            )

    # move model to device
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # build or load Wdq*
    if args.wdq_star_path is not None:
        print(f"📦 Loading Wdq*: {args.wdq_star_path}")
        wdq_star = torch.load(args.wdq_star_path, map_location="cpu")
    else:
        print(f"📦 Loading theta_star: {args.theta_star_path}")
        theta_star = torch.load(args.theta_star_path, map_location="cpu")
        if args.bits == 1:
            print("⚠️ bits=1 rebuilds the fixed group-wise (mu,beta) quantizer; theta params are ignored.")
        print(f"🛠️ Rebuilding Wdq* from theta_star (bits={args.bits}, gs={group_size}, clip_pct={args.clip_pct}, rounding={args.rounding})")
        wdq_star = build_wdq_dict_from_theta_star(
            model=model,
            theta_star=theta_star,
            b=args.bits,
            group_size=group_size,
            clip_pct=args.clip_pct,
            rounding=args.rounding,
            device=device,
        )
        del theta_star
        gc.collect()

    # inject Wdq* baseline
    apply_wdq_star(model, wdq_star)

    # patch AB* wrappers
    model = patch_layerwise_ab_from_step2p5(
        model,
        low_rank_ab=low_rank_ab,
        low_rank_abbar=low_rank_abbar,
        calib_s=calib_s,
        calib_h_lowrank=calib_h_lowrank,
        alpha=1.0,  # will toggle below
        ab_compute=args.ab_compute,
    )

    prompts = [
        "Hello, my name is",
        "The quick brown fox",
        "In a shocking finding, scientists discovered that",
    ]

    # -------------------------
    # Eval 1) Wdq*-only (alpha=0)
    # -------------------------
    print("\n=== EVAL: Wdq*-only (alpha=0) ===")
    for m in model.modules():
        if isinstance(m, (AddLowRankCorrection, AddLowRankCorrectionFP32)):
            m.alpha = 0.0
    ppl_wdq, t_wdq = evaluate_ppl_wikitext2(
        model,
        tok,
        device,
        "Wdq*-only",
        stride=args.ppl_stride,
        max_tokens=args.ppl_max_tokens,
    )

    gen_wdq = None
    if not args.skip_gen:
        print("⏱️ Measuring generation metrics (Wdq*-only)...")
        gen_wdq = measure_generation_metrics(
            model, tok, device,
            prompts=prompts,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=args.gen_do_sample,
            num_beams=args.gen_num_beams,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            repeats=args.gen_repeats,
        )
        print(f"  • TTFB(median) = {gen_wdq['ttfb_ms_median']:.1f} ms")
        print(f"  • tok/s(median)= {gen_wdq['tok_s_median']:.2f}")

    # -------------------------
    # Eval 2) Wdq* + AB* (alpha=1)
    # -------------------------
    print(f"\n=== EVAL: Wdq* + AB* (alpha={args.ab_alpha:g}) ===")
    for m in model.modules():
        if isinstance(m, (AddLowRankCorrection, AddLowRankCorrectionFP32)):
            m.alpha = float(args.ab_alpha)
    ppl_ab, t_ab = evaluate_ppl_wikitext2(
        model,
        tok,
        device,
        "Wdq*+AB*",
        stride=args.ppl_stride,
        max_tokens=args.ppl_max_tokens,
    )

    gen_ab = None
    if not args.skip_gen:
        print("⏱️ Measuring generation metrics (Wdq*+AB*)...")
        gen_ab = measure_generation_metrics(
            model, tok, device,
            prompts=prompts,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=args.gen_do_sample,
            num_beams=args.gen_num_beams,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            repeats=args.gen_repeats,
        )
        print(f"  • TTFB(median) = {gen_ab['ttfb_ms_median']:.1f} ms")
        print(f"  • tok/s(median)= {gen_ab['tok_s_median']:.2f}")

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 110)
    print(f"FINAL SUMMARY | model={args.model_name} | activation=fp16 fixed | layerwise AB* | no grouping/no caching")
    print("-" * 110)

    def fmt_gen(g):
        if not g:
            return ("-", "-")
        return (f"{g['ttfb_ms_median']:.1f}", f"{g['tok_s_median']:.2f}")

    ttfb_wdq, tok_wdq = fmt_gen(gen_wdq)
    ttfb_ab, tok_ab = fmt_gen(gen_ab)

    print(f"{'Method':<20} | {'PPL':>10} | {'EvalTime(s)':>10} | {'TTFB(ms)':>10} | {'tok/s':>10}")
    print("-" * 110)
    print(f"{'Wdq*-only':<20} | {ppl_wdq:>10.4f} | {t_wdq:>10.2f} | {ttfb_wdq:>10} | {tok_wdq:>10}")
    method_ab = f"Wdq*+AB* (a={args.ab_alpha:g})"
    print(f"{method_ab:<20} | {ppl_ab:>10.4f} | {t_ab:>10.2f} | {ttfb_ab:>10} | {tok_ab:>10}")

    dppl = ppl_wdq - ppl_ab
    print("-" * 110)
    print(f"ΔPPL (Wdq*-only - Wdq*+AB*) = {dppl:+.4f}")
    if gen_wdq and gen_ab:
        dttfb = gen_ab["ttfb_ms_median"] - gen_wdq["ttfb_ms_median"]
        dtok = gen_ab["tok_s_median"] - gen_wdq["tok_s_median"]
        print(f"ΔTTFB(ms) (AB - base)      = {dttfb:+.1f}")
        print(f"Δtok/s (AB - base)         = {dtok:+.2f}")
    print("=" * 110)

    # cleanup
    del model, wdq_star, low_rank_ab, low_rank_abbar, calib_s, calib_h_lowrank
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
