#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
verify_three_stage_uv_optimization.py

개선판:
  - Step1: Procrustes-based initializer only
  - Step2: direct U,V optimization for weighted residual
  - Step3: IRLS-like spectral refinement on top of Step2 result

핵심 목적:
  선택한 몇 개 레이어에서,
    quant-only / weighted rank-r SVD / proposed 3-stage
  를 비교하여,
    weighted residual surrogate 기준으로
  실제 개선되는지 확인한다.

평가:
  Ew0 = (W - Wq) * sqrt_diag
  target objective:
      min_{rank(Z)<=r} ||Ew0 - Z||_F^2
  + optional weak anchor / geometry prior

주의:
  - baseline_weighted_svd는 위 surrogate의 최적해라 매우 강함.
  - 따라서 이 스크립트에서 제안법이 baseline을 완전히 이기기 어렵다.
  - 하지만 "좋은 initialization + constrained factor optimization + refinement"가
    baseline에 근접하거나, downstream metric 연결시 유리한지 볼 수 있다.
    
CUDA_VISIBLE_DEVICES=2 python test/verify_three_stage_uv_optimization.py \
  --model_id meta-llama/Llama-3.1-8B \
  --calib_s_path ./output/llama3_8b/calib_sqrtdiag.pt \
  --bits 2 \
  --group_size 128 \
  --layer_regex "model\.layers\.(0|1)\.self_attn\.q_proj\.weight" \
  --rank 32 \
  --max_layers 2 \
  --quant_mode hessian \
  --uv_steps 200 \
  --uv_lr 5e-3 \
  --lam_anchor 1e-3 \
  --irls_lam 5e-3 \
  --device cuda \
  --device_map cpu \
  --out_dir ./analysis_uv_three_stage_small
"""

import os
import re
import gc
import json
import time
import math
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e


# ============================================================
# Target filter
# ============================================================
TARGET_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "fc1", "fc2",
}

def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        isinstance(name, str)
        and isinstance(tensor, torch.Tensor)
        and tensor.ndim == 2
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )

def module_name_from_weight(full_weight_name: str) -> str:
    return full_weight_name[:-len(".weight")]


# ============================================================
# Utils
# ============================================================
def log(msg: str) -> None:
    print(msg, flush=True)

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return dev

def load_pt(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


# ============================================================
# Calibration load
# ============================================================
def load_calib_map(calib_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    payload = load_pt(calib_path)
    calib = payload.get("cov_ops", payload)

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in calib.items():
        if not isinstance(v, dict):
            continue

        sqrt_diag = None
        hdiag = None

        if "sqrt" in v:
            sqrt_diag = v["sqrt"].to(torch.float32).cpu()
            hdiag = sqrt_diag.pow(2)
        elif "s" in v:
            sqrt_diag = v["s"].to(torch.float32).cpu()
            hdiag = sqrt_diag.pow(2)
        elif "var" in v:
            hdiag = v["var"].to(torch.float32).cpu().clamp_min(0.0)
            sqrt_diag = torch.sqrt(hdiag.clamp_min(1e-12))
        elif "inv_s" in v:
            inv_s = v["inv_s"].to(torch.float32).cpu().clamp_min(1e-12)
            sqrt_diag = 1.0 / inv_s
            hdiag = sqrt_diag.pow(2)

        if sqrt_diag is None or hdiag is None:
            continue

        out[k] = {
            "sqrt": sqrt_diag.contiguous(),
            "hessian_diag": hdiag.contiguous(),
        }

    if not out:
        raise RuntimeError(f"No usable entries found in calib file: {calib_path}")
    return out

def lookup_calib_entry(
    calib_map: Dict[str, Dict[str, torch.Tensor]],
    full_weight_name: str,
) -> Optional[Dict[str, torch.Tensor]]:
    entry = calib_map.get(full_weight_name)
    if entry is None:
        entry = calib_map.get(module_name_from_weight(full_weight_name))
    return entry


# ============================================================
# Group helpers
# ============================================================
def _to_groups(W: torch.Tensor, group_size: int):
    O, I = W.shape
    pad = (group_size - (I % group_size)) % group_size
    if pad:
        W = F.pad(W, (0, pad))
    O2, Ipad = W.shape
    G = Ipad // group_size
    return W.view(O2, G, group_size), O2, G, group_size, I, pad

def _from_groups(Xg: torch.Tensor, orig_I: int) -> torch.Tensor:
    O, G, S = Xg.shape
    return Xg.reshape(O, G * S)[:, :orig_I]


# ============================================================
# Lloyd-Max
# ============================================================
@torch.no_grad()
def _kth_quantiles_lastdim(X_flat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    N, S = X_flat.shape
    probs = probs.clamp(0.0, 1.0).to(device=X_flat.device, dtype=torch.float32)
    ks = (probs * (S - 1)).round().to(torch.int64) + 1
    ks = ks.clamp(1, S).tolist()
    outs = [torch.kthvalue(X_flat, k, dim=1).values for k in ks]
    return torch.stack(outs, dim=1)

@torch.no_grad()
def _lloyd_centroid_update(
    x: torch.Tensor,
    idx: torch.Tensor,
    levels: int,
    prev_cb: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    new_cb = prev_cb.clone()

    if sample_weight is None:
        numer = torch.zeros_like(new_cb)
        denom = torch.zeros_like(new_cb)
        numer.scatter_add_(1, idx, x)
        denom.scatter_add_(1, idx, torch.ones_like(x))
        valid = denom > 0
        mean = numer / denom.clamp_min(1.0)
        return torch.where(valid, mean, new_cb)

    numer = torch.zeros_like(new_cb)
    denom = torch.zeros_like(new_cb)
    numer.scatter_add_(1, idx, x * sample_weight)
    denom.scatter_add_(1, idx, sample_weight)
    valid = denom > 0
    weighted_mean = numer / denom.clamp_min(1e-12)
    new_cb = torch.where(valid, weighted_mean, new_cb)

    fallback = ~valid
    if fallback.any():
        numer_plain = torch.zeros_like(new_cb)
        denom_plain = torch.zeros_like(new_cb)
        numer_plain.scatter_add_(1, idx, x)
        denom_plain.scatter_add_(1, idx, torch.ones_like(x))
        plain_mean = numer_plain / denom_plain.clamp_min(1.0)
        new_cb = torch.where(fallback, plain_mean, new_cb)

    return new_cb

@torch.no_grad()
def _lloyd_max_codebook_per_group(
    X_flat: torch.Tensor,
    levels: int,
    max_iter: int = 12,
    tol: float = 1e-4,
    chunk_groups: int = 4096,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    N, S = X_flat.shape
    cg = max(1, int(chunk_groups))
    codebook = torch.empty((N, levels), device=X_flat.device, dtype=X_flat.dtype)

    if sample_weight is not None:
        if sample_weight.shape != X_flat.shape:
            raise ValueError(f"sample_weight shape mismatch: {sample_weight.shape} vs {X_flat.shape}")
        sample_weight = sample_weight.to(device=X_flat.device, dtype=X_flat.dtype).clamp_min(0.0)

    for start in range(0, N, cg):
        end = min(start + cg, N)
        x = X_flat[start:end]
        w = sample_weight[start:end] if sample_weight is not None else None

        if levels == 1:
            if w is None:
                cb = x.mean(dim=1, keepdim=True)
            else:
                denom = w.sum(dim=1, keepdim=True)
                numer = (x * w).sum(dim=1, keepdim=True)
                cb = torch.where(denom > 0, numer / denom.clamp_min(1e-12), x.mean(dim=1, keepdim=True))
            codebook[start:end] = cb
            continue

        probs = (torch.arange(levels, device=x.device, dtype=torch.float32) + 0.5) / float(levels)
        cb = _kth_quantiles_lastdim(x, probs)
        cb, _ = torch.sort(cb, dim=1)

        for _ in range(max_iter):
            mid = (cb[:, :-1] + cb[:, 1:]) * 0.5
            idx = torch.sum(x.unsqueeze(-1) > mid.unsqueeze(1), dim=-1).to(torch.long)
            new_cb = _lloyd_centroid_update(x, idx, levels, cb, sample_weight=w)
            new_cb, _ = torch.sort(new_cb, dim=1)
            delta = (new_cb - cb).abs().amax()
            cb = new_cb
            if float(delta.item()) <= tol:
                break

        codebook[start:end] = cb

    return codebook

@torch.no_grad()
def _assign_codes_by_midpoints(
    X_flat: torch.Tensor,
    codebook: torch.Tensor,
    chunk_groups: int = 4096,
) -> torch.Tensor:
    N, S = X_flat.shape
    L = codebook.shape[1]
    out = torch.empty((N, S), device=X_flat.device, dtype=torch.int64)
    if L == 1:
        out.zero_()
        return out

    cg = max(1, int(chunk_groups))
    for start in range(0, N, cg):
        end = min(start + cg, N)
        x = X_flat[start:end]
        cb = codebook[start:end]
        mid = (cb[:, :-1] + cb[:, 1:]) * 0.5
        idx = torch.sum(x.unsqueeze(-1) > mid.unsqueeze(1), dim=-1).to(torch.long)
        out[start:end] = idx
    return out

@torch.no_grad()
def _dequant_from_codebook_and_codes(codebook: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.long:
        codes = codes.to(torch.long)
    return torch.gather(codebook, dim=1, index=codes)

@torch.no_grad()
def lloyd_asym_nonuniform_quantize(
    W: torch.Tensor,
    b: int,
    group_size: int,
    lloyd_iter: int = 12,
    chunk_groups: int = 4096,
    hessian_diag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    assert b in (1, 2, 3, 4)
    Wg, O, G, S, orig_I, pad = _to_groups(W, group_size)
    X_flat = Wg.reshape(-1, S)
    L = 1 << b

    H_flat = None
    if hessian_diag is not None:
        hdiag = hessian_diag.detach().to(device=W.device, dtype=torch.float32).flatten()
        if hdiag.numel() != orig_I:
            raise ValueError(f"hessian_diag mismatch: expected {orig_I}, got {hdiag.numel()}")
        if pad:
            hdiag = F.pad(hdiag, (0, pad), value=0.0)
        H_flat = hdiag.view(1, G, S).expand(O, -1, -1).reshape(-1, S).contiguous()

    cb_flat = _lloyd_max_codebook_per_group(
        X_flat=X_flat,
        levels=L,
        max_iter=lloyd_iter,
        tol=1e-4,
        chunk_groups=chunk_groups,
        sample_weight=H_flat,
    )
    codes_flat = _assign_codes_by_midpoints(X_flat, cb_flat, chunk_groups=chunk_groups)
    Xq_flat = _dequant_from_codebook_and_codes(cb_flat, codes_flat)
    Xq = Xq_flat.reshape(O, G, S)
    Wq = _from_groups(Xq, orig_I)

    meta = {
        "bits": int(b),
        "group_size": int(group_size),
        "levels": int(L),
        "orig_shape": (int(O), int(orig_I)),
        "uses_hessian_weighting": bool(hessian_diag is not None),
    }
    return Wq, meta


# ============================================================
# Model extraction
# ============================================================
@torch.no_grad()
def snapshot_selected_weights_to_cpu(
    model_id: str,
    revision: Optional[str],
    trust_remote_code: bool,
    load_dtype: torch.dtype,
    device_map: Optional[str],
    layer_regex: Optional[str],
    max_layers: int,
) -> Dict[str, torch.Tensor]:
    log(f"[load] model={model_id}, device_map={device_map}, dtype={load_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    layer_re = re.compile(layer_regex) if layer_regex else None
    selected: Dict[str, torch.Tensor] = {}

    for name, param in model.state_dict().items():
        if getattr(param, "is_meta", False):
            raise RuntimeError(f"meta tensor detected for {name}. Re-run with --device_map none or cpu.")
        if not is_target_weight(name, param):
            continue
        if layer_re and not layer_re.search(name):
            continue
        selected[name] = param.detach().to("cpu", dtype=torch.float32)
        if max_layers > 0 and len(selected) >= max_layers:
            break

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not selected:
        raise RuntimeError("No selected target layers found.")
    return selected


# ============================================================
# Linear algebra helpers
# ============================================================
@torch.no_grad()
def top_right_subspace(X: torch.Tensor, r: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    rr = min(r, Vh.shape[0])
    return Vh[:rr].transpose(0, 1).contiguous()

@torch.no_grad()
def alignment_score(Qa: torch.Tensor, Qb: torch.Tensor) -> float:
    rr = min(Qa.shape[1], Qb.shape[1])
    if rr <= 0:
        return 0.0
    M = Qa[:, :rr].transpose(0, 1) @ Qb[:, :rr]
    return float((M.pow(2).sum() / float(rr)).item())

@torch.no_grad()
def principal_cosines(Qa: torch.Tensor, Qb: torch.Tensor) -> torch.Tensor:
    rr = min(Qa.shape[1], Qb.shape[1])
    if rr <= 0:
        return torch.zeros(0, dtype=torch.float32, device=Qa.device)
    M = Qa[:, :rr].transpose(0, 1) @ Qb[:, :rr]
    return torch.linalg.svdvals(M)

@torch.no_grad()
def procrustes_rotation(Qsrc: torch.Tensor, Qtgt: torch.Tensor) -> torch.Tensor:
    rr = min(Qsrc.shape[1], Qtgt.shape[1])
    if rr <= 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=Qsrc.device)
    M = Qsrc[:, :rr].transpose(0, 1) @ Qtgt[:, :rr]
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh

@torch.no_grad()
def best_rank_r_svd_approx(X: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    rr = min(r, S.numel())
    Ur = U[:, :rr]
    Sr = S[:rr]
    Vr = Vh[:rr].transpose(0, 1)
    Xr = (Ur * Sr.unsqueeze(0)) @ Vr.transpose(0, 1)
    return Xr, Ur, Vr

@torch.no_grad()
def evr_at_r(X: torch.Tensor, r: int) -> float:
    s = torch.linalg.svdvals(X)
    if s.numel() == 0:
        return 0.0
    s2 = s.pow(2)
    rr = min(r, s2.numel())
    return float((s2[:rr].sum() / s2.sum().clamp_min(1e-12)).item())

@torch.no_grad()
def stable_rank(X: torch.Tensor) -> float:
    if X.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(X)
    if s.numel() == 0:
        return 0.0
    return float((X.norm(p="fro").pow(2) / s[0].pow(2).clamp_min(1e-12)).item())

@torch.no_grad()
def capture_ratio(X: torch.Tensor, Q: torch.Tensor) -> float:
    total = X.pow(2).sum().clamp_min(1e-12)
    proj = X @ Q
    return float((proj.pow(2).sum() / total).item())


# ============================================================
# Step1: initializer only
# ============================================================
@torch.no_grad()
def step1_initializer_from_procrustes(
    Ew: torch.Tensor,
    Ww: torch.Tensor,
    rank: int,
) -> Dict[str, torch.Tensor]:
    """
    Step1은 최종 해가 아니라 UV 초기화 생성용.
    - Qe: residual subspace
    - Qw: weight subspace
    - Qe를 Qw 방향으로 rotation
    - 그 basis를 사용해 UV init 생성
    """
    Qe = top_right_subspace(Ew, rank)
    Qw = top_right_subspace(Ww, rank)
    rr = min(Qe.shape[1], Qw.shape[1])

    Qe = Qe[:, :rr]
    Qw = Qw[:, :rr]

    R = procrustes_rotation(Qe, Qw)
    Qinit = Qe @ R

    # U init from least squares on fixed V=Qinit
    V0 = Qinit
    U0 = Ew @ V0

    Z0 = U0 @ V0.transpose(0, 1)

    return {
        "U0": U0,
        "V0": V0,
        "Qw": Qw,
        "Qe": Qe,
        "R": R,
        "Z0": Z0,
    }


# ============================================================
# Step2: direct U,V optimization
# ============================================================
def orthonormalize_columns(X: torch.Tensor) -> torch.Tensor:
    Q, R = torch.linalg.qr(X, mode="reduced")
    sign = torch.sign(torch.diag(R))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q * sign.unsqueeze(0)

def cosine_anchor_penalty(V: torch.Tensor, Qw: torch.Tensor) -> torch.Tensor:
    """
    1 - alignment score
    """
    rr = min(V.shape[1], Qw.shape[1])
    M = V[:, :rr].transpose(0, 1) @ Qw[:, :rr]
    align = (M.pow(2).sum() / float(rr)).clamp(0.0, 1.0)
    return 1.0 - align

def smooth_row_energy_penalty(U: torch.Tensor) -> torch.Tensor:
    """
    optional mild stabilization:
    large row-energy spikes 완화
    """
    row_e = U.pow(2).sum(dim=1)
    return row_e.var()

def optimize_uv_direct(
    Ew: torch.Tensor,
    U0: torch.Tensor,
    V0: torch.Tensor,
    Qw: torch.Tensor,
    num_steps: int = 300,
    lr: float = 1e-2,
    lam_u: float = 1e-5,
    lam_v: float = 1e-5,
    lam_anchor: float = 1e-2,
    lam_rowvar: float = 1e-5,
    reorth_every: int = 20,
) -> Dict[str, torch.Tensor]:
    """
    minimize:
      ||Ew - U V^T||_F^2
      + lam_u ||U||_F^2
      + lam_v ||V||_F^2
      + lam_anchor * (1 - align(V, Qw))
      + lam_rowvar * Var(row_energy(U))
    """
    with torch.enable_grad():
        U = U0.clone().detach().requires_grad_(True)
        V = V0.clone().detach().requires_grad_(True)

        opt = torch.optim.Adam([U, V], lr=lr)

        best = {
            "loss": float("inf"),
            "U": U0.clone(),
            "V": V0.clone(),
            "Z": U0 @ V0.transpose(0, 1),
        }

        for step in range(num_steps):
            opt.zero_grad(set_to_none=True)

            Z = U @ V.transpose(0, 1)
            rec = (Ew - Z).pow(2).sum()
            reg_u = U.pow(2).sum()
            reg_v = V.pow(2).sum()
            pen_anchor = cosine_anchor_penalty(V, Qw)
            pen_rowvar = smooth_row_energy_penalty(U)

            loss = rec + lam_u * reg_u + lam_v * reg_v + lam_anchor * pen_anchor + lam_rowvar * pen_rowvar
            loss.backward()
            opt.step()

            if reorth_every > 0 and ((step + 1) % reorth_every == 0):
                with torch.no_grad():
                    V.copy_(orthonormalize_columns(V))
                    U.copy_(Ew @ V)

            cur = float(loss.item())
            if cur < best["loss"]:
                with torch.no_grad():
                    Vbest = orthonormalize_columns(V.detach())
                    Ubest = Ew @ Vbest
                    Zbest = Ubest @ Vbest.transpose(0, 1)
                best = {
                    "loss": cur,
                    "U": Ubest.clone(),
                    "V": Vbest.clone(),
                    "Z": Zbest.clone(),
                }

    return best


# ============================================================
# Step3: IRLS-like refinement on Step2 result
# ============================================================
@torch.no_grad()
def refine_uv_with_schatten_irls(
    Ew: torch.Tensor,
    Z_init: torch.Tensor,
    rank: int,
    p: float = 0.5,
    lam: float = 1e-2,
    eps: float = 1e-6,
    iters: int = 12,
    mix_with_data: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    spectral refinement:
      min_Z  0.5||Ew-Z||_F^2 + lam ||Z||_{S_p}^p

    heuristic IRLS-like update.
    """
    Z = Z_init.clone()
    m, n = Ew.shape
    rr = min(rank, m, n)

    for _ in range(iters):
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        U = U[:, :rr]
        S = S[:rr]
        V = Vh[:rr].transpose(0, 1)

        d = (S.pow(2) + eps).pow((p * 0.5) - 1.0)
        alpha = torch.sum(U * (Ew @ V), dim=0)

        s_new = (mix_with_data * alpha) / (1.0 + lam * d)
        Z = (U * s_new.unsqueeze(0)) @ V.transpose(0, 1)

    U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
    rr = min(rank, S.numel())
    U = U[:, :rr]
    S = S[:rr]
    V = Vh[:rr].transpose(0, 1)
    Z = (U * S.unsqueeze(0)) @ V.transpose(0, 1)

    return {
        "U": U * torch.sqrt(S.unsqueeze(0)),
        "V": V * torch.sqrt(S.unsqueeze(0)),
        "Z": Z,
    }


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def evaluate_candidate(
    name: str,
    Ew0: torch.Tensor,
    Ww: torch.Tensor,
    Zw: torch.Tensor,
    rank: int,
) -> Dict[str, float]:
    Rw = Ew0 - Zw
    Qw = top_right_subspace(Ww, rank)
    Qr = top_right_subspace(Rw, rank)

    out = {
        "method": name,
        "weighted_residual_fro": float(torch.norm(Rw, p="fro").item()),
        "weighted_surrogate_loss": float(Rw.pow(2).sum().item()),
        "remaining_stable_rank": stable_rank(Rw),
        "remaining_evr@r": evr_at_r(Rw, rank),
        "capture_by_Wsubspace@r": capture_ratio(Rw, Qw),
        "align_to_W@r": alignment_score(Qr, Qw),
    }

    cosines = principal_cosines(Qr, Qw)
    out["mean_principal_cos@r"] = float(cosines.mean().item()) if cosines.numel() > 0 else 0.0
    out["min_principal_cos@r"] = float(cosines.min().item()) if cosines.numel() > 0 else 0.0
    return out


# ============================================================
# Plot
# ============================================================
def save_method_compare_plot(df: pd.DataFrame, metric: str, out_path: str, title: str):
    pivot = df.pivot(index="label", columns="method", values=metric)
    methods = list(pivot.columns)
    labels = list(pivot.index)

    plt.figure(figsize=(max(9, len(labels) * 0.7), 5.0))
    xs = list(range(len(labels)))
    width = 0.8 / max(1, len(methods))

    for i, m in enumerate(methods):
        offs = [x - 0.4 + width * (i + 0.5) for x in xs]
        plt.bar(offs, pivot[m].tolist(), width=width, label=m)

    plt.xticks(xs, labels, rotation=45, ha="right")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# Main per-layer pipeline
# ============================================================
@torch.no_grad()
def run_pipeline_on_layer(
    W: torch.Tensor,
    sqrt_diag: torch.Tensor,
    hdiag: torch.Tensor,
    bits: int,
    group_size: int,
    lloyd_iter: int,
    chunk_groups: int,
    rank: int,
    quant_mode: str,
    uv_steps: int,
    uv_lr: float,
    lam_u: float,
    lam_v: float,
    lam_anchor: float,
    lam_rowvar: float,
    reorth_every: int,
    irls_p: float,
    irls_lam: float,
    irls_iters: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    Ww = W * sqrt_diag.unsqueeze(0)

    quant_cfgs = []
    if quant_mode in ("plain", "both"):
        quant_cfgs.append(("plain_quant", None))
    if quant_mode in ("hessian", "both"):
        quant_cfgs.append(("hessian_quant", hdiag))

    for quant_name, qhdiag in quant_cfgs:
        Wq, _ = lloyd_asym_nonuniform_quantize(
            W=W,
            b=bits,
            group_size=group_size,
            lloyd_iter=lloyd_iter,
            chunk_groups=chunk_groups,
            hessian_diag=qhdiag,
        )

        Ew0 = (W - Wq) * sqrt_diag.unsqueeze(0)

        # 0) quant only
        rows.append(evaluate_candidate(
            name=f"{quant_name}/quant_only",
            Ew0=Ew0,
            Ww=Ww,
            Zw=torch.zeros_like(Ew0),
            rank=rank,
        ))

        # 1) baseline weighted SVD
        Zsvd, _, _ = best_rank_r_svd_approx(Ew0, rank)
        rows.append(evaluate_candidate(
            name=f"{quant_name}/baseline_weighted_svd",
            Ew0=Ew0,
            Ww=Ww,
            Zw=Zsvd,
            rank=rank,
        ))

        # 2) Step1 initializer only
        s1 = step1_initializer_from_procrustes(Ew0, Ww, rank)
        rows.append(evaluate_candidate(
            name=f"{quant_name}/step1_init_only",
            Ew0=Ew0,
            Ww=Ww,
            Zw=s1["Z0"],
            rank=rank,
        ))

        # 3) Step2 UV direct optimization
        s2 = optimize_uv_direct(
            Ew=Ew0,
            U0=s1["U0"],
            V0=s1["V0"],
            Qw=s1["Qw"],
            num_steps=uv_steps,
            lr=uv_lr,
            lam_u=lam_u,
            lam_v=lam_v,
            lam_anchor=lam_anchor,
            lam_rowvar=lam_rowvar,
            reorth_every=reorth_every,
        )
        rows.append(evaluate_candidate(
            name=f"{quant_name}/step2_uv_opt",
            Ew0=Ew0,
            Ww=Ww,
            Zw=s2["Z"],
            rank=rank,
        ))

        # 4) Step3 IRLS refine
        s3 = refine_uv_with_schatten_irls(
            Ew=Ew0,
            Z_init=s2["Z"],
            rank=rank,
            p=irls_p,
            lam=irls_lam,
            iters=irls_iters,
        )
        rows.append(evaluate_candidate(
            name=f"{quant_name}/step3_uv_irls_refine",
            Ew0=Ew0,
            Ww=Ww,
            Zw=s3["Z"],
            rank=rank,
        ))

    return rows


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("Verify improved three-stage UV optimization on selected layers")

    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--calib_s_path", required=True)
    ap.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--quant_mode", type=str, default="both", choices=["plain", "hessian", "both"])

    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=64)

    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)

    ap.add_argument("--uv_steps", type=int, default=300)
    ap.add_argument("--uv_lr", type=float, default=1e-2)
    ap.add_argument("--lam_u", type=float, default=1e-5)
    ap.add_argument("--lam_v", type=float, default=1e-5)
    ap.add_argument("--lam_anchor", type=float, default=1e-2)
    ap.add_argument("--lam_rowvar", type=float, default=1e-5)
    ap.add_argument("--reorth_every", type=int, default=20)

    ap.add_argument("--irls_p", type=float, default=0.5)
    ap.add_argument("--irls_lam", type=float, default=1e-2)
    ap.add_argument("--irls_iters", type=int, default=12)

    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--device_map", type=str, default="cpu")
    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)

    if args.dtype == "bf16":
        load_dtype = torch.bfloat16
    elif args.dtype in ("fp16", "float16"):
        load_dtype = torch.float16
    elif args.dtype in ("fp32", "float32"):
        load_dtype = torch.float32
    else:
        load_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    dm_raw = str(args.device_map).strip().lower()
    resolved_device_map = None if dm_raw in {"", "none", "null"} else args.device_map

    calib_map = load_calib_map(args.calib_s_path)
    log(f"[load] calib entries: {len(calib_map)}")

    weights = snapshot_selected_weights_to_cpu(
        model_id=args.model_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        load_dtype=load_dtype,
        device_map=resolved_device_map,
        layer_regex=args.layer_regex,
        max_layers=args.max_layers,
    )
    log(f"[select] selected layers: {len(weights)}")

    all_rows = []
    selected_names = sorted(weights.keys())

    for idx, wkey in enumerate(selected_names, 1):
        t0 = time.time()
        log(f"[layer] {idx}/{len(selected_names)} {wkey}")

        entry = lookup_calib_entry(calib_map, wkey)
        if entry is None:
            log(f"[warn] missing calib entry -> skip {wkey}")
            continue

        W = weights[wkey].to(device=device, dtype=torch.float32)
        sqrt_diag = entry["sqrt"].to(device=device, dtype=torch.float32)
        hdiag = entry["hessian_diag"].to(device=device, dtype=torch.float32)

        if sqrt_diag.numel() != W.shape[1]:
            log(f"[warn] sqrt mismatch -> skip {wkey}")
            continue
        if hdiag.numel() != W.shape[1]:
            log(f"[warn] hdiag mismatch -> skip {wkey}")
            continue

        layer_rows = run_pipeline_on_layer(
            W=W,
            sqrt_diag=sqrt_diag,
            hdiag=hdiag,
            bits=args.bits,
            group_size=args.group_size,
            lloyd_iter=args.lloyd_iter,
            chunk_groups=args.chunk_groups,
            rank=args.rank,
            quant_mode=args.quant_mode,
            uv_steps=args.uv_steps,
            uv_lr=args.uv_lr,
            lam_u=args.lam_u,
            lam_v=args.lam_v,
            lam_anchor=args.lam_anchor,
            lam_rowvar=args.lam_rowvar,
            reorth_every=args.reorth_every,
            irls_p=args.irls_p,
            irls_lam=args.irls_lam,
            irls_iters=args.irls_iters,
        )

        short_label = wkey.replace(".weight", "").replace("model.layers.", "L")
        for row in layer_rows:
            row["weight_key"] = wkey
            row["label"] = short_label
            row["bits"] = int(args.bits)
            row["rank"] = int(args.rank)
            row["group_size"] = int(args.group_size)

        all_rows.extend(layer_rows)

        del W
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log(f"[done] {wkey} in {time.time()-t0:.2f}s")

    if not all_rows:
        raise RuntimeError("No valid rows were produced.")

    df = pd.DataFrame(all_rows)
    per_layer_csv = os.path.join(args.out_dir, "uv_three_stage_per_layer_metrics.csv")
    df.to_csv(per_layer_csv, index=False)

    summary_rows = []
    for method, sub in df.groupby("method"):
        summary_rows.append({
            "method": method,
            "num_rows": int(len(sub)),
            "mean_weighted_residual_fro": float(sub["weighted_residual_fro"].mean()),
            "mean_weighted_surrogate_loss": float(sub["weighted_surrogate_loss"].mean()),
            "mean_remaining_stable_rank": float(sub["remaining_stable_rank"].mean()),
            "mean_remaining_evr@r": float(sub["remaining_evr@r"].mean()),
            "mean_capture_by_Wsubspace@r": float(sub["capture_by_Wsubspace@r"].mean()),
            "mean_align_to_W@r": float(sub["align_to_W@r"].mean()),
            "mean_mean_principal_cos@r": float(sub["mean_principal_cos@r"].mean()),
            "mean_min_principal_cos@r": float(sub["min_principal_cos@r"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("method")
    summary_csv = os.path.join(args.out_dir, "uv_three_stage_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    best_df = df.sort_values(["label", "weighted_surrogate_loss"]).groupby("label", as_index=False).first()
    best_csv = os.path.join(args.out_dir, "uv_best_method_per_layer.csv")
    best_df.to_csv(best_csv, index=False)

    for metric in [
        "weighted_residual_fro",
        "weighted_surrogate_loss",
        "remaining_stable_rank",
        "capture_by_Wsubspace@r",
        "align_to_W@r",
    ]:
        save_method_compare_plot(
            df=df,
            metric=metric,
            out_path=os.path.join(args.out_dir, f"{metric}_compare.png"),
            title=f"{metric} by method",
        )

    result_json = {
        "num_layers": int(df["label"].nunique()),
        "rank": int(args.rank),
        "bits": int(args.bits),
        "group_size": int(args.group_size),
        "quant_mode": args.quant_mode,
    }
    for _, row in summary_df.iterrows():
        m = row["method"]
        result_json[m] = {
            "mean_weighted_residual_fro": float(row["mean_weighted_residual_fro"]),
            "mean_weighted_surrogate_loss": float(row["mean_weighted_surrogate_loss"]),
            "mean_remaining_stable_rank": float(row["mean_remaining_stable_rank"]),
            "mean_capture_by_Wsubspace@r": float(row["mean_capture_by_Wsubspace@r"]),
            "mean_align_to_W@r": float(row["mean_align_to_W@r"]),
        }

    with open(os.path.join(args.out_dir, "uv_three_stage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    log("\n[done] saved:")
    log(f"  - {per_layer_csv}")
    log(f"  - {summary_csv}")
    log(f"  - {best_csv}")
    log(f"  - {os.path.join(args.out_dir, 'uv_three_stage_summary.json')}")

    log("\n[reading guide]")
    log("1) 가장 중요한 지표는 mean_weighted_surrogate_loss")
    log("2) baseline_weighted_svd는 surrogate 최적해라 매우 강함")
    log("3) step2_uv_opt가 step1_init_only보다 좋아져야 initializer가 의미 있음")
    log("4) step3_uv_irls_refine가 step2_uv_opt보다 좋아져야 refinement가 의미 있음")
    log("5) baseline을 못 이겨도, 이후 output-MSE/PPL 연결에서 유리한지 추가 검증 가능")


if __name__ == "__main__":
    main()
