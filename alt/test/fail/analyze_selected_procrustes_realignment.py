#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_selected_procrustes_realignment.py

목적:
  - calib sqrt만 있고, 미리 뽑아둔 quantized/residual이 없다고 가정
  - 선택한 몇 개 레이어를 원본 모델에서 직접 읽어
    plain Lloyd-Max vs Hessian-weighted Lloyd-Max 를 수행
  - residual weighted matrix E_w = (W - Wq) * sqrt_diag 에 대해
      (1) 원본 W_w = W * sqrt_diag 와의 subspace alignment 측정
      (2) Procrustes closed-form 정렬 가능성 검증
      (3) 정렬 전/후 reference-subspace capture 비교
  - "공간 재배치" 아이디어가 최소한 subspace mismatch 관점에서 의미 있는지 확인

핵심 아이디어:
  - Vw_r : W_w 의 top-r right singular subspace
  - Ve_r : E_w 의 top-r right singular subspace
  - alignment score = || Ve^T Vw ||_F^2 / r
  - Procrustes R* = argmin || Ve R - Vw ||_F
      with closed form R*=UV^T from Ve^T Vw = UΣV^T
  - rotation 자체는 singular value를 바꾸지 않으므로,
    여기서는 "subspace mismatch를 줄이는가"와
    "Vw-subspace로 투영했을 때 capture가 좋아지는가"를 본다.

예시:
CUDA_VISIBLE_DEVICES=2 python test/analyze_selected_procrustes_realignment.py \
  --model_id meta-llama/Llama-3.1-8B \
  --calib_s_path ./output/llama3_8b/calib_sqrtdiag.pt \
  --bits 2 \
  --group_size 128 \
  --layer_regex "model\\.layers\\.(0|4|8|12)\\.(self_attn\\.q_proj|mlp\\.down_proj)\\.weight" \
  --rank_list 16,32,64,128 \
  --max_layers 8 \
  --device cuda \
  --device_map cpu \
  --out_dir ./analysis_selected_procrustes
  
CUDA_VISIBLE_DEVICES=2 python test/analyze_selected_procrustes_realignment.py \
  --model_id meta-llama/Llama-3.1-8B \
  --calib_s_path ./output/llama3_8b/calib_sqrtdiag.pt \
  --bits 2 \
  --group_size 128 \
  --layer_regex "model\.layers\.(0|1)\.self_attn\.q_proj\.weight" \
  --rank_list 16,32,64 \
  --max_layers 2 \
  --device cuda \
  --device_map cpu \
  --out_dir ./analysis_qproj_small
"""

import os
import re
import gc
import json
import math
import time
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
    return full_weight_name[: -len(".weight")]


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
# Model weight extraction
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
def parse_rank_list(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    vals = sorted(set(v for v in vals if v > 0))
    if not vals:
        raise ValueError("rank_list must contain positive integers")
    return vals

@torch.no_grad()
def top_right_subspace(X: torch.Tensor, r: int, device: torch.device) -> torch.Tensor:
    """
    returns V_r in R^{n x r}, orthonormal columns
    """
    X = X.to(device=device, dtype=torch.float32)
    try:
        Vh = torch.linalg.svd(X, full_matrices=False).Vh
    except RuntimeError:
        Vh = torch.linalg.svd(X.cpu(), full_matrices=False).Vh
    rr = min(r, Vh.shape[0])
    return Vh[:rr].transpose(0, 1).to(torch.float32).cpu()

@torch.no_grad()
def alignment_score(Qe: torch.Tensor, Qw: torch.Tensor) -> float:
    """
    Qe: [n,r], Qw: [n,r]
    score = ||Qe^T Qw||_F^2 / r  in [0,1]
    """
    rr = min(Qe.shape[1], Qw.shape[1])
    if rr <= 0:
        return 0.0
    M = Qe[:, :rr].transpose(0, 1) @ Qw[:, :rr]
    return float((M.pow(2).sum() / float(rr)).item())

@torch.no_grad()
def principal_cosines(Qe: torch.Tensor, Qw: torch.Tensor) -> torch.Tensor:
    """
    singular values of Qe^T Qw = cos(theta_i)
    """
    rr = min(Qe.shape[1], Qw.shape[1])
    if rr <= 0:
        return torch.zeros(0, dtype=torch.float32)
    M = Qe[:, :rr].transpose(0, 1) @ Qw[:, :rr]
    s = torch.linalg.svdvals(M).to(torch.float32).cpu()
    return s

@torch.no_grad()
def procrustes_rotation(Qe: torch.Tensor, Qw: torch.Tensor) -> torch.Tensor:
    """
    R* = argmin ||Qe R - Qw||_F, with R orthogonal
    closed form from SVD(Qe^T Qw)=UΣV^T -> R=UV^T
    """
    rr = min(Qe.shape[1], Qw.shape[1])
    if rr <= 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    M = Qe[:, :rr].transpose(0, 1) @ Qw[:, :rr]
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh
    return R.to(torch.float32).cpu()

@torch.no_grad()
def reference_capture_ratio(X: torch.Tensor, Qref: torch.Tensor) -> float:
    """
    X total energy 중에서 reference subspace Qref 가 캡처하는 비율:
      || X Qref ||_F^2 / ||X||_F^2
    X: [m,n], Qref: [n,r]
    """
    total = X.pow(2).sum().item()
    if total <= 0:
        return 0.0
    proj = X @ Qref
    cap = proj.pow(2).sum().item()
    return float(cap / max(total, 1e-12))

@torch.no_grad()
def best_rank_evr(X: torch.Tensor, r: int) -> Tuple[float, float]:
    """
    returns:
      evr@r
      best rank-r residual fro
    """
    S = torch.linalg.svdvals(X).to(torch.float32).cpu()
    if S.numel() == 0:
        return 0.0, 0.0
    s2 = S.pow(2)
    total = float(s2.sum().item())
    rr = min(r, S.numel())
    captured = float(s2[:rr].sum().item())
    evr = captured / max(total, 1e-12)
    res = math.sqrt(max(total - captured, 0.0))
    return float(evr), float(res)


# ============================================================
# Per-layer analysis
# ============================================================
@torch.no_grad()
def analyze_one_residual_against_W(
    W: torch.Tensor,
    Wq: torch.Tensor,
    sqrt_diag: torch.Tensor,
    ranks: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """
    W_w = W * sqrt_diag
    E_w = (W-Wq) * sqrt_diag

    For each r:
      - best-rank EVR of E_w
      - alignment score between top-r subspaces of W_w and E_w
      - principal cosine stats
      - Procrustes-improved basis mismatch
      - reference-subspace capture:
          capture_by_Wsubspace  = ||E_w Qw||^2 / ||E_w||^2
          capture_by_Esubspace  = ||E_w Qe||^2 / ||E_w||^2 = EVR@r
          after Procrustes basis rotation in latent coords, mismatch only
            (energy itself unchanged, but basis discrepancy can reduce)
    """
    W = W.to(device=device, dtype=torch.float32)
    Wq = Wq.to(device=device, dtype=torch.float32)
    sqrt_diag = sqrt_diag.to(device=device, dtype=torch.float32)

    E = W - Wq
    Ww = (W * sqrt_diag.unsqueeze(0)).cpu()
    Ew = (E * sqrt_diag.unsqueeze(0)).cpu()

    out: Dict[str, float] = {}
    out["fro_weighted"] = float(torch.norm(Ew, p="fro").item())
    out["stable_rank"] = float(
        (torch.norm(Ew, p="fro").pow(2) / torch.linalg.svdvals(Ew)[0].pow(2).clamp_min(1e-12)).item()
    ) if Ew.numel() > 0 else 0.0

    for r in ranks:
        Qw = top_right_subspace(Ww, r, device=device)   # [n,r]
        Qe = top_right_subspace(Ew, r, device=device)   # [n,r]
        rr = min(Qw.shape[1], Qe.shape[1])

        evr_r, best_res_r = best_rank_evr(Ew, r)
        align_before = alignment_score(Qe, Qw)

        cosines = principal_cosines(Qe, Qw)
        mean_cos = float(cosines.mean().item()) if cosines.numel() > 0 else 0.0
        min_cos = float(cosines.min().item()) if cosines.numel() > 0 else 0.0

        # Procrustes in basis coordinates
        R = procrustes_rotation(Qe, Qw)
        if rr > 0:
            Qe_rot = Qe[:, :rr] @ R
            align_after = alignment_score(Qe_rot, Qw[:, :rr])

            # basis mismatch before/after
            mismatch_before = float(torch.norm(Qe[:, :rr] - Qw[:, :rr], p="fro").item())
            mismatch_after = float(torch.norm(Qe_rot - Qw[:, :rr], p="fro").item())
        else:
            align_after = align_before
            mismatch_before = 0.0
            mismatch_after = 0.0

        # Reference-subspace capture:
        # energy captured if we use W-subspace vs E-subspace
        cap_by_Wsub = reference_capture_ratio(Ew, Qw[:, :rr] if rr > 0 else Qw)
        cap_by_Esub = reference_capture_ratio(Ew, Qe[:, :rr] if rr > 0 else Qe)  # should be ~= EVR
        cap_gap = cap_by_Esub - cap_by_Wsub

        out[f"evr@{r}"] = float(evr_r)
        out[f"best_rank{r}_residual_fro"] = float(best_res_r)

        out[f"align_to_W@{r}"] = float(align_before)
        out[f"align_to_W_after_proc@{r}"] = float(align_after)
        out[f"align_gain_proc@{r}"] = float(align_after - align_before)

        out[f"mean_principal_cos@{r}"] = float(mean_cos)
        out[f"min_principal_cos@{r}"] = float(min_cos)

        out[f"basis_mismatch_before@{r}"] = float(mismatch_before)
        out[f"basis_mismatch_after@{r}"] = float(mismatch_after)
        out[f"basis_mismatch_gain@{r}"] = float(mismatch_before - mismatch_after)

        out[f"capture_by_Wsubspace@{r}"] = float(cap_by_Wsub)
        out[f"capture_by_Esubspace@{r}"] = float(cap_by_Esub)
        out[f"capture_gap_EminusW@{r}"] = float(cap_gap)

    return out


# ============================================================
# Plot
# ============================================================
def scatter_plot(x, y, xlabel, ylabel, title, out_path, diagonal=True):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=18, alpha=0.75)
    if diagonal and len(x) > 0 and len(y) > 0:
        mn = min(min(x), min(y))
        mx = max(max(x), max(y))
        plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def bar_compare(names, a, b, label_a, label_b, title, out_path):
    plt.figure(figsize=(max(8, len(names) * 0.7), 4.8))
    xs = list(range(len(names)))
    width = 0.38
    plt.bar([x - width/2 for x in xs], a, width=width, label=label_a)
    plt.bar([x + width/2 for x in xs], b, width=width, label=label_b)
    plt.xticks(xs, names, rotation=45, ha="right")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("Selected-layer Procrustes realignment analysis")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--calib_s_path", required=True)
    ap.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--group_size", type=int, default=128)

    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=8)

    ap.add_argument("--rank_list", type=str, default="16,32,64,128")
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)

    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"])
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--device_map", type=str, default="cpu",
                    help='recommend "cpu" or "none" for simple state_dict extraction')

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = resolve_device(args.device)
    ranks = parse_rank_list(args.rank_list)

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

    rows = []
    selected_names = sorted(weights.keys())

    for idx, wkey in enumerate(selected_names, 1):
        t0 = time.time()
        mod = module_name_from_weight(wkey)
        log(f"[analyze] {idx}/{len(selected_names)} {wkey}")

        W_cpu = weights[wkey]
        entry = lookup_calib_entry(calib_map, wkey)
        if entry is None:
            log(f"[warn] missing calib entry -> skip {wkey}")
            continue

        sqrt_diag = entry["sqrt"]
        hdiag = entry["hessian_diag"]

        if sqrt_diag.numel() != W_cpu.shape[1]:
            log(f"[warn] sqrt dim mismatch -> skip {wkey}: W={tuple(W_cpu.shape)}, sqrt={tuple(sqrt_diag.shape)}")
            continue
        if hdiag.numel() != W_cpu.shape[1]:
            log(f"[warn] hdiag dim mismatch -> skip {wkey}: W={tuple(W_cpu.shape)}, hdiag={tuple(hdiag.shape)}")
            continue

        W = W_cpu.to(device=device, dtype=torch.float32)

        # plain Lloyd
        Wq_plain, _ = lloyd_asym_nonuniform_quantize(
            W=W,
            b=int(args.bits),
            group_size=int(args.group_size),
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
            hessian_diag=None,
        )

        # hessian Lloyd
        Wq_hess, _ = lloyd_asym_nonuniform_quantize(
            W=W,
            b=int(args.bits),
            group_size=int(args.group_size),
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
            hessian_diag=hdiag.to(device=device, dtype=torch.float32),
        )

        plain_metrics = analyze_one_residual_against_W(
            W=W,
            Wq=Wq_plain,
            sqrt_diag=sqrt_diag,
            ranks=ranks,
            device=device,
        )
        hess_metrics = analyze_one_residual_against_W(
            W=W,
            Wq=Wq_hess,
            sqrt_diag=sqrt_diag,
            ranks=ranks,
            device=device,
        )

        row = {
            "weight_key": wkey,
            "module": mod,
            "out_features": int(W.shape[0]),
            "in_features": int(W.shape[1]),
            "numel": int(W.numel()),
            "bits": int(args.bits),
            "group_size": int(args.group_size),

            "plain_fro_weighted": plain_metrics["fro_weighted"],
            "hess_fro_weighted": hess_metrics["fro_weighted"],
            "fro_weighted_ratio_hess_over_plain": (
                hess_metrics["fro_weighted"] / max(plain_metrics["fro_weighted"], 1e-12)
            ),

            "plain_stable_rank": plain_metrics["stable_rank"],
            "hess_stable_rank": hess_metrics["stable_rank"],
        }

        for r in ranks:
            # low-rankness
            row[f"plain_evr@{r}"] = plain_metrics[f"evr@{r}"]
            row[f"hess_evr@{r}"] = hess_metrics[f"evr@{r}"]
            row[f"evr_gap_hess_minus_plain@{r}"] = (
                hess_metrics[f"evr@{r}"] - plain_metrics[f"evr@{r}"]
            )

            row[f"plain_best_rank{r}_residual_fro"] = plain_metrics[f"best_rank{r}_residual_fro"]
            row[f"hess_best_rank{r}_residual_fro"] = hess_metrics[f"best_rank{r}_residual_fro"]

            # alignment
            row[f"plain_align_to_W@{r}"] = plain_metrics[f"align_to_W@{r}"]
            row[f"hess_align_to_W@{r}"] = hess_metrics[f"align_to_W@{r}"]
            row[f"align_gap_hess_minus_plain@{r}"] = (
                hess_metrics[f"align_to_W@{r}"] - plain_metrics[f"align_to_W@{r}"]
            )

            # after Procrustes (basis mismatch reduction)
            row[f"plain_align_to_W_after_proc@{r}"] = plain_metrics[f"align_to_W_after_proc@{r}"]
            row[f"hess_align_to_W_after_proc@{r}"] = hess_metrics[f"align_to_W_after_proc@{r}"]

            row[f"plain_align_gain_proc@{r}"] = plain_metrics[f"align_gain_proc@{r}"]
            row[f"hess_align_gain_proc@{r}"] = hess_metrics[f"align_gain_proc@{r}"]

            row[f"plain_basis_mismatch_before@{r}"] = plain_metrics[f"basis_mismatch_before@{r}"]
            row[f"hess_basis_mismatch_before@{r}"] = hess_metrics[f"basis_mismatch_before@{r}"]
            row[f"plain_basis_mismatch_after@{r}"] = plain_metrics[f"basis_mismatch_after@{r}"]
            row[f"hess_basis_mismatch_after@{r}"] = hess_metrics[f"basis_mismatch_after@{r}"]
            row[f"plain_basis_mismatch_gain@{r}"] = plain_metrics[f"basis_mismatch_gain@{r}"]
            row[f"hess_basis_mismatch_gain@{r}"] = hess_metrics[f"basis_mismatch_gain@{r}"]

            # principal cosines
            row[f"plain_mean_principal_cos@{r}"] = plain_metrics[f"mean_principal_cos@{r}"]
            row[f"hess_mean_principal_cos@{r}"] = hess_metrics[f"mean_principal_cos@{r}"]

            # capture by W-subspace vs E-subspace
            row[f"plain_capture_by_Wsubspace@{r}"] = plain_metrics[f"capture_by_Wsubspace@{r}"]
            row[f"hess_capture_by_Wsubspace@{r}"] = hess_metrics[f"capture_by_Wsubspace@{r}"]
            row[f"plain_capture_by_Esubspace@{r}"] = plain_metrics[f"capture_by_Esubspace@{r}"]
            row[f"hess_capture_by_Esubspace@{r}"] = hess_metrics[f"capture_by_Esubspace@{r}"]

            row[f"plain_capture_gap_EminusW@{r}"] = plain_metrics[f"capture_gap_EminusW@{r}"]
            row[f"hess_capture_gap_EminusW@{r}"] = hess_metrics[f"capture_gap_EminusW@{r}"]

        rows.append(row)

        del W, Wq_plain, Wq_hess
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[done] {wkey} in {time.time()-t0:.2f}s")

    if not rows:
        raise RuntimeError("No rows analyzed.")

    df = pd.DataFrame(rows)
    per_layer_csv = os.path.join(args.out_dir, "per_layer_metrics.csv")
    df.to_csv(per_layer_csv, index=False)

    summary = {
        "num_layers": int(len(df)),
        "selected_layers": selected_names,
        "mean_plain_fro_weighted": float(df["plain_fro_weighted"].mean()),
        "mean_hess_fro_weighted": float(df["hess_fro_weighted"].mean()),
        "mean_fro_weighted_ratio_hess_over_plain": float(df["fro_weighted_ratio_hess_over_plain"].mean()),
        "mean_plain_stable_rank": float(df["plain_stable_rank"].mean()),
        "mean_hess_stable_rank": float(df["hess_stable_rank"].mean()),
    }

    for r in ranks:
        summary[f"mean_plain_evr@{r}"] = float(df[f"plain_evr@{r}"].mean())
        summary[f"mean_hess_evr@{r}"] = float(df[f"hess_evr@{r}"].mean())
        summary[f"mean_evr_gap_hess_minus_plain@{r}"] = float(df[f"evr_gap_hess_minus_plain@{r}"].mean())

        summary[f"mean_plain_align_to_W@{r}"] = float(df[f"plain_align_to_W@{r}"].mean())
        summary[f"mean_hess_align_to_W@{r}"] = float(df[f"hess_align_to_W@{r}"].mean())
        summary[f"mean_align_gap_hess_minus_plain@{r}"] = float(df[f"align_gap_hess_minus_plain@{r}"].mean())

        summary[f"mean_plain_basis_mismatch_gain@{r}"] = float(df[f"plain_basis_mismatch_gain@{r}"].mean())
        summary[f"mean_hess_basis_mismatch_gain@{r}"] = float(df[f"hess_basis_mismatch_gain@{r}"].mean())

        summary[f"mean_plain_capture_by_Wsubspace@{r}"] = float(df[f"plain_capture_by_Wsubspace@{r}"].mean())
        summary[f"mean_hess_capture_by_Wsubspace@{r}"] = float(df[f"hess_capture_by_Wsubspace@{r}"].mean())
        summary[f"mean_plain_capture_gap_EminusW@{r}"] = float(df[f"plain_capture_gap_EminusW@{r}"].mean())
        summary[f"mean_hess_capture_gap_EminusW@{r}"] = float(df[f"hess_capture_gap_EminusW@{r}"].mean())

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # plots
    for r in ranks:
        scatter_plot(
            df[f"plain_evr@{r}"].tolist(),
            df[f"hess_evr@{r}"].tolist(),
            xlabel=f"plain EVR@{r}",
            ylabel=f"hess EVR@{r}",
            title=f"EVR@{r}: plain vs hessian",
            out_path=os.path.join(args.out_dir, f"evr_scatter_rank{r}.png"),
        )

        scatter_plot(
            df[f"plain_align_to_W@{r}"].tolist(),
            df[f"hess_align_to_W@{r}"].tolist(),
            xlabel=f"plain align-to-W@{r}",
            ylabel=f"hess align-to-W@{r}",
            title=f"Alignment to W-subspace @ {r}",
            out_path=os.path.join(args.out_dir, f"align_scatter_rank{r}.png"),
        )

        scatter_plot(
            df[f"plain_capture_by_Wsubspace@{r}"].tolist(),
            df[f"hess_capture_by_Wsubspace@{r}"].tolist(),
            xlabel=f"plain capture by W-subspace @ {r}",
            ylabel=f"hess capture by W-subspace @ {r}",
            title=f"Residual energy captured by W-subspace @ {r}",
            out_path=os.path.join(args.out_dir, f"captureW_scatter_rank{r}.png"),
        )

        scatter_plot(
            df[f"plain_basis_mismatch_gain@{r}"].tolist(),
            df[f"hess_basis_mismatch_gain@{r}"].tolist(),
            xlabel=f"plain Procrustes mismatch gain @ {r}",
            ylabel=f"hess Procrustes mismatch gain @ {r}",
            title=f"Procrustes mismatch reduction @ {r}",
            out_path=os.path.join(args.out_dir, f"proc_gain_scatter_rank{r}.png"),
        )

    # layer-wise bar plot for one representative rank
    rep_r = 64 if 64 in ranks else ranks[min(len(ranks)-1, 0)]
    short_names = [n.replace(".weight", "").replace("model.layers.", "L") for n in df["weight_key"].tolist()]
    bar_compare(
        short_names,
        df[f"plain_align_to_W@{rep_r}"].tolist(),
        df[f"hess_align_to_W@{rep_r}"].tolist(),
        label_a=f"plain align@{rep_r}",
        label_b=f"hess align@{rep_r}",
        title=f"Layer-wise alignment to W-subspace @ {rep_r}",
        out_path=os.path.join(args.out_dir, f"layerwise_align_rank{rep_r}.png"),
    )
    bar_compare(
        short_names,
        df[f"plain_capture_by_Wsubspace@{rep_r}"].tolist(),
        df[f"hess_capture_by_Wsubspace@{rep_r}"].tolist(),
        label_a=f"plain captureW@{rep_r}",
        label_b=f"hess captureW@{rep_r}",
        title=f"Layer-wise residual energy captured by W-subspace @ {rep_r}",
        out_path=os.path.join(args.out_dir, f"layerwise_captureW_rank{rep_r}.png"),
    )

    log("\n[done] saved:")
    log(f"  - {per_layer_csv}")
    log(f"  - {os.path.join(args.out_dir, 'summary.json')}")

    log("\n[quick reading guide]")
    log("1) fro_weighted_ratio_hess_over_plain < 1  -> Hessian quant residual weighted norm is smaller")
    log("2) mean_evr_gap_hess_minus_plain@64 < 0   -> Hessian residual is less low-rank recoverable")
    log("3) mean_align_gap_hess_minus_plain@64 < 0 -> Hessian residual subspace is less aligned with W-subspace")
    log("4) mean_hess_capture_by_Wsubspace@64 < mean_plain_capture_by_Wsubspace@64")
    log("   -> W principal subspace explains Hessian residual less well")
    log("5) basis_mismatch_gain@r > 0 -> closed-form Procrustes can reduce subspace mismatch")
    log("   (이건 '재정렬 여지'가 있다는 뜻이지, 곧바로 EVR 개선을 의미하진 않음)")


if __name__ == "__main__":
    main()