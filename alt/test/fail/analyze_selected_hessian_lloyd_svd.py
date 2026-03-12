#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_selected_hessian_lloyd_svd.py

목적:
  - 미리 quantized weights / residual을 저장해두지 않았다고 가정
  - 원본 모델에서 선택한 몇 개 레이어 weight를 직접 읽어
    plain Lloyd-Max vs Hessian-weighted Lloyd-Max 를 각각 수행
  - residual / weighted residual의 low-rankness 및
    원본 W principal subspace와의 alignment를 비교

입력:
  - HF model_id
  - calib_sqrtdiag.pt  (이미 있음)
  - 선택 레이어 조건 (regex 또는 max_layers)

출력:
  - per_layer_metrics.csv
  - summary.json
  - optional scatter plots

예시:
CUDA_VISIBLE_DEVICES=2 python analyze_selected_hessian_lloyd_svd.py \
  --model_id meta-llama/Llama-3.1-8B \
  --calib_s_path ./output/llama3_8b/calib_sqrtdiag.pt \
  --bits 2 \
  --group_size 128 \
  --layer_regex "model\\.layers\\.(0|4|8|12)\\.(self_attn\\.q_proj|mlp\\.down_proj)\\.weight" \
  --rank_list 16,32,64,128 \
  --max_layers 8 \
  --device cuda \
  --out_dir ./analysis_selected_layers

핵심 metric:
  - weighted fro norm of residual
  - EVR@r on E_w = E * sqrt_diag
  - best rank-r weighted residual fro
  - stable rank
  - alignment score between top-r right singular subspace of:
      (a) original weighted W_w = W * sqrt_diag
      (b) residual weighted E_w
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
# Logging / device
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


# ============================================================
# Load calibration Hessian / sqrt diag
# ============================================================
def load_pt(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")

def load_calib_map(calib_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    expected entry:
      key -> {
        's' or 'sqrt' or 'var'
      }
    normalize to:
      {
        key: {
          'sqrt': [I],
          'hessian_diag': [I]
        }
      }
    """
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
# Group reshape helpers
# ============================================================
def _to_groups(W: torch.Tensor, group_size: int):
    """
    W: [O,I] -> [O,G,S]
    """
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
# Lloyd-Max helpers
# ============================================================
@torch.no_grad()
def _kth_quantiles_lastdim(X_flat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """
    X_flat: [N,S], probs: [L]
    returns [N,L]
    """
    N, S = X_flat.shape
    probs = probs.clamp(0.0, 1.0).to(device=X_flat.device, dtype=torch.float32)
    ks = (probs * (S - 1)).round().to(torch.int64) + 1
    ks = ks.clamp(1, S).tolist()
    outs = [torch.kthvalue(X_flat, k, dim=1).values for k in ks]
    return torch.stack(outs, dim=1)

@torch.no_grad()
def _lloyd_centroid_update(
    x: torch.Tensor,                  # [N,S]
    idx: torch.Tensor,                # [N,S]
    levels: int,
    prev_cb: torch.Tensor,            # [N,L]
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
    X_flat: torch.Tensor,               # [N,S]
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
    X_flat: torch.Tensor,      # [N,S]
    codebook: torch.Tensor,    # [N,L]
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
    W: torch.Tensor,                  # [O,I]
    b: int,
    group_size: int,
    lloyd_iter: int = 12,
    chunk_groups: int = 4096,
    hessian_diag: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    returns:
      Wq: [O,I]
      meta: small dict
    """
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
        X_flat,
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
# Layer extraction
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
            raise RuntimeError(
                f"meta tensor detected for {name}. Re-run with --device_map none or cpu."
            )
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
# Metrics
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
def compute_subspace_alignment_score(
    A: torch.Tensor,   # [m,n]
    B: torch.Tensor,   # [m,n]
    r: int,
    device: torch.device,
) -> float:
    """
    Compare top-r right singular subspaces of A and B.
    score in [0,1], higher = more aligned
    """
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)

    try:
        Va = torch.linalg.svd(A, full_matrices=False).Vh[:r].transpose(0, 1)  # [n,r]
        Vb = torch.linalg.svd(B, full_matrices=False).Vh[:r].transpose(0, 1)  # [n,r]
    except RuntimeError:
        A = A.cpu()
        B = B.cpu()
        Va = torch.linalg.svd(A, full_matrices=False).Vh[:r].transpose(0, 1)
        Vb = torch.linalg.svd(B, full_matrices=False).Vh[:r].transpose(0, 1)

    rr = min(Va.shape[1], Vb.shape[1], r)
    if rr <= 0:
        return 0.0
    M = Va[:, :rr].transpose(0, 1) @ Vb[:, :rr]  # [r,r]
    score = (M.pow(2).sum() / float(rr)).item()
    return float(score)

@torch.no_grad()
def compute_spectrum_metrics(
    W: torch.Tensor,          # [O,I]
    Wq: torch.Tensor,         # [O,I]
    sqrt_diag: torch.Tensor,  # [I]
    ranks: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """
    analyze residual E = W - Wq on weighted space E_w = E * sqrt_diag
    also compare top-r right singular space of W_w and E_w
    """
    W = W.to(device=device, dtype=torch.float32)
    Wq = Wq.to(device=device, dtype=torch.float32)
    sqrt_diag = sqrt_diag.to(device=device, dtype=torch.float32)

    E = W - Wq
    Ww = W * sqrt_diag.unsqueeze(0)
    Ew = E * sqrt_diag.unsqueeze(0)

    fro_raw = torch.norm(E, p="fro").item()
    fro_weighted = torch.norm(Ew, p="fro").item()

    try:
        S = torch.linalg.svdvals(Ew)
    except RuntimeError:
        S = torch.linalg.svdvals(Ew.cpu())
    S = S.to(torch.float32).cpu()

    s2 = S.pow(2)
    total_energy = float(s2.sum().item())
    eps = 1e-12

    out = {
        "fro_raw": float(fro_raw),
        "fro_weighted": float(fro_weighted),
        "weighted_total_energy": float(total_energy),
        "rank_full": int(S.numel()),
        "sigma1": float(S[0].item()) if S.numel() > 0 else 0.0,
        "top1_share": float(s2[0].item() / max(total_energy, eps)) if S.numel() > 0 else 0.0,
        "top8_share": float(s2[: min(8, S.numel())].sum().item() / max(total_energy, eps)) if S.numel() > 0 else 0.0,
        "top32_share": float(s2[: min(32, S.numel())].sum().item() / max(total_energy, eps)) if S.numel() > 0 else 0.0,
        "stable_rank": float((Ew.norm(p="fro").pow(2) / S[0].pow(2).clamp_min(1e-12)).item()) if S.numel() > 0 else 0.0,
    }

    cumsum = torch.cumsum(s2, dim=0)
    for r in ranks:
        rr = min(r, S.numel())
        captured = float(cumsum[rr - 1].item()) if rr > 0 else 0.0
        evr = captured / max(total_energy, eps)
        residual_energy = max(total_energy - captured, 0.0)
        recovery_fro = math.sqrt(residual_energy)

        out[f"evr@{r}"] = float(evr)
        out[f"best_rank{r}_residual_fro"] = float(recovery_fro)
        out[f"align_to_W@{r}"] = float(
            compute_subspace_alignment_score(Ww, Ew, r, device=device)
        )

    return out


# ============================================================
# Plots
# ============================================================
def scatter_plot(
    x, y, xlabel, ylabel, title, out_path, diagonal=True
):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=15, alpha=0.75)
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


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("Analyze selected layers: plain vs hessian Lloyd + SVD metrics")
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
                    help='recommend "cpu" or "none" for easier state loading')

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
        Wq_plain, meta_plain = lloyd_asym_nonuniform_quantize(
            W,
            b=int(args.bits),
            group_size=int(args.group_size),
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
            hessian_diag=None,
        )

        # hessian Lloyd
        Wq_hess, meta_hess = lloyd_asym_nonuniform_quantize(
            W,
            b=int(args.bits),
            group_size=int(args.group_size),
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
            hessian_diag=hdiag.to(device=device, dtype=torch.float32),
        )

        plain_metrics = compute_spectrum_metrics(
            W=W,
            Wq=Wq_plain,
            sqrt_diag=sqrt_diag,
            ranks=ranks,
            device=device,
        )
        hess_metrics = compute_spectrum_metrics(
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

            "plain_fro_raw": plain_metrics["fro_raw"],
            "hess_fro_raw": hess_metrics["fro_raw"],
            "plain_fro_weighted": plain_metrics["fro_weighted"],
            "hess_fro_weighted": hess_metrics["fro_weighted"],
            "fro_weighted_ratio_hess_over_plain": (
                hess_metrics["fro_weighted"] / max(plain_metrics["fro_weighted"], 1e-12)
            ),

            "plain_top1_share": plain_metrics["top1_share"],
            "hess_top1_share": hess_metrics["top1_share"],
            "plain_top8_share": plain_metrics["top8_share"],
            "hess_top8_share": hess_metrics["top8_share"],
            "plain_top32_share": plain_metrics["top32_share"],
            "hess_top32_share": hess_metrics["top32_share"],

            "top1_share_ratio_hess_over_plain": (
                hess_metrics["top1_share"] / max(plain_metrics["top1_share"], 1e-12)
            ),
            "top8_share_ratio_hess_over_plain": (
                hess_metrics["top8_share"] / max(plain_metrics["top8_share"], 1e-12)
            ),
            "top32_share_ratio_hess_over_plain": (
                hess_metrics["top32_share"] / max(plain_metrics["top32_share"], 1e-12)
            ),

            "plain_stable_rank": plain_metrics["stable_rank"],
            "hess_stable_rank": hess_metrics["stable_rank"],
        }

        for r in ranks:
            row[f"plain_evr@{r}"] = plain_metrics[f"evr@{r}"]
            row[f"hess_evr@{r}"] = hess_metrics[f"evr@{r}"]
            row[f"evr_gap_hess_minus_plain@{r}"] = (
                hess_metrics[f"evr@{r}"] - plain_metrics[f"evr@{r}"]
            )

            row[f"plain_best_rank{r}_residual_fro"] = plain_metrics[f"best_rank{r}_residual_fro"]
            row[f"hess_best_rank{r}_residual_fro"] = hess_metrics[f"best_rank{r}_residual_fro"]
            row[f"best_rank{r}_residual_ratio_hess_over_plain"] = (
                hess_metrics[f"best_rank{r}_residual_fro"] / max(plain_metrics[f"best_rank{r}_residual_fro"], 1e-12)
            )

            row[f"plain_align_to_W@{r}"] = plain_metrics[f"align_to_W@{r}"]
            row[f"hess_align_to_W@{r}"] = hess_metrics[f"align_to_W@{r}"]
            row[f"align_gap_hess_minus_plain@{r}"] = (
                hess_metrics[f"align_to_W@{r}"] - plain_metrics[f"align_to_W@{r}"]
            )

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
        "mean_plain_fro_weighted": float(df["plain_fro_weighted"].mean()),
        "mean_hess_fro_weighted": float(df["hess_fro_weighted"].mean()),
        "mean_fro_weighted_ratio_hess_over_plain": float(df["fro_weighted_ratio_hess_over_plain"].mean()),
        "mean_plain_top32_share": float(df["plain_top32_share"].mean()),
        "mean_hess_top32_share": float(df["hess_top32_share"].mean()),
        "mean_plain_stable_rank": float(df["plain_stable_rank"].mean()),
        "mean_hess_stable_rank": float(df["hess_stable_rank"].mean()),
        "selected_layers": selected_names,
    }
    for r in ranks:
        summary[f"mean_plain_evr@{r}"] = float(df[f"plain_evr@{r}"].mean())
        summary[f"mean_hess_evr@{r}"] = float(df[f"hess_evr@{r}"].mean())
        summary[f"mean_evr_gap_hess_minus_plain@{r}"] = float(df[f"evr_gap_hess_minus_plain@{r}"].mean())

        summary[f"mean_plain_best_rank{r}_residual_fro"] = float(df[f"plain_best_rank{r}_residual_fro"].mean())
        summary[f"mean_hess_best_rank{r}_residual_fro"] = float(df[f"hess_best_rank{r}_residual_fro"].mean())

        summary[f"mean_plain_align_to_W@{r}"] = float(df[f"plain_align_to_W@{r}"].mean())
        summary[f"mean_hess_align_to_W@{r}"] = float(df[f"hess_align_to_W@{r}"].mean())
        summary[f"mean_align_gap_hess_minus_plain@{r}"] = float(df[f"align_gap_hess_minus_plain@{r}"].mean())

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
            title=f"Subspace alignment to W @ {r}",
            out_path=os.path.join(args.out_dir, f"align_scatter_rank{r}.png"),
        )
        scatter_plot(
            df[f"plain_best_rank{r}_residual_fro"].tolist(),
            df[f"hess_best_rank{r}_residual_fro"].tolist(),
            xlabel=f"plain best-rank{r} residual fro",
            ylabel=f"hess best-rank{r} residual fro",
            title=f"Best rank-{r} weighted residual",
            out_path=os.path.join(args.out_dir, f"recovery_scatter_rank{r}.png"),
        )

    log("\n[done] saved:")
    log(f"  - {per_layer_csv}")
    log(f"  - {os.path.join(args.out_dir, 'summary.json')}")

    log("\n[quick reading guide]")
    log("1) fro_weighted_ratio_hess_over_plain < 1  -> Hessian quant residual weighted norm is smaller")
    log("2) mean_evr_gap_hess_minus_plain@64 < 0   -> Hessian residual is less low-rank recoverable")
    log("3) mean_align_gap_hess_minus_plain@64 < 0 -> Hessian residual subspace is less aligned with original W subspace")
    log("4) stable rank 증가                        -> residual spectrum is flatter")


if __name__ == "__main__":
    main()