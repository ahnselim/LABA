#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_selected_three_stage_residual_repair.py

Diagnostic validator for a 3-stage residual repair pipeline on a few selected
layers. This is intentionally a lightweight, torch-only proxy rather than a
paper-exact implementation.

Stages:
  1) Procrustes space realignment with a ProMises-style spectral prior
  2) Hessian-preconditioned Riemannian CG on the Stiefel manifold
  3) HM-IRLS-style Schatten-p spectral shrinkage refinement

Input residual:
  E_w = (W - Wq) * sqrt_diag

Main comparisons per selected layer / rank / quant mode:
  - input_full       : original weighted residual
  - baseline_svd     : best rank-r weighted SVD
  - stage1_realigned : full-rank Procrustes-realigned residual
  - stage2_rcg       : rank-r Hessian-preconditioned manifold solution
  - stage3_hm_irls   : rank-r Schatten-p refined solution

Outputs:
  - per_stage_metrics.csv
  - summary.json
  - stage comparison plots

Example:
CUDA_VISIBLE_DEVICES=2 python test/analyze_selected_three_stage_residual_repair.py \
  --model_id meta-llama/Llama-3.1-8B \
  --calib_s_path ./output/llama3_8b/calib_sqrtdiag.pt \
  --bits 2 \
  --group_size 128 \
  --quant_modes plain,hessian \
  --layer_regex "model\\.layers\\.(0|4|8|12)\\.(self_attn\\.q_proj|mlp\\.down_proj)\\.weight" \
  --rank_list 16,32,64 \
  --max_layers 4 \
  --device cuda \
  --device_map cpu \
  --out_dir ./analysis_three_stage
"""

import argparse
import gc
import json
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F


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


@torch.no_grad()
def _kth_quantiles_lastdim(X_flat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    _, S = X_flat.shape
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
    del levels
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
    N, _ = X_flat.shape
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
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError("transformers가 필요합니다: pip install transformers") from e

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
    X = X.to(device=device, dtype=torch.float32)
    try:
        Vh = torch.linalg.svd(X, full_matrices=False).Vh
    except RuntimeError:
        Vh = torch.linalg.svd(X.cpu(), full_matrices=False).Vh
    rr = min(r, Vh.shape[0])
    return Vh[:rr].transpose(0, 1).to(torch.float32).cpu()


@torch.no_grad()
def alignment_score(Qe: torch.Tensor, Qw: torch.Tensor) -> float:
    rr = min(Qe.shape[1], Qw.shape[1])
    if rr <= 0:
        return 0.0
    M = Qe[:, :rr].transpose(0, 1) @ Qw[:, :rr]
    return float((M.pow(2).sum() / float(rr)).item())


@torch.no_grad()
def principal_cosines(Qe: torch.Tensor, Qw: torch.Tensor) -> torch.Tensor:
    rr = min(Qe.shape[1], Qw.shape[1])
    if rr <= 0:
        return torch.zeros(0, dtype=torch.float32)
    M = Qe[:, :rr].transpose(0, 1) @ Qw[:, :rr]
    return torch.linalg.svdvals(M).to(torch.float32).cpu()


@torch.no_grad()
def reference_capture_ratio(X: torch.Tensor, Qref: torch.Tensor) -> float:
    total = X.pow(2).sum().item()
    if total <= 0:
        return 0.0
    proj = X @ Qref
    cap = proj.pow(2).sum().item()
    return float(cap / max(total, 1e-12))


@torch.no_grad()
def best_rank_evr(X: torch.Tensor, r: int) -> Tuple[float, float]:
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


STAGE_ORDER = [
    "input_full",
    "baseline_svd",
    "stage1_realigned",
    "stage2_rcg",
    "stage3_hm_irls",
]


def parse_quant_modes(raw: str) -> List[str]:
    modes = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in {"plain", "hessian"}:
            raise ValueError(f"Unsupported quant mode: {item}")
        modes.append(item)
    modes = sorted(set(modes), key=lambda x: ["plain", "hessian"].index(x))
    if not modes:
        raise ValueError("quant_modes must contain at least one of plain,hessian")
    return modes


def fro_norm(x: torch.Tensor) -> float:
    return float(torch.norm(x, p="fro").item())


def stable_rank(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    s = torch.linalg.svdvals(x).to(torch.float32).cpu()
    if s.numel() == 0:
        return 0.0
    return float((torch.norm(x, p="fro").pow(2) / s[0].pow(2).clamp_min(1e-12)).item())


def orthonormalize(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    q, r = torch.linalg.qr(x, mode="reduced")
    if r.numel() == 0:
        return q
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(0, 1))


@torch.no_grad()
def truncated_svd_matrix(x: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    rr = min(rank, s.numel())
    if rr <= 0:
        return torch.zeros_like(x), torch.zeros(0, dtype=torch.float32)
    xr = (u[:, :rr] * s[:rr]) @ vh[:rr, :]
    return xr.to(torch.float32), s[:rr].to(torch.float32).cpu()


@torch.no_grad()
def weighted_axis_strengths(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    coeff = x @ basis
    strength = torch.sqrt(coeff.pow(2).sum(dim=0).clamp_min(1e-12))
    return strength.to(torch.float32).cpu()


@torch.no_grad()
def promises_rotation(
    qe: torch.Tensor,
    qw: torch.Tensor,
    axis_strength: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    rr = min(qe.shape[1], qw.shape[1])
    if rr <= 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    cross = qe[:, :rr].transpose(0, 1) @ qw[:, :rr]
    axis_strength = axis_strength[:rr].to(torch.float32)
    axis_strength = axis_strength / axis_strength.sum().clamp_min(1e-12)
    prior = torch.diag(axis_strength)
    target = cross + float(kappa) * prior
    u, _, vh = torch.linalg.svd(target, full_matrices=False)
    return (u @ vh).to(torch.float32).cpu()


@torch.no_grad()
def stage1_realignment(
    ew: torch.Tensor,
    ww: torch.Tensor,
    rank: int,
    device: torch.device,
    kappa: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    qe = top_right_subspace(ew, rank, device=device)
    qw = top_right_subspace(ww, rank, device=device)
    rr = min(qe.shape[1], qw.shape[1])
    if rr <= 0:
        return ew.clone(), {
            "stage1_rr": 0,
            "stage1_rotation_trace": 0.0,
            "stage1_transfer_energy_ratio": 0.0,
        }

    qe = qe[:, :rr].to(torch.float32)
    qw = qw[:, :rr].to(torch.float32)
    axis_strength = weighted_axis_strengths(ww, qw)
    rot = promises_rotation(qe, qw, axis_strength=axis_strength, kappa=kappa)

    coeff = ew @ qe
    tail = ew - coeff @ qe.transpose(0, 1)
    aligned = tail + (coeff @ rot) @ qw.transpose(0, 1)

    transfer_ratio = float(
        (((coeff @ rot) @ qw.transpose(0, 1)).pow(2).sum() / ew.pow(2).sum().clamp_min(1e-12)).item()
    )
    meta = {
        "stage1_rr": int(rr),
        "stage1_rotation_trace": float(torch.trace(rot).item()),
        "stage1_transfer_energy_ratio": transfer_ratio,
    }
    return aligned.to(torch.float32).cpu(), meta


def project_tangent(q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return z - q @ sym(q.transpose(0, 1) @ z)


def objective_value(x: torch.Tensor, q: torch.Tensor) -> float:
    return float((x @ q).pow(2).sum().item())


def riemannian_gradient(
    x: torch.Tensor,
    q: torch.Tensor,
    inv_metric_diag: torch.Tensor,
) -> torch.Tensor:
    euclid = 2.0 * (x.transpose(0, 1) @ (x @ q))
    precond = inv_metric_diag.unsqueeze(1) * euclid
    return project_tangent(q, precond)


@torch.no_grad()
def rcg_optimize_basis(
    x: torch.Tensor,
    init_q: torch.Tensor,
    hdiag: torch.Tensor,
    steps: int,
    init_step_size: float,
    min_step_size: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if init_q.numel() == 0:
        return init_q.clone(), {
            "stage2_obj": 0.0,
            "stage2_step_count": 0,
            "stage2_last_step": 0.0,
        }

    q = orthonormalize(init_q.to(torch.float32))
    inv_metric_diag = 1.0 / hdiag.to(torch.float32).clamp_min(1e-8)
    grad = riemannian_gradient(x, q, inv_metric_diag)
    direction = -grad
    obj = objective_value(x, q)
    accepted_steps = 0
    last_step = 0.0

    for _ in range(max(0, int(steps))):
        step = float(init_step_size)
        accepted = False
        cand_q = q
        cand_obj = obj

        while step >= float(min_step_size):
            test_q = orthonormalize(q + step * direction)
            test_obj = objective_value(x, test_q)
            if test_obj >= obj - 1e-10:
                accepted = True
                cand_q = test_q
                cand_obj = test_obj
                break
            step *= 0.5

        if not accepted:
            break

        accepted_steps += 1
        last_step = step

        new_grad = riemannian_gradient(x, cand_q, inv_metric_diag)
        transported_grad = project_tangent(cand_q, grad)
        transported_dir = project_tangent(cand_q, direction)

        beta_num = (new_grad * (new_grad - transported_grad)).sum()
        beta_den = grad.pow(2).sum().clamp_min(1e-12)
        beta = max(0.0, float((beta_num / beta_den).item()))

        direction = -new_grad + beta * transported_dir
        q = cand_q
        grad = new_grad
        obj = cand_obj

    meta = {
        "stage2_obj": float(obj),
        "stage2_step_count": int(accepted_steps),
        "stage2_last_step": float(last_step),
    }
    return q.to(torch.float32).cpu(), meta


@torch.no_grad()
def hm_irls_refine(
    x: torch.Tensor,
    rank: int,
    p: float,
    lam: float,
    steps: int,
    eps: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    current, s_init = truncated_svd_matrix(x, rank)
    if current.numel() == 0:
        return current, {
            "stage3_removed_energy": 0.0,
            "stage3_mean_shrink": 0.0,
        }

    mean_shrink = 1.0
    for _ in range(max(0, int(steps))):
        u, s, vh = torch.linalg.svd(current, full_matrices=False)
        rr = min(rank, s.numel())
        if rr <= 0:
            break

        sr = s[:rr].to(torch.float32)
        weights = (sr.pow(2) + float(eps)).pow(float(p) * 0.5 - 0.5)
        scale = float(lam) * weights / weights.mean().clamp_min(1e-12)
        sr_new = torch.clamp(sr - scale, min=0.0)
        mean_shrink = float((sr_new / sr.clamp_min(1e-12)).mean().item())
        current = (u[:, :rr] * sr_new) @ vh[:rr, :]

    removed_energy = float((s_init.pow(2).sum().item() - current.pow(2).sum().item()))
    meta = {
        "stage3_removed_energy": removed_energy,
        "stage3_mean_shrink": float(mean_shrink),
    }
    return current.to(torch.float32).cpu(), meta


@torch.no_grad()
def evaluate_stage(
    candidate: torch.Tensor,
    original: torch.Tensor,
    ww: torch.Tensor,
    rank: int,
    device: torch.device,
) -> Dict[str, float]:
    candidate = candidate.to(torch.float32).cpu()
    original = original.to(torch.float32).cpu()
    qw = top_right_subspace(ww, rank, device=device)
    if float(candidate.pow(2).sum().item()) <= 1e-20:
        qx = torch.zeros((candidate.shape[1], 0), dtype=torch.float32)
    else:
        qx = top_right_subspace(candidate, rank, device=device)
    rr = min(qw.shape[1], qx.shape[1])
    qx_rr = qx[:, :rr] if rr > 0 else qx
    qw_rr = qw[:, :rr] if rr > 0 else qw

    align = alignment_score(qx_rr, qw_rr)
    cosines = principal_cosines(qx_rr, qw_rr)
    mean_cos = float(cosines.mean().item()) if cosines.numel() > 0 else 0.0
    min_cos = float(cosines.min().item()) if cosines.numel() > 0 else 0.0

    original_energy = original.pow(2).sum().clamp_min(1e-12)
    candidate_energy = candidate.pow(2).sum()
    recon_err = torch.norm(original - candidate, p="fro")
    capture_original = reference_capture_ratio(original, qx_rr)
    energy_in_w_subspace = reference_capture_ratio(candidate, qw_rr)
    self_evr, self_best_res = best_rank_evr(candidate, rank)

    return {
        "weighted_fro": float(torch.norm(candidate, p="fro").item()),
        "weighted_energy_ratio_vs_input": float((candidate_energy / original_energy).item()),
        "weighted_recon_error_to_input": float(recon_err.item()),
        "weighted_recon_error_ratio_to_input": float((recon_err.pow(2) / original_energy).item()),
        "capture_input_by_method_subspace": float(capture_original),
        "candidate_energy_in_W_subspace": float(energy_in_w_subspace),
        "align_to_W": float(align),
        "mean_principal_cos": float(mean_cos),
        "min_principal_cos": float(min_cos),
        "stable_rank": float(stable_rank(candidate)),
        "self_evr": float(self_evr),
        "self_best_rank_residual_fro": float(self_best_res),
    }


def plot_stage_metric(df: pd.DataFrame, quant_mode: str, metric: str, out_path: str) -> None:
    sub = df[df["quant_mode"] == quant_mode].copy()
    if sub.empty:
        return

    pivot = (
        sub.groupby(["rank", "stage"], as_index=False)[metric]
        .mean()
        .pivot(index="stage", columns="rank", values=metric)
        .reindex(STAGE_ORDER)
    )
    if pivot.empty:
        return

    plt.figure(figsize=(8.0, 4.8))
    xs = list(range(len(pivot.index)))
    for rank in pivot.columns:
        plt.plot(xs, pivot[rank].tolist(), marker="o", label=f"rank {int(rank)}")
    plt.xticks(xs, pivot.index, rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} by stage ({quant_mode})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser("Selected-layer three-stage residual repair validator")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--calib_s_path", required=True)
    ap.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--quant_modes", type=str, default="plain,hessian")

    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=4)

    ap.add_argument("--rank_list", type=str, default="16,32,64")
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)

    ap.add_argument("--promises_kappa", type=float, default=0.25)
    ap.add_argument("--rcg_steps", type=int, default=20)
    ap.add_argument("--rcg_init_step", type=float, default=1.0)
    ap.add_argument("--rcg_min_step", type=float, default=1e-4)
    ap.add_argument("--hm_p", type=float, default=0.5)
    ap.add_argument("--hm_lambda", type=float, default=0.025)
    ap.add_argument("--hm_steps", type=int, default=5)
    ap.add_argument("--hm_eps", type=float, default=1e-6)

    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--device_map", type=str, default="cpu")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ranks = parse_rank_list(args.rank_list)
    quant_modes = parse_quant_modes(args.quant_modes)
    device = resolve_device(args.device)

    if args.dtype == "bf16":
        load_dtype = torch.bfloat16
    elif args.dtype in {"fp16", "float16"}:
        load_dtype = torch.float16
    elif args.dtype in {"fp32", "float32"}:
        load_dtype = torch.float32
    else:
        load_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

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
        log(f"[analyze] {idx}/{len(selected_names)} {wkey}")

        entry = lookup_calib_entry(calib_map, wkey)
        if entry is None:
            log(f"[warn] missing calib entry -> skip {wkey}")
            continue

        sqrt_diag = entry["sqrt"].to(torch.float32).cpu()
        hdiag = entry["hessian_diag"].to(torch.float32).cpu()
        w_cpu = weights[wkey].to(torch.float32).cpu()

        if sqrt_diag.numel() != w_cpu.shape[1]:
            log(f"[warn] sqrt dim mismatch -> skip {wkey}")
            continue
        if hdiag.numel() != w_cpu.shape[1]:
            log(f"[warn] hdiag dim mismatch -> skip {wkey}")
            continue

        w_dev = w_cpu.to(device=device, dtype=torch.float32)
        ww = (w_cpu * sqrt_diag.unsqueeze(0)).to(torch.float32).cpu()

        quantized: Dict[str, torch.Tensor] = {}
        if "plain" in quant_modes:
            wq_plain, _ = lloyd_asym_nonuniform_quantize(
                W=w_dev,
                b=int(args.bits),
                group_size=int(args.group_size),
                lloyd_iter=int(args.lloyd_iter),
                chunk_groups=int(args.chunk_groups),
                hessian_diag=None,
            )
            quantized["plain"] = wq_plain.to(torch.float32).cpu()
        if "hessian" in quant_modes:
            wq_hess, _ = lloyd_asym_nonuniform_quantize(
                W=w_dev,
                b=int(args.bits),
                group_size=int(args.group_size),
                lloyd_iter=int(args.lloyd_iter),
                chunk_groups=int(args.chunk_groups),
                hessian_diag=hdiag.to(device=device, dtype=torch.float32),
            )
            quantized["hessian"] = wq_hess.to(torch.float32).cpu()

        for quant_mode, wq_cpu in quantized.items():
            ew = ((w_cpu - wq_cpu) * sqrt_diag.unsqueeze(0)).to(torch.float32).cpu()
            input_fro = fro_norm(ew)
            row_base = {
                "weight_key": wkey,
                "module": module_name_from_weight(wkey),
                "quant_mode": quant_mode,
                "out_features": int(w_cpu.shape[0]),
                "in_features": int(w_cpu.shape[1]),
                "bits": int(args.bits),
                "group_size": int(args.group_size),
                "input_weighted_fro": float(input_fro),
            }

            for rank in ranks:
                stage1_full, stage1_meta = stage1_realignment(
                    ew=ew,
                    ww=ww,
                    rank=rank,
                    device=device,
                    kappa=float(args.promises_kappa),
                )
                baseline_rank, baseline_svals = truncated_svd_matrix(ew, rank)
                q_init = top_right_subspace(stage1_full, rank, device=device)
                q_rcg, stage2_meta = rcg_optimize_basis(
                    x=stage1_full,
                    init_q=q_init,
                    hdiag=hdiag,
                    steps=int(args.rcg_steps),
                    init_step_size=float(args.rcg_init_step),
                    min_step_size=float(args.rcg_min_step),
                )
                stage2_rank = (stage1_full @ q_rcg) @ q_rcg.transpose(0, 1)
                stage3_rank, stage3_meta = hm_irls_refine(
                    x=stage2_rank,
                    rank=rank,
                    p=float(args.hm_p),
                    lam=float(args.hm_lambda),
                    steps=int(args.hm_steps),
                    eps=float(args.hm_eps),
                )

                candidates = {
                    "input_full": ew,
                    "baseline_svd": baseline_rank,
                    "stage1_realigned": stage1_full,
                    "stage2_rcg": stage2_rank,
                    "stage3_hm_irls": stage3_rank,
                }

                for stage_name, candidate in candidates.items():
                    metrics = evaluate_stage(
                        candidate=candidate,
                        original=ew,
                        ww=ww,
                        rank=rank,
                        device=device,
                    )
                    row = dict(row_base)
                    row.update(
                        {
                            "rank": int(rank),
                            "stage": stage_name,
                            "baseline_sigma_sum": float(baseline_svals.sum().item()) if baseline_svals.numel() > 0 else 0.0,
                            "stage1_rr": int(stage1_meta["stage1_rr"]),
                            "stage1_rotation_trace": float(stage1_meta["stage1_rotation_trace"]),
                            "stage1_transfer_energy_ratio": float(stage1_meta["stage1_transfer_energy_ratio"]),
                            "stage2_obj": float(stage2_meta["stage2_obj"]),
                            "stage2_step_count": int(stage2_meta["stage2_step_count"]),
                            "stage2_last_step": float(stage2_meta["stage2_last_step"]),
                            "stage3_removed_energy": float(stage3_meta["stage3_removed_energy"]),
                            "stage3_mean_shrink": float(stage3_meta["stage3_mean_shrink"]),
                        }
                    )
                    row.update(metrics)
                    rows.append(row)

        del w_dev
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(f"[done] {wkey} in {time.time() - t0:.2f}s")

    if not rows:
        raise RuntimeError("No rows analyzed.")

    df = pd.DataFrame(rows)
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    df = df.sort_values(["weight_key", "quant_mode", "rank", "stage"]).reset_index(drop=True)

    per_stage_csv = os.path.join(args.out_dir, "per_stage_metrics.csv")
    df.to_csv(per_stage_csv, index=False)

    summary_rows = (
        df.groupby(["quant_mode", "rank", "stage"], as_index=False)
        .agg(
            mean_align_to_W=("align_to_W", "mean"),
            mean_capture_input_by_method_subspace=("capture_input_by_method_subspace", "mean"),
            mean_candidate_energy_in_W_subspace=("candidate_energy_in_W_subspace", "mean"),
            mean_recon_error_ratio=("weighted_recon_error_ratio_to_input", "mean"),
            mean_weighted_energy_ratio=("weighted_energy_ratio_vs_input", "mean"),
            mean_stable_rank=("stable_rank", "mean"),
            mean_self_evr=("self_evr", "mean"),
            mean_stage2_steps=("stage2_step_count", "mean"),
            mean_stage3_removed_energy=("stage3_removed_energy", "mean"),
        )
        .sort_values(["quant_mode", "rank", "stage"])
    )

    summary_payload = {
        "num_layers": int(df["weight_key"].nunique()),
        "selected_layers": selected_names,
        "quant_modes": quant_modes,
        "ranks": [int(r) for r in ranks],
        "promises_kappa": float(args.promises_kappa),
        "rcg_steps": int(args.rcg_steps),
        "hm_p": float(args.hm_p),
        "hm_lambda": float(args.hm_lambda),
        "summary_rows": summary_rows.to_dict(orient="records"),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    for quant_mode in quant_modes:
        plot_stage_metric(
            df,
            quant_mode=quant_mode,
            metric="align_to_W",
            out_path=os.path.join(args.out_dir, f"{quant_mode}_align_to_W.png"),
        )
        plot_stage_metric(
            df,
            quant_mode=quant_mode,
            metric="capture_input_by_method_subspace",
            out_path=os.path.join(args.out_dir, f"{quant_mode}_capture_input.png"),
        )
        plot_stage_metric(
            df,
            quant_mode=quant_mode,
            metric="weighted_recon_error_ratio_to_input",
            out_path=os.path.join(args.out_dir, f"{quant_mode}_recon_error_ratio.png"),
        )

    log("\n[done] saved:")
    log(f"  - {per_stage_csv}")
    log(f"  - {os.path.join(args.out_dir, 'summary.json')}")
    log("\n[reading guide]")
    log("1) align_to_W up            -> repaired low-rank space follows original W principal axes better")
    log("2) capture_input_by_method_subspace up")
    log("   -> that stage's right subspace explains the original weighted residual better")
    log("3) weighted_recon_error_ratio_to_input down")
    log("   -> stage output stays closer to the original weighted residual")
    log("4) candidate_energy_in_W_subspace up")
    log("   -> more energy has been concentrated into the W reference subspace")
    log("5) stage3_removed_energy > 0")
    log("   -> HM-IRLS-style refinement actually suppressed weak singular directions")


if __name__ == "__main__":
    main()
