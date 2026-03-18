#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_residual_lowrank_visual.py

Build three geometry-first visualizations comparing OAS vs second-moment residuals:

  1. Mean cumulative EVR curve
  2. Rank-vs-best-reconstruction-error curve
  3. Layer-wise scatter: EVR@r advantage vs AB gain advantage

Default analysis space is the weighted residual:
  R_d = (W - W_q) * D

Example:
CUDA_VISIBLE_DEVICES=1 python test/compare_residual_lowrank_visual.py \
  --model_id meta-llama/Llama-3.1-8B \
  --step1_dir_oas ./output/llama3_8b/step1_quant/1bit \
  --calib_oas ./output/llama3_8b/calib_sqrtdiag.pt \
  --low_rank_path_oas ./output/llama3_8b/step3_svd/1bit/low_rank_ab.pt \
  --step1_dir_second ./output/llama3_8b_64/step1_quant/1bit \
  --calib_second ./output/llama3_8b_64/calib_sqrtdiag.pt \
  --low_rank_path_second ./output/llama3_8b_64/step3_svd/1bit/low_rank_ab.pt \
  --out_dir ./output/geometry_lowrank/1bit \
  --rank_eval 64 \
  --max_rank_plot 256 \
  --device cuda \
  --model_device_map auto \
  --selected_blocks 0 4 8 12
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e


TARGET_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "fc1", "fc2",
}

MODULE_ORDER = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "o_proj": 3,
    "out_proj": 4,
    "gate_proj": 5,
    "up_proj": 6,
    "down_proj": 7,
    "fc1": 8,
    "fc2": 9,
}

ATTN_TYPES = {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}
MLP_TYPES = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"}


def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )


def module_type_from_key(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def module_family(module_type: str) -> str:
    if module_type in ATTN_TYPES:
        return "attention"
    if module_type in MLP_TYPES:
        return "mlp"
    return "other"


def extract_block_index(name: str) -> Optional[int]:
    patterns = (
        r"\bmodel\.layers\.(\d+)\.",
        r"\bencoder\.layers\.(\d+)\.",
        r"\blayers\.(\d+)\.",
    )
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def sort_key(name: str) -> Tuple[int, int, str]:
    bidx = extract_block_index(name)
    suffix = module_type_from_key(name)
    return (10**9 if bidx is None else bidx, MODULE_ORDER.get(suffix, 10**6), name)


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(max(b, eps))


def parse_int_list(values: Optional[Sequence[str]]) -> List[int]:
    out: List[int] = []
    if not values:
        return out
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if not piece:
                continue
            out.append(int(piece))
    return out


def parse_str_list(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    if not values:
        return out
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out


def dequant_from_codebook_codes(
    codebook_ogq: torch.Tensor,
    qcodes_ogs: torch.Tensor,
    orig_i: int,
) -> torch.Tensor:
    o, g, q = codebook_ogq.shape
    _, _, s = qcodes_ogs.shape
    cb = codebook_ogq.reshape(o * g, q)
    idx = qcodes_ogs.reshape(o * g, s).long()
    xq = torch.gather(cb, dim=1, index=idx).reshape(o, g, s)
    return xq.reshape(o, g * s)[:, :orig_i]


def load_step1_artifacts(step1_dir: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, dict]]:
    p = Path(step1_dir).resolve()
    codebook_path = p / "codebook.pt"
    qcodes_path = p / "qcodes.pt"
    meta_path = p / "meta.pt"
    if not (codebook_path.exists() and qcodes_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"step1_dir must contain codebook.pt, qcodes.pt, meta.pt: {p}")
    codebooks = torch.load(codebook_path, map_location="cpu")
    qcodes = torch.load(qcodes_path, map_location="cpu")
    metas = torch.load(meta_path, map_location="cpu")
    return codebooks, qcodes, metas


def load_calib_map(calib_path: str) -> Dict[str, dict]:
    payload = torch.load(calib_path, map_location="cpu")
    calib_map = payload.get("cov_ops", payload)
    if not isinstance(calib_map, dict):
        raise TypeError(f"Unsupported calib payload type: {type(calib_map)!r}")
    return calib_map


def load_low_rank_ab(low_rank_path: str) -> Dict[str, dict]:
    payload = torch.load(low_rank_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported low_rank_ab payload type: {type(payload)!r}")
    return payload


def lookup_calib_entry(calib_map: Dict[str, dict], full_weight_name: str) -> dict:
    entry = calib_map.get(full_weight_name)
    if entry is None and full_weight_name.endswith(".weight"):
        entry = calib_map.get(full_weight_name[: -len(".weight")])
    if entry is None:
        raise KeyError(f"Missing calibration entry for {full_weight_name}")
    return entry


def load_diag_weight(entry: dict, eps: float) -> torch.Tensor:
    if "s" in entry:
        d = entry["s"].to(torch.float32)
    elif "sqrt" in entry:
        d = entry["sqrt"].to(torch.float32)
    elif "var" in entry:
        d = torch.sqrt(entry["var"].to(torch.float32).clamp_min(0.0))
    elif "inv_s" in entry:
        d = 1.0 / entry["inv_s"].to(torch.float32).clamp_min(float(eps))
    else:
        raise KeyError("calib entry must include one of: s, sqrt, var, inv_s")
    return d.clamp_min(float(eps)).contiguous()


def reconstruct_ab(entry: dict, out_dim: int, in_dim: int, device: torch.device) -> Tuple[torch.Tensor, int]:
    if not isinstance(entry, dict):
        raise TypeError(f"low_rank_ab entry must be dict, got {type(entry)!r}")
    a = entry.get("A", None)
    b = entry.get("B", None)
    if a is None or b is None:
        raise KeyError("low_rank_ab entry must contain A and B")
    a = a.to(device=device, dtype=torch.float32)
    b = b.to(device=device, dtype=torch.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if a.shape[0] != out_dim or b.shape[1] != in_dim:
        raise ValueError(
            f"AB shape mismatch: A={tuple(a.shape)} B={tuple(b.shape)} expected ({out_dim}, r) and (r, {in_dim})"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"AB rank mismatch: A={tuple(a.shape)} B={tuple(b.shape)}")
    return a @ b, int(a.shape[1])


def _snapshot_state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if getattr(v, "is_meta", False):
            raise NotImplementedError(f"meta tensor in state_dict: {k}")
        state[k] = v.detach().to("cpu")
    return state


def load_model_state(args: argparse.Namespace) -> Tuple[torch.device, Dict[str, torch.Tensor]]:
    if args.dtype_w == "fp16":
        load_dtype = torch.float16
    elif args.dtype_w == "bf16":
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dm_raw = str(args.model_device_map).strip().lower()
    resolved_model_device_map = None if dm_raw in {"", "none", "null"} else args.model_device_map

    print(
        f"[LowRankVisual] loading original model: {args.model_id} "
        f"(device_map={resolved_model_device_map}, device={device})",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=args.trust_remote_code,
        device_map=resolved_model_device_map,
        low_cpu_mem_usage=True,
    )
    try:
        state = _snapshot_state_to_cpu(model)
        del model
    except NotImplementedError:
        if resolved_model_device_map is None:
            raise
        print("[LowRankVisual] Detected meta tensors. Re-loading model on CPU.", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
            trust_remote_code=args.trust_remote_code,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        state = _snapshot_state_to_cpu(model)
        del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return device, state


def pad_curve(values: np.ndarray, target_len: int, fill_value: float) -> np.ndarray:
    out = np.full(target_len, fill_value, dtype=np.float64)
    if values.size > 0:
        out[: min(target_len, values.size)] = values[:target_len]
    return out


def svd_curve_metrics(m: torch.Tensor, max_rank_plot: int) -> Dict[str, np.ndarray | float | int]:
    if m.numel() == 0:
        return {
            "rank_cap": 0,
            "energy_total": 0.0,
            "cum_evr": np.zeros(max_rank_plot, dtype=np.float64),
            "recon_err": np.ones(max_rank_plot, dtype=np.float64),
            "norm_singular": np.zeros(max_rank_plot, dtype=np.float64),
        }

    s = torch.linalg.svdvals(m)
    if s.numel() == 0:
        return {
            "rank_cap": 0,
            "energy_total": 0.0,
            "cum_evr": np.zeros(max_rank_plot, dtype=np.float64),
            "recon_err": np.ones(max_rank_plot, dtype=np.float64),
            "norm_singular": np.zeros(max_rank_plot, dtype=np.float64),
        }

    s = s.to(torch.float64)
    s2 = s * s
    total = float(s2.sum().item())
    rank_cap = int(s.numel())
    cum = torch.cumsum(s2, dim=0) / max(total, 1e-12)
    recon = 1.0 - cum
    norm_s = s / max(float(s.sum().item()), 1e-12)

    return {
        "rank_cap": rank_cap,
        "energy_total": total,
        "cum_evr": pad_curve(cum.cpu().numpy(), target_len=max_rank_plot, fill_value=1.0),
        "recon_err": pad_curve(recon.cpu().numpy(), target_len=max_rank_plot, fill_value=0.0),
        "norm_singular": pad_curve(norm_s.cpu().numpy(), target_len=max_rank_plot, fill_value=0.0),
    }


def energy_capture_ratio(before: torch.Tensor, after: torch.Tensor) -> float:
    before_e = float((before.to(torch.float64) ** 2).sum().item())
    after_e = float((after.to(torch.float64) ** 2).sum().item())
    return safe_div(before_e - after_e, before_e)


def aggregate_mean_curve(rows: Sequence[dict], method: str, family: Optional[str], stage: str, key: str) -> np.ndarray:
    selected = [row for row in rows if family in {None, "all"} or row["module_family"] == family]
    if not selected:
        return np.zeros(0, dtype=np.float64)
    arr = np.stack([np.asarray(row[method][stage][key], dtype=np.float64) for row in selected], axis=0)
    return arr.mean(axis=0)


def aggregate_mean_scalar(rows: Sequence[dict], key: str, family: Optional[str] = None) -> float:
    vals = []
    for row in rows:
        if family not in {None, "all"} and row["module_family"] != family:
            continue
        if key in row:
            vals.append(float(row[key]))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def build_method_context(
    *,
    label: str,
    step1_dir: str,
    calib_path: str,
    low_rank_path: str,
) -> dict:
    print(f"[LowRankVisual][{label}] loading step1: {step1_dir}", flush=True)
    codebooks, qcodes, metas = load_step1_artifacts(step1_dir)
    print(f"[LowRankVisual][{label}] loading calib: {calib_path}", flush=True)
    calib = load_calib_map(calib_path)
    print(f"[LowRankVisual][{label}] loading low_rank_ab: {low_rank_path}", flush=True)
    low_rank_ab = load_low_rank_ab(low_rank_path)
    return {
        "label": label,
        "step1_dir": str(Path(step1_dir).resolve()),
        "calib_path": str(Path(calib_path).resolve()),
        "low_rank_path": str(Path(low_rank_path).resolve()),
        "codebooks": codebooks,
        "qcodes": qcodes,
        "metas": metas,
        "calib": calib,
        "low_rank_ab": low_rank_ab,
    }


def resolve_keys(args: argparse.Namespace, state: Dict[str, torch.Tensor], ctxs: Sequence[dict]) -> List[str]:
    layer_re = re.compile(args.layer_regex) if args.layer_regex else None
    selected_layers = set(parse_str_list(args.selected_layers))
    selected_blocks = set(parse_int_list(args.selected_blocks))
    keyset = set(state.keys())
    for ctx in ctxs:
        keyset &= set(ctx["codebooks"].keys())
        keyset &= set(ctx["qcodes"].keys())
        keyset &= set(ctx["metas"].keys())
        keyset &= set(ctx["low_rank_ab"].keys())
        keyset &= set(ctx["calib"].keys()) | {k + ".weight" for k in ctx["calib"].keys() if not str(k).endswith(".weight")}

    keys: List[str] = []
    for key in keyset:
        if not is_target_weight(key, state[key]):
            continue
        if layer_re and not layer_re.search(key):
            continue
        if selected_layers and key not in selected_layers:
            continue
        if selected_blocks:
            block_idx = extract_block_index(key)
            if block_idx is None or block_idx not in selected_blocks:
                continue
        keys.append(key)

    keys = sorted(set(keys), key=sort_key)
    if args.max_layers > 0:
        keys = keys[: int(args.max_layers)]
    if not keys:
        raise RuntimeError("No matched layers found.")
    return keys


def build_analysis_matrix(residual: torch.Tensor, d: torch.Tensor, space: str) -> torch.Tensor:
    if space == "raw":
        return residual.to(torch.float32)
    return (residual.to(torch.float32) * d.unsqueeze(0)).contiguous()


@torch.no_grad()
def analyze_layer(
    key: str,
    state: Dict[str, torch.Tensor],
    device: torch.device,
    method_ctxs: Dict[str, dict],
    args: argparse.Namespace,
) -> dict:
    w_cpu = state[key].to(torch.float32)
    out_dim, in_dim = int(w_cpu.shape[0]), int(w_cpu.shape[1])

    base = {
        "layer": key,
        "block_idx": -1 if extract_block_index(key) is None else int(extract_block_index(key)),
        "module_type": module_type_from_key(key),
        "module_family": module_family(module_type_from_key(key)),
        "out_dim": out_dim,
        "in_dim": in_dim,
        "rank_eval": int(args.rank_eval),
        "residual_space": str(args.residual_space),
    }

    w = w_cpu.to(device)
    method_rows: Dict[str, dict] = {}

    for label, ctx in method_ctxs.items():
        meta = ctx["metas"][key]
        orig_i = int(tuple(meta["orig_shape"])[1])
        if orig_i != in_dim:
            raise RuntimeError(f"weight/meta mismatch on {key}: weight_in={in_dim} meta_in={orig_i}")

        codebook = ctx["codebooks"][key].to(device=device, dtype=torch.float32)
        qcodes = ctx["qcodes"][key].to(device=device)
        wq = dequant_from_codebook_codes(codebook, qcodes, orig_i=orig_i)
        calib_entry = lookup_calib_entry(ctx["calib"], key)
        d = load_diag_weight(calib_entry, eps=float(args.eps)).to(device=device, dtype=torch.float32)
        if d.numel() != in_dim:
            raise RuntimeError(f"diag size mismatch on {key}: expected {in_dim}, got {d.numel()}")

        ab, ab_rank_used = reconstruct_ab(ctx["low_rank_ab"][key], out_dim=out_dim, in_dim=in_dim, device=device)

        residual_before = (w - wq).to(torch.float32)
        residual_after = (w - (wq + ab)).to(torch.float32)
        analyze_before = build_analysis_matrix(residual_before, d=d, space=str(args.residual_space))
        analyze_after = build_analysis_matrix(residual_after, d=d, space=str(args.residual_space))

        before_curves = svd_curve_metrics(analyze_before, max_rank_plot=int(args.max_rank_plot))
        after_curves = svd_curve_metrics(analyze_after, max_rank_plot=int(args.max_rank_plot))
        rank_eval_idx = max(0, min(int(args.rank_eval), int(args.max_rank_plot)) - 1)

        method_rows[label] = {
            "bits": int(meta.get("bits", -1)),
            "group_size": int(meta.get("group_size", -1)),
            "ab_rank_used": int(ab_rank_used),
            "before": before_curves,
            "after": after_curves,
            "evr_at_rank_eval_before": float(before_curves["cum_evr"][rank_eval_idx]) if int(args.max_rank_plot) > 0 else 0.0,
            "evr_at_rank_eval_after": float(after_curves["cum_evr"][rank_eval_idx]) if int(args.max_rank_plot) > 0 else 0.0,
            "ab_gain": energy_capture_ratio(analyze_before, analyze_after),
            "before_energy": float((analyze_before.to(torch.float64) ** 2).sum().item()),
            "after_energy": float((analyze_after.to(torch.float64) ** 2).sum().item()),
        }

        del codebook, qcodes, wq, d, ab, residual_before, residual_after, analyze_before, analyze_after

    row = dict(base)
    row["oas"] = method_rows["oas"]
    row["second"] = method_rows["second"]
    row["delta_evr_at_rank_eval_before"] = float(
        row["oas"]["evr_at_rank_eval_before"] - row["second"]["evr_at_rank_eval_before"]
    )
    row["delta_ab_gain"] = float(row["oas"]["ab_gain"] - row["second"]["ab_gain"])
    row["winner_evr"] = "oas" if row["delta_evr_at_rank_eval_before"] > 0 else ("second" if row["delta_evr_at_rank_eval_before"] < 0 else "tie")
    row["winner_ab_gain"] = "oas" if row["delta_ab_gain"] > 0 else ("second" if row["delta_ab_gain"] < 0 else "tie")
    return row


def plot_mean_curve_figure(rows: Sequence[dict], args: argparse.Namespace, out_path: Path) -> None:
    x = np.arange(1, int(args.max_rank_plot) + 1)
    families = [("all", "All"), ("attention", "Attention"), ("mlp", "MLP")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {"oas": "#1f77b4", "second": "#d62728"}

    for ax, (family_key, title) in zip(axes, families):
        for method in ("oas", "second"):
            y = aggregate_mean_curve(rows, method=method, family=family_key, stage="before", key="cum_evr")
            if y.size == 0:
                continue
            ax.plot(x, y, label=method.upper(), color=colors[method], linewidth=2.2)
            idx = int(args.rank_eval) - 1
            if 0 <= idx < y.size:
                ax.scatter([int(args.rank_eval)], [y[idx]], color=colors[method], s=28, zorder=3)
                ax.text(
                    int(args.rank_eval) + 2,
                    float(y[idx]),
                    f"{method.upper()} {y[idx]:.3f}",
                    color=colors[method],
                    fontsize=9,
                    va="center",
                )
        ax.axvline(int(args.rank_eval), color="0.5", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Rank")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Cumulative explained variance ratio")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Mean cumulative EVR ({args.residual_space} residual)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction_curve_figure(rows: Sequence[dict], args: argparse.Namespace, out_path: Path) -> None:
    x = np.arange(1, int(args.max_rank_plot) + 1)
    families = [("all", "All"), ("attention", "Attention"), ("mlp", "MLP")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = {"oas": "#1f77b4", "second": "#d62728"}

    for ax, (family_key, title) in zip(axes, families):
        for method in ("oas", "second"):
            y = aggregate_mean_curve(rows, method=method, family=family_key, stage="before", key="recon_err")
            if y.size == 0:
                continue
            ax.plot(x, y, label=method.upper(), color=colors[method], linewidth=2.2)
            idx = int(args.rank_eval) - 1
            if 0 <= idx < y.size:
                ax.scatter([int(args.rank_eval)], [y[idx]], color=colors[method], s=28, zorder=3)
                ax.text(
                    int(args.rank_eval) + 2,
                    float(y[idx]),
                    f"{method.upper()} {y[idx]:.3f}",
                    color=colors[method],
                    fontsize=9,
                    va="center",
                )
        ax.axvline(int(args.rank_eval), color="0.5", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Rank")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Best rank-r reconstruction error")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Rank vs reconstruction error ({args.residual_space} residual)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def correlation_safe(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def plot_scatter_figure(rows: Sequence[dict], args: argparse.Namespace, out_path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"attention": "#2ca02c", "mlp": "#ff7f0e", "other": "#7f7f7f"}

    xs_all: List[float] = []
    ys_all: List[float] = []
    by_family: Dict[str, List[Tuple[float, float]]] = {"attention": [], "mlp": [], "other": []}

    for row in rows:
        x = float(row["delta_evr_at_rank_eval_before"])
        y = float(row["delta_ab_gain"])
        fam = str(row["module_family"])
        if fam not in by_family:
            by_family[fam] = []
        by_family[fam].append((x, y))
        xs_all.append(x)
        ys_all.append(y)

    for fam, pairs in by_family.items():
        if not pairs:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.scatter(xs, ys, s=34, alpha=0.82, color=colors.get(fam, "#7f7f7f"), label=fam)

    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1.0)

    if len(xs_all) >= 2:
        coeff = np.polyfit(np.asarray(xs_all), np.asarray(ys_all), deg=1)
        xs_line = np.linspace(min(xs_all), max(xs_all), 100)
        ys_line = coeff[0] * xs_line + coeff[1]
        ax.plot(xs_line, ys_line, color="black", linewidth=1.5, alpha=0.8)

    corr_all = correlation_safe(xs_all, ys_all)
    corr_attn = correlation_safe(
        [p[0] for p in by_family.get("attention", [])],
        [p[1] for p in by_family.get("attention", [])],
    )
    corr_mlp = correlation_safe(
        [p[0] for p in by_family.get("mlp", [])],
        [p[1] for p in by_family.get("mlp", [])],
    )

    ax.set_title(f"Geometry advantage vs AB gain advantage @ rank {args.rank_eval}")
    ax.set_xlabel(f"Delta EVR@{args.rank_eval} before AB (OAS - Second)")
    ax.set_ylabel("Delta AB gain (OAS - Second)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.03,
        0.97,
        f"corr(all)={corr_all:.3f}\ncorr(attn)={corr_attn:.3f}\ncorr(mlp)={corr_mlp:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "corr_all": corr_all,
        "corr_attention": corr_attn,
        "corr_mlp": corr_mlp,
    }


def write_layerwise_csv(rows: Sequence[dict], out_path: Path) -> None:
    flat_rows: List[dict] = []
    for row in rows:
        flat_rows.append({
            "layer": row["layer"],
            "block_idx": row["block_idx"],
            "module_type": row["module_type"],
            "module_family": row["module_family"],
            "out_dim": row["out_dim"],
            "in_dim": row["in_dim"],
            "rank_eval": row["rank_eval"],
            "residual_space": row["residual_space"],
            "oas_bits": row["oas"]["bits"],
            "second_bits": row["second"]["bits"],
            "oas_ab_rank_used": row["oas"]["ab_rank_used"],
            "second_ab_rank_used": row["second"]["ab_rank_used"],
            "oas_evr_at_rank_eval_before": row["oas"]["evr_at_rank_eval_before"],
            "second_evr_at_rank_eval_before": row["second"]["evr_at_rank_eval_before"],
            "oas_evr_at_rank_eval_after": row["oas"]["evr_at_rank_eval_after"],
            "second_evr_at_rank_eval_after": row["second"]["evr_at_rank_eval_after"],
            "oas_ab_gain": row["oas"]["ab_gain"],
            "second_ab_gain": row["second"]["ab_gain"],
            "delta_evr_at_rank_eval_before": row["delta_evr_at_rank_eval_before"],
            "delta_ab_gain": row["delta_ab_gain"],
            "winner_evr": row["winner_evr"],
            "winner_ab_gain": row["winner_ab_gain"],
        })

    if not flat_rows:
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        wr.writeheader()
        for row in flat_rows:
            wr.writerow(row)


def build_summary(rows: Sequence[dict], args: argparse.Namespace, scatter_stats: dict, keys: Sequence[str], ctxs: Dict[str, dict]) -> dict:
    summary = {
        "model_id": args.model_id,
        "rank_eval": int(args.rank_eval),
        "max_rank_plot": int(args.max_rank_plot),
        "num_layers": len(rows),
        "residual_space": str(args.residual_space),
        "step1_dir_oas": ctxs["oas"]["step1_dir"],
        "calib_oas": ctxs["oas"]["calib_path"],
        "low_rank_path_oas": ctxs["oas"]["low_rank_path"],
        "step1_dir_second": ctxs["second"]["step1_dir"],
        "calib_second": ctxs["second"]["calib_path"],
        "low_rank_path_second": ctxs["second"]["low_rank_path"],
        "matched_layers": list(keys),
        "mean_oas_evr_at_rank_eval_before": aggregate_mean_scalar(
            [{"value": row["oas"]["evr_at_rank_eval_before"]} for row in rows], key="value"
        ),
        "mean_second_evr_at_rank_eval_before": aggregate_mean_scalar(
            [{"value": row["second"]["evr_at_rank_eval_before"]} for row in rows], key="value"
        ),
        "mean_oas_ab_gain": aggregate_mean_scalar(
            [{"value": row["oas"]["ab_gain"]} for row in rows], key="value"
        ),
        "mean_second_ab_gain": aggregate_mean_scalar(
            [{"value": row["second"]["ab_gain"]} for row in rows], key="value"
        ),
        "mean_delta_evr_at_rank_eval_before": aggregate_mean_scalar(rows, key="delta_evr_at_rank_eval_before"),
        "mean_delta_ab_gain": aggregate_mean_scalar(rows, key="delta_ab_gain"),
        "oas_evr_win_count": int(sum(1 for row in rows if row["winner_evr"] == "oas")),
        "second_evr_win_count": int(sum(1 for row in rows if row["winner_evr"] == "second")),
        "oas_ab_gain_win_count": int(sum(1 for row in rows if row["winner_ab_gain"] == "oas")),
        "second_ab_gain_win_count": int(sum(1 for row in rows if row["winner_ab_gain"] == "second")),
        "scatter": scatter_stats,
    }

    family_breakdown: Dict[str, dict] = {}
    for family in ("attention", "mlp"):
        sub = [row for row in rows if row["module_family"] == family]
        family_breakdown[family] = {
            "num_layers": len(sub),
            "mean_delta_evr_at_rank_eval_before": aggregate_mean_scalar(sub, key="delta_evr_at_rank_eval_before"),
            "mean_delta_ab_gain": aggregate_mean_scalar(sub, key="delta_ab_gain"),
            "mean_oas_evr_at_rank_eval_before": aggregate_mean_scalar(
                [{"value": row["oas"]["evr_at_rank_eval_before"]} for row in sub], key="value"
            ),
            "mean_second_evr_at_rank_eval_before": aggregate_mean_scalar(
                [{"value": row["second"]["evr_at_rank_eval_before"]} for row in sub], key="value"
            ),
            "mean_oas_ab_gain": aggregate_mean_scalar(
                [{"value": row["oas"]["ab_gain"]} for row in sub], key="value"
            ),
            "mean_second_ab_gain": aggregate_mean_scalar(
                [{"value": row["second"]["ab_gain"]} for row in sub], key="value"
            ),
        }
    summary["family_breakdown"] = family_breakdown
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Compare OAS vs second residual low-rank friendliness visuals")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--step1_dir_oas", required=True)
    ap.add_argument("--calib_oas", required=True)
    ap.add_argument("--low_rank_path_oas", required=True)

    ap.add_argument("--step1_dir_second", required=True)
    ap.add_argument("--calib_second", required=True)
    ap.add_argument("--low_rank_path_second", required=True)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rank_eval", type=int, default=64)
    ap.add_argument("--max_rank_plot", type=int, default=256)
    ap.add_argument("--residual_space", choices=["weighted", "raw"], default="weighted")
    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--selected_layers", nargs="*", default=None, help="Exact weight keys to analyze. Comma-separated values are also accepted.")
    ap.add_argument("--selected_blocks", nargs="*", default=None, help="Transformer block indices to analyze. Comma-separated values are also accepted.")
    ap.add_argument("--max_layers", type=int, default=0)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto")
    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--eps", type=float, default=1e-8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.rank_eval) <= 0:
        raise ValueError("--rank_eval must be >= 1")
    if int(args.max_rank_plot) < int(args.rank_eval):
        raise ValueError("--max_rank_plot must be >= --rank_eval")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device, state = load_model_state(args)
    ctxs = {
        "oas": build_method_context(
            label="oas",
            step1_dir=args.step1_dir_oas,
            calib_path=args.calib_oas,
            low_rank_path=args.low_rank_path_oas,
        ),
        "second": build_method_context(
            label="second",
            step1_dir=args.step1_dir_second,
            calib_path=args.calib_second,
            low_rank_path=args.low_rank_path_second,
        ),
    }

    keys = resolve_keys(args, state=state, ctxs=list(ctxs.values()))
    print(f"[LowRankVisual] matched layers: {len(keys)}", flush=True)

    rows: List[dict] = []
    for idx, key in enumerate(keys, start=1):
        print(f"[LowRankVisual] ({idx}/{len(keys)}) analyzing: {key}", flush=True)
        row = analyze_layer(key=key, state=state, device=device, method_ctxs=ctxs, args=args)
        rows.append(row)

        if torch.cuda.is_available() and (idx % 4 == 0 or idx == len(keys)):
            torch.cuda.empty_cache()
        if idx % 4 == 0 or idx == len(keys):
            gc.collect()

    layerwise_csv = out_dir / "layerwise_lowrank_compare.csv"
    write_layerwise_csv(rows, layerwise_csv)

    evr_png = out_dir / "figure_a_mean_cumulative_evr.png"
    recon_png = out_dir / "figure_b_rank_vs_recon_error.png"
    scatter_png = out_dir / "figure_c_evr_advantage_vs_ab_gain.png"

    plot_mean_curve_figure(rows, args, evr_png)
    plot_reconstruction_curve_figure(rows, args, recon_png)
    scatter_stats = plot_scatter_figure(rows, args, scatter_png)

    summary = build_summary(rows, args, scatter_stats=scatter_stats, keys=keys, ctxs=ctxs)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] ✅", flush=True)
    print(f"  csv     : {layerwise_csv}", flush=True)
    print(f"  fig A   : {evr_png}", flush=True)
    print(f"  fig B   : {recon_png}", flush=True)
    print(f"  fig C   : {scatter_png}", flush=True)
    print(f"  summary : {summary_path}", flush=True)
    print("\n[Quick summary]", flush=True)
    print(
        f"  mean EVR@{args.rank_eval} before AB (OAS / Second) = "
        f"{summary['mean_oas_evr_at_rank_eval_before']:.6f} / {summary['mean_second_evr_at_rank_eval_before']:.6f}",
        flush=True,
    )
    print(
        f"  mean AB gain (OAS / Second) = "
        f"{summary['mean_oas_ab_gain']:.6f} / {summary['mean_second_ab_gain']:.6f}",
        flush=True,
    )
    print(
        f"  delta (EVR / AB gain) = "
        f"{summary['mean_delta_evr_at_rank_eval_before']:.6f} / {summary['mean_delta_ab_gain']:.6f}",
        flush=True,
    )
    print(
        f"  scatter corr(all/attn/mlp) = "
        f"{scatter_stats['corr_all']:.3f} / {scatter_stats['corr_attention']:.3f} / {scatter_stats['corr_mlp']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
