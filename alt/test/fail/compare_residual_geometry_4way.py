#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_residual_geometry_4way.py

목적:
  OAS-calibrated Step1 quant 결과와 second-moment-calibrated Step1 quant 결과를 동시에 불러와,
  residual geometry를 아래 4개 조합으로 비교한다.

    1) R_oas * D_oas
    2) R_oas * D_second
    3) R_second * D_oas
    4) R_second * D_second

  where
    R_oas    = W - Wq_oas
    R_second = W - Wq_second

분석 항목:
  A. weighted residual magnitude
     - ||R D||_F
     - mean((R D)^2)

  B. singular spectrum / low-rank friendliness
     - EVR@k
     - stable rank
     - top singular energy shares

  C. weighted residual column concentration
     - column energy top-k share
     - Gini
     - normalized entropy

  D. diag weight spread
     - D_oas, D_second 각각의 cond / percentile spread / gini

  E. second-moment mean contribution
     - E[x^2] = Var(x) + mean^2
     - mean^2 / E[x^2] ratio stats

출력:
  - layerwise_compare_4way.csv
  - summary.json

예시:
CUDA_VISIBLE_DEVICES=2 python test/fail/compare_residual_geometry_4way.py \
  --model_id meta-llama/Llama-3.1-8B \
  --step1_dir_oas ./output/llama3_8b/step1_quant/2bit \
  --step1_dir_second ./output/llama3_8b_64/step1_quant/2bit \
  --calib_oas ./output/llama3_8b/calib_sqrtdiag.pt \
  --calib_second ./output/llama3_8b_64/calib_sqrtdiag.pt \
  --out_dir ./output/llama3_8b_64/compare_4way/2bit \
  --device cuda \
  --model_device_map auto \
  --dtype_w fp16 \
  --rank_list 16 32 64 128
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e


# ------------------------------------------------------------
# Target helpers
# ------------------------------------------------------------
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


def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )


def _snapshot_state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if getattr(v, "is_meta", False):
            raise NotImplementedError(f"meta tensor in state_dict: {k}")
        state[k] = v.detach().to("cpu")
    return state


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
    suffix = name.split(".")[-2] if "." in name else ""
    return (10**9 if bidx is None else bidx, MODULE_ORDER.get(suffix, 10**6), name)


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


# ------------------------------------------------------------
# Calib helpers
# ------------------------------------------------------------
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


def maybe_get_mean_var(entry: dict) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    mean = entry.get("mean", None)
    var = entry.get("var", None)
    if mean is not None:
        mean = mean.to(torch.float32).contiguous()
    if var is not None:
        var = var.to(torch.float32).contiguous()
    return mean, var


# ------------------------------------------------------------
# Math/stat helpers
# ------------------------------------------------------------
def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(max(b, eps))


def tensor_percentile(x: torch.Tensor, q: float) -> float:
    x = x.reshape(-1)
    if x.numel() == 0:
        return 0.0
    q = float(max(0.0, min(100.0, q)))
    if x.numel() == 1:
        return float(x.item())
    k = int(round((q / 100.0) * (x.numel() - 1)))
    k = max(0, min(k, x.numel() - 1))
    vals, _ = torch.sort(x)
    return float(vals[k].item())


def gini_from_nonnegative(x: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.reshape(-1).to(torch.float64).clamp_min(0.0)
    n = x.numel()
    if n == 0:
        return 0.0
    s = float(x.sum().item())
    if s <= eps:
        return 0.0
    xs, _ = torch.sort(x)
    idx = torch.arange(1, n + 1, device=xs.device, dtype=torch.float64)
    g = (2.0 * torch.sum(idx * xs) / (n * torch.sum(xs))) - (n + 1.0) / n
    return float(g.item())


def normalized_entropy_from_nonnegative(x: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.reshape(-1).to(torch.float64).clamp_min(0.0)
    s = float(x.sum().item())
    n = x.numel()
    if n == 0 or s <= eps:
        return 0.0
    p = x / s
    ent = -(p * p.clamp_min(eps).log()).sum().item()
    return float(ent / max(math.log(n), eps))


@torch.no_grad()
def residual_magnitude_metrics(m: torch.Tensor) -> Dict[str, float]:
    sq = m * m
    return {
        "fro": float(torch.norm(m, p="fro").item()),
        "mean_sq": float(sq.mean().item()),
        "sum_sq": float(sq.sum().item()),
        "max_abs": float(m.abs().max().item()) if m.numel() > 0 else 0.0,
    }


@torch.no_grad()
def svd_energy_metrics(m: torch.Tensor, rank_list: List[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if m.numel() == 0:
        return out

    s = torch.linalg.svdvals(m)
    if s.numel() == 0:
        return out

    s2 = s * s
    total = float(s2.sum().item())
    top = float(s2.max().item())

    out["stable_rank"] = safe_div(total, top)

    for k in [1, 4, 8, 16, 32, 64, 128]:
        kk = min(k, s2.numel())
        out[f"sv_top{k}_share"] = safe_div(float(s2[:kk].sum().item()), total)

    for r in rank_list:
        rr = min(int(r), s2.numel())
        out[f"evr_at_{r}"] = safe_div(float(s2[:rr].sum().item()), total)

    return out


@torch.no_grad()
def column_energy_metrics(m: torch.Tensor) -> Dict[str, float]:
    col_e = (m * m).sum(dim=0).to(torch.float32)
    total = float(col_e.sum().item())
    out: Dict[str, float] = {}

    if col_e.numel() == 0 or total <= 0.0:
        for k in [1, 4, 8, 16, 32, 64, 128]:
            out[f"col_top{k}_share"] = 0.0
        out["col_gini"] = 0.0
        out["col_entropy_norm"] = 0.0
        out["col_energy_total"] = 0.0
        return out

    col_sorted, _ = torch.sort(col_e, descending=True)
    for k in [1, 4, 8, 16, 32, 64, 128]:
        kk = min(k, col_sorted.numel())
        out[f"col_top{k}_share"] = safe_div(float(col_sorted[:kk].sum().item()), total)

    out["col_gini"] = gini_from_nonnegative(col_e)
    out["col_entropy_norm"] = normalized_entropy_from_nonnegative(col_e)
    out["col_energy_total"] = total
    return out


@torch.no_grad()
def diag_spread_metrics(d: torch.Tensor) -> Dict[str, float]:
    d = d.reshape(-1).to(torch.float32).clamp_min(1e-12)
    out: Dict[str, float] = {}
    out["d_min"] = float(d.min().item())
    out["d_max"] = float(d.max().item())
    out["d_mean"] = float(d.mean().item())
    out["d_std"] = float(d.std(unbiased=False).item())
    out["d_cond_maxmin"] = safe_div(out["d_max"], out["d_min"])

    p5 = tensor_percentile(d, 5)
    p50 = tensor_percentile(d, 50)
    p95 = tensor_percentile(d, 95)
    p99 = tensor_percentile(d, 99)

    out["d_p5"] = p5
    out["d_p50"] = p50
    out["d_p95"] = p95
    out["d_p99"] = p99
    out["d_p95_over_p5"] = safe_div(p95, p5)
    out["d_p99_over_p50"] = safe_div(p99, p50)
    out["d_gini"] = gini_from_nonnegative(d)
    return out


@torch.no_grad()
def mean_contribution_metrics(mean: Optional[torch.Tensor], ex2: Optional[torch.Tensor]) -> Dict[str, float]:
    out = {
        "mean_ratio_avg": 0.0,
        "mean_ratio_p50": 0.0,
        "mean_ratio_p90": 0.0,
        "mean_ratio_max": 0.0,
    }
    if mean is None or ex2 is None:
        return out

    ex2 = ex2.to(torch.float32).clamp_min(1e-12)
    mu2 = mean.to(torch.float32).pow(2)
    ratio = (mu2 / ex2).clamp(min=0.0, max=1e6).reshape(-1)

    out["mean_ratio_avg"] = float(ratio.mean().item())
    out["mean_ratio_p50"] = tensor_percentile(ratio, 50)
    out["mean_ratio_p90"] = tensor_percentile(ratio, 90)
    out["mean_ratio_max"] = float(ratio.max().item())
    return out


def add_prefixed(row: dict, prefix: str, items: Dict[str, float]) -> None:
    for k, v in items.items():
        row[f"{prefix}_{k}"] = float(v)


def aggregate_mean(rows: List[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


# ------------------------------------------------------------
# I/O loading
# ------------------------------------------------------------
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


def load_context(args: argparse.Namespace) -> dict:
    print(f"[4Way] loading step1 OAS artifacts: {args.step1_dir_oas}", flush=True)
    codebooks_oas, qcodes_oas, metas_oas = load_step1_artifacts(args.step1_dir_oas)

    print(f"[4Way] loading step1 second artifacts: {args.step1_dir_second}", flush=True)
    codebooks_sec, qcodes_sec, metas_sec = load_step1_artifacts(args.step1_dir_second)

    print(f"[4Way] loading calib_oas: {args.calib_oas}", flush=True)
    calib_oas_payload = torch.load(args.calib_oas, map_location="cpu")
    calib_oas: Dict[str, dict] = calib_oas_payload.get("cov_ops", calib_oas_payload)

    print(f"[4Way] loading calib_second: {args.calib_second}", flush=True)
    calib_sec_payload = torch.load(args.calib_second, map_location="cpu")
    calib_sec: Dict[str, dict] = calib_sec_payload.get("cov_ops", calib_sec_payload)

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
        f"[4Way] loading original model: {args.model_id} "
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
        print("[4Way] Detected meta tensors. Re-loading model on CPU.", flush=True)
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

    layer_re = re.compile(args.layer_regex) if args.layer_regex else None
    keys: List[str] = []

    keyset = (
        set(codebooks_oas.keys())
        & set(qcodes_oas.keys())
        & set(metas_oas.keys())
        & set(codebooks_sec.keys())
        & set(qcodes_sec.keys())
        & set(metas_sec.keys())
        & set(calib_oas.keys())
        & set(calib_sec.keys())
        & set(state.keys())
    )

    for key in keyset:
        if not is_target_weight(key, state[key]):
            continue
        if layer_re and not layer_re.search(key):
            continue
        keys.append(key)

    keys = sorted(keys, key=sort_key)
    if args.max_layers > 0:
        keys = keys[: int(args.max_layers)]
    if not keys:
        raise RuntimeError("No matched layers found.")

    print(f"[4Way] matched layers: {len(keys)}", flush=True)

    return {
        "device": device,
        "state": state,
        "keys": keys,
        "codebooks_oas": codebooks_oas,
        "qcodes_oas": qcodes_oas,
        "metas_oas": metas_oas,
        "codebooks_sec": codebooks_sec,
        "qcodes_sec": qcodes_sec,
        "metas_sec": metas_sec,
        "calib_oas": calib_oas,
        "calib_sec": calib_sec,
    }


# ------------------------------------------------------------
# Layer analysis
# ------------------------------------------------------------
@torch.no_grad()
def analyze_layer(key: str, ctx: dict, args: argparse.Namespace) -> dict:
    device = ctx["device"]
    state = ctx["state"]

    codebooks_oas = ctx["codebooks_oas"]
    qcodes_oas = ctx["qcodes_oas"]
    metas_oas = ctx["metas_oas"]

    codebooks_sec = ctx["codebooks_sec"]
    qcodes_sec = ctx["qcodes_sec"]
    metas_sec = ctx["metas_sec"]

    calib_oas = ctx["calib_oas"]
    calib_sec = ctx["calib_sec"]

    meta_oas = metas_oas[key]
    meta_sec = metas_sec[key]

    orig_i_oas = int(tuple(meta_oas["orig_shape"])[1])
    orig_i_sec = int(tuple(meta_sec["orig_shape"])[1])

    if orig_i_oas != orig_i_sec:
        raise RuntimeError(f"orig_I mismatch between oas/sec meta on {key}: {orig_i_oas} vs {orig_i_sec}")

    orig_i = orig_i_oas
    w_cpu = state[key].to(torch.float32)
    if w_cpu.shape[1] != orig_i:
        raise RuntimeError(f"weight/meta mismatch on {key}: weight_in={w_cpu.shape[1]} meta_in={orig_i}")

    w = w_cpu.to(device)

    # reconstruct Wq_oas
    cb_oas = codebooks_oas[key].to(device=device, dtype=torch.float32)
    qc_oas = qcodes_oas[key].to(device=device)
    wq_oas = dequant_from_codebook_codes(cb_oas, qc_oas, orig_i=orig_i)

    # reconstruct Wq_second
    cb_sec = codebooks_sec[key].to(device=device, dtype=torch.float32)
    qc_sec = qcodes_sec[key].to(device=device)
    wq_sec = dequant_from_codebook_codes(cb_sec, qc_sec, orig_i=orig_i)

    # residuals
    r_oas = (w - wq_oas).to(torch.float32)
    r_sec = (w - wq_sec).to(torch.float32)

    # diag weights
    d_oas = load_diag_weight(calib_oas[key], eps=float(args.eps)).to(device)
    d_sec = load_diag_weight(calib_sec[key], eps=float(args.eps)).to(device)

    if d_oas.numel() != orig_i or d_sec.numel() != orig_i:
        raise RuntimeError(f"diag size mismatch on {key}")

    # 4-way weighted residuals
    ro_do = r_oas * d_oas.unsqueeze(0)   # R_oas D_oas
    ro_ds = r_oas * d_sec.unsqueeze(0)   # R_oas D_second
    rs_do = r_sec * d_oas.unsqueeze(0)   # R_second D_oas
    rs_ds = r_sec * d_sec.unsqueeze(0)   # R_second D_second

    row = {
        "layer": key,
        "bits_oas": int(meta_oas["bits"]),
        "bits_second": int(meta_sec["bits"]),
        "group_size_oas": int(meta_oas["group_size"]),
        "group_size_second": int(meta_sec["group_size"]),
        "out_dim": int(w.shape[0]),
        "in_dim": int(w.shape[1]),

        "residual_fro_oas": float(torch.norm(r_oas, p="fro").item()),
        "residual_fro_second": float(torch.norm(r_sec, p="fro").item()),
        "delta_residual_fro_oas_minus_second": float(torch.norm(r_oas, p="fro").item() - torch.norm(r_sec, p="fro").item()),
    }

    rank_list = [int(x) for x in args.rank_list]

    # residual-only delta
    raw_sq_oas = float((r_oas * r_oas).mean().item())
    raw_sq_sec = float((r_sec * r_sec).mean().item())
    row["residual_mean_sq_oas"] = raw_sq_oas
    row["residual_mean_sq_second"] = raw_sq_sec
    row["delta_residual_mean_sq_oas_minus_second"] = raw_sq_oas - raw_sq_sec

    # 4-way metrics
    combos = {
        "ro_do": ro_do,
        "ro_ds": ro_ds,
        "rs_do": rs_do,
        "rs_ds": rs_ds,
    }

    for prefix, m in combos.items():
        add_prefixed(row, prefix, residual_magnitude_metrics(m))
        add_prefixed(row, prefix, svd_energy_metrics(m, rank_list))
        add_prefixed(row, prefix, column_energy_metrics(m))

    # diag metrics
    add_prefixed(row, "do", diag_spread_metrics(d_oas.detach().cpu()))
    add_prefixed(row, "ds", diag_spread_metrics(d_sec.detach().cpu()))

    # mean contribution: second-moment 쪽만 의미가 큼
    mean_sec, ex2_sec = maybe_get_mean_var(calib_sec[key])
    add_prefixed(row, "second", mean_contribution_metrics(mean_sec, ex2_sec))

    # key deltas for interpretation
    for r in rank_list:
        k_rodo = f"ro_do_evr_at_{r}"
        k_rsds = f"rs_ds_evr_at_{r}"
        k_rods = f"ro_ds_evr_at_{r}"
        k_rsdo = f"rs_do_evr_at_{r}"

        if k_rodo in row and k_rsds in row:
            row[f"delta_self_evr_at_{r}_rodo_minus_rsds"] = float(row[k_rodo] - row[k_rsds])
        if k_rods in row and k_rsdo in row:
            row[f"delta_cross_evr_at_{r}_rods_minus_rsdo"] = float(row[k_rods] - row[k_rsdo])

    # self metric comparison: each quantizer under its own metric
    row["delta_self_fro_rodo_minus_rsds"] = float(row["ro_do_fro"] - row["rs_ds_fro"])
    row["delta_self_mean_sq_rodo_minus_rsds"] = float(row["ro_do_mean_sq"] - row["rs_ds_mean_sq"])
    row["delta_self_stable_rank_rodo_minus_rsds"] = float(row.get("ro_do_stable_rank", 0.0) - row.get("rs_ds_stable_rank", 0.0))
    row["delta_self_col_gini_rodo_minus_rsds"] = float(row.get("ro_do_col_gini", 0.0) - row.get("rs_ds_col_gini", 0.0))

    # same metric comparison: under OAS metric only / under second metric only
    row["delta_oas_metric_fro_ro_minus_rs"] = float(row["ro_do_fro"] - row["rs_do_fro"])
    row["delta_second_metric_fro_ro_minus_rs"] = float(row["ro_ds_fro"] - row["rs_ds_fro"])

    for r in rank_list:
        row[f"delta_oas_metric_evr_at_{r}_ro_minus_rs"] = float(
            row.get(f"ro_do_evr_at_{r}", 0.0) - row.get(f"rs_do_evr_at_{r}", 0.0)
        )
        row[f"delta_second_metric_evr_at_{r}_ro_minus_rs"] = float(
            row.get(f"ro_ds_evr_at_{r}", 0.0) - row.get(f"rs_ds_evr_at_{r}", 0.0)
        )

    # metric mismatch sensitivity for each residual
    row["delta_metric_for_ro_fro_do_minus_ds"] = float(row["ro_do_fro"] - row["ro_ds_fro"])
    row["delta_metric_for_rs_fro_do_minus_ds"] = float(row["rs_do_fro"] - row["rs_ds_fro"])
    row["delta_diag_cond_do_minus_ds"] = float(row["do_d_cond_maxmin"] - row["ds_d_cond_maxmin"])
    row["delta_diag_p99p50_do_minus_ds"] = float(row["do_d_p99_over_p50"] - row["ds_d_p99_over_p50"])

    for r in rank_list:
        row[f"delta_metric_for_ro_evr_at_{r}_do_minus_ds"] = float(
            row.get(f"ro_do_evr_at_{r}", 0.0) - row.get(f"ro_ds_evr_at_{r}", 0.0)
        )
        row[f"delta_metric_for_rs_evr_at_{r}_do_minus_ds"] = float(
            row.get(f"rs_do_evr_at_{r}", 0.0) - row.get(f"rs_ds_evr_at_{r}", 0.0)
        )

    # cleanup
    del w, wq_oas, wq_sec
    del r_oas, r_sec
    del ro_do, ro_ds, rs_do, rs_ds
    del d_oas, d_sec
    del cb_oas, qc_oas, cb_sec, qc_sec

    return row


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("4-way residual geometry comparison for OAS vs second-moment Step1")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--step1_dir_oas", required=True)
    ap.add_argument("--step1_dir_second", required=True)
    ap.add_argument("--calib_oas", required=True)
    ap.add_argument("--calib_second", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto")
    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=0)
    ap.add_argument("--rank_list", type=int, nargs="+", default=[16, 32, 64, 128])

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_context(args)

    rows: List[dict] = []
    for idx, key in enumerate(ctx["keys"], start=1):
        print(f"[4Way] ({idx}/{len(ctx['keys'])}) analyzing: {key}", flush=True)
        row = analyze_layer(key, ctx, args)
        rows.append(row)

        if torch.cuda.is_available() and (idx % 4 == 0 or idx == len(ctx["keys"])):
            torch.cuda.empty_cache()
        if idx % 4 == 0 or idx == len(ctx["keys"]):
            gc.collect()

    csv_path = out_dir / "layerwise_compare_4way.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)

    summary = {
        "model_id": args.model_id,
        "step1_dir_oas": str(Path(args.step1_dir_oas).resolve()),
        "step1_dir_second": str(Path(args.step1_dir_second).resolve()),
        "calib_oas": str(Path(args.calib_oas).resolve()),
        "calib_second": str(Path(args.calib_second).resolve()),
        "out_dir": str(out_dir),
        "num_layers": len(rows),

        # self/self
        "mean_ro_do_fro": aggregate_mean(rows, "ro_do_fro"),
        "mean_rs_ds_fro": aggregate_mean(rows, "rs_ds_fro"),
        "mean_delta_self_fro_rodo_minus_rsds": aggregate_mean(rows, "delta_self_fro_rodo_minus_rsds"),

        "mean_ro_do_evr_at_64": aggregate_mean(rows, "ro_do_evr_at_64"),
        "mean_rs_ds_evr_at_64": aggregate_mean(rows, "rs_ds_evr_at_64"),
        "mean_delta_self_evr_at_64_rodo_minus_rsds": aggregate_mean(rows, "delta_self_evr_at_64_rodo_minus_rsds"),

        "mean_ro_do_stable_rank": aggregate_mean(rows, "ro_do_stable_rank"),
        "mean_rs_ds_stable_rank": aggregate_mean(rows, "rs_ds_stable_rank"),

        # same metric comparison
        "mean_ro_do_vs_rs_do_evr_at_64_delta": aggregate_mean(rows, "delta_oas_metric_evr_at_64_ro_minus_rs"),
        "mean_ro_ds_vs_rs_ds_evr_at_64_delta": aggregate_mean(rows, "delta_second_metric_evr_at_64_ro_minus_rs"),

        # mismatch sensitivity
        "mean_metric_for_ro_evr_at_64_do_minus_ds": aggregate_mean(rows, "delta_metric_for_ro_evr_at_64_do_minus_ds"),
        "mean_metric_for_rs_evr_at_64_do_minus_ds": aggregate_mean(rows, "delta_metric_for_rs_evr_at_64_do_minus_ds"),

        # diag spread
        "mean_do_d_cond_maxmin": aggregate_mean(rows, "do_d_cond_maxmin"),
        "mean_ds_d_cond_maxmin": aggregate_mean(rows, "ds_d_cond_maxmin"),
        "mean_do_d_p99_over_p50": aggregate_mean(rows, "do_d_p99_over_p50"),
        "mean_ds_d_p99_over_p50": aggregate_mean(rows, "ds_d_p99_over_p50"),

        # second mean term
        "mean_second_mean_ratio_avg": aggregate_mean(rows, "second_mean_ratio_avg"),
        "mean_second_mean_ratio_p90": aggregate_mean(rows, "second_mean_ratio_p90"),

        # raw residual
        "mean_residual_fro_oas": aggregate_mean(rows, "residual_fro_oas"),
        "mean_residual_fro_second": aggregate_mean(rows, "residual_fro_second"),
        "mean_delta_residual_fro_oas_minus_second": aggregate_mean(rows, "delta_residual_fro_oas_minus_second"),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] ✅")
    print(f"  csv    : {csv_path}")
    print(f"  summary: {summary_path}")

    if rows:
        print("\n[Quick summary]")
        print(f"  self Fro      | ro_do={summary['mean_ro_do_fro']:.6f} | rs_ds={summary['mean_rs_ds_fro']:.6f}")
        print(f"  self EVR@64   | ro_do={summary['mean_ro_do_evr_at_64']:.6f} | rs_ds={summary['mean_rs_ds_evr_at_64']:.6f}")
        print(f"  stable-rank   | ro_do={summary['mean_ro_do_stable_rank']:.6f} | rs_ds={summary['mean_rs_ds_stable_rank']:.6f}")
        print(f"  d p99/p50     | do={summary['mean_do_d_p99_over_p50']:.6f} | ds={summary['mean_ds_d_p99_over_p50']:.6f}")
        print(f"  mean^2/E[x^2] | second avg={summary['mean_second_mean_ratio_avg']:.6f} | p90={summary['mean_second_mean_ratio_p90']:.6f}")


if __name__ == "__main__":
    main()