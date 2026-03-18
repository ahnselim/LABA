#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
analyze_ab_reduction.py

목적:
  Step1 quantization 결과와 Step3 low-rank AB correction을 불러와,
  특정 calibration metric(OAS / second / self) 기준에서
  weighted residual reduction을 레이어별로 분석한다.

핵심 비교:
  W        : original weight
  Wq       : step1 quantized weight
  AB       : step3 low-rank correction
  R0       : W - Wq
  R1       : W - (Wq + AB)
  R0D, R1D : right-weighted residual with chosen diag metric D

출력:
  - layerwise_ab_reduction.csv
  - summary.json
  
CUDA_VISIBLE_DEVICES=1 python test/analyze_ab_reduction.py \
  --model_id meta-llama/Llama-3.1-8B \
  --step1_dir ./output/llama3_8b/step1_quant/2bit \
  --calib_path ./output/llama3_8b/calib_sqrtdiag.pt \
  --low_rank_path ./output/llama3_8b/step3_svd/2bit/low_rank_ab.pt \
  --label oas \
  --compare_step1_dir ./output/llama3_8b_64/step1_quant/2bit \
  --compare_calib_path ./output/llama3_8b_64/calib_sqrtdiag.pt \
  --compare_low_rank_path ./output/llama3_8b_64/step3_svd/2bit/low_rank_ab.pt \
  --compare_label second \
  --calib_oas ./output/llama3_8b/calib_sqrtdiag.pt \
  --calib_second ./output/llama3_8b_64/calib_sqrtdiag.pt \
  --metric_source self \
  --rank_eval 64 \
  --out_dir ./output/geometry_causal/2bit/oas_vs_second_ab_reduction


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


def sort_key(name: str) -> Tuple[int, int, str]:
    bidx = extract_block_index(name)
    suffix = module_type_from_key(name)
    return (10**9 if bidx is None else bidx, MODULE_ORDER.get(suffix, 10**6), name)


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(max(b, eps))


def add_prefixed(row: dict, prefix: str, items: Dict[str, float]) -> None:
    for k, v in items.items():
        row[f"{prefix}_{k}"] = float(v)


def aggregate_mean(rows: List[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


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


def load_low_rank_ab(low_rank_path: str) -> Dict[str, dict]:
    payload = torch.load(low_rank_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported low_rank_ab payload type: {type(payload)!r}")
    return payload


def load_calib_map(calib_path: str) -> Dict[str, dict]:
    payload = torch.load(calib_path, map_location="cpu")
    calib_map = payload.get("cov_ops", payload)
    if not isinstance(calib_map, dict):
        raise TypeError(f"Unsupported calib payload type: {type(calib_map)!r}")
    return calib_map


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


@torch.no_grad()
def svd_energy_metrics(m: torch.Tensor, rank_eval: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if m.numel() == 0:
        out[f"evr_at_{rank_eval}"] = 0.0
        out["stable_rank"] = 0.0
        return out

    s = torch.linalg.svdvals(m)
    if s.numel() == 0:
        out[f"evr_at_{rank_eval}"] = 0.0
        out["stable_rank"] = 0.0
        return out

    s2 = s * s
    total = float(s2.sum().item())
    top = float(s2.max().item()) if s2.numel() > 0 else 0.0
    rr = min(int(rank_eval), int(s2.numel()))

    out[f"evr_at_{rank_eval}"] = safe_div(float(s2[:rr].sum().item()), total)
    out["stable_rank"] = safe_div(total, top)
    return out


@torch.no_grad()
def residual_magnitude_metrics(before: torch.Tensor, after: torch.Tensor) -> Dict[str, float]:
    before_sq = before * before
    after_sq = after * after

    before_fro = float(torch.norm(before, p="fro").item())
    after_fro = float(torch.norm(after, p="fro").item())
    before_sum_sq = float(before_sq.sum().item())
    after_sum_sq = float(after_sq.sum().item())
    before_mean_sq = float(before_sq.mean().item())
    after_mean_sq = float(after_sq.mean().item())

    return {
        "before_fro": before_fro,
        "after_fro": after_fro,
        "before_mean_sq": before_mean_sq,
        "after_mean_sq": after_mean_sq,
        "before_sum_sq": before_sum_sq,
        "after_sum_sq": after_sum_sq,
        "fro_reduction_abs": before_fro - after_fro,
        "fro_reduction_ratio": safe_div(before_fro - after_fro, before_fro),
        "mean_sq_reduction_abs": before_mean_sq - after_mean_sq,
        "mean_sq_reduction_ratio": safe_div(before_mean_sq - after_mean_sq, before_mean_sq),
        "captured_weighted_energy": safe_div(before_sum_sq - after_sum_sq, before_sum_sq),
    }


@torch.no_grad()
def overfocus_metrics(d_eval: torch.Tensor, d_self: torch.Tensor, residual: torch.Tensor) -> Dict[str, float]:
    rho = (d_eval.to(torch.float32) / d_self.to(torch.float32).clamp_min(1e-12)).reshape(-1)
    rho = rho.clamp_min(0.0)

    residual_col_e = (residual.to(torch.float32) * residual.to(torch.float32)).sum(dim=0)
    total_col_e = float(residual_col_e.sum().item())

    top1 = float(rho.max().item()) if rho.numel() > 0 else 0.0
    if rho.numel() == 0:
        p95 = 0.0
    else:
        vals, _ = torch.sort(rho)
        idx = min(max(int(round(0.95 * (vals.numel() - 1))), 0), vals.numel() - 1)
        p95 = float(vals[idx].item())

    rho_norm = rho / rho.sum().clamp_min(1e-12)
    ent = float(-(rho_norm * rho_norm.clamp_min(1e-12).log()).sum().item()) if rho.numel() else 0.0
    ent_norm = safe_div(ent, math.log(max(int(rho.numel()), 2)))

    rho_sorted, idx_sorted = torch.sort(rho, descending=True)
    out: Dict[str, float] = {
        "rho_top1": top1,
        "rho_p95": p95,
        "rho_entropy_norm": ent_norm,
    }
    if rho.numel() == 0:
        out["rho_gini"] = 0.0
    else:
        xs, _ = torch.sort(rho.to(torch.float64))
        n = xs.numel()
        idx = torch.arange(1, n + 1, device=xs.device, dtype=torch.float64)
        denom = torch.sum(xs)
        out["rho_gini"] = float(((2.0 * torch.sum(idx * xs) / (n * denom.clamp_min(1e-12))) - (n + 1.0) / n).item())

    for k in [1, 4, 8, 16, 32, 64, 128]:
        kk = min(k, int(idx_sorted.numel()))
        if kk == 0 or total_col_e <= 0.0:
            out[f"rho_top{k}_residual_col_energy_share"] = 0.0
        else:
            share = float(residual_col_e[idx_sorted[:kk]].sum().item())
            out[f"rho_top{k}_residual_col_energy_share"] = safe_div(share, total_col_e)
    return out


def resolve_eval_calib_map_for(
    metric_source: str,
    calib_path: Optional[str],
    calib_oas: Optional[str],
    calib_second: Optional[str],
) -> Tuple[Dict[str, dict], str]:
    metric_source = str(metric_source).lower()
    if metric_source == "self":
        if not calib_path:
            raise ValueError("--metric_source self requires --calib_path")
        return load_calib_map(calib_path), str(Path(calib_path).resolve())
    if metric_source == "oas":
        if not calib_oas:
            raise ValueError("--metric_source oas requires --calib_oas")
        return load_calib_map(calib_oas), str(Path(calib_oas).resolve())
    if metric_source == "second":
        if not calib_second:
            raise ValueError("--metric_source second requires --calib_second")
        return load_calib_map(calib_second), str(Path(calib_second).resolve())
    raise ValueError(f"Unknown metric_source: {metric_source}")


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
        f"[AB-Reduction] loading original model: {args.model_id} "
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
        print("[AB-Reduction] Detected meta tensors. Re-loading model on CPU.", flush=True)
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


def load_method_context(
    *,
    args: argparse.Namespace,
    step1_dir: str,
    low_rank_path: str,
    calib_path: Optional[str],
    state: Dict[str, torch.Tensor],
    device: torch.device,
    label: str,
) -> dict:
    print(f"[AB-Reduction][{label}] loading step1 artifacts: {step1_dir}", flush=True)
    codebooks, qcodes, metas = load_step1_artifacts(step1_dir)

    print(f"[AB-Reduction][{label}] loading low_rank_ab: {low_rank_path}", flush=True)
    low_rank_ab = load_low_rank_ab(low_rank_path)

    eval_calib, eval_calib_path = resolve_eval_calib_map_for(
        metric_source=args.metric_source,
        calib_path=calib_path,
        calib_oas=args.calib_oas,
        calib_second=args.calib_second,
    )
    print(f"[AB-Reduction][{label}] eval metric source={args.metric_source} -> {eval_calib_path}", flush=True)

    self_calib = None
    if calib_path:
        print(f"[AB-Reduction][{label}] loading self calib: {calib_path}", flush=True)
        self_calib = load_calib_map(calib_path)

    keyset = set(codebooks.keys()) & set(qcodes.keys()) & set(metas.keys()) & set(low_rank_ab.keys()) & set(eval_calib.keys()) & set(state.keys())
    if self_calib is not None:
        keyset &= set(self_calib.keys())

    return {
        "device": device,
        "state": state,
        "keyset": keyset,
        "codebooks": codebooks,
        "qcodes": qcodes,
        "metas": metas,
        "low_rank_ab": low_rank_ab,
        "eval_calib": eval_calib,
        "self_calib": self_calib,
        "eval_calib_path": eval_calib_path,
        "label": label,
        "step1_dir": str(Path(step1_dir).resolve()),
        "low_rank_path": str(Path(low_rank_path).resolve()),
        "self_calib_path": (None if calib_path is None else str(Path(calib_path).resolve())),
    }


def resolve_keys(args: argparse.Namespace, state: Dict[str, torch.Tensor], method_contexts: List[dict]) -> List[str]:
    layer_re = re.compile(args.layer_regex) if args.layer_regex else None
    keyset = set(state.keys())
    for ctx in method_contexts:
        keyset &= set(ctx["keyset"])

    keys: List[str] = []
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
    return keys


@torch.no_grad()
def analyze_layer(key: str, ctx: dict, args: argparse.Namespace) -> dict:
    device = ctx["device"]
    state = ctx["state"]
    codebooks = ctx["codebooks"]
    qcodes_dict = ctx["qcodes"]
    metas = ctx["metas"]
    low_rank_ab = ctx["low_rank_ab"]
    eval_calib = ctx["eval_calib"]
    self_calib = ctx["self_calib"]

    meta = metas[key]
    orig_i = int(tuple(meta["orig_shape"])[1])

    w_cpu = state[key].to(torch.float32)
    if w_cpu.shape[1] != orig_i:
        raise RuntimeError(f"weight/meta mismatch on {key}: weight_in={w_cpu.shape[1]} meta_in={orig_i}")

    w = w_cpu.to(device)
    codebook = codebooks[key].to(device=device, dtype=torch.float32)
    qcodes = qcodes_dict[key].to(device=device)
    wq = dequant_from_codebook_codes(codebook, qcodes, orig_i=orig_i)
    ab, rank_used = reconstruct_ab(low_rank_ab[key], out_dim=int(w.shape[0]), in_dim=int(w.shape[1]), device=device)

    d_eval = load_diag_weight(eval_calib[key], eps=float(args.eps)).to(device=device, dtype=torch.float32)
    if d_eval.numel() != orig_i:
        raise RuntimeError(f"eval diag size mismatch on {key}: expected {orig_i}, got {d_eval.numel()}")

    d_self = None
    if self_calib is not None and key in self_calib:
        d_self = load_diag_weight(self_calib[key], eps=float(args.eps)).to(device=device, dtype=torch.float32)
        if d_self.numel() != orig_i:
            raise RuntimeError(f"self diag size mismatch on {key}: expected {orig_i}, got {d_self.numel()}")

    r0 = (w - wq).to(torch.float32)
    r1 = (w - (wq + ab)).to(torch.float32)
    r0d = r0 * d_eval.unsqueeze(0)
    r1d = r1 * d_eval.unsqueeze(0)

    row = {
        "layer": key,
        "block_idx": -1 if extract_block_index(key) is None else int(extract_block_index(key)),
        "module_type": module_type_from_key(key),
        "module_family": module_family(module_type_from_key(key)),
        "out_dim": int(w.shape[0]),
        "in_dim": int(w.shape[1]),
        "bits": int(meta.get("bits", -1)),
        "group_size": int(meta.get("group_size", -1)),
        "rank_eval": int(args.rank_eval),
        "ab_rank_used": int(rank_used),
        "metric_source": str(args.metric_source),
    }

    add_prefixed(row, "weighted", residual_magnitude_metrics(r0d, r1d))

    before_spec = svd_energy_metrics(r0d, rank_eval=int(args.rank_eval))
    after_spec = svd_energy_metrics(r1d, rank_eval=int(args.rank_eval))
    row[f"before_evr_at_{args.rank_eval}"] = float(before_spec[f"evr_at_{args.rank_eval}"])
    row[f"after_evr_at_{args.rank_eval}"] = float(after_spec[f"evr_at_{args.rank_eval}"])
    row["before_stable_rank"] = float(before_spec["stable_rank"])
    row["after_stable_rank"] = float(after_spec["stable_rank"])
    row[f"delta_evr_at_{args.rank_eval}"] = float(row[f"after_evr_at_{args.rank_eval}"] - row[f"before_evr_at_{args.rank_eval}"])
    row["delta_stable_rank"] = float(row["after_stable_rank"] - row["before_stable_rank"])

    row["raw_residual_fro_before"] = float(torch.norm(r0, p="fro").item())
    row["raw_residual_fro_after"] = float(torch.norm(r1, p="fro").item())
    row["raw_ab_fro"] = float(torch.norm(ab, p="fro").item())
    row["raw_wq_fro"] = float(torch.norm(wq, p="fro").item())

    if d_self is not None:
        add_prefixed(row, "overfocus", overfocus_metrics(d_eval=d_eval, d_self=d_self, residual=r0))

    low_rank_meta = low_rank_ab[key].get("meta", {}) if isinstance(low_rank_ab[key], dict) else {}
    if isinstance(low_rank_meta, dict):
        for k in ("rank", "rank_used", "objective_weighted", "weighted_residual_energy", "weighted_evr_at_rank", "bits", "group_size"):
            if k in low_rank_meta and low_rank_meta[k] is not None:
                value = low_rank_meta[k]
                row[f"step3_{k}"] = float(value) if isinstance(value, float) else int(value)

    del w, wq, ab, r0, r1, r0d, r1d, codebook, qcodes, d_eval
    if d_self is not None:
        del d_self

    return row


def prefixed_row(row: dict, prefix: str, keep_keys: Tuple[str, ...] = ("layer", "block_idx", "module_type", "module_family", "out_dim", "in_dim", "rank_eval", "metric_source")) -> dict:
    out: dict = {}
    for key in keep_keys:
        if key in row:
            out[key] = row[key]
    for key, value in row.items():
        if key in keep_keys:
            continue
        out[f"{prefix}_{key}"] = value
    return out


def compare_rows(row_a: dict, row_b: dict, label_a: str, label_b: str, rank_eval: int) -> dict:
    base = prefixed_row(row_a, label_a)
    other = prefixed_row(row_b, label_b)
    row = dict(base)
    row.update(other)

    delta_specs = [
        ("weighted_captured_weighted_energy", "higher"),
        ("weighted_fro_reduction_ratio", "higher"),
        ("weighted_mean_sq_reduction_ratio", "higher"),
        (f"after_evr_at_{rank_eval}", "higher"),
        ("after_stable_rank", "lower"),
        ("weighted_after_fro", "lower"),
        ("weighted_after_mean_sq", "lower"),
    ]
    for metric, better in delta_specs:
        ka = f"{label_a}_{metric}"
        kb = f"{label_b}_{metric}"
        if ka not in row or kb not in row:
            continue
        delta = float(row[ka]) - float(row[kb])
        row[f"delta_{label_a}_minus_{label_b}_{metric}"] = delta
        if better == "higher":
            row[f"winner_{metric}"] = label_a if delta > 0 else (label_b if delta < 0 else "tie")
        else:
            row[f"winner_{metric}"] = label_a if delta < 0 else (label_b if delta > 0 else "tie")
    return row


def build_summary(rows: List[dict], args: argparse.Namespace, ctx: dict) -> dict:
    summary = {
        "model_id": args.model_id,
        "step1_dir": str(Path(args.step1_dir).resolve()),
        "low_rank_path": str(Path(args.low_rank_path).resolve()),
        "metric_source": str(args.metric_source),
        "eval_calib_path": str(ctx["eval_calib_path"]),
        "self_calib_path": None if args.calib_path is None else str(Path(args.calib_path).resolve()),
        "num_layers": len(rows),
        "rank_eval": int(args.rank_eval),
        "mean_before_fro": aggregate_mean(rows, "weighted_before_fro"),
        "mean_after_fro": aggregate_mean(rows, "weighted_after_fro"),
        "mean_before_mean_sq": aggregate_mean(rows, "weighted_before_mean_sq"),
        "mean_after_mean_sq": aggregate_mean(rows, "weighted_after_mean_sq"),
        "mean_fro_reduction_abs": aggregate_mean(rows, "weighted_fro_reduction_abs"),
        "mean_fro_reduction_ratio": aggregate_mean(rows, "weighted_fro_reduction_ratio"),
        "mean_mean_sq_reduction_ratio": aggregate_mean(rows, "weighted_mean_sq_reduction_ratio"),
        "mean_captured_weighted_energy": aggregate_mean(rows, "weighted_captured_weighted_energy"),
        f"mean_before_evr_at_{args.rank_eval}": aggregate_mean(rows, f"before_evr_at_{args.rank_eval}"),
        f"mean_after_evr_at_{args.rank_eval}": aggregate_mean(rows, f"after_evr_at_{args.rank_eval}"),
        "mean_before_stable_rank": aggregate_mean(rows, "before_stable_rank"),
        "mean_after_stable_rank": aggregate_mean(rows, "after_stable_rank"),
        "num_positive_energy_capture": int(sum(1 for r in rows if float(r.get("weighted_captured_weighted_energy", 0.0)) > 0.0)),
        "num_negative_energy_capture": int(sum(1 for r in rows if float(r.get("weighted_captured_weighted_energy", 0.0)) < 0.0)),
    }

    attn_rows = [r for r in rows if r.get("module_family") == "attention"]
    mlp_rows = [r for r in rows if r.get("module_family") == "mlp"]
    summary["attn_mean_captured_weighted_energy"] = aggregate_mean(attn_rows, "weighted_captured_weighted_energy")
    summary["mlp_mean_captured_weighted_energy"] = aggregate_mean(mlp_rows, "weighted_captured_weighted_energy")
    summary["attn_mean_fro_reduction_ratio"] = aggregate_mean(attn_rows, "weighted_fro_reduction_ratio")
    summary["mlp_mean_fro_reduction_ratio"] = aggregate_mean(mlp_rows, "weighted_fro_reduction_ratio")

    module_breakdown: Dict[str, dict] = {}
    for module in sorted({str(r.get("module_type", "unknown")) for r in rows}):
        sub = [r for r in rows if r.get("module_type") == module]
        module_breakdown[module] = {
            "num_layers": len(sub),
            "mean_captured_weighted_energy": aggregate_mean(sub, "weighted_captured_weighted_energy"),
            "mean_fro_reduction_ratio": aggregate_mean(sub, "weighted_fro_reduction_ratio"),
            "mean_mean_sq_reduction_ratio": aggregate_mean(sub, "weighted_mean_sq_reduction_ratio"),
            f"mean_before_evr_at_{args.rank_eval}": aggregate_mean(sub, f"before_evr_at_{args.rank_eval}"),
            f"mean_after_evr_at_{args.rank_eval}": aggregate_mean(sub, f"after_evr_at_{args.rank_eval}"),
            "mean_before_stable_rank": aggregate_mean(sub, "before_stable_rank"),
            "mean_after_stable_rank": aggregate_mean(sub, "after_stable_rank"),
        }
    summary["module_breakdown"] = module_breakdown

    if rows and "overfocus_rho_top1" in rows[0]:
        summary["mean_overfocus_rho_top1"] = aggregate_mean(rows, "overfocus_rho_top1")
        summary["mean_overfocus_rho_p95"] = aggregate_mean(rows, "overfocus_rho_p95")
        summary["mean_overfocus_rho_gini"] = aggregate_mean(rows, "overfocus_rho_gini")
        summary["mean_overfocus_rho_entropy_norm"] = aggregate_mean(rows, "overfocus_rho_entropy_norm")

    return summary


def build_compare_summary(rows: List[dict], args: argparse.Namespace, ctx_a: dict, ctx_b: dict, label_a: str, label_b: str) -> dict:
    summary = {
        "model_id": args.model_id,
        "metric_source": str(args.metric_source),
        "eval_calib_path": str(ctx_a["eval_calib_path"]),
        "num_layers": len(rows),
        "rank_eval": int(args.rank_eval),
        "method_a": {
            "label": label_a,
            "step1_dir": ctx_a["step1_dir"],
            "low_rank_path": ctx_a["low_rank_path"],
            "self_calib_path": ctx_a["self_calib_path"],
            "mean_captured_weighted_energy": aggregate_mean(rows, f"{label_a}_weighted_captured_weighted_energy"),
            "mean_fro_reduction_ratio": aggregate_mean(rows, f"{label_a}_weighted_fro_reduction_ratio"),
            "mean_mean_sq_reduction_ratio": aggregate_mean(rows, f"{label_a}_weighted_mean_sq_reduction_ratio"),
            f"mean_after_evr_at_{args.rank_eval}": aggregate_mean(rows, f"{label_a}_after_evr_at_{args.rank_eval}"),
            "mean_after_stable_rank": aggregate_mean(rows, f"{label_a}_after_stable_rank"),
        },
        "method_b": {
            "label": label_b,
            "step1_dir": ctx_b["step1_dir"],
            "low_rank_path": ctx_b["low_rank_path"],
            "self_calib_path": ctx_b["self_calib_path"],
            "mean_captured_weighted_energy": aggregate_mean(rows, f"{label_b}_weighted_captured_weighted_energy"),
            "mean_fro_reduction_ratio": aggregate_mean(rows, f"{label_b}_weighted_fro_reduction_ratio"),
            "mean_mean_sq_reduction_ratio": aggregate_mean(rows, f"{label_b}_weighted_mean_sq_reduction_ratio"),
            f"mean_after_evr_at_{args.rank_eval}": aggregate_mean(rows, f"{label_b}_after_evr_at_{args.rank_eval}"),
            "mean_after_stable_rank": aggregate_mean(rows, f"{label_b}_after_stable_rank"),
        },
        "delta_summary": {
            "mean_delta_captured_weighted_energy": aggregate_mean(rows, f"delta_{label_a}_minus_{label_b}_weighted_captured_weighted_energy"),
            "mean_delta_fro_reduction_ratio": aggregate_mean(rows, f"delta_{label_a}_minus_{label_b}_weighted_fro_reduction_ratio"),
            "mean_delta_mean_sq_reduction_ratio": aggregate_mean(rows, f"delta_{label_a}_minus_{label_b}_weighted_mean_sq_reduction_ratio"),
            f"mean_delta_after_evr_at_{args.rank_eval}": aggregate_mean(rows, f"delta_{label_a}_minus_{label_b}_after_evr_at_{args.rank_eval}"),
            "mean_delta_after_stable_rank": aggregate_mean(rows, f"delta_{label_a}_minus_{label_b}_after_stable_rank"),
        },
    }

    winner_metrics = [
        "weighted_captured_weighted_energy",
        "weighted_fro_reduction_ratio",
        "weighted_mean_sq_reduction_ratio",
        f"after_evr_at_{args.rank_eval}",
        "after_stable_rank",
        "weighted_after_fro",
        "weighted_after_mean_sq",
    ]
    win_counts: Dict[str, dict] = {}
    for metric in winner_metrics:
        key = f"winner_{metric}"
        wins = {label_a: 0, label_b: 0, "tie": 0}
        for row in rows:
            winner = str(row.get(key, "tie"))
            wins[winner] = wins.get(winner, 0) + 1
        win_counts[metric] = wins
    summary["win_counts"] = win_counts

    modules = sorted({str(r.get("module_type", "unknown")) for r in rows})
    module_breakdown: Dict[str, dict] = {}
    for module in modules:
        sub = [r for r in rows if r.get("module_type") == module]
        module_breakdown[module] = {
            f"{label_a}_mean_captured_weighted_energy": aggregate_mean(sub, f"{label_a}_weighted_captured_weighted_energy"),
            f"{label_b}_mean_captured_weighted_energy": aggregate_mean(sub, f"{label_b}_weighted_captured_weighted_energy"),
            f"delta_{label_a}_minus_{label_b}_captured_weighted_energy": aggregate_mean(sub, f"delta_{label_a}_minus_{label_b}_weighted_captured_weighted_energy"),
            f"{label_a}_wins_energy": int(sum(1 for r in sub if r.get("winner_weighted_captured_weighted_energy") == label_a)),
            f"{label_b}_wins_energy": int(sum(1 for r in sub if r.get("winner_weighted_captured_weighted_energy") == label_b)),
        }
    summary["module_breakdown"] = module_breakdown
    return summary


def main() -> None:
    ap = argparse.ArgumentParser("Analyze weighted residual reduction from Step3 AB correction")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--step1_dir", required=True)
    ap.add_argument("--calib_path", default=None, help="Self calibration path for the current method")
    ap.add_argument("--label", default="method_a")
    ap.add_argument("--compare_step1_dir", default=None)
    ap.add_argument("--compare_calib_path", default=None, help="Self calibration path for compare method when metric_source=self")
    ap.add_argument("--compare_low_rank_path", default=None)
    ap.add_argument("--compare_label", default="method_b")
    ap.add_argument("--calib_oas", default=None, help="Optional OAS calibration path for re-evaluation")
    ap.add_argument("--calib_second", default=None, help="Optional second-moment calibration path for re-evaluation")
    ap.add_argument("--low_rank_path", required=True)
    ap.add_argument("--metric_source", default="self", choices=["self", "oas", "second"])
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--rank_eval", type=int, default=64)
    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=0)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto")
    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--eps", type=float, default=1e-8)

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_mode = bool(args.compare_step1_dir or args.compare_low_rank_path or args.compare_calib_path)
    if compare_mode and not (args.compare_step1_dir and args.compare_low_rank_path):
        raise ValueError("compare mode requires both --compare_step1_dir and --compare_low_rank_path")
    if compare_mode and args.metric_source == "self" and not args.compare_calib_path:
        raise ValueError("--metric_source self compare mode requires --compare_calib_path")

    device, state = load_model_state(args)
    ctx = load_method_context(
        args=args,
        step1_dir=args.step1_dir,
        low_rank_path=args.low_rank_path,
        calib_path=args.calib_path,
        state=state,
        device=device,
        label=str(args.label),
    )
    method_contexts = [ctx]
    compare_ctx = None
    if compare_mode:
        compare_ctx = load_method_context(
            args=args,
            step1_dir=args.compare_step1_dir,
            low_rank_path=args.compare_low_rank_path,
            calib_path=args.compare_calib_path,
            state=state,
            device=device,
            label=str(args.compare_label),
        )
        method_contexts.append(compare_ctx)

    keys = resolve_keys(args, state=state, method_contexts=method_contexts)
    print(f"[AB-Reduction] matched layers: {len(keys)}", flush=True)

    rows: List[dict] = []
    for idx, key in enumerate(keys, start=1):
        print(f"[AB-Reduction] ({idx}/{len(keys)}) analyzing: {key}", flush=True)
        row_a = analyze_layer(key, ctx, args)
        if compare_ctx is None:
            rows.append(row_a)
        else:
            row_b = analyze_layer(key, compare_ctx, args)
            rows.append(compare_rows(row_a, row_b, label_a=str(args.label), label_b=str(args.compare_label), rank_eval=int(args.rank_eval)))

        if torch.cuda.is_available() and (idx % 4 == 0 or idx == len(keys)):
            torch.cuda.empty_cache()
        if idx % 4 == 0 or idx == len(keys):
            gc.collect()

    csv_path = out_dir / "layerwise_ab_reduction.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            for row in rows:
                wr.writerow(row)

    if compare_ctx is None:
        summary = build_summary(rows, args, ctx)
    else:
        summary = build_compare_summary(rows, args, ctx, compare_ctx, label_a=str(args.label), label_b=str(args.compare_label))
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] ✅", flush=True)
    print(f"  csv    : {csv_path}", flush=True)
    print(f"  summary: {summary_path}", flush=True)
    if rows:
        print("\n[Quick summary]", flush=True)
        if compare_ctx is None:
            print(
                f"  mean captured weighted energy = {summary['mean_captured_weighted_energy']:.6f}",
                flush=True,
            )
            print(
                f"  mean fro reduction ratio      = {summary['mean_fro_reduction_ratio']:.6f}",
                flush=True,
            )
            print(
                f"  attn/mlp captured energy      = {summary['attn_mean_captured_weighted_energy']:.6f} / {summary['mlp_mean_captured_weighted_energy']:.6f}",
                flush=True,
            )
        else:
            print(
                f"  mean captured energy ({args.label}/{args.compare_label}) = "
                f"{summary['method_a']['mean_captured_weighted_energy']:.6f} / "
                f"{summary['method_b']['mean_captured_weighted_energy']:.6f}",
                flush=True,
            )
            print(
                f"  mean delta captured energy ({args.label}-{args.compare_label}) = "
                f"{summary['delta_summary']['mean_delta_captured_weighted_energy']:.6f}",
                flush=True,
            )
            print(
                f"  energy win count ({args.label}/{args.compare_label}/tie) = "
                f"{summary['win_counts']['weighted_captured_weighted_energy'][args.label]} / "
                f"{summary['win_counts']['weighted_captured_weighted_energy'][args.compare_label]} / "
                f"{summary['win_counts']['weighted_captured_weighted_energy']['tie']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
