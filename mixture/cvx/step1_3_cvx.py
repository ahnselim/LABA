#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 1_3 CVX (Measured-Proxy Budget Optimizer)

Objective (recommended measured proxy):
    min_{b_j in bits}  sum_j loss[j, b_j]
    s.t.               sum_j w_j * b_j <= B

where
- loss[j, b] comes from prebake-measured CSV (layer-wise residual/loss table),
  or is auto-measured in this script by reusing step1_2 prebake measurement.
- w_j comes from sensitivity CSV (typically numel(w_j))
- B is set by --avg_bits * sum_j w_j or --total_bits.

This replaces model proxies like C*alpha*2^{-2b} with directly measured per-bit loss.

Outputs:
- <output_dir>/bit_assign.csv         (Step4-compatible: layer_name, R_int)
- <output_dir>/step1_3_cvx_report.csv (detailed per-layer report)
- <output_dir>/bit_assign_meta.txt    (summary stats)

CUDA_VISIBLE_DEVICES=0 \
python cvx/step1_3_cvx.py \
  --sens_csv ./output_7b/output_step1_cvx/step1_1/layerwise_sensitivity.csv \
  --auto_measure_loss \
  --model_id huggyllama/llama-7b \
  --prebake_root ./output_7b/output_step0_prebake \
  --avg_bits 2.0 \
  --output_dir ./output_7b/output_step1_cvx/step1_3_cvx

CUDA_VISIBLE_DEVICES=0 \
python cvx/step1_3_cvx.py \
  --sens_csv ./output_7b/output_step1_cvx/step1_1/layerwise_sensitivity.csv \
  --auto_measure_loss \
  --model_id huggyllama/llama-7b \
  --prebake_root ./output_7b/output_step0_prebake \
  --avg_bits 2.0 \
  --c_col C_mean_per_batch \
--normalize_loss_by_refbit --norm_ref_bit 4 --use_cj --cj_clip_min 1e-3 \
  --norm_eps 1e-9 \
  --output_dir ./output_7b/output_step1_cvx/step1_3_cvx



"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Layer:
    name: str
    w: int
    C: float = 1.0


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows


def parse_bits_arg(bits_arg: str) -> Tuple[int, ...]:
    bits = tuple(sorted(set(int(x.strip()) for x in bits_arg.split(",") if x.strip())))
    if not bits:
        raise ValueError("--bits must contain at least one integer.")
    return bits


def _find_name_col(header: Dict[str, str], label: str) -> str:
    name_cols = ["layer_name", "module", "name"]
    found = next((c for c in name_cols if c in header), None)
    if not found:
        raise ValueError(
            f"Cannot find a layer-name column (layer_name/module/name) in {label}."
        )
    return found


def load_sens_table(
    path: str,
    w_col: str = "numel(w_j)",
    c_col: Optional[str] = None,
) -> List[Layer]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"sens_csv empty: {path}")

    name_col = _find_name_col(rows[0], "sens_csv")

    if w_col not in rows[0]:
        fallback_w_cols = ["w_j", "numel(w_j)", "numel", "params", "weight_count", "#(W_j)"]
        found_w = next((c for c in fallback_w_cols if c in rows[0]), None)
        if not found_w:
            raise ValueError(
                f"Cannot find weight column in sens_csv. requested={w_col}, "
                f"available={list(rows[0].keys())}"
            )
        w_col = found_w

    use_c = c_col is not None
    if use_c and c_col not in rows[0]:
        alt_c_cols = ["C_mean_per_batch", "C_sum", "C_per_param", "C_j", "Cprime_j"]
        found_c = next((c for c in alt_c_cols if c in rows[0]), None)
        if found_c is None:
            raise ValueError(
                f"Cannot find C column in sens_csv. requested={c_col}, available={list(rows[0].keys())}"
            )
        c_col = found_c

    layers: List[Layer] = []
    for r in rows:
        nm = r.get(name_col, "").strip()
        if not nm:
            continue
        try:
            w = int(float(r[w_col]))
        except Exception as e:
            raise ValueError(f"Invalid weight value in sens_csv for layer={nm}: {r.get(w_col)}") from e
        if w <= 0:
            raise ValueError(f"Weight count must be >0, got layer={nm}, w={w}")
        C = 1.0
        if use_c:
            try:
                C = float(r[c_col])  # type: ignore[index]
            except Exception as e:
                raise ValueError(
                    f"Invalid C value in sens_csv for layer={nm}, col={c_col}: {r.get(c_col)}"
                ) from e
        layers.append(Layer(name=nm, w=w, C=C))

    if not layers:
        raise ValueError(f"No valid rows found in sens_csv: {path}")
    return layers


def load_loss_table(path: str, bits: Tuple[int, ...]) -> Dict[str, Dict[int, float]]:
    """Expected: layer_name + b1/b2/... or L1/L2/... or loss1/loss2/..."""
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"loss_table_csv empty: {path}")

    name_col = _find_name_col(rows[0], "loss_table_csv")

    bit_col: Dict[int, str] = {}
    for col in rows[0].keys():
        key = col.strip().lower().replace(" ", "").replace("_", "")
        m = re.match(r"^(?:b|l|loss)(\d+)$", key)
        if m:
            b = int(m.group(1))
            if b not in bit_col:
                bit_col[b] = col

    missing = [b for b in bits if b not in bit_col]
    if missing:
        raise ValueError(
            f"loss_table_csv missing columns for bits={missing}; available_bits={sorted(bit_col.keys())}"
        )

    table: Dict[str, Dict[int, float]] = {}
    for r in rows:
        nm = r.get(name_col, "").strip()
        if not nm:
            continue
        row: Dict[int, float] = {}
        for b in bits:
            try:
                row[b] = float(r[bit_col[b]])
            except Exception as e:
                raise ValueError(
                    f"Invalid loss value: layer={nm}, bit={b}, value={r.get(bit_col[b])}"
                ) from e

        table[nm] = row
        if nm.endswith(".weight"):
            table[nm[:-7]] = row
        else:
            table[nm + ".weight"] = row

    if not table:
        raise ValueError(f"No valid rows found in loss_table_csv: {path}")
    return table


def load_loss_table_from_alpha_csv(path: str, bits: Tuple[int, ...]) -> Dict[str, Dict[int, float]]:
    """
    Parse step1_2 alpha CSV and build measured loss table by:
      loss[j,b] := Lres_weighted (aka Lab)
    """
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"alpha/loss csv empty: {path}")

    name_col = next((c for c in ("module", "layer_name", "name", "full_name") if c in rows[0]), None)
    if name_col is None:
        raise ValueError(f"Cannot find layer name column in {path}")

    bit_col = "bit" if "bit" in rows[0] else None
    if bit_col is None:
        raise ValueError(f"Cannot find bit column in {path}")

    loss_col = next(
        (
            c
            for c in (
                "Lres_weighted",
                "Lab",
                "L_ab_weighted",
                "L_ab",
                "loss",
                "loss_weighted",
            )
            if c in rows[0]
        ),
        None,
    )
    if loss_col is None:
        raise ValueError(
            f"Cannot find measured loss column in {path}. expected one of "
            f"[Lres_weighted, Lab, L_ab_weighted, L_ab, loss, loss_weighted]"
        )

    bits_set = set(bits)
    table: Dict[str, Dict[int, float]] = {}
    for r in rows:
        nm = r.get(name_col, "").strip()
        if not nm:
            continue
        if nm.endswith(".weight"):
            nm = nm[:-7]
        try:
            b = int(float(r[bit_col]))
        except Exception:
            continue
        if b not in bits_set:
            continue
        try:
            lv = float(r[loss_col])
        except Exception:
            continue
        if nm not in table:
            table[nm] = {}
        table[nm][b] = lv

    missing_layers = [k for k, row in table.items() if any(b not in row for b in bits)]
    if missing_layers:
        ex = missing_layers[0]
        miss = [b for b in bits if b not in table[ex]]
        raise ValueError(
            f"alpha/loss csv has incomplete bit rows. example layer={ex}, missing_bits={miss}"
        )

    if not table:
        raise ValueError(f"No valid measured rows in {path}")

    out: Dict[str, Dict[int, float]] = {}
    for nm, row in table.items():
        out[nm] = {b: float(row[b]) for b in bits}
        out[nm + ".weight"] = out[nm]
    return out


def build_measured_loss_table_via_step12(
    output_dir: str,
    bits: Tuple[int, ...],
    model_id: str,
    prebake_root: str,
    revision: Optional[str],
    trust_remote_code: bool,
    dtype: str,
    device_map: str,
    seed: int,
    dataset: str,
    dataset_config: Optional[str],
    split: str,
    use_streaming: bool,
    nsamples: int,
    seqlen: int,
    reuse_calib: bool,
    calib_cache_dir: str,
    calib_batch_size: int,
    keep_calib_on_device: bool,
    empty_cache_interval: int,
    strict_prebake: bool,
) -> Tuple[str, str]:
    # Lazy import so normal loss_table_csv mode does not require transformers/datasets.
    if __package__:
        from .step1_2_alpha_estimation import (
            Step12AlphaPrebakeConfig,
            run as run_step12_alpha,
        )
    else:
        from step1_2_alpha_estimation import (
            Step12AlphaPrebakeConfig,
            run as run_step12_alpha,
        )

    auto_dir = Path(output_dir) / "auto_measure_step1_2"
    auto_dir.mkdir(parents=True, exist_ok=True)

    outs = run_step12_alpha(
        Step12AlphaPrebakeConfig(
            model_id=str(model_id),
            prebake_root=str(prebake_root),
            output_dir=str(auto_dir),
            revision=revision,
            trust_remote_code=bool(trust_remote_code),
            dtype=str(dtype),
            device_map=str(device_map),
            seed=int(seed),
            dataset=str(dataset),
            dataset_config=dataset_config,
            split=str(split),
            use_streaming=bool(use_streaming),
            nsamples=int(nsamples),
            seqlen=int(seqlen),
            reuse_calib=bool(reuse_calib),
            calib_cache_dir=str(calib_cache_dir),
            bits=tuple(int(b) for b in bits),
            calib_batch_size=int(calib_batch_size),
            keep_calib_on_device=bool(keep_calib_on_device),
            empty_cache_interval=int(empty_cache_interval),
            strict_prebake=bool(strict_prebake),
        )
    )
    alpha_csv = str(outs["alpha_csv"])
    by_name = load_loss_table_from_alpha_csv(alpha_csv, bits=bits)

    pivot_path = str(Path(output_dir) / "auto_measured_loss_table.csv")
    # Write compact reusable loss table CSV.
    rows = []
    for nm in sorted(k for k in by_name.keys() if not k.endswith(".weight")):
        row = {"layer_name": nm}
        for b in bits:
            row[f"b{b}"] = by_name[nm][b]
        rows.append(row)
    with open(pivot_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["layer_name"] + [f"b{b}" for b in bits]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    return alpha_csv, pivot_path


def total_bits(layers: List[Layer], assign: List[int]) -> float:
    return float(sum(layers[j].w * assign[j] for j in range(len(assign))))


def total_loss(loss_table: List[Dict[int, float]], assign: List[int]) -> float:
    return float(sum(loss_table[j][assign[j]] for j in range(len(assign))))


def make_base_loss_row(
    raw_row: Dict[int, float],
    bits: Tuple[int, ...],
    normalize_by_refbit: bool,
    norm_ref_bit: int,
    norm_eps: float,
) -> Dict[int, float]:
    if not normalize_by_refbit:
        return {b: float(raw_row[b]) for b in bits}
    denom = max(float(raw_row[norm_ref_bit]), float(norm_eps))
    return {b: float(raw_row[b]) / denom for b in bits}


def argmin_with_penalty(loss_row: Dict[int, float], wj: int, mu: float, bits: Tuple[int, ...]) -> int:
    best_b = bits[0]
    best_v = float("inf")
    for b in bits:
        v = float(loss_row[b]) + float(mu) * float(wj) * float(b)
        # Tie-break by smaller bit for deterministic monotone S(mu)
        if (v < best_v - 1e-18) or (abs(v - best_v) <= 1e-18 and b < best_b):
            best_v = v
            best_b = b
    return best_b


def solve_mu_assignment(
    layers: List[Layer],
    loss_table: List[Dict[int, float]],
    bits: Tuple[int, ...],
    B: float,
    max_iter: int = 80,
) -> Tuple[float, List[int], str]:
    n = len(layers)
    bmin = bits[0]

    assign_min = [bmin] * n
    S_min = total_bits(layers, assign_min)
    if B <= S_min + 1e-12:
        return float("inf"), assign_min, "all_min_bits"

    # mu=0: each layer picks pure minimum measured loss bit
    assign0 = [argmin_with_penalty(loss_table[j], layers[j].w, 0.0, bits) for j in range(n)]
    S0 = total_bits(layers, assign0)
    if S0 <= B + 1e-12:
        return 0.0, assign0, "unconstrained_feasible"

    # Binary search mu so that S(mu) <= B and as close as possible.
    mu_lo = 0.0
    mu_hi = 1.0

    def assign_for(mu: float) -> Tuple[List[int], float]:
        a = [argmin_with_penalty(loss_table[j], layers[j].w, mu, bits) for j in range(n)]
        return a, total_bits(layers, a)

    a_hi, s_hi = assign_for(mu_hi)
    for _ in range(80):
        if s_hi <= B + 1e-12:
            break
        mu_hi *= 2.0
        a_hi, s_hi = assign_for(mu_hi)
        if mu_hi > 1e20:
            break

    best_assign = a_hi
    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        a_mid, s_mid = assign_for(mu_mid)
        if s_mid <= B + 1e-12:
            mu_hi = mu_mid
            best_assign = a_mid
        else:
            mu_lo = mu_mid
        if abs(mu_hi - mu_lo) <= 1e-12 * max(1.0, mu_hi):
            break

    return mu_hi, best_assign, "ok"


def _next_bit(cur: int, bits: Tuple[int, ...]) -> Optional[int]:
    idx = bits.index(cur)
    return bits[idx + 1] if idx + 1 < len(bits) else None


def _prev_bit(cur: int, bits: Tuple[int, ...]) -> Optional[int]:
    idx = bits.index(cur)
    return bits[idx - 1] if idx > 0 else None


def improve_under_budget(
    layers: List[Layer],
    loss_table: List[Dict[int, float]],
    bits: Tuple[int, ...],
    B: float,
    assign: List[int],
) -> List[int]:
    """
    Given a feasible assignment (<=B), greedily spend remaining budget on
    best marginal moves (minimum delta_loss per added bit) to approach B.
    """
    n = len(assign)
    S = total_bits(layers, assign)
    if S >= B - 1e-12:
        return assign

    heap: List[Tuple[float, float, int, int, int]] = []

    def push_candidate(j: int) -> None:
        cur = assign[j]
        nxt = _next_bit(cur, bits)
        if nxt is None:
            return
        dbits = float(layers[j].w * (nxt - cur))
        if dbits <= 0:
            return
        dloss = float(loss_table[j][nxt] - loss_table[j][cur])
        score = dloss / dbits
        heapq.heappush(heap, (score, dloss, j, cur, nxt))

    for j in range(n):
        push_candidate(j)

    while heap:
        _, _, j, old_bit, new_bit = heapq.heappop(heap)
        if assign[j] != old_bit:
            continue

        dbits = float(layers[j].w * (new_bit - old_bit))
        if S + dbits > B + 1e-12:
            continue

        assign[j] = new_bit
        S += dbits
        push_candidate(j)

    return assign


def repair_over_budget(
    layers: List[Layer],
    loss_table: List[Dict[int, float]],
    bits: Tuple[int, ...],
    B: float,
    assign: List[int],
) -> List[int]:
    """If assignment exceeds budget, greedily reduce bits with minimum loss increase/bit."""
    S = total_bits(layers, assign)
    if S <= B + 1e-12:
        return assign

    heap: List[Tuple[float, float, int, int, int]] = []

    def push_candidate(j: int) -> None:
        cur = assign[j]
        prv = _prev_bit(cur, bits)
        if prv is None:
            return
        dbits = float(layers[j].w * (cur - prv))
        if dbits <= 0:
            return
        dloss = float(loss_table[j][prv] - loss_table[j][cur])
        score = dloss / dbits
        heapq.heappush(heap, (score, dloss, j, cur, prv))

    for j in range(len(assign)):
        push_candidate(j)

    while S > B + 1e-12 and heap:
        _, _, j, old_bit, new_bit = heapq.heappop(heap)
        if assign[j] != old_bit:
            continue
        dbits = float(layers[j].w * (old_bit - new_bit))
        assign[j] = new_bit
        S -= dbits
        push_candidate(j)

    return assign


def optimize_budget(
    layers: List[Layer],
    loss_table: List[Dict[int, float]],
    bits: Tuple[int, ...],
    B: float,
    max_iter: int,
) -> Tuple[List[int], Dict[str, float]]:
    mu, assign, status = solve_mu_assignment(
        layers=layers,
        loss_table=loss_table,
        bits=bits,
        B=B,
        max_iter=max_iter,
    )

    assign = repair_over_budget(layers, loss_table, bits, B, assign)
    assign = improve_under_budget(layers, loss_table, bits, B, assign)

    S = total_bits(layers, assign)
    L = total_loss(loss_table, assign)
    return assign, {
        "mu": float(mu),
        "status": status,
        "total_bits": float(S),
        "total_loss": float(L),
    }


def _bit_hist(assign: List[int], bits: Tuple[int, ...]) -> Dict[str, int]:
    hist = {str(b): 0 for b in bits}
    for b in assign:
        hist[str(int(b))] = hist.get(str(int(b)), 0) + 1
    return hist


def main() -> None:
    ap = argparse.ArgumentParser("Step1_3 CVX with measured proxy loss (budget mode)")
    ap.add_argument("--sens_csv", required=True, help="Sensitivity CSV (for layer_name and weights)")
    ap.add_argument(
        "--loss_table_csv",
        default=None,
        help="Measured per-layer loss table CSV: layer_name + b1/b2/... or L1/L2/... or loss1/loss2/...",
    )
    ap.add_argument(
        "--auto_measure_loss",
        action="store_true",
        help="If --loss_table_csv is not given, measure per-layer per-bit loss in this script via step1_2 prebake path.",
    )

    # auto-measure options (reused from step1_2)
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--prebake_root", default=None, help="Step0 prebake root containing bit1..bit4/")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
    )
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default=True,
    )
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--reuse_calib", action="store_true")
    ap.add_argument("--calib_cache_dir", default="./artifacts/bitmin")
    ap.add_argument("--calib_batch_size", type=int, default=1)
    ap.add_argument("--keep_calib_on_device", action="store_true")
    ap.add_argument("--empty_cache_interval", type=int, default=0)
    ap.add_argument("--strict_prebake", action="store_true")

    ap.add_argument("--w_col", default="numel(w_j)", help="Weight-count column in sens_csv")
    ap.add_argument(
        "--use_cj",
        action="store_true",
        help="Use C_j weighting in objective: loss_proxy[j,b] = C_j * loss_raw[j,b]",
    )
    ap.add_argument(
        "--c_col",
        default="C_mean_per_batch",
        help="C_j column in sens_csv (used only when --use_cj is set)",
    )
    ap.add_argument(
        "--cj_clip_min",
        type=float,
        default=0.0,
        help="Clip C_j from below before multiplying (default: 0.0)",
    )
    ap.add_argument(
        "--normalize_loss_by_refbit",
        action="store_true",
        help="Use base loss normalization: loss_base[j,b] = loss_raw[j,b] / (loss_raw[j,ref_bit] + eps).",
    )
    ap.add_argument(
        "--norm_ref_bit",
        type=int,
        default=None,
        help="Reference bit for normalization (default: max allowed bit).",
    )
    ap.add_argument(
        "--norm_eps",
        type=float,
        default=1e-12,
        help="Epsilon added to normalization denominator.",
    )
    ap.add_argument("--bits", default="1,2,3,4", help="Allowed bits, e.g. 1,2,3,4")
    ap.add_argument("--avg_bits", type=float, default=None, help="Target average bits")
    ap.add_argument("--total_bits", type=float, default=None, help="Target total bits budget")
    ap.add_argument("--max_iter", type=int, default=80, help="Mu-search iterations")
    ap.add_argument("--output_dir", default="./artifacts/bitmin/step3_cvx")

    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.avg_bits is None and args.total_bits is None:
        raise ValueError("Provide --avg_bits or --total_bits")
    if args.avg_bits is not None and args.total_bits is not None:
        raise ValueError("Use only one of --avg_bits or --total_bits")

    bits = parse_bits_arg(args.bits)
    if float(args.norm_eps) <= 0.0:
        raise ValueError("--norm_eps must be > 0.")
    norm_ref_bit = int(args.norm_ref_bit) if args.norm_ref_bit is not None else int(bits[-1])
    if args.normalize_loss_by_refbit and norm_ref_bit not in bits:
        raise ValueError(
            f"--norm_ref_bit={norm_ref_bit} is not in allowed bits={list(bits)}"
        )

    layers = load_sens_table(
        args.sens_csv,
        w_col=args.w_col,
        c_col=(args.c_col if args.use_cj else None),
    )

    measured_alpha_csv = None
    auto_loss_csv = None
    loss_table_csv_used = args.loss_table_csv
    if args.loss_table_csv:
        by_name = load_loss_table(args.loss_table_csv, bits=bits)
    else:
        if not args.auto_measure_loss:
            raise ValueError(
                "No --loss_table_csv provided. Use --auto_measure_loss "
                "(with --model_id and --prebake_root) to measure loss in this script."
            )
        if not args.model_id or not args.prebake_root:
            raise ValueError(
                "--auto_measure_loss requires both --model_id and --prebake_root."
            )
        print("[step1_3_cvx] Measuring loss table in-code via step1_2 prebake flow ...")
        measured_alpha_csv, auto_loss_csv = build_measured_loss_table_via_step12(
            output_dir=args.output_dir,
            bits=bits,
            model_id=args.model_id,
            prebake_root=args.prebake_root,
            revision=args.revision,
            trust_remote_code=bool(args.trust_remote_code),
            dtype=args.dtype,
            device_map=args.device_map,
            seed=int(args.seed),
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            use_streaming=bool(args.use_streaming),
            nsamples=int(args.nsamples),
            seqlen=int(args.seqlen),
            reuse_calib=bool(args.reuse_calib),
            calib_cache_dir=args.calib_cache_dir,
            calib_batch_size=int(args.calib_batch_size),
            keep_calib_on_device=bool(args.keep_calib_on_device),
            empty_cache_interval=int(args.empty_cache_interval),
            strict_prebake=bool(args.strict_prebake),
        )
        loss_table_csv_used = auto_loss_csv
        by_name = load_loss_table(auto_loss_csv, bits=bits)

    loss_table_raw: List[Dict[int, float]] = []
    loss_table_base: List[Dict[int, float]] = []
    loss_table: List[Dict[int, float]] = []
    missing: List[str] = []
    for L in layers:
        row = by_name.get(L.name)
        if row is None:
            missing.append(L.name)
        else:
            raw_row = {b: float(row[b]) for b in bits}
            loss_table_raw.append(raw_row)
            base_row = make_base_loss_row(
                raw_row=raw_row,
                bits=bits,
                normalize_by_refbit=bool(args.normalize_loss_by_refbit),
                norm_ref_bit=norm_ref_bit,
                norm_eps=float(args.norm_eps),
            )
            loss_table_base.append(base_row)
            if args.use_cj:
                c_eff = max(float(L.C), float(args.cj_clip_min))
                loss_table.append({b: float(c_eff * base_row[b]) for b in bits})
            else:
                loss_table.append(base_row)

    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"loss_table_csv missing {len(missing)} layers from sens_csv. sample: {sample}"
        )

    w_sum = float(sum(L.w for L in layers))
    if args.avg_bits is not None:
        B = float(args.avg_bits) * w_sum
        requested_avg = float(args.avg_bits)
    else:
        B = float(args.total_bits)
        requested_avg = float(B / max(1.0, w_sum))

    B_min = float(bits[0] * w_sum)
    B_max = float(bits[-1] * w_sum)
    if B < B_min:
        print(
            f"[step1_3_cvx] Requested budget below min-bit floor. clamped: {B:.6e} -> {B_min:.6e}",
            file=sys.stderr,
        )
        B = B_min
    if B > B_max:
        print(
            f"[step1_3_cvx] Requested budget above max-bit ceiling. clamped: {B:.6e} -> {B_max:.6e}",
            file=sys.stderr,
        )
        B = B_max

    assign, info = optimize_budget(
        layers=layers,
        loss_table=loss_table,
        bits=bits,
        B=B,
        max_iter=int(args.max_iter),
    )

    total_bits_final = float(info["total_bits"])
    total_loss_final = float(info["total_loss"])
    avg_bits_final = total_bits_final / max(1.0, w_sum)

    bit_assign_path = os.path.join(args.output_dir, "bit_assign.csv")
    with open(bit_assign_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "layer_name",
                "w_j",
                "C_j",
                "C_eff",
                "R_int",
                "loss_selected_proxy",
                "loss_selected_base",
                "loss_selected_raw",
            ]
        )
        for j, L in enumerate(layers):
            b = int(assign[j])
            c_eff = max(float(L.C), float(args.cj_clip_min)) if args.use_cj else 1.0
            wr.writerow(
                [
                    L.name,
                    L.w,
                    f"{float(L.C):.9e}",
                    f"{float(c_eff):.9e}",
                    b,
                    f"{loss_table[j][b]:.9e}",
                    f"{loss_table_base[j][b]:.9e}",
                    f"{loss_table_raw[j][b]:.9e}",
                ]
            )

    report_path = os.path.join(args.output_dir, "step1_3_cvx_report.csv")
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        header = [
            "layer_name",
            "w_j",
            "C_j",
            "C_eff",
            "R_int",
            "loss_selected_proxy",
            "loss_selected_base",
            "loss_selected_raw",
            "bit_contribution",
        ]
        header += [f"loss_proxy_b{b}" for b in bits]
        header += [f"loss_base_b{b}" for b in bits]
        header += [f"loss_raw_b{b}" for b in bits]
        wr.writerow(header)
        for j, L in enumerate(layers):
            b = int(assign[j])
            c_eff = max(float(L.C), float(args.cj_clip_min)) if args.use_cj else 1.0
            row = [
                L.name,
                L.w,
                f"{float(L.C):.9e}",
                f"{float(c_eff):.9e}",
                b,
                f"{loss_table[j][b]:.9e}",
                f"{loss_table_base[j][b]:.9e}",
                f"{loss_table_raw[j][b]:.9e}",
                f"{float(L.w * b):.3f}",
            ]
            row += [f"{loss_table[j][bb]:.9e}" for bb in bits]
            row += [f"{loss_table_base[j][bb]:.9e}" for bb in bits]
            row += [f"{loss_table_raw[j][bb]:.9e}" for bb in bits]
            wr.writerow(row)

    meta_path = os.path.join(args.output_dir, "bit_assign_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("=== Step1_3 CVX Measured-Proxy Summary ===\n")
        f.write(f"status            : {info['status']}\n")
        f.write(f"mu                : {info['mu']:.9e}\n")
        f.write(f"bits_allowed      : {list(bits)}\n")
        f.write(f"use_cj            : {bool(args.use_cj)}\n")
        f.write(f"c_col             : {args.c_col}\n")
        f.write(f"cj_clip_min       : {float(args.cj_clip_min):.9e}\n")
        f.write(f"normalize_by_ref  : {bool(args.normalize_loss_by_refbit)}\n")
        f.write(f"norm_ref_bit      : {int(norm_ref_bit)}\n")
        f.write(f"norm_eps          : {float(args.norm_eps):.9e}\n")
        f.write(f"loss_table_csv    : {loss_table_csv_used}\n")
        if measured_alpha_csv:
            f.write(f"measured_alpha_csv: {measured_alpha_csv}\n")
        f.write(f"requested_B       : {B:.6e}\n")
        f.write(f"requested_avg     : {requested_avg:.6f}\n")
        f.write(f"final_total_bits  : {total_bits_final:.6e}\n")
        f.write(f"final_avg_bits    : {avg_bits_final:.6f}\n")
        f.write(f"final_total_loss  : {total_loss_final:.9e}\n")
        f.write(f"num_layers        : {len(layers)}\n")
        f.write(f"bit_hist          : {json.dumps(_bit_hist(assign, bits), ensure_ascii=True)}\n")

    print("[step1_3_cvx] Done")
    print(f"  bit_assign.csv : {bit_assign_path}")
    print(f"  report.csv     : {report_path}")
    print(f"  meta.txt       : {meta_path}")
    print(f"  requested avg  : {requested_avg:.6f}")
    print(f"  final avg      : {avg_bits_final:.6f}")
    print(f"  final loss     : {total_loss_final:.9e}")
    if measured_alpha_csv:
        print(f"  measured alpha : {measured_alpha_csv}")
    if auto_loss_csv:
        print(f"  auto loss csv  : {auto_loss_csv}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Compatibility wrapper API (same style as step1_3_bit_optimization.py)
# ---------------------------------------------------------------------------


def _invoke_local_main(argv: Sequence[str]) -> subprocess.CompletedProcess:
    argv = list(argv)
    args = [str(sys.executable), str(Path(__file__).resolve())] + argv
    prev_argv = sys.argv[:]
    exit_code = 0
    try:
        sys.argv = [str(Path(__file__).resolve())] + argv
        try:
            main()
        except SystemExit as e:
            code = e.code
            if code is None:
                exit_code = 0
            elif isinstance(code, int):
                exit_code = int(code)
            else:
                print(code, file=sys.stderr)
                exit_code = 1
    finally:
        sys.argv = prev_argv
    return subprocess.CompletedProcess(args=args, returncode=int(exit_code))


@dataclass
class Step13CvxConfig:
    sens_csv: str
    output_dir: str
    loss_table_csv: Optional[str] = None
    auto_measure_loss: bool = False
    model_id: Optional[str] = None
    prebake_root: Optional[str] = None
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    device_map: str = "auto"
    seed: int = 42
    dataset: str = "DKYoon/SlimPajama-6B"
    dataset_config: Optional[str] = None
    split: str = "train"
    use_streaming: bool = True
    nsamples: int = 64
    seqlen: int = 2048
    reuse_calib: bool = False
    calib_cache_dir: str = "./artifacts/bitmin"
    calib_batch_size: int = 1
    keep_calib_on_device: bool = False
    empty_cache_interval: int = 0
    strict_prebake: bool = False
    w_col: str = "numel(w_j)"
    use_cj: bool = False
    c_col: str = "C_mean_per_batch"
    cj_clip_min: float = 0.0
    normalize_loss_by_refbit: bool = False
    norm_ref_bit: Optional[int] = None
    norm_eps: float = 1e-12
    bits: str = "1,2,3,4"
    avg_bits: Optional[float] = None
    total_bits: Optional[float] = None
    max_iter: int = 80
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


def build_command(cfg: Step13CvxConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--sens_csv",
        str(cfg.sens_csv),
        "--w_col",
        str(cfg.w_col),
        "--c_col",
        str(cfg.c_col),
        "--cj_clip_min",
        str(float(cfg.cj_clip_min)),
        "--norm_eps",
        str(float(cfg.norm_eps)),
        "--bits",
        str(cfg.bits),
        "--max_iter",
        str(int(cfg.max_iter)),
        "--output_dir",
        str(cfg.output_dir),
    ]
    if cfg.loss_table_csv is not None:
        cmd += ["--loss_table_csv", str(cfg.loss_table_csv)]
    if cfg.auto_measure_loss:
        cmd += ["--auto_measure_loss"]
    if cfg.use_cj:
        cmd += ["--use_cj"]
    if cfg.normalize_loss_by_refbit:
        cmd += ["--normalize_loss_by_refbit"]
    if cfg.norm_ref_bit is not None:
        cmd += ["--norm_ref_bit", str(int(cfg.norm_ref_bit))]
    if cfg.model_id is not None:
        cmd += ["--model_id", str(cfg.model_id)]
    if cfg.prebake_root is not None:
        cmd += ["--prebake_root", str(cfg.prebake_root)]
    if cfg.revision is not None:
        cmd += ["--revision", str(cfg.revision)]
    if cfg.trust_remote_code:
        cmd += ["--trust_remote_code"]
    cmd += ["--dtype", str(cfg.dtype)]
    cmd += ["--device_map", str(cfg.device_map)]
    cmd += ["--seed", str(int(cfg.seed))]
    cmd += ["--dataset", str(cfg.dataset)]
    if cfg.dataset_config is not None:
        cmd += ["--dataset_config", str(cfg.dataset_config)]
    cmd += ["--split", str(cfg.split)]
    cmd += ["--use_streaming", "true" if cfg.use_streaming else "false"]
    cmd += ["--nsamples", str(int(cfg.nsamples))]
    cmd += ["--seqlen", str(int(cfg.seqlen))]
    if cfg.reuse_calib:
        cmd += ["--reuse_calib"]
    cmd += ["--calib_cache_dir", str(cfg.calib_cache_dir)]
    cmd += ["--calib_batch_size", str(int(cfg.calib_batch_size))]
    if cfg.keep_calib_on_device:
        cmd += ["--keep_calib_on_device"]
    cmd += ["--empty_cache_interval", str(int(cfg.empty_cache_interval))]
    if cfg.strict_prebake:
        cmd += ["--strict_prebake"]
    if cfg.avg_bits is not None:
        cmd += ["--avg_bits", str(float(cfg.avg_bits))]
    if cfg.total_bits is not None:
        cmd += ["--total_bits", str(float(cfg.total_bits))]
    return cmd


def run(cfg: Step13CvxConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp
