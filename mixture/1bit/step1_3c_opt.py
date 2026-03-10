#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 1_3c — Discrete Bit Optimization with measured loss proxy

핵심 아이디어:
  - 연속 근사(C*alpha*2^{-2R}) 대신 측정값 기반 이산 목적함수 사용
      min  sum_j C_adj[j] * Lres(j, b_j)
      s.t. sum_j w_j * b_j <= B                      (budget 모드)
           sum_j C_adj[j] * Lres(j, b_j) <= L_target (target 모드)

  - Lres(j,b)는 step1_2 alpha CSV의 Lres_weighted(우선), 혹은
    alpha*Lq_weighted 등의 fallback으로 구성.
  - step1/step2에서 누락된 값은 본 스크립트 내부에서 자동 보완.
  - mixed ablation directory에서도 사용 가능:
      bit-1 rows may come from the group-wise `(mu,beta)` 1-bit pipeline,
      while bits 2/3/4 stay on the baseline pipeline.
  
python cvx/step1_3c_opt.py \
  --sens_csv ./output_7b/output_step1_cvx/step1_1/layerwise_sensitivity.csv \
  --alpha_csv ./output_7b/output_step1_cvx/step1_2/alpha_layerwise_prebake.csv \
  --mode budget --avg_bits 2.00 \
  --normalize_lres_by_refbit --norm_ref_bit 4 \
  --cj_transform power --cj_power 0.7 \
  --proxy_shape marginal_gain --marginal_gain_power 1.85 \
  --output_dir ./output_7b/output_step1_cvx/step1_3c_opt_final
  
python cvx/step1_3c_opt.py \
  --sens_csv ./output/output_step1_cvx/step1_1/layerwise_sensitivity.csv \
  --alpha_csv ./output/output_step1_cvx/step1_2/alpha_layerwise_prebake.csv \
  --mode budget --avg_bits 2.00 \
  --cj_transform power --cj_power 0.7 \
  --normalize_lres_by_refbit --norm_ref_bit 4 \
  --proxy_shape marginal_gain --marginal_gain_power 1.85 \
  --output_dir ./output/output_step1_cvx/step1_3c_opt_2

"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# -------------------------
# Data structures
# -------------------------
@dataclass
class LayerRec:
    name: str
    w: float
    c_raw: float
    c_adj: float
    lres_raw: Dict[int, float]
    lres_base: Dict[int, float]
    proxy_base: Dict[int, float]
    loss_proxy: Dict[int, float]


# -------------------------
# CSV helpers
# -------------------------
def _read_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows


def _pick_existing(cols: Sequence[str], row0: Dict[str, str]) -> Optional[str]:
    return next((c for c in cols if c in row0), None)


def parse_bits(bits_arg: str) -> Tuple[int, ...]:
    bits = sorted(set(int(x.strip()) for x in bits_arg.split(",") if x.strip()))
    if not bits:
        raise ValueError("--bits must not be empty.")
    if any((b < 1 or b > 8) for b in bits):
        raise ValueError(f"--bits out of range: {bits}")
    return tuple(bits)


def parse_sensitivity(path: str, c_col: str, w_col: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"sens_csv is empty: {path}")

    name_col = _pick_existing(("layer_name", "module", "name"), rows[0])
    if name_col is None:
        raise ValueError("sens_csv must contain one of: layer_name/module/name")

    if w_col not in rows[0]:
        w_col = _pick_existing(("numel(w_j)", "w_j", "numel", "params", "weight_count"), rows[0]) or w_col
    if w_col not in rows[0]:
        raise ValueError(f"weight column not found in sens_csv. requested={w_col}")

    c_candidates = [c_col, "C_mean_per_batch", "C_sum", "C_per_param", "C_j", "Cprime_j"]
    c_candidates = [c for i, c in enumerate(c_candidates) if c and c not in c_candidates[:i]]

    names: List[str] = []
    c_vals: List[float] = []
    w_vals: List[float] = []

    for r in rows:
        nm = (r.get(name_col) or "").strip()
        if not nm:
            continue
        try:
            w = float(r[w_col])
        except Exception as e:
            raise ValueError(f"invalid w value: layer={nm}, col={w_col}, value={r.get(w_col)}") from e
        if w <= 0:
            continue

        c_v = None
        for cc in c_candidates:
            if cc in r and r.get(cc, "").strip() != "":
                try:
                    c_v = float(r[cc])
                    break
                except Exception:
                    pass
        if c_v is None:
            # step1에 C가 없다면 중립값 1.0 사용 (내부 계산 fallback)
            c_v = 1.0

        names.append(nm[:-7] if nm.endswith(".weight") else nm)
        c_vals.append(float(c_v))
        w_vals.append(float(w))

    if not names:
        raise ValueError(f"no valid rows in sens_csv: {path}")
    return names, np.asarray(c_vals, dtype=np.float64), np.asarray(w_vals, dtype=np.float64)


def _extract_lres_from_alpha_row(r: Dict[str, str]) -> Optional[float]:
    direct_cols = ("Lres_weighted", "Lab", "L_ab_weighted", "L_ab", "loss_weighted", "loss")
    for c in direct_cols:
        v = r.get(c, "")
        if v != "":
            try:
                x = float(v)
                if math.isfinite(x) and x >= 0.0:
                    return x
            except Exception:
                pass

    # fallback: alpha * Lq_weighted (or Lq)
    alpha = None
    for c in ("alpha",):
        v = r.get(c, "")
        if v != "":
            try:
                alpha = float(v)
                break
            except Exception:
                pass
    lq = None
    for c in ("Lq_weighted", "Lq", "quant_error", "Lq_raw"):
        v = r.get(c, "")
        if v != "":
            try:
                lq = float(v)
                break
            except Exception:
                pass

    if alpha is not None and lq is not None and math.isfinite(alpha) and math.isfinite(lq):
        x = max(0.0, float(alpha) * float(lq))
        return x
    if alpha is not None and math.isfinite(alpha):
        return max(0.0, float(alpha))
    return None


def parse_alpha_as_lres_table(path: str, bits: Tuple[int, ...]) -> Dict[str, Dict[int, float]]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"alpha_csv is empty: {path}")

    name_col = _pick_existing(("module", "layer_name", "name", "full_name"), rows[0])
    bit_col = _pick_existing(("bit", "bits", "R_int"), rows[0])
    if name_col is None or bit_col is None:
        raise ValueError("alpha_csv must contain name(module/layer_name/...) and bit columns.")

    bits_set = set(bits)
    table: Dict[str, Dict[int, float]] = {}
    for r in rows:
        raw_nm = (r.get(name_col) or "").strip()
        if not raw_nm:
            continue
        nm = raw_nm[:-7] if raw_nm.endswith(".weight") else raw_nm

        try:
            b = int(float(r.get(bit_col, "")))
        except Exception:
            continue
        if b not in bits_set:
            continue

        lres = _extract_lres_from_alpha_row(r)
        if lres is None:
            continue

        if nm not in table:
            table[nm] = {}
        prev = table[nm].get(b, None)
        # If multiple rows exist for the same (module, bit), keep the better measured residual.
        table[nm][b] = float(lres) if prev is None else min(float(prev), float(lres))

    return table


def parse_loss_table(path: str, bits: Tuple[int, ...]) -> Dict[str, Dict[int, float]]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"loss_table_csv is empty: {path}")

    name_col = _pick_existing(("layer_name", "module", "name"), rows[0])
    if name_col is None:
        raise ValueError("loss_table_csv must contain one of: layer_name/module/name")

    col_map: Dict[int, str] = {}
    for c in rows[0].keys():
        key = c.strip().lower().replace("_", "").replace(" ", "")
        if key.startswith("b") and key[1:].isdigit():
            col_map[int(key[1:])] = c
        elif key.startswith("l") and key[1:].isdigit():
            col_map[int(key[1:])] = c
        elif key.startswith("loss") and key[4:].isdigit():
            col_map[int(key[4:])] = c

    miss = [b for b in bits if b not in col_map]
    if miss:
        raise ValueError(f"loss_table_csv missing bit columns for bits={miss}")

    out: Dict[str, Dict[int, float]] = {}
    for r in rows:
        raw_nm = (r.get(name_col) or "").strip()
        if not raw_nm:
            continue
        nm = raw_nm[:-7] if raw_nm.endswith(".weight") else raw_nm
        rec: Dict[int, float] = {}
        ok = True
        for b in bits:
            try:
                x = float(r[col_map[b]])
            except Exception:
                ok = False
                break
            if not math.isfinite(x) or x < 0.0:
                ok = False
                break
            rec[b] = float(x)
        if ok:
            out[nm] = rec
    return out


# -------------------------
# Fallback generation (missing step1/2 values)
# -------------------------
def _ensure_monotone_nonincreasing(bits: Tuple[int, ...], row: Dict[int, float], eps: float) -> Dict[int, float]:
    vals = {b: max(float(row[b]), eps) for b in bits}
    prev = vals[bits[0]]
    for b in bits[1:]:
        if vals[b] > prev:
            vals[b] = prev
        prev = vals[b]
    return vals


def _fit_log_linear_predict(known_bits: List[int], known_vals: List[float], target_b: int) -> float:
    xb = np.asarray(known_bits, dtype=np.float64)
    y = np.log(np.maximum(np.asarray(known_vals, dtype=np.float64), 1e-30))
    if len(known_bits) >= 2:
        m, c = np.polyfit(xb, y, 1)
        return float(np.exp(m * float(target_b) + c))
    # one-point fallback: 1bit 증가마다 1/4 오차(=6dB/bit) 가정
    b0, y0 = float(xb[0]), float(known_vals[0])
    return float(y0 * (0.25 ** (float(target_b) - b0)))


def fill_missing_lres(
    names: List[str],
    bits: Tuple[int, ...],
    raw_table: Dict[str, Dict[int, float]],
    eps: float,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, int]]:
    # global median curve (bit별) for full-missing layers
    global_curve: Dict[int, float] = {}
    for b in bits:
        vals = []
        for nm in names:
            if nm in raw_table and b in raw_table[nm]:
                v = float(raw_table[nm][b])
                if math.isfinite(v) and v >= 0.0:
                    vals.append(v)
        if vals:
            global_curve[b] = float(np.median(np.asarray(vals, dtype=np.float64)))
        else:
            # absolute fallback if no measured value exists at this bit
            global_curve[b] = float(0.25 ** (b - bits[0]))
    global_curve = _ensure_monotone_nonincreasing(bits, global_curve, eps)

    out: Dict[str, Dict[int, float]] = {}
    stat = {
        "full_missing_layers": 0,
        "partial_missing_layers": 0,
        "predicted_points": 0,
        "global_curve_fallback_points": 0,
    }

    for nm in names:
        if nm not in raw_table or len(raw_table[nm]) == 0:
            out[nm] = dict(global_curve)
            stat["full_missing_layers"] += 1
            stat["global_curve_fallback_points"] += len(bits)
            continue

        src = raw_table[nm]
        known_bits = sorted([b for b in bits if b in src and math.isfinite(float(src[b])) and float(src[b]) >= 0.0])
        if not known_bits:
            out[nm] = dict(global_curve)
            stat["full_missing_layers"] += 1
            stat["global_curve_fallback_points"] += len(bits)
            continue

        row: Dict[int, float] = {}
        if len(known_bits) < len(bits):
            stat["partial_missing_layers"] += 1

        known_vals = [max(float(src[b]), eps) for b in known_bits]
        for b in bits:
            if b in src and math.isfinite(float(src[b])) and float(src[b]) >= 0.0:
                row[b] = max(float(src[b]), eps)
            else:
                row[b] = max(_fit_log_linear_predict(known_bits, known_vals, b), eps)
                stat["predicted_points"] += 1

        out[nm] = _ensure_monotone_nonincreasing(bits, row, eps)

    return out, stat


# -------------------------
# C_j transformation
# -------------------------
def apply_c_transform(
    c_raw: np.ndarray,
    mode: str,
    cj_power: float,
    cj_clip_min: float,
    cj_floor_ratio: float,
) -> np.ndarray:
    c = np.maximum(c_raw.astype(np.float64), 0.0)
    eps = 1e-30

    if mode == "none":
        c_adj = c.copy()
    elif mode == "sqrt":
        c_adj = np.sqrt(c)
    elif mode == "log1p_mean":
        mean_c = float(np.mean(c)) if np.any(c > 0) else 1.0
        c_adj = np.log1p(c / max(mean_c, eps))
    elif mode == "power":
        c_adj = np.power(c, float(cj_power))
    else:
        raise ValueError(f"unsupported --cj_transform: {mode}")

    c_adj = np.maximum(c_adj, float(cj_clip_min))
    if cj_floor_ratio > 0.0:
        pos = c_adj[c_adj > 0]
        if pos.size > 0:
            floor = float(np.median(pos)) * float(cj_floor_ratio)
            c_adj = np.maximum(c_adj, floor)
    return c_adj


def build_marginal_proxy_curve(
    lres_base: Dict[int, float],
    bits: Tuple[int, ...],
    gain_power: float,
) -> Dict[int, float]:
    """
    Build proxy curve from adjacent marginal gains.
      gain(b->next) = max(L(b)-L(next), 0)
      warped_gain   = gain ** gain_power

    For gain_power=1.0, this exactly recovers the original curve.
    gain_power>1.0 amplifies shape differences across layers.
    """
    out: Dict[int, float] = {}
    bmax = bits[-1]
    out[bmax] = float(lres_base[bmax])
    for i in range(len(bits) - 2, -1, -1):
        b = bits[i]
        nxt = bits[i + 1]
        g = max(float(lres_base[b]) - float(lres_base[nxt]), 0.0)
        g_warp = float(g ** float(gain_power)) if g > 0.0 else 0.0
        out[b] = float(out[nxt] + g_warp)
    return out


# -------------------------
# Optimization
# -------------------------
def _nearest_bit(bits: Tuple[int, ...], target: int) -> int:
    return min(bits, key=lambda b: (abs(b - int(target)), b))


def _next_bit(bits: Tuple[int, ...], b: int) -> Optional[int]:
    i = bits.index(b)
    if i + 1 < len(bits):
        return bits[i + 1]
    return None


def _prev_bit(bits: Tuple[int, ...], b: int) -> Optional[int]:
    i = bits.index(b)
    if i - 1 >= 0:
        return bits[i - 1]
    return None


def total_bits(layers: List[LayerRec], assign: List[int]) -> float:
    return float(sum(layers[i].w * assign[i] for i in range(len(assign))))


def total_loss(layers: List[LayerRec], assign: List[int]) -> float:
    return float(sum(layers[i].loss_proxy[assign[i]] for i in range(len(assign))))


def optimize_budget(
    layers: List[LayerRec],
    bits: Tuple[int, ...],
    budget_bits: float,
    init_bit: int,
) -> Tuple[List[int], Dict[str, float]]:
    bmin, bmax = bits[0], bits[-1]
    S_min = float(sum(Lr.w * bmin for Lr in layers))
    S_max = float(sum(Lr.w * bmax for Lr in layers))
    B = min(max(float(budget_bits), S_min), S_max)
    n = len(layers)

    def run_from_start(start_bit: int) -> Tuple[List[int], float, float, int]:
        assign = [int(start_bit) for _ in range(n)]
        S = total_bits(layers, assign)
        L = total_loss(layers, assign)
        moves = 0

        # over-budget: choose minimum loss increase / bit to downgrade
        while S > B + 1e-12:
            best = None  # (score, dloss, dbits, idx, new_bit)
            for j, lr in enumerate(layers):
                cur = assign[j]
                prv = _prev_bit(bits, cur)
                if prv is None:
                    continue
                dbits = lr.w * float(cur - prv)
                dloss = lr.loss_proxy[prv] - lr.loss_proxy[cur]  # >=0 ideally
                score = dloss / max(dbits, 1e-30)
                cand = (score, dloss, dbits, j, prv)
                if (best is None) or (cand[0] < best[0]):
                    best = cand
            if best is None:
                break
            _, dloss, dbits, j, nb = best
            assign[j] = nb
            S -= dbits
            L += dloss
            moves += 1

        # under-budget: choose maximum loss reduction / bit to upgrade
        while True:
            best = None  # (score, gain, dbits, idx, new_bit)
            for j, lr in enumerate(layers):
                cur = assign[j]
                nxt = _next_bit(bits, cur)
                if nxt is None:
                    continue
                dbits = lr.w * float(nxt - cur)
                if S + dbits > B + 1e-12:
                    continue
                gain = lr.loss_proxy[cur] - lr.loss_proxy[nxt]  # >=0 ideally
                score = gain / max(dbits, 1e-30)
                cand = (score, gain, dbits, j, nxt)
                if (best is None) or (cand[0] > best[0]):
                    best = cand
            if best is None:
                break
            _, gain, dbits, j, nb = best
            assign[j] = nb
            S += dbits
            L -= gain
            moves += 1

        return assign, float(S), float(L), int(moves)

    start_set = {int(_nearest_bit(bits, init_bit)), int(bmin), int(bmax)}
    candidates: List[Tuple[List[int], float, float, int, int]] = []
    for st in sorted(start_set):
        a, s, l, m = run_from_start(st)
        candidates.append((a, s, l, m, st))

    # Select best feasible candidate:
    # 1) lower proxy loss, 2) higher bit usage (closer to budget), 3) fewer moves.
    best = min(candidates, key=lambda x: (x[2], B - x[1], x[3]))
    assign, S, L, moves, best_start = best
    return assign, {
        "total_bits": float(S),
        "total_loss": float(L),
        "moves": float(moves),
        "best_start_bit": float(best_start),
    }


def optimize_target(
    layers: List[LayerRec],
    bits: Tuple[int, ...],
    target_loss: float,
) -> Tuple[List[int], Dict[str, float]]:
    n = len(layers)
    bmin = bits[0]
    assign = [bmin for _ in range(n)]
    S = total_bits(layers, assign)
    L = total_loss(layers, assign)
    moves = 0

    if L <= target_loss + 1e-12:
        return assign, {"total_bits": float(S), "total_loss": float(L), "moves": float(moves), "status": "all_bmin_satisfies"}

    while L > target_loss + 1e-12:
        best = None  # (score, gain, dbits, idx, new_bit)
        for j, lr in enumerate(layers):
            cur = assign[j]
            nxt = _next_bit(bits, cur)
            if nxt is None:
                continue
            dbits = lr.w * float(nxt - cur)
            gain = lr.loss_proxy[cur] - lr.loss_proxy[nxt]
            score = gain / max(dbits, 1e-30)
            cand = (score, gain, dbits, j, nxt)
            if (best is None) or (cand[0] > best[0]):
                best = cand
        if best is None:
            return assign, {"total_bits": float(S), "total_loss": float(L), "moves": float(moves), "status": "infeasible_target"}
        _, gain, dbits, j, nb = best
        assign[j] = nb
        S += dbits
        L -= gain
        moves += 1

    return assign, {"total_bits": float(S), "total_loss": float(L), "moves": float(moves), "status": "ok"}


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Step 1_3c discrete optimizer with measured Lres")
    ap.add_argument("--sens_csv", required=True)
    ap.add_argument("--alpha_csv", default=None, help="step1_2 alpha CSV (module,bit,Lres_weighted/...)")
    ap.add_argument("--loss_table_csv", default=None, help="Optional pivot loss CSV with b1/b2/... columns")

    ap.add_argument("--C_col", default="C_mean_per_batch")
    ap.add_argument("--w_col", default="numel(w_j)")
    ap.add_argument("--bits", default="1,2,3,4",
                    help="candidate bits; in this ablation directory bit1 may be the mu-beta variant")

    ap.add_argument("--mode", choices=("budget", "target"), default=None)
    ap.add_argument("--avg_bits", type=float, default=None)
    ap.add_argument("--total_bits", type=float, default=None)
    ap.add_argument("--target_residual", type=float, default=None)
    ap.add_argument("--target_ratio", type=float, default=0.5)
    ap.add_argument("--target_ref_bit", type=int, default=None, help="If set, target_ratio applies to L(ref_bit)")
    ap.add_argument("--init_bit", type=int, default=2, help="budget mode start bit")

    ap.add_argument("--normalize_lres_by_refbit", action="store_true")
    ap.add_argument("--norm_ref_bit", type=int, default=None)
    ap.add_argument("--norm_eps", type=float, default=1e-12)
    ap.add_argument(
        "--proxy_shape",
        choices=("absolute", "marginal_gain"),
        default="absolute",
        help=(
            "absolute: use Lres base directly, "
            "marginal_gain: rebuild curve from adjacent gains (shape-sensitive)"
        ),
    )
    ap.add_argument(
        "--marginal_gain_power",
        type=float,
        default=1.0,
        help="Only used when --proxy_shape=marginal_gain. >1 sharpens shape differences.",
    )

    ap.add_argument("--cj_transform", choices=("none", "sqrt", "log1p_mean", "power"), default="log1p_mean")
    ap.add_argument("--cj_power", type=float, default=0.5)
    ap.add_argument("--cj_clip_min", type=float, default=1e-12)
    ap.add_argument("--cj_floor_ratio", type=float, default=0.0)

    ap.add_argument("--output_dir", default="./artifacts/bitmin/step3c")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    bits = parse_bits(args.bits)
    if args.normalize_lres_by_refbit and args.norm_ref_bit is None:
        args.norm_ref_bit = int(bits[-1])
    if args.normalize_lres_by_refbit and int(args.norm_ref_bit) not in bits:
        raise ValueError(f"--norm_ref_bit={args.norm_ref_bit} must be in bits={bits}")
    if args.norm_eps <= 0.0:
        raise ValueError("--norm_eps must be > 0")
    if args.marginal_gain_power <= 0.0:
        raise ValueError("--marginal_gain_power must be > 0")

    # mode selection
    mode = args.mode
    if mode is None:
        mode = "budget" if (args.avg_bits is not None or args.total_bits is not None) else "target"

    # load step1 sensitivity
    names, c_raw_arr, w_arr = parse_sensitivity(args.sens_csv, c_col=args.C_col, w_col=args.w_col)
    c_adj_arr = apply_c_transform(
        c_raw=c_raw_arr,
        mode=str(args.cj_transform),
        cj_power=float(args.cj_power),
        cj_clip_min=float(args.cj_clip_min),
        cj_floor_ratio=float(args.cj_floor_ratio),
    )

    # load step2 measured loss
    if args.loss_table_csv:
        raw_lres = parse_loss_table(args.loss_table_csv, bits=bits)
    elif args.alpha_csv:
        raw_lres = parse_alpha_as_lres_table(args.alpha_csv, bits=bits)
    else:
        raise ValueError("Provide either --alpha_csv or --loss_table_csv")

    filled_lres, fill_stat = fill_missing_lres(
        names=names,
        bits=bits,
        raw_table=raw_lres,
        eps=float(args.norm_eps),
    )

    # build layer objects
    layers: List[LayerRec] = []
    for i, nm in enumerate(names):
        lraw = {b: float(filled_lres[nm][b]) for b in bits}
        if args.normalize_lres_by_refbit:
            ref = max(float(lraw[int(args.norm_ref_bit)]), float(args.norm_eps))
            lbase = {b: float(lraw[b] / ref) for b in bits}
        else:
            lbase = dict(lraw)

        if args.proxy_shape == "marginal_gain":
            pbase = build_marginal_proxy_curve(
                lres_base=lbase,
                bits=bits,
                gain_power=float(args.marginal_gain_power),
            )
        else:
            pbase = dict(lbase)

        lproxy = {b: float(c_adj_arr[i] * pbase[b]) for b in bits}
        layers.append(
            LayerRec(
                name=nm,
                w=float(w_arr[i]),
                c_raw=float(c_raw_arr[i]),
                c_adj=float(c_adj_arr[i]),
                lres_raw=lraw,
                lres_base=lbase,
                proxy_base=pbase,
                loss_proxy=lproxy,
            )
        )

    w_sum = float(np.sum(w_arr))
    bmin, bmax = bits[0], bits[-1]
    B_min = float(w_sum * bmin)
    B_max = float(w_sum * bmax)

    # references
    L_bmin = float(sum(l.loss_proxy[bmin] for l in layers))
    L_bmax = float(sum(l.loss_proxy[bmax] for l in layers))
    ref_bit = int(args.target_ref_bit) if args.target_ref_bit is not None else bmin
    L_ref = float(sum(l.loss_proxy[ref_bit] for l in layers))

    if mode == "budget":
        if args.avg_bits is None and args.total_bits is None:
            raise ValueError("budget mode needs --avg_bits or --total_bits")
        if args.avg_bits is not None and args.total_bits is not None:
            raise ValueError("Use only one of --avg_bits or --total_bits")
        B_req = float(args.total_bits) if args.total_bits is not None else float(args.avg_bits) * w_sum
        assign, info = optimize_budget(
            layers=layers,
            bits=bits,
            budget_bits=B_req,
            init_bit=int(args.init_bit),
        )
        status = "ok"
        req_avg_bits = B_req / max(w_sum, 1e-30)
    else:
        if args.target_residual is not None:
            L_target = float(args.target_residual)
        else:
            L_target = float(args.target_ratio) * float(L_ref)
        assign, info = optimize_target(
            layers=layers,
            bits=bits,
            target_loss=L_target,
        )
        status = str(info.get("status", "ok"))
        req_avg_bits = float("nan")
        B_req = float("nan")

    S_final = float(info["total_bits"])
    L_final = float(info["total_loss"])
    avg_final = S_final / max(w_sum, 1e-30)

    # save outputs
    bit_assign_csv = os.path.join(args.output_dir, "bit_assign.csv")
    with open(bit_assign_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        header = [
            "layer_name",
            "w_j",
            "C_j",
            "C_adj",
            "R_int",
            "loss_proxy_selected",
            "Lres_selected_raw",
            "Lres_selected_base",
            "proxy_base_selected",
        ]
        header += [f"Lres_raw_b{b}" for b in bits]
        header += [f"Lres_base_b{b}" for b in bits]
        header += [f"proxy_base_b{b}" for b in bits]
        header += [f"loss_proxy_b{b}" for b in bits]
        wr.writerow(header)
        for i, lr in enumerate(layers):
            b = int(assign[i])
            row = [
                lr.name,
                f"{lr.w:.0f}",
                f"{lr.c_raw:.9e}",
                f"{lr.c_adj:.9e}",
                b,
                f"{lr.loss_proxy[b]:.9e}",
                f"{lr.lres_raw[b]:.9e}",
                f"{lr.lres_base[b]:.9e}",
                f"{lr.proxy_base[b]:.9e}",
            ]
            row += [f"{lr.lres_raw[bb]:.9e}" for bb in bits]
            row += [f"{lr.lres_base[bb]:.9e}" for bb in bits]
            row += [f"{lr.proxy_base[bb]:.9e}" for bb in bits]
            row += [f"{lr.loss_proxy[bb]:.9e}" for bb in bits]
            wr.writerow(row)

    hist: Dict[int, int] = {}
    for b in assign:
        hist[int(b)] = hist.get(int(b), 0) + 1

    meta_path = os.path.join(args.output_dir, "bit_assign_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("=== Step1_3c Discrete Optimization Summary ===\n")
        f.write(f"mode                 : {mode}\n")
        f.write(f"status               : {status}\n")
        f.write(f"bits                 : {list(bits)}\n")
        f.write(f"num_layers           : {len(layers)}\n")
        f.write(f"w_sum                : {w_sum:.9e}\n")
        f.write(f"S_min(all bmin)      : {B_min:.9e}\n")
        f.write(f"S_max(all bmax)      : {B_max:.9e}\n")
        if mode == "budget":
            f.write(f"B_requested          : {B_req:.9e}\n")
            f.write(f"avg_bits_requested   : {req_avg_bits:.9f}\n")
        else:
            f.write(f"L_target             : {L_target:.9e}\n")
            f.write(f"L_ref(bit={ref_bit}) : {L_ref:.9e}\n")
        f.write(f"L_bmin               : {L_bmin:.9e}\n")
        f.write(f"L_bmax               : {L_bmax:.9e}\n")
        f.write(f"S_final              : {S_final:.9e}\n")
        f.write(f"avg_bits_final       : {avg_final:.9f}\n")
        f.write(f"L_final              : {L_final:.9e}\n")
        f.write(f"moves                : {int(info.get('moves', 0.0))}\n")
        f.write(f"cj_transform         : {args.cj_transform}\n")
        f.write(f"cj_power             : {float(args.cj_power):.6f}\n")
        f.write(f"cj_clip_min          : {float(args.cj_clip_min):.9e}\n")
        f.write(f"cj_floor_ratio       : {float(args.cj_floor_ratio):.6f}\n")
        f.write(f"normalize_lres       : {bool(args.normalize_lres_by_refbit)}\n")
        if args.normalize_lres_by_refbit:
            f.write(f"norm_ref_bit         : {int(args.norm_ref_bit)}\n")
        f.write(f"proxy_shape          : {args.proxy_shape}\n")
        f.write(f"marginal_gain_power  : {float(args.marginal_gain_power):.6f}\n")
        f.write(f"missing_full_layers  : {fill_stat['full_missing_layers']}\n")
        f.write(f"missing_partial      : {fill_stat['partial_missing_layers']}\n")
        f.write(f"predicted_points     : {fill_stat['predicted_points']}\n")
        f.write(f"global_fallback_pts  : {fill_stat['global_curve_fallback_points']}\n")
        f.write(f"bit_hist             : {dict(sorted(hist.items()))}\n")

    print("\n[step1_3c_opt] ===== Summary =====")
    print(f"mode             : {mode}")
    print(f"status           : {status}")
    if mode == "budget":
        print(f"avg_bits(req)    : {req_avg_bits:.6f}")
    print(f"avg_bits(final)  : {avg_final:.6f}")
    print(f"L_final(proxy)   : {L_final:.6e}")
    print(f"bit_hist         : {dict(sorted(hist.items()))}")
    print(f"proxy_shape      : {args.proxy_shape} (gain_power={float(args.marginal_gain_power):.3f})")
    print(
        "[step1_3c_opt] missing fill stats:",
        {
            "full_missing_layers": fill_stat["full_missing_layers"],
            "partial_missing_layers": fill_stat["partial_missing_layers"],
            "predicted_points": fill_stat["predicted_points"],
            "global_curve_fallback_points": fill_stat["global_curve_fallback_points"],
        },
    )
    print(f"[step1_3c_opt] Saved: {bit_assign_csv}")
    print(f"[step1_3c_opt] Saved: {meta_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Compatibility wrapper API (same style as other step scripts)
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
class Step13COptConfig:
    sens_csv: str
    output_dir: str
    alpha_csv: Optional[str] = None
    loss_table_csv: Optional[str] = None
    C_col: str = "C_mean_per_batch"
    w_col: str = "numel(w_j)"
    bits: str = "1,2,3,4"
    mode: Optional[str] = None
    avg_bits: Optional[float] = None
    total_bits: Optional[float] = None
    target_residual: Optional[float] = None
    target_ratio: float = 0.5
    target_ref_bit: Optional[int] = None
    init_bit: int = 2
    normalize_lres_by_refbit: bool = False
    norm_ref_bit: Optional[int] = None
    norm_eps: float = 1e-12
    proxy_shape: str = "absolute"
    marginal_gain_power: float = 1.0
    cj_transform: str = "log1p_mean"
    cj_power: float = 0.5
    cj_clip_min: float = 1e-12
    cj_floor_ratio: float = 0.0
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


def build_command(cfg: Step13COptConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--sens_csv",
        str(cfg.sens_csv),
        "--C_col",
        str(cfg.C_col),
        "--w_col",
        str(cfg.w_col),
        "--bits",
        str(cfg.bits),
        "--target_ratio",
        str(float(cfg.target_ratio)),
        "--init_bit",
        str(int(cfg.init_bit)),
        "--norm_eps",
        str(float(cfg.norm_eps)),
        "--proxy_shape",
        str(cfg.proxy_shape),
        "--marginal_gain_power",
        str(float(cfg.marginal_gain_power)),
        "--cj_transform",
        str(cfg.cj_transform),
        "--cj_power",
        str(float(cfg.cj_power)),
        "--cj_clip_min",
        str(float(cfg.cj_clip_min)),
        "--cj_floor_ratio",
        str(float(cfg.cj_floor_ratio)),
        "--output_dir",
        str(cfg.output_dir),
    ]
    if cfg.alpha_csv is not None:
        cmd += ["--alpha_csv", str(cfg.alpha_csv)]
    if cfg.loss_table_csv is not None:
        cmd += ["--loss_table_csv", str(cfg.loss_table_csv)]
    if cfg.mode is not None:
        cmd += ["--mode", str(cfg.mode)]
    if cfg.avg_bits is not None:
        cmd += ["--avg_bits", str(float(cfg.avg_bits))]
    if cfg.total_bits is not None:
        cmd += ["--total_bits", str(float(cfg.total_bits))]
    if cfg.target_residual is not None:
        cmd += ["--target_residual", str(float(cfg.target_residual))]
    if cfg.target_ref_bit is not None:
        cmd += ["--target_ref_bit", str(int(cfg.target_ref_bit))]
    if cfg.normalize_lres_by_refbit:
        cmd += ["--normalize_lres_by_refbit"]
    if cfg.norm_ref_bit is not None:
        cmd += ["--norm_ref_bit", str(int(cfg.norm_ref_bit))]
    return cmd


def run(cfg: Step13COptConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp
