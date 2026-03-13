#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build static_info_v3.json for BRP surrogate (dynamic alpha per bit)

Input:
  - sens_csv: layerwise sensitivity CSV
  - alpha_csv: alpha table CSV (either wide: alpha1/alpha2/alpha3/alpha4, or long: bit + alpha)

Output:
  - output_dir/static_info_v3.json
  - output_dir/alpha_table_flat.csv (debug)
  
python dataset/build_static_info_dynamic_alpha.py \
  --sens_csv ./output/output_step1_cvx/step1_1/layerwise_sensitivity.csv \
  --alpha_csv ./output/output_step1_cvx/step1_2/alpha_layerwise_rankvar.csv \
  --model_id meta-llama/Llama-3.2-3B \
  --avg_bits_target 2.50 \
  --output_dir ./output/static_info_v3

"""

import os
import re
import json
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _has_col(df: pd.DataFrame, name: str) -> bool:
    return _norm(name) in {_norm(c) for c in df.columns}


def _get_col(df: pd.DataFrame, *cands: str) -> Optional[str]:
    # normalized exact match first, then substring
    cols = list(df.columns)
    nmap = {_norm(c): c for c in cols}

    for cand in cands:
        nc = _norm(cand)
        if nc in nmap:
            return nmap[nc]

    # substring match
    nkeys = list(nmap.keys())
    for cand in cands:
        nc = _norm(cand)
        for nk in nkeys:
            if nc in nk or nk in nc:
                return nmap[nk]
    return None


def _infer_sens_cols(sens: pd.DataFrame) -> Tuple[str, str, str]:
    layer_col = _get_col(sens, "layer_name", "layer", "module", "name")
    if layer_col is None:
        raise KeyError(f"[sens] cannot find layer column. cols={list(sens.columns)}")

    W_col = _get_col(sens, "numel(w_j)", "numel_w_j", "numel", "num_params", "n_params", "params", "W", "w")
    if W_col is None:
        raise KeyError(f"[sens] cannot find W column. cols={list(sens.columns)}")

    # prefer per_param if present, else sum, else mean
    C_col = _get_col(sens, "C_per_param", "C_sum", "C_mean_per_batch", "C_mean", "C")
    if C_col is None:
        raise KeyError(f"[sens] cannot find C column. cols={list(sens.columns)}")

    return layer_col, W_col, C_col


def _infer_alpha_cols(alpha: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Force Step2 long format:
      - layer key: module (preferred) else full_name (strip .weight)
      - bit col: bit
      - alpha col: alpha
    """
    bit_col = _get_col(alpha, "bit", "bits", "b")
    a_col = _get_col(alpha, "alpha", "alpha_val", "alpha_value", "a")
    if bit_col is None or a_col is None:
        raise KeyError(f"[alpha] need bit/alpha columns. cols={list(alpha.columns)}")

    # prefer module
    layer_col = _get_col(alpha, "module")
    if layer_col is None:
        layer_col = _get_col(alpha, "layer_name", "layer", "full_name", "name")
    if layer_col is None:
        raise KeyError(f"[alpha] cannot find layer/module column. cols={list(alpha.columns)}")

    return layer_col, bit_col, a_col


def _strip_weight_suffix(x: str) -> str:
    x = str(x)
    if x.endswith(".weight"):
        return x[:-len(".weight")]
    return x


# -----------------------------
# Main builder
# -----------------------------
def build_static_info(
    sens_csv: str,
    alpha_csv: str,
    model_id: str,
    avg_bits_target: float,
) -> Tuple[Dict, pd.DataFrame]:

    sens = pd.read_csv(sens_csv)
    alpha = pd.read_csv(alpha_csv)

    # --- sens parsing ---
    sens_layer_col, W_col, C_col = _infer_sens_cols(sens)
    sens_layers = sens[sens_layer_col].astype(str).tolist()

    W_map: Dict[str, int] = {}
    C_map: Dict[str, float] = {}
    for ln, w, c in zip(sens_layers, sens[W_col].tolist(), sens[C_col].tolist()):
        W_map[str(ln)] = int(float(w))
        C_map[str(ln)] = float(c)

    # --- alpha parsing (Step2 long format) ---
    a_layer_col, bit_col, a_col = _infer_alpha_cols(alpha)

    tmp = alpha[[a_layer_col, bit_col, a_col]].copy()
    tmp[a_layer_col] = tmp[a_layer_col].astype(str)

    # if using full_name, strip ".weight" so it matches sens/module naming
    if _norm(a_layer_col) in {_norm("full_name"), _norm("name")} and (tmp[a_layer_col].str.contains(r"\.weight$").any()):
        tmp[a_layer_col] = tmp[a_layer_col].map(_strip_weight_suffix)

    tmp[bit_col] = tmp[bit_col].astype(int)
    tmp[a_col] = tmp[a_col].astype(float)

    bits = [1, 2, 3, 4]
    # keep only bits 1/2/3/4
    tmp = tmp[tmp[bit_col].isin(bits)].copy()

    if tmp.empty:
        raise RuntimeError("[alpha] filtered alpha table is empty after keeping bits in {1,2,3,4}")

    # pivot: index=layer(module), columns=bit, values=alpha
    piv = tmp.pivot_table(index=a_layer_col, columns=bit_col, values=a_col, aggfunc="mean")

    # ensure all bit columns exist
    for b in bits:
        if b not in piv.columns:
            piv[b] = np.nan

    # reindex to sens layer order (critical)
    piv = piv.reindex(sens_layers)

    missing_layers = [ln for ln in sens_layers if ln not in piv.index or (ln in piv.index and piv.loc[ln].isna().any())]
    if missing_layers:
        # show first few with diagnostics
        ex = missing_layers[:10]
        raise ValueError(
            f"[alpha] missing some layer/bit entries after pivot+reindex. "
            f"missing_count={len(missing_layers)} examples={ex}\n"
            f"Hint: check naming mismatch between sens layer_name and alpha module/full_name."
        )

    # build alpha_map
    alpha_map: Dict[str, Dict[str, float]] = {}
    for ln in sens_layers:
        row = piv.loc[ln]
        a1 = float(np.clip(row[1], 1e-6, 1.0 - 1e-6))
        a2 = float(np.clip(row[2], 1e-6, 1.0 - 1e-6))
        a3 = float(np.clip(row[3], 1e-6, 1.0 - 1e-6))
        a4 = float(np.clip(row[4], 1e-6, 1.0 - 1e-6))
        alpha_map[str(ln)] = {"1": a1, "2": a2, "3": a3, "4": a4}

    # quick sanity: how many layers have (almost) identical alphas?
    eq_cnt = 0
    for ln in sens_layers:
        a1 = alpha_map[ln]["1"]
        a2 = alpha_map[ln]["2"]
        a3 = alpha_map[ln]["3"]
        a4 = alpha_map[ln]["4"]
        if max(abs(a1 - a2), abs(a1 - a3), abs(a1 - a4), abs(a2 - a3), abs(a2 - a4), abs(a3 - a4)) < 1e-8:
            eq_cnt += 1

    flat = pd.DataFrame(
        [
            {
                "layer_name": ln,
                "W": W_map[ln],
                "C": C_map[ln],
                "alpha1": alpha_map[ln]["1"],
                "alpha2": alpha_map[ln]["2"],
                "alpha3": alpha_map[ln]["3"],
                "alpha4": alpha_map[ln]["4"],
            }
            for ln in sens_layers
        ]
    )

    static_info = {
        "model_id": model_id,
        "avg_bits_target": float(avg_bits_target),
        "layer_names": sens_layers,
        "C_map": C_map,
        "W_map": W_map,
        "alpha_map": alpha_map,
        "meta": {
            "sens_csv": sens_csv,
            "alpha_csv": alpha_csv,
            "sens_cols_used": {"layer": sens_layer_col, "W": W_col, "C": C_col},
            "alpha_cols_used": {"layer": a_layer_col, "bit": bit_col, "alpha": a_col},
            "note": f"layers_with_identical_alpha_1_2_3_4 (tol=1e-8): {eq_cnt}/{len(sens_layers)}",
        },
    }
    return static_info, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_csv", type=str, required=True)
    ap.add_argument("--alpha_csv", type=str, required=True)
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--avg_bits_target", type=float, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="static_info_v3.json")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    static_info, flat = build_static_info(
        sens_csv=args.sens_csv,
        alpha_csv=args.alpha_csv,
        model_id=args.model_id,
        avg_bits_target=args.avg_bits_target,
    )

    out_json = os.path.join(args.output_dir, args.out_name)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(static_info, f, indent=2, ensure_ascii=False)

    out_flat = os.path.join(args.output_dir, "alpha_table_flat.csv")
    flat.to_csv(out_flat, index=False)

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_flat}")
    print(f"[Info] sens cols used: {static_info['meta']['sens_cols_used']}")
    print(f"[Info] alpha cols used: {static_info['meta']['alpha_cols_used']}")
    print(f"[Info] {static_info['meta']['note']}")


if __name__ == "__main__":
    main()
