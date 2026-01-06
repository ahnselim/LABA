#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3b — Simulated Annealing (SA) Refinement for Integer Bit Allocation
===========================================================================

변경점(요청 반영)
----------------
- Step3와 동일한 방식으로 **폴더(`--output_dir`)에 결과 저장**:
  • <output_dir>/step3b_sa_report.csv     : 상세 리포트
  • <output_dir>/bit_assign.csv           : Step4 호환 (layer_name,R_int)
  • <output_dir>/bit_assign_meta.txt      : 요약/통계
- `--out_csv`는 선택(백워드 호환). 주면 리포트를 그 경로에도 추가 저장.

개요
----
연속 워터필링(Convex) 결과를 시드로 사용하고, **담금질(Simulated Annealing)** 기반의
이산 탐색으로 정수 비트 {2,3,4} 할당을 **예산형(budget)** 또는 **타깃손실형(target)**에서
더 최적화합니다.

핵심 아이디어
--------------
1) 가능한 경우, 각 레이어 j에 대해 b∈{2,3,4}에 대한 **실측 잔여 테이블** L_j(b)를 사용.
2) 실측이 없으면 **모형식** L_j(b)=C_j * α_j(b) * 2^{-2b} 사용.
3) **예산형**: Σ w_j b_j = B 제약을 항상 유지하는 **비트 스왑(+1/-1)** 탐색.
4) **타깃형**: 총 비트 최소 + 제약 L≤L_tgt. F_λ=Σ w b + λ·max(0, L-L_tgt)로 SA.

입출력
------
입력 CSV(유연 파싱):
  • --sens_csv: layer_name, w_j(또는 numel...), C_j(기본) 또는 Cprime_j(옵션)
  • --loss_table_csv (선택): layer_name, b2,b3,b4 (또는 L2,L3,L4)
  • --alpha_csv (선택): module, bit, alpha — 없으면 --alpha_default 사용
  • --init_assign_csv (선택): layer_name, b_init 또는 R_int — 시드로 사용

출력(폴더):
  • <output_dir>/step3b_sa_report.csv
  • <output_dir>/bit_assign.csv
  • <output_dir>/bit_assign_meta.txt

python montecarlo/step3b_sa_refine.py \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --mode budget --avg_bits 2.50 \
  --steps_per_temp 2000  --max_temps 400   --rho 0.995   --restarts 2 \
  --output_dir ../artifacts/montecarlo/step3b_refine

"""

from __future__ import annotations
import argparse
import csv
import math
import os
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

KAPPA = 2.0 * math.log(2.0)  # so that 2^{-2R} = exp(-KAPPA * R)

# -----------------------------
# Data structures
# -----------------------------


@dataclass
class Layer:
    name: str
    w: int  # weight count (numel)
    C: float  # sensitivity (or base coefficient)


@dataclass
class Problem:
    layers: List[Layer]
    bits_set: Tuple[int, ...]
    min_bit: int
    max_bit: int
    # loss_table[j][b] -> L_j(b)
    loss_table: List[Dict[int, float]]


# -----------------------------
# CSV helpers
# -----------------------------


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        # csv.Sniffer는 가끔 실패할 수 있어 단순 DictReader로 처리
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in r.items()})
    return rows


# -----------------------------
# Loading & building tables
# -----------------------------


def load_sens_table(path: str, sens_col: str = "C_mean_per_batch") -> List[Layer]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"sens_csv empty: {path}")
    layers: List[Layer] = []

    weight_cols = ["w_j", "numel(w_j)", "numel", "params", "#(W_j)", "weight_count"]
    name_cols = ["layer_name", "module", "name"]

    # detect name
    found_name = next((c for c in name_cols if c in rows[0]), None)
    if not found_name:
        raise ValueError(
            "Cannot find a name column (layer_name/module/name) in sens_csv"
        )

    # detect weight
    found_w = next((c for c in weight_cols if c in rows[0]), None)
    if not found_w:
        raise ValueError(
            "Cannot find a weight column among: %s" % ", ".join(weight_cols)
        )

    # detect sensitivity column
    if sens_col not in rows[0]:
        alternatives = ["C_j", "Cprime_j", "C_sum", "C_mean_per_batch", "C_per_param"]
        sens_col = next((alt for alt in alternatives if alt in rows[0]), None)
        if not sens_col:
            raise ValueError("Cannot find sensitivity column in sens_csv.")

    for r in rows:
        name = r[found_name]
        try:
            w = int(float(r[found_w]))
        except Exception:
            w = int(round(float(r[found_w])))
        C = float(r[sens_col])
        layers.append(Layer(name=name, w=w, C=C))
    return layers


def load_alpha_table(path: str) -> Dict[str, Dict[int, float]]:
    """Return alpha_map[layer_name][bit] = alpha.
    If a row has 'module' in ('GLOBAL','*'), use as global fallback per bit.
    """
    rows = _read_csv_rows(path)
    alpha_map: Dict[str, Dict[int, float]] = {}
    global_row: Dict[int, float] = {}

    for r in rows:
        keys = {k.lower(): k for k in r.keys()}
        mod_key = (
            keys.get("module", None)
            or keys.get("layer_name", None)
            or keys.get("name", None)
        )
        bit_key = keys.get("bit", None)
        alp_key = keys.get("alpha", None)
        if not (mod_key and bit_key and alp_key):
            continue
        module = r[mod_key]
        bit = int(float(r[bit_key]))
        alpha = float(r[alp_key])
        if module.upper() in ("*", "GLOBAL"):
            global_row[bit] = alpha
        else:
            alpha_map.setdefault(module, {})[bit] = alpha

    for m in list(alpha_map.keys()):
        for b in (2, 3, 4):
            if b not in alpha_map[m] and b in global_row:
                alpha_map[m][b] = global_row[b]
    if global_row:
        alpha_map["__GLOBAL__"] = global_row
    return alpha_map


def load_loss_table(path: str) -> Dict[str, Dict[int, float]]:
    """Expected columns: layer_name + (b2,b3,b4) or (L2,L3,L4) or (loss2,loss3,loss4)."""
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"loss_table_csv empty: {path}")

    name_cols = ["layer_name", "module", "name"]
    found_name = next((c for c in name_cols if c in rows[0]), None)
    if not found_name:
        raise ValueError("Cannot find a name column in loss_table_csv")

    candidates = [
        ("b2", "b3", "b4"),
        ("L2", "L3", "L4"),
        ("loss2", "loss3", "loss4"),
        ("l2", "l3", "l4"),
    ]
    found_triplet = next(
        (tpl for tpl in candidates if all(t in rows[0] for t in tpl)), None
    )
    if not found_triplet:
        raise ValueError("Cannot find (b2,b3,b4) or (L2,L3,L4) in loss_table_csv")

    (c2, c3, c4) = found_triplet
    table: Dict[str, Dict[int, float]] = {}
    for r in rows:
        name = r[found_name]
        table[name] = {
            2: float(r[c2]),
            3: float(r[c3]),
            4: float(r[c4]),
        }
    return table


def build_loss_table(
    layers: List[Layer],
    bits: Tuple[int, ...],
    loss_table_csv: Optional[str],
    alpha_csv: Optional[str],
    alpha_default: float,
) -> Tuple[List[Dict[int, float]], Dict[int, float]]:
    """
    Returns (loss_table_per_layer, alpha_global_per_bit)
    - loss_table_per_layer[j][b] = L_j(b)
    - alpha_global_per_bit[b] = alpha used globally when per-layer missing
    """
    loss_table: List[Dict[int, float]] = []
    alpha_global: Dict[int, float] = {b: alpha_default for b in bits}

    if loss_table_csv:
        by_name = load_loss_table(loss_table_csv)
        for L in layers:
            row = by_name.get(L.name)
            if row is None:
                raise ValueError(f"loss_table_csv missing layer {L.name}")
            loss_table.append({b: float(row[b]) for b in bits})
        return loss_table, alpha_global

    alpha_map: Dict[str, Dict[int, float]] = {}
    if alpha_csv and os.path.exists(alpha_csv):
        alpha_map = load_alpha_table(alpha_csv)
        if "__GLOBAL__" in alpha_map:
            for b, val in alpha_map["__GLOBAL__"].items():
                alpha_global[b] = val

    for L in layers:
        row: Dict[int, float] = {}
        for b in bits:
            a = alpha_map.get(L.name, {}).get(b, alpha_global.get(b, alpha_default))
            row[b] = float(L.C) * float(a) * math.exp(-KAPPA * float(b))
        loss_table.append(row)

    return loss_table, alpha_global


# -----------------------------
# Continuous water-filling (budget/target)
# -----------------------------


def waterfill_budget(
    Cp: List[float], w: List[int], B: float, Rmin: float, Rmax: float
) -> List[float]:
    """min Σ Cp_j e^{-KAPPA R_j} s.t. Σ w_j R_j ≤ B, R∈[Rmin,Rmax]"""
    n = len(Cp)
    min_sum = Rmin * sum(w)
    max_sum = Rmax * sum(w)
    if B <= min_sum:
        return [Rmin] * n
    if B >= max_sum:
        return [Rmax] * n

    def total_bits(lam: float) -> float:
        s = 0.0
        for j in range(n):
            val = (1.0 / KAPPA) * math.log((KAPPA * Cp[j]) / (lam * w[j]))
            R = max(Rmin, min(Rmax, val))
            s += w[j] * R
        return s

    lam_lo, lam_hi = 1e-12, 1e12
    while total_bits(lam_lo) > B:
        lam_lo *= 10.0
        if lam_lo > 1e60:
            break
    while total_bits(lam_hi) < B:
        lam_hi /= 10.0
        if lam_hi < 1e-60:
            break

    for _ in range(100):
        lam = math.sqrt(lam_lo * lam_hi)
        tb = total_bits(lam)
        if abs(tb - B) / max(1.0, B) < 1e-6:
            break
        if tb > B:
            lam_lo = lam
        else:
            lam_hi = lam

    lam = math.sqrt(lam_lo * lam_hi)
    R = []
    for j in range(n):
        val = (1.0 / KAPPA) * math.log((KAPPA * Cp[j]) / (lam * w[j]))
        Rj = max(Rmin, min(Rmax, val))
        R.append(Rj)
    return R


def waterfill_target_find_budget(
    Cp: List[float], w: List[int], L_tgt: float, Rmin: float, Rmax: float
) -> Tuple[float, List[float]]:
    """Find minimal B s.t. min_{Σ wR ≤ B} Σ Cp e^{-KAPPA R} ≤ L_tgt."""
    Wsum = float(sum(w))
    B_lo = Rmin * Wsum
    B_hi = Rmax * Wsum

    def L_of_B(B: float) -> float:
        R = waterfill_budget(Cp, w, B, Rmin, Rmax)
        return sum(Cp[j] * math.exp(-KAPPA * R[j]) for j in range(len(Cp)))

    L_hi = L_of_B(B_hi)
    if L_hi > L_tgt:
        Rmax_vec = [Rmax] * len(Cp)
        return B_hi, Rmax_vec

    for _ in range(60):
        B_mid = 0.5 * (B_lo + B_hi)
        L_mid = L_of_B(B_mid)
        if L_mid <= L_tgt:
            B_hi = B_mid
        else:
            B_lo = B_mid
        if abs(B_hi - B_lo) / max(1.0, B_hi) < 1e-6:
            break

    B_star = B_hi
    R_star = waterfill_budget(Cp, w, B_star, Rmin, Rmax)
    return B_star, R_star


# -----------------------------
# Integerization helpers
# -----------------------------


def round_and_fix_budget(
    R: List[float],
    w: List[int],
    loss_table: List[Dict[int, float]],
    bits: Tuple[int, ...],
    B: float,
) -> List[int]:
    """Round to integers and adjust to match budget B using ΔL per weight."""
    n = len(R)
    min_b, max_b = bits[0], bits[-1]
    b = [max(min_b, min(max_b, int(round(Rj)))) for Rj in R]

    def total_bits(bv: List[int]) -> float:
        return sum(w[j] * bv[j] for j in range(n))

    B_now = total_bits(b)
    target = float(B)

    if abs(B_now - target) < 0.5:
        return b

    if B_now > target:
        while B_now > target + 1e-9:
            candidates = []
            for j in range(n):
                if b[j] > min_b:
                    dL = loss_table[j][b[j] - 1] - loss_table[j][b[j]]
                    eff = dL / max(1.0, w[j])
                    candidates.append((eff, dL, j))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1]))
            _, _, jbest = candidates[0]
            b[jbest] -= 1
            B_now -= w[jbest]
        return b
    else:
        while B_now < target - 1e-9:
            candidates = []
            for j in range(n):
                if b[j] < max_b:
                    dL = loss_table[j][b[j] + 1] - loss_table[j][b[j]]
                    eff = dL / max(1.0, w[j])
                    candidates.append((eff, dL, j))
            if not candidates:
                break
            candidates.sort(key=lambda x: (x[0], x[1]))
            _, _, jbest = candidates[0]
            b[jbest] += 1
            B_now += w[jbest]
        return b


# -----------------------------
# SA core
# -----------------------------


def total_loss(problem: Problem, b: List[int]) -> float:
    return sum(problem.loss_table[j][b[j]] for j in range(len(b)))


def total_bits(problem: Problem, b: List[int]) -> int:
    return sum(problem.layers[j].w * b[j] for j in range(len(b)))


def propose_swap_budget(
    problem: Problem, b: List[int], rng: random.Random
) -> Tuple[Optional[Tuple[int, int]], float]:
    min_b, max_b = problem.min_bit, problem.max_bit
    n = len(b)
    for _ in range(64):
        j = rng.randrange(n)
        k = rng.randrange(n)
        if j == k:
            continue
        if b[j] < max_b and b[k] > min_b:
            d = (problem.loss_table[j][b[j] + 1] - problem.loss_table[j][b[j]]) + (
                problem.loss_table[k][b[k] - 1] - problem.loss_table[k][b[k]]
            )
            return (j, k), d
    return None, 0.0


def auto_T0_budget(
    problem: Problem, b: List[int], rng: random.Random, samples: int = 256
) -> float:
    deltas = []
    for _ in range(samples):
        sel, d = propose_swap_budget(problem, b, rng)
        if sel is not None:
            deltas.append(abs(d))
    if not deltas:
        return 1.0
    return max(1e-9, statistics.pstdev(deltas))


def sa_budget(
    problem: Problem,
    b_init: List[int],
    T0: float,
    rho: float,
    steps_per_temp: int,
    max_temps: int,
    rng: random.Random,
) -> Tuple[List[int], float]:
    b = b_init[:]
    L = total_loss(problem, b)
    best_b = b[:]
    best_L = L

    T = T0
    no_improve = 0
    for _ in range(max_temps):
        accept_cnt = 0
        for _ in range(steps_per_temp):
            sel, dL = propose_swap_budget(problem, b, rng)
            if sel is None:
                break
            j, k = sel
            if dL < 0 or rng.random() < math.exp(-dL / max(1e-12, T)):
                b[j] += 1
                b[k] -= 1
                L += dL
                accept_cnt += 1
                if L < best_L - 1e-12:
                    best_L = L
                    best_b = b[:]
        no_improve = no_improve + 1 if accept_cnt == 0 else 0
        T *= rho
        if no_improve >= 8:
            break
    return best_b, best_L


def energy_target(
    problem: Problem, b: List[int], lam: float, L_tgt: float
) -> Tuple[float, float, int]:
    L = total_loss(problem, b)
    B = total_bits(problem, b)
    E = float(B) + lam * max(0.0, L - L_tgt)
    return E, L, B


def propose_move_target(
    problem: Problem, b: List[int], rng: random.Random, p_decrease: float = 0.7
) -> Tuple[int, int]:
    min_b, max_b = problem.min_bit, problem.max_bit
    n = len(b)
    for _ in range(64):
        j = rng.randrange(n)
        if rng.random() < p_decrease:
            if b[j] > min_b:
                return j, -1
        else:
            if b[j] < max_b:
                return j, +1
    for j in range(n):
        if b[j] > min_b:
            return j, -1
    for j in range(n):
        if b[j] < max_b:
            return j, +1
    return -1, 0


def auto_T0_target(
    problem: Problem,
    b: List[int],
    lam: float,
    L_tgt: float,
    rng: random.Random,
    samples: int = 256,
) -> float:
    deltas = []
    E0, _, _ = energy_target(problem, b, lam, L_tgt)
    for _ in range(samples):
        j, d = propose_move_target(problem, b, rng)
        if j < 0 or d == 0:
            continue
        old_b = b[j]
        new_b = old_b + d
        if new_b < problem.min_bit or new_b > problem.max_bit:
            continue
        Ldiff = problem.loss_table[j][new_b] - problem.loss_table[j][old_b]
        Bdiff = problem.layers[j].w * (new_b - old_b)
        new_L = total_loss(problem, b) + Ldiff
        new_E = float(total_bits(problem, b) + Bdiff) + lam * max(0.0, new_L - L_tgt)
        deltas.append(abs(new_E - E0))
    if not deltas:
        return 1.0
    return max(1e-9, statistics.pstdev(deltas))


def sa_target(
    problem: Problem,
    b_init: List[int],
    L_tgt: float,
    lam0: float,
    T0: float,
    rho: float,
    steps_per_temp: int,
    max_temps: int,
    rng: random.Random,
    lam_up: float = 1.3,
    lam_down: float = 0.95,
) -> Tuple[List[int], float, int]:
    b = b_init[:]
    lam = lam0
    E, L, B = energy_target(problem, b, lam, L_tgt)
    best_b = b[:]
    best_E, best_L, best_B = E, L, B

    T = T0
    no_improve = 0
    for _ in range(max_temps):
        accept_cnt = 0
        for _ in range(steps_per_temp):
            j, d = propose_move_target(problem, b, rng)
            if j < 0 or d == 0:
                break
            old_b = b[j]
            new_b = old_b + d
            Ldiff = problem.loss_table[j][new_b] - problem.loss_table[j][old_b]
            Bdiff = problem.layers[j].w * d
            new_L = L + Ldiff
            new_B = B + Bdiff
            new_E = float(new_B) + lam * max(0.0, new_L - L_tgt)
            Ediff = new_E - E
            if Ediff < 0 or rng.random() < math.exp(-Ediff / max(1e-12, T)):
                b[j] = new_b
                E, L, B = new_E, new_L, new_B
                accept_cnt += 1
                if E < best_E - 1e-12 or (
                    abs(E - best_E) < 1e-12 and L <= L_tgt and new_B < best_B
                ):
                    best_E, best_L, best_B = E, L, B
                    best_b = b[:]
        lam = lam * (1.3 if L > L_tgt else 0.95)
        T *= rho
        no_improve = no_improve + 1 if accept_cnt == 0 else 0
        if no_improve >= 8:
            break
    return best_b, best_L, best_B


# -----------------------------
# Main
# -----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Step3b — SA refinement for integer bit allocation (folder outputs)"
    )
    ap.add_argument(
        "--sens_csv",
        type=str,
        required=True,
        help="CSV with layer_name, w_j (or numel), C_j (or Cprime_j)",
    )
    ap.add_argument(
        "--sens_col",
        type=str,
        default="C_mean_per_batch",
        help="Sensitivity column to use (default: C_mean_per_batch). Alternatives auto-detected.",
    )

    ap.add_argument(
        "--loss_table_csv",
        type=str,
        default=None,
        help="Measured per-layer loss table CSV with columns b2,b3,b4 (or L2,L3,L4). Preferred if provided.",
    )
    ap.add_argument(
        "--alpha_csv",
        type=str,
        default=None,
        help="Optional alpha table: module,bit,alpha",
    )
    ap.add_argument(
        "--alpha_default",
        type=float,
        default=1.0,
        help="Fallback alpha when missing (default: 1.0)",
    )

    ap.add_argument(
        "--init_assign_csv",
        type=str,
        default=None,
        help="Optional seed assignment CSV with layer_name and b_init or R_int",
    )

    ap.add_argument(
        "--mode",
        type=str,
        default="budget",
        choices=["budget", "target"],
        help="Optimization mode",
    )
    ap.add_argument(
        "--avg_bits", type=float, default=None, help="(budget) average bits per weight"
    )
    ap.add_argument(
        "--total_bits",
        type=float,
        default=None,
        help="(budget) total bit budget Σ w_j b_j",
    )
    ap.add_argument(
        "--target_loss",
        type=float,
        default=None,
        help="(target) target total loss L_tgt",
    )

    ap.add_argument(
        "--bits",
        type=str,
        default="2,3,4",
        help="Allowed bits, comma-separated (default: 2,3,4)",
    )

    ap.add_argument(
        "--T0", type=str, default="auto", help='Initial temperature (float) or "auto"'
    )
    ap.add_argument(
        "--rho", type=float, default=0.995, help="Cooling factor (default: 0.995)"
    )
    ap.add_argument(
        "--steps_per_temp",
        type=int,
        default=50,
        help="SA steps per temperature (default: 50)",
    )
    ap.add_argument(
        "--max_temps",
        type=int,
        default=400,
        help="Max temperature epochs (default: 400)",
    )
    ap.add_argument(
        "--restarts", type=int, default=2, help="Number of SA restarts (default: 2)"
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")

    # 출력 방식: Step3와 동일하게 폴더 중심
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write outputs (report, bit_assign, meta)",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="(Optional, backward-compat) also write report CSV to this path",
    )

    args = ap.parse_args()
    rng = random.Random(args.seed)

    bits = tuple(sorted(set(int(x) for x in args.bits.split(","))))
    min_bit, max_bit = bits[0], bits[-1]

    layers = load_sens_table(args.sens_csv, sens_col=args.sens_col)
    w = [L.w for L in layers]

    loss_table, alpha_global = build_loss_table(
        layers=layers,
        bits=bits,
        loss_table_csv=args.loss_table_csv,
        alpha_csv=args.alpha_csv,
        alpha_default=args.alpha_default,
    )

    problem = Problem(
        layers=layers,
        bits_set=bits,
        min_bit=min_bit,
        max_bit=max_bit,
        loss_table=loss_table,
    )

    # Build Cprime for continuous seeding (using global alpha at bit=3 by default)
    a_ref = alpha_global.get(3, args.alpha_default)
    Cprime = [L.C * a_ref for L in layers]

    # ---------- Initialize & run SA ----------
    if args.mode == "budget":
        if args.avg_bits is not None:
            B = float(sum(w)) * float(args.avg_bits)
        elif args.total_bits is not None:
            B = float(args.total_bits)
        else:
            raise ValueError("budget mode requires --avg_bits or --total_bits")

        R_cont = waterfill_budget(
            Cprime, w, B, Rmin=float(min_bit), Rmax=float(max_bit)
        )
        b_seed = round_and_fix_budget(R_cont, w, loss_table, bits, B)

        if args.init_assign_csv and os.path.exists(args.init_assign_csv):
            rows = _read_csv_rows(args.init_assign_csv)
            idx = {L.name: i for i, L in enumerate(layers)}
            for r in rows:
                nm = r.get("layer_name") or r.get("module") or r.get("name")
                if nm in idx:
                    j = idx[nm]
                    if r.get("b_init"):
                        b_seed[j] = int(float(r["b_init"]))
                    elif r.get("R_int"):
                        b_seed[j] = int(float(r["R_int"]))
            # fix to exact budget
            curB = total_bits(problem, b_seed)
            if abs(curB - B) > 1e-6:
                b_seed = round_and_fix_budget(
                    [float(x) for x in b_seed], w, loss_table, bits, B
                )

        T0 = (
            auto_T0_budget(problem, b_seed, rng)
            if str(args.T0).lower() == "auto"
            else float(args.T0)
        )

        best_b, best_L = None, float("inf")
        for _ in range(max(1, args.restarts)):
            bb, LL = sa_budget(
                problem,
                b_seed[:],
                T0=T0,
                rho=args.rho,
                steps_per_temp=args.steps_per_temp,
                max_temps=args.max_temps,
                rng=rng,
            )
            if LL < best_L:
                best_L, best_b = LL, bb
        final_b = best_b
        final_L = total_loss(problem, final_b)
        final_B = total_bits(problem, final_b)
        target_desc = (
            f"avg_bits={args.avg_bits}"
            if args.avg_bits is not None
            else f"total_bits={args.total_bits}"
        )

    else:
        if args.target_loss is None:
            raise ValueError("target mode requires --target_loss")
        L_tgt = float(args.target_loss)

        B_star, R_star = waterfill_target_find_budget(
            Cprime, w, L_tgt, Rmin=float(min_bit), Rmax=float(max_bit)
        )
        b_seed = round_and_fix_budget(R_star, w, loss_table, bits, B_star)

        lam0 = 1.0
        T0 = (
            auto_T0_target(problem, b_seed, lam=lam0, L_tgt=L_tgt, rng=rng)
            if str(args.T0).lower() == "auto"
            else float(args.T0)
        )

        best_b, best_L, best_B = None, float("inf"), 10**30
        for _ in range(max(1, args.restarts)):
            bb, LL, BB = sa_target(
                problem,
                b_seed[:],
                L_tgt=L_tgt,
                lam0=lam0,
                T0=T0,
                rho=args.rho,
                steps_per_temp=args.steps_per_temp,
                max_temps=args.max_temps,
                rng=rng,
            )
            if LL <= L_tgt:
                if best_b is None or BB < best_B or (BB == best_B and LL < best_L):
                    best_b, best_L, best_B = bb, LL, BB
            elif best_b is None:
                best_b, best_L, best_B = bb, LL, BB
        final_b = best_b
        final_L = total_loss(problem, final_b)
        final_B = total_bits(problem, final_b)
        target_desc = f"L_tgt={L_tgt:.6e}"

    # ---------- Write outputs (folder like Step3) ----------
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "step3b_sa_report.csv")
    bitassign_path = os.path.join(args.output_dir, "bit_assign.csv")
    meta_path = os.path.join(args.output_dir, "bit_assign_meta.txt")

    # detailed report
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer_name",
                "w_j",
                "L2",
                "L3",
                "L4",
                "b_init_or_seed",
                "b_final",
                "loss_final",
                "Δloss_final",
            ]
        )
        report_seed = locals().get("b_seed", final_b)
        for j, Lyr in enumerate(problem.layers):
            L2 = problem.loss_table[j][2]
            L3 = problem.loss_table[j][3]
            L4 = problem.loss_table[j][4]
            b0 = report_seed[j] if j < len(report_seed) else final_b[j]
            bf = final_b[j]
            lf = problem.loss_table[j][bf]
            dl = lf - problem.loss_table[j][b0]
            writer.writerow(
                [
                    Lyr.name,
                    Lyr.w,
                    f"{L2:.6e}",
                    f"{L3:.6e}",
                    f"{L4:.6e}",
                    b0,
                    bf,
                    f"{lf:.6e}",
                    f"{dl:.6e}",
                ]
            )
        Wsum = sum(w)
        avg_bits = float(final_B) / max(1.0, float(Wsum))
        writer.writerow(
            [
                "__TOTAL__",
                Wsum,
                "",
                "",
                "",
                "",
                "",
                f"{final_L:.6e}",
                f"B={final_B}; avg_bits={avg_bits:.6f}",
            ]
        )

    # step4-compatible bit_assign.csv
    with open(bitassign_path, "w", newline="", encoding="utf-8") as f2:
        w2 = csv.writer(f2)
        w2.writerow(["layer_name", "R_int"])
        for j, Lyr in enumerate(problem.layers):
            w2.writerow([Lyr.name, int(final_b[j])])

    # meta summary
    with open(meta_path, "w", encoding="utf-8") as f:
        mode = args.mode
        Wsum = sum(w)
        avg_bits = float(final_B) / max(1.0, float(Wsum))
        f.write("=== Step3b SA Optimization Summary ===\n")
        f.write(f"mode             : {mode}\n")
        f.write(f"target/budget    : {target_desc}\n")
        f.write(f"result bits      : total_bits={final_B} | avg_bits={avg_bits:.6f}\n")
        f.write(f"result loss      : total_loss={final_L:.6e}\n")
        f.write(f"bits set         : {bits}\n")
        f.write(f"T0,rho,steps/T   : {T0},{args.rho},{args.steps_per_temp}\n")
        f.write(f"max_temps,restarts: {args.max_temps},{args.restarts}\n")

    # optional backward-compat out_csv
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        # copy the same report
        with open(report_path, "r", encoding="utf-8") as src, open(
            args.out_csv, "w", encoding="utf-8"
        ) as dst:
            dst.write(src.read())

    print(f"[Step3b-SA] Wrote report         : {report_path}")
    if args.out_csv:
        print(f"[Step3b-SA] Wrote report (extra): {args.out_csv}")
    print(f"[Step3b-SA] Wrote bit_assign     : {bitassign_path}  (use this for Step4)")
    print(f"[Step3b-SA] Wrote meta           : {meta_path}")
    print("[Step3b-SA] Done.")


if __name__ == "__main__":
    main()
