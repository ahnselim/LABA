#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3 — Bit Optimization (Continuous → Integer), layerwise (no grouping)

모드:
 • target (기본): L_target를 맞추며 총비트 Σ w_j R_j 최소화
 • budget     : 평균비트(= --avg_bits) 제약을 맞추며 잔여손실 Σ C'·2^{-2R} 최소화

입력:
 • --sens_csv (Step1 결과): columns=["layer_name","numel(w_j)","C_sum","C_mean_per_batch","C_per_param","batches"]
 • --alpha_csv (선택, Step2 결과): columns=["module","bit","alpha", ...]
    - --alpha_bit로 사용할 α(b) 선택 (기본 3). 없으면 --alpha_default 사용(기본 1.0)

사용 예:
  # (A) target 모드 (기존과 동일)
  python step3_bit_optimization.py \
    --sens_csv ./artifacts/bitmin/step1/layerwise_sensitivity.csv \
    --alpha_csv ./artifacts/bitmin/step2/alpha_layerwise_rank64.csv \
    --alpha_bit 3 \
    --C_col C_mean_per_batch \
    --target_ratio 0.40 \
    --bmin 2 --bmax 4 \
    --output_dir ./artifacts/bitmin/step3

  # (B) budget 모드 (평균 비트 = 2.25)
  python step3_bit_optimization.py \
    --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
    --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
    --alpha_bit 3 \
    --C_col C_mean_per_batch \
    --avg_bits 2.50 \
    --bmin 2 --bmax 4 \
    --output_dir ../artifacts/montecarlo/step3_budget
"""

import os, csv, math, argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# -------------------------
# 로딩 유틸
# -------------------------
def load_sensitivity_csv(
    path: str, C_col: str, w_col: str
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Step1 CSV 로드 → (names, C, w)"""
    names: List[str] = []
    C_list: List[float] = []
    w_list: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = row.get("layer_name") or row.get("module") or row.get("name")
            if name is None:
                raise ValueError("Step1 CSV에 'layer_name' 열이 필요합니다.")
            try:
                C = float(row[C_col])
            except Exception:
                raise ValueError(
                    f"지정한 C_col='{C_col}' 열을 찾을 수 없거나 float 변환 실패."
                )

            w_raw = row.get(w_col)
            w = float(w_raw) if w_raw is not None else 1.0

            names.append(name)
            C_list.append(C)
            w_list.append(w)

    return names, np.array(C_list, dtype=np.float64), np.array(w_list, dtype=np.float64)


def load_alpha_csv(path: str, target_bit: int) -> Dict[str, float]:
    """Step2 CSV 로드 → module별 α(b)
    기대 열: 'module','bit','alpha'
    반환: {module: alpha_float}
    """
    amap: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                b = int(row.get("bit", "").strip())
            except Exception:
                continue
            if b != target_bit:
                continue

            mod = row.get("module")
            if not mod:
                fn = row.get("full_name", "")
                if fn.endswith(".weight"):
                    mod = fn[:-7]
            if not mod:
                continue

            try:
                a = float(row["alpha"])
            except Exception:
                a = float(row.get("alpha", "1.0"))
            amap[mod] = a
    return amap


# -------------------------
# target 모드: λ-이분 (잔여 목표, 비용 최소)
# -------------------------
def compute_residual_target(
    Cp: np.ndarray, w: np.ndarray, lam: float, bmin: float, bmax: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """target 모드에서
    R(λ) = clamp( 0.5*log2(λ·C'/w), [bmin,bmax] )
    L(λ) = Σ C'·2^{-2R}
    """
    Cp_pos = np.maximum(Cp, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = lam * Cp_pos / np.maximum(w, 1e-30)
        log2v = 0.5 * np.log2(np.maximum(base, 1e-300))
        R_cont = log2v
        R_clamped = np.clip(R_cont, bmin, bmax)
        L = float(np.sum(Cp_pos * (2.0 ** (-2.0 * R_clamped))))
    return R_cont, R_clamped, L


def solve_lambda_for_target(
    Cp: np.ndarray,
    w: np.ndarray,
    L_target: float,
    bmin: int,
    bmax: int,
    max_iter: int = 64,
    tol_rel: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray, float, dict]:
    """λ 이분탐색으로 L ≈ L_target (λ↑ → R↑ → L↓)"""
    L_bmin = float(np.sum(Cp * (2.0 ** (-2.0 * bmin))))
    L_bmax = float(np.sum(Cp * (2.0 ** (-2.0 * bmax))))
    info = {"L_bmin": L_bmin, "L_bmax": L_bmax}

    if L_bmin <= L_target:
        lam = 0.0
        R_cont = np.full_like(Cp, bmin, dtype=np.float64)
        R_clamped = R_cont.copy()
        return lam, R_cont, R_clamped, L_bmin, {**info, "status": "all_bmin_satisfies"}

    if L_bmax > L_target:
        lam = float("inf")
        R_cont = np.full_like(Cp, bmax, dtype=np.float64)
        R_clamped = R_cont.copy()
        return lam, R_cont, R_clamped, L_bmax, {**info, "status": "infeasible_target"}

    lam_lo, lam_hi = 1e-9, 1e9
    for _ in range(20):
        _, _, L_lo = compute_residual_target(Cp, w, lam_lo, bmin, bmax)
        if L_lo >= L_target:
            lam_lo *= 0.1
        else:
            break

    for _ in range(20):
        _, _, L_hi = compute_residual_target(Cp, w, lam_hi, bmin, bmax)
        if L_hi <= L_target:
            break
        lam_hi *= 10.0

    for _ in range(max_iter):
        lam_mid = math.sqrt(lam_lo * lam_hi)
        R_cont, R_clamped, L_mid = compute_residual_target(Cp, w, lam_mid, bmin, bmax)
        if abs(L_mid - L_target) <= max(1e-12, tol_rel * L_target):
            return lam_mid, R_cont, R_clamped, L_mid, {**info, "status": "ok"}
        if L_mid > L_target:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    return lam_mid, R_cont, R_clamped, L_mid, {**info, "status": "max_iter"}


def greedy_integer_refine_target(
    Cp: np.ndarray,
    w: np.ndarray,
    b_init: np.ndarray,
    L_target: float,
    bmin: int,
    bmax: int,
) -> Tuple[np.ndarray, float, float]:
    """target 모드 정수화: 비용(총비트) 최소화, L ≤ L_target 유지"""
    b = b_init.astype(np.int64).copy()
    L_cur = float(np.sum(Cp * (2.0 ** (-2.0 * b))))
    S_cur = float(np.sum(w * b))

    if L_cur > L_target:
        # 잔여 감소/비용(Δdown/w) 큰 순으로 1비트↑
        while L_cur > L_target:
            mask = b < bmax
            if not np.any(mask):
                break
            b_sel = b[mask]
            Cp_sel = Cp[mask]
            w_sel = w[mask]

            delta = Cp_sel * (2.0 ** (-2.0 * b_sel) - 2.0 ** (-2.0 * (b_sel + 1)))
            score = delta / np.maximum(w_sel, 1e-30)
            idx_local = int(np.argmax(score))
            idx = np.where(mask)[0][idx_local]

            b[idx] += 1
            L_cur -= float(delta[idx_local])
            S_cur += float(w[idx])

    else:
        # 비용 줄이기: 잔여 증가/비용(Δup/w) 최소 항부터 1비트↓ (제약 유지)
        while True:
            mask = b > bmin
            if not np.any(mask):
                break
            b_sel = b[mask]
            Cp_sel = Cp[mask]
            w_sel = w[mask]

            delta_up = Cp_sel * (2.0 ** (-2.0 * (b_sel - 1)) - 2.0 ** (-2.0 * b_sel))
            score = delta_up / np.maximum(w_sel, 1e-30)
            order = np.argsort(score)  # 작은 순

            progressed = False
            for idx_local in order:
                idx = np.where(mask)[0][idx_local]
                new_L = L_cur + float(delta_up[idx_local])
                if new_L <= L_target + 1e-12:
                    b[idx] -= 1
                    L_cur = new_L
                    S_cur -= float(w[idx])
                    progressed = True
                    break
            if not progressed:
                break

    return b, L_cur, S_cur


# -------------------------
# budget 모드: μ-이분 (비트 예산, 손실 최소)
# -------------------------
def compute_bits_budget(
    Cp: np.ndarray, w: np.ndarray, mu: float, bmin: float, bmax: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """budget 모드에서
    R(μ) = clamp( 0.5*log2(C'/(μ·w)), [bmin,bmax] )
    S(μ) = Σ w·R, L(μ) = Σ C'·2^{-2R}
    """
    Cp_pos = np.maximum(Cp, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.maximum(Cp_pos / np.maximum(mu * w, 1e-30), 1e-300)
        log2v = 0.5 * np.log2(base)
        R_cont = log2v
        R_clamped = np.clip(R_cont, bmin, bmax)
        S = float(np.sum(w * R_clamped))
        L = float(np.sum(Cp_pos * (2.0 ** (-2.0 * R_clamped))))
    return R_cont, R_clamped, S, L


def solve_mu_for_budget(
    Cp: np.ndarray,
    w: np.ndarray,
    B: float,
    bmin: int,
    bmax: int,
    max_iter: int = 64,
    tol_rel: float = 1e-6,
) -> Tuple[float, np.ndarray, np.ndarray, float, float, dict]:
    """μ 이분탐색으로 S ≈ B (μ↑ → R↓ → S↓)"""
    S_bmin = float(np.sum(w * bmin))
    S_bmax = float(np.sum(w * bmax))
    info = {"S_bmin": S_bmin, "S_bmax": S_bmax}

    if B <= S_bmin + 1e-12:
        mu = float("inf")
        R_cont = np.full_like(Cp, bmin, dtype=np.float64)
        R_clamped = R_cont.copy()
        L = float(np.sum(Cp * (2.0 ** (-2.0 * bmin))))
        return mu, R_cont, R_clamped, S_bmin, L, {**info, "status": "all_bmin_budget"}

    if B >= S_bmax - 1e-12:
        mu = 0.0
        R_cont = np.full_like(Cp, bmax, dtype=np.float64)
        R_clamped = R_cont.copy()
        L = float(np.sum(Cp * (2.0 ** (-2.0 * bmax))))
        return mu, R_cont, R_clamped, S_bmax, L, {**info, "status": "all_bmax_budget"}

    mu_lo, mu_hi = 1e-9, 1e9

    # lo 확장: S_lo > B가 되도록
    for _ in range(20):
        _, _, S_lo, _ = compute_bits_budget(Cp, w, mu_lo, bmin, bmax)
        if S_lo > B:
            mu_lo *= 0.1
        else:
            break

    # hi 확장: S_hi < B가 되도록
    for _ in range(20):
        _, _, S_hi, _ = compute_bits_budget(Cp, w, mu_hi, bmin, bmax)
        if S_hi < B:
            mu_hi *= 10.0
        else:
            break

    for _ in range(max_iter):
        mu_mid = math.sqrt(mu_lo * mu_hi)
        R_cont, R_clamped, S_mid, L_mid = compute_bits_budget(Cp, w, mu_mid, bmin, bmax)
        if abs(S_mid - B) <= max(1e-9, tol_rel * max(1.0, B)):
            return mu_mid, R_cont, R_clamped, S_mid, L_mid, {**info, "status": "ok"}
        if S_mid > B:
            mu_lo = mu_mid  # 비트 과다 → μ↑ 필요 → lo를 올림
        else:
            mu_hi = mu_mid

    return mu_mid, R_cont, R_clamped, S_mid, L_mid, {**info, "status": "max_iter"}


def greedy_integer_refine_budget(
    Cp: np.ndarray,
    w: np.ndarray,
    b_init: np.ndarray,
    B: float,
    bmin: int,
    bmax: int,
) -> Tuple[np.ndarray, float, float]:
    """budget 모드 정수화: 비트 예산 B를 충족하며 잔여손실 L 최소화"""
    b = b_init.astype(np.int64).copy()
    S_cur = float(np.sum(w * b))
    L_cur = float(np.sum(Cp * (2.0 ** (-2.0 * b))))

    if S_cur > B:
        # 줄이기: 잔여 증가/비용(Δup/w) 최소 항부터 1비트↓
        while S_cur > B + 1e-9:
            mask = b > bmin
            if not np.any(mask):
                break
            b_sel = b[mask]
            Cp_sel = Cp[mask]
            w_sel = w[mask]

            delta_up = Cp_sel * (2.0 ** (-2.0 * (b_sel - 1)) - 2.0 ** (-2.0 * b_sel))
            score = delta_up / np.maximum(w_sel, 1e-30)
            idx_local = int(np.argmin(score))
            idx = np.where(mask)[0][idx_local]

            b[idx] -= 1
            L_cur += float(delta_up[idx_local])
            S_cur -= float(w[idx])

    elif S_cur < B - 1e-9:
        # 늘리기: 잔여 감소/비용(Δdown/w) 최대 항부터 1비트↑
        while S_cur < B - 1e-9:
            mask = b < bmax
            if not np.any(mask):
                break
            b_sel = b[mask]
            Cp_sel = Cp[mask]
            w_sel = w[mask]

            delta_down = Cp_sel * (2.0 ** (-2.0 * b_sel) - 2.0 ** (-2.0 * (b_sel + 1)))
            score = delta_down / np.maximum(w_sel, 1e-30)
            idx_local = int(np.argmax(score))
            idx = np.where(mask)[0][idx_local]

            b[idx] += 1
            L_cur -= float(delta_down[idx_local])
            S_cur += float(w[idx])

    return b, L_cur, S_cur


# -------------------------
# 메인
# -------------------------
def main():
    ap = argparse.ArgumentParser("Step 3 — Bit Optimization (Continuous → Integer)")
    ap.add_argument("--sens_csv", required=True, help="Step1 CSV path")
    ap.add_argument("--alpha_csv", default=None, help="(Optional) Step2 CSV path")
    ap.add_argument(
        "--alpha_bit",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="alpha(b)에서 사용할 bit",
    )
    ap.add_argument(
        "--C_col",
        default="C_mean_per_batch",
        help="Step1에서 사용할 C 컬럼 (예: C_mean_per_batch, C_sum, C_per_param)",
    )
    ap.add_argument(
        "--w_col",
        default="numel(w_j)",
        help="Step1에서 사용할 비용 컬럼 (기본: numel(w_j))",
    )

    # 모드 선택
    ap.add_argument(
        "--mode",
        choices=["target", "budget"],
        default=None,
        help="명시하면 해당 모드로 강제. 미지정이면 --avg_bits 유무로 자동 선택.",
    )

    # target 모드 옵션
    ap.add_argument("--target_residual", type=float, default=None, help="절대 L_target")
    ap.add_argument(
        "--target_ratio",
        type=float,
        default=0.50,
        help="L_target = target_ratio * Σ C'·2^{-2·bmin} (abs가 없을 때 사용)",
    )

    # budget 모드 옵션
    ap.add_argument(
        "--avg_bits",
        type=float,
        default=None,
        help="평균 비트(예: 2.25). 지정 시 budget 모드로 동작",
    )
    ap.add_argument(
        "--alpha_default",
        type=float,
        default=1.0,
        help="alpha CSV에 항목 없을 때 기본 α",
    )
    ap.add_argument("--bmin", type=int, default=2)
    ap.add_argument("--bmax", type=int, default=4)
    ap.add_argument("--max_iter", type=int, default=64)
    ap.add_argument("--tol_rel", type=float, default=1e-6)
    ap.add_argument("--output_dir", default="./artifacts/bitmin/step3")

    args = ap.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 모드 결정
    mode = args.mode
    if mode is None:
        mode = "budget" if args.avg_bits is not None else "target"

    # 1) 데이터 로드
    names, C, w = load_sensitivity_csv(
        args.sens_csv, C_col=args.C_col, w_col=args.w_col
    )
    n = len(names)

    if args.alpha_csv:
        amap = load_alpha_csv(args.alpha_csv, target_bit=args.alpha_bit)
        alpha = np.array(
            [float(amap.get(name, args.alpha_default)) for name in names],
            dtype=np.float64,
        )
    else:
        alpha = np.full(n, args.alpha_default, dtype=np.float64)

    # 2) C' 구성
    Cp = alpha * C
    Cp[Cp < 0.0] = 0.0  # 안전 클램프
    bmin, bmax = int(args.bmin), int(args.bmax)

    # 공통 통계
    L_bmin = float(np.sum(Cp * (2.0 ** (-2.0 * bmin))))
    L_bmax = float(np.sum(Cp * (2.0 ** (-2.0 * bmax))))
    S_bmin = float(np.sum(w * bmin))
    S_bmax = float(np.sum(w * bmax))

    # 3) 연속 해 계산 (모드별)
    meta_extra = {}
    if mode == "target":
        if args.target_residual is not None:
            L_target = float(args.target_residual)
        else:
            L_target = float(args.target_ratio) * L_bmin

        lam, R_cont, R_clamped, L_cont, info = solve_lambda_for_target(
            Cp, w, L_target, bmin, bmax, max_iter=args.max_iter, tol_rel=args.tol_rel
        )
        status = info.get("status", "ok")

        # 정수화 (비용 최소, L ≤ L_target 유지)
        b_init = np.clip(np.floor(R_clamped + 1e-12), bmin, bmax).astype(np.int64)
        b_int, L_int, S_int = greedy_integer_refine_target(
            Cp, w, b_init, L_target, bmin, bmax
        )

        meta_extra.update(
            {
                "mode": "target",
                "lambda": lam,
                "L_target": L_target,
                "L_cont": L_cont,
            }
        )

    else:
        # mode == "budget"
        if args.avg_bits is None:
            raise ValueError("budget 모드에는 --avg_bits 를 지정해야 합니다.")
        avg_bits = float(args.avg_bits)
        B = avg_bits * float(np.sum(w))

        mu, R_cont, R_clamped, S_cont, L_cont, info = solve_mu_for_budget(
            Cp, w, B, bmin, bmax, max_iter=args.max_iter, tol_rel=args.tol_rel
        )
        status = info.get("status", "ok")

        # 정수화 (비트예산 충족, 잔여손실 최소)
        b_init = np.clip(np.floor(R_clamped + 1e-12), bmin, bmax).astype(np.int64)
        b_int, L_int, S_int = greedy_integer_refine_budget(Cp, w, b_init, B, bmin, bmax)

        meta_extra.update(
            {
                "mode": "budget",
                "mu": mu,
                "avg_bits": avg_bits,
                "B_budget": B,
                "S_cont": S_cont,
                "L_cont": L_cont,
            }
        )

    # 4) 저장
    total_bits_int = float(np.sum(w * b_int))
    total_bits_cont = float(np.sum(w * R_clamped))

    csv_path = os.path.join(args.output_dir, "bit_assign.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "layer_name",
                "w_j",
                "C_j",
                "alpha_j",
                "Cprime_j",
                "R_cont",
                "R_clamped",
                "R_int",
                "term_residual_int=C'·2^{-2R_int}",
            ]
        )
        for i in range(n):
            term = float(Cp[i] * (2.0 ** (-2.0 * b_int[i])))
            wr.writerow(
                [
                    names[i],
                    f"{w[i]:.0f}",
                    f"{C[i]:.6e}",
                    f"{alpha[i]:.6e}",
                    f"{Cp[i]:.6e}",
                    f"{float(R_cont[i]):.6f}",
                    f"{float(R_clamped[i]):.6f}",
                    int(b_int[i]),
                    f"{term:.6e}",
                ]
            )

    meta_path = os.path.join(args.output_dir, "bit_assign_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("=== Step3 Bit Optimization Summary ===\n")
        f.write(f"mode : {meta_extra.get('mode')}\n")
        if meta_extra.get("mode") == "target":
            f.write(f"lambda : {meta_extra['lambda']:.6e}\n")
            f.write(f"L_target : {meta_extra['L_target']:.6e}\n")
            f.write(f"L_cont(≈target) : {meta_extra['L_cont']:.6e}\n")
        else:
            f.write(f"mu : {meta_extra['mu']:.6e}\n")
            f.write(f"avg_bits (req) : {meta_extra['avg_bits']:.6f}\n")
            f.write(f"B_budget (req) : {meta_extra['B_budget']:.6e}\n")
            f.write(f"S_cont(≈budget) : {meta_extra['S_cont']:.6e}\n")
            f.write(f"L_cont : {meta_extra['L_cont']:.6e}\n")

        f.write(f"status : {status}\n")
        f.write(f"L(bmin) : {L_bmin:.6e}\n")
        f.write(f"L(bmax) : {L_bmax:.6e}\n")
        f.write(f"S(bmin) : {S_bmin:.6e}\n")
        f.write(f"S(bmax) : {S_bmax:.6e}\n")
        f.write(f"L_int(final) : {L_int:.6e}\n")
        f.write(f"S_int(final) : {S_int:.6e}\n")
        f.write(f"Total bits (int) : {total_bits_int:.3e}\n")
        f.write(f"Total bits (cont): {total_bits_cont:.3e}\n")

        active = int(np.sum((R_clamped > bmin + 1e-9) & (R_clamped < bmax - 1e-9)))
        f.write(f"Active set size : {active}\n")

    # 콘솔 요약
    print("\n[Step3] ===== Optimization Summary =====")
    print(f"mode : {meta_extra.get('mode')}")
    if meta_extra.get("mode") == "target":
        print(f"lambda : {meta_extra['lambda']:.6e}")
        print(f"L_target : {meta_extra['L_target']:.6e}")
        print(f"L(bmin) : {L_bmin:.6e} | L(bmax): {L_bmax:.6e}")
        print(f"L_cont(≈target) : {meta_extra['L_cont']:.6e}")
        print(f"L_int(final) : {L_int:.6e}")
    else:
        print(f"mu : {meta_extra['mu']:.6e}")
        print(f"avg_bits (req) : {meta_extra['avg_bits']:.6f}")
        print(f"B_budget (req) : {meta_extra['B_budget']:.6e}")
        print(f"S(bmin) : {S_bmin:.6e} | S(bmax): {S_bmax:.6e}")
        print(f"S_cont(≈budget) : {meta_extra['S_cont']:.6e}")
        print(f"S_int(final) : {S_int:.6e}")
        print(f"L_int(final) : {L_int:.6e}")

    print(f"Total bits (int) : {total_bits_int:.3e} | (cont): {total_bits_cont:.3e}")
    uniq, cnts = np.unique(b_int, return_counts=True)
    print(
        "[Step3] Bit histogram:",
        dict(zip([int(u) for u in uniq], [int(c) for c in cnts])),
    )


if __name__ == "__main__":
    main()
