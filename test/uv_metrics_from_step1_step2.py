#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
uv_metrics_from_step1_step2.py (FINAL)

✅ 목적
Step1 + Step2 산출물만으로 Rw = (W - Wq) ⊙ s^T 를 자동 구성하고,
"uv OFF vs uv ON"을 **step3_5_v2의 정의 그대로** 맞춰서 다음 3가지 지표를 비교합니다:

1) CapturedEnergy@rankR / NRMSE@rankR  (Rw 기준)
2) (옵션) Best-so-far objective curve  (step3_5 로그 2개 주면)
3) Residual Row/Col energy CV          (잔차 균형화)

✅ 입력
- step1_dir:
  - (A) quant_error.pt  (추천)  dict[key] = (W - Wq) [O,I] fp32/cpu
    또는
  - (B) original_weights.pt + quantized_weights.pt  dict[key]=W, Wq
- calib_sqrtdiag.pt:
  dict[key] = {"s": Tensor[I], ...}

✅ uv OFF / uv ON 정의 (중요)
- OFF: u=v=1 고정, pred = SVD_rank(Rw)
- ON : step3_5 방식 반복
        rbar = u^{-1} Rw v^{-1}
        (A,B) = SVD_rank(rbar)
        (u,v) = argmin ||Rw - diag(u) (A B) diag(v)||_F^2   (ALS closed-form)
      pred = diag(u) (A B) diag(v)

사용 예시:
CUDA_VISIBLE_DEVICES=1 python uv_metrics_from_step1_step2.py \
  --step1_dir ./output/4bit_asym_nonuniform_base \
  --calib_s  ./output/4bit_asym_nonuniform_base/calib_sqrtdiag.pt \
  --layer_regex "model.layers.0.*(q_proj|k_proj|v_proj).weight" \
  --rank 64 --device cuda \
  --out_dir ./uv_metrics_layer0

옵션: step3_5 로그로 best-so-far curve 비교
  --log_uv_off /path/to/uv_off/train_log.jsonl \
  --log_uv_on  /path/to/uv_on/train_log.jsonl \
  --log_field  mse_weighted

"""

import argparse, json, os, re
import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------
# IO helpers
# ---------------------------

def load_pt_dict(path: str):
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a dict .pt")
    return obj

def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ---------------------------
# Math helpers
# ---------------------------

def frob_norm_sq(X: torch.Tensor) -> torch.Tensor:
    return (X * X).sum()

def best_so_far_curve(vals):
    out = []
    cur = None
    for v in vals:
        cur = v if cur is None else min(cur, v)
        out.append(cur)
    return out

def cv_energy_rows_cols(R: torch.Tensor):
    e_row = torch.sqrt((R * R).sum(dim=1) + 1e-12)
    e_col = torch.sqrt((R * R).sum(dim=0) + 1e-12)
    cv_row = (e_row.std(unbiased=False) / (e_row.mean() + 1e-12)).item()
    cv_col = (e_col.std(unbiased=False) / (e_col.mean() + 1e-12)).item()
    return cv_row, cv_col

@torch.no_grad()
def captured_energy_and_nrmse(Rw: torch.Tensor, Rw_hat: torch.Tensor):
    denom = frob_norm_sq(Rw) + 1e-12
    resid = Rw - Rw_hat
    num = frob_norm_sq(resid)
    captured = (1.0 - (num / denom)).item()
    nrmse = torch.sqrt(num / denom).item()
    return captured, nrmse, resid


# ---------------------------
# Rank-r SVD (A,B split) with fallbacks
# ---------------------------

@torch.no_grad()
def rank_r_svd_AB(m: torch.Tensor, r: int):
    """
    step3_5 스타일:
      m ≈ A B, A=[O,r], B=[r,I] where A=U*sqrt(S), B=sqrt(S)*Vh
    lowrank가 없으면 full svd로 fallback.
    """
    o, i = m.shape
    r_eff = min(int(r), o, i)
    if r_eff <= 0:
        raise ValueError("rank must be positive")

    # try lowrank if available
    if hasattr(torch.linalg, "svd_lowrank"):
        try:
            u, s, v = torch.linalg.svd_lowrank(m, q=r_eff, niter=2)
            sroot = torch.sqrt(s.clamp_min(0.0))
            a = u * sroot.unsqueeze(0)
            b = sroot.unsqueeze(1) * v.T
            return a, b
        except Exception:
            pass

    if hasattr(torch, "svd_lowrank"):
        try:
            u, s, v = torch.svd_lowrank(m, q=r_eff, niter=2)
            sroot = torch.sqrt(s.clamp_min(0.0))
            a = u * sroot.unsqueeze(0)
            b = sroot.unsqueeze(1) * v.T
            return a, b
        except Exception:
            pass

    # fallback: full svd
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    u = u[:, :r_eff]
    s = s[:r_eff]
    vh = vh[:r_eff, :]
    sroot = torch.sqrt(s.clamp_min(0.0))
    a = u * sroot.unsqueeze(0)
    b = sroot.unsqueeze(1) * vh
    return a, b


# ---------------------------
# uv closed-form (step3_5 style)
# ---------------------------

@torch.no_grad()
def update_uv_closed_form_like_step3(
    rw: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    uv_iters: int = 2,
    eps: float = 1e-8,
    normalize_mode: str = "mean_abs",
):
    """
    minimize || rw - diag(u) (a@b) diag(v) ||_F^2 by ALS closed-form updates.
    normalize_mode: {"none","mean_abs","rms","median_abs"}
    """
    m = a @ b
    u = u.clone()
    v = v.clone()

    for _ in range(max(1, int(uv_iters))):
        mv = m * v.unsqueeze(0)
        num_u = (rw * mv).sum(dim=1)
        den_u = (mv * mv).sum(dim=1).clamp_min(eps)
        u = num_u / den_u

        mu = u.unsqueeze(1) * m
        num_v = (rw * mu).sum(dim=0)
        den_v = (mu * mu).sum(dim=0).clamp_min(eps)
        v = num_v / den_v

        if normalize_mode != "none":
            u_abs = u.abs()
            if normalize_mode == "mean_abs":
                alpha = u_abs.mean().clamp_min(eps)
            elif normalize_mode == "rms":
                alpha = torch.sqrt(torch.mean(u * u)).clamp_min(eps)
            elif normalize_mode == "median_abs":
                alpha = torch.median(u_abs).clamp_min(eps)
            else:
                raise ValueError(f"unknown normalize_mode: {normalize_mode}")
            u = u / alpha
            v = v * alpha

    return u, v


# ---------------------------
# Fit models (OFF / ON)
# ---------------------------

@torch.no_grad()
def fit_ab_only(rw: torch.Tensor, rank: int):
    """
    OFF: u=v=1, AB = SVD_rank(rw)
    pred = AB
    """
    a, b = rank_r_svd_AB(rw, rank)
    pred = a @ b
    u = torch.ones((rw.shape[0],), device=rw.device, dtype=rw.dtype)
    v = torch.ones((rw.shape[1],), device=rw.device, dtype=rw.dtype)
    return pred, a, b, u, v

@torch.no_grad()
def fit_uvab_step3_style(
    rw: torch.Tensor,
    rank: int,
    uv_outer: int = 3,
    uv_iters: int = 2,
    eps: float = 1e-8,
    normalize_mode: str = "mean_abs",
):
    """
    ON: step3_5 스타일 반복
      rbar = u^{-1} rw v^{-1}
      (a,b) = SVD_rank(rbar)
      (u,v) = ALS closed-form to fit rw ≈ diag(u)(a@b)diag(v)
    """
    o, i = rw.shape
    u = torch.ones((o,), device=rw.device, dtype=rw.dtype)
    v = torch.ones((i,), device=rw.device, dtype=rw.dtype)

    # init AB on rw
    a, b = rank_r_svd_AB(rw, rank)

    for _ in range(max(1, int(uv_outer))):
        # rbar = u^{-1} rw v^{-1} (sign-stable)
        u_sign = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
        v_sign = torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
        u_inv = 1.0 / (u_sign * u.abs().clamp_min(eps))
        v_inv = 1.0 / (v_sign * v.abs().clamp_min(eps))
        rbar = (u_inv.unsqueeze(1) * rw) * v_inv.unsqueeze(0)

        # AB update on rbar
        a, b = rank_r_svd_AB(rbar, rank)

        # uv update to fit rw
        u, v = update_uv_closed_form_like_step3(
            rw=rw, a=a, b=b, u=u, v=v,
            uv_iters=uv_iters, eps=eps, normalize_mode=normalize_mode
        )

    pred = (u.unsqueeze(1) * (a @ b)) * v.unsqueeze(0)
    return pred, a, b, u, v


# ---------------------------
# Logs (optional)
# ---------------------------

def load_objectives_from_log(path, field="mse_weighted"):
    vals = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if field in d:
                    vals.append(float(d[field]))
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and field in obj and isinstance(obj[field], list):
            vals = [float(x) for x in obj[field]]
        elif isinstance(obj, list):
            for d in obj:
                if isinstance(d, dict) and field in d:
                    vals.append(float(d[field]))
                elif isinstance(d, (int, float)):
                    vals.append(float(d))
    else:
        raise ValueError("log must be .jsonl or .json")

    if len(vals) == 0:
        raise ValueError(f"No objective values found: field='{field}', path={path}")
    return vals


# ---------------------------
# Build Rw from Step1+Step2
# ---------------------------

@torch.no_grad()
def build_Rw_for_key(key: str, err_dict, W_dict, Wq_dict, calib_dict, device):
    """
    key: full weight name ending with ".weight"
    returns Rw on device float32
    """
    if key not in calib_dict:
        return None

    s = calib_dict[key]["s"]
    if not torch.is_tensor(s):
        return None
    s = s.to(dtype=torch.float32, device=device)

    if err_dict is not None and key in err_dict:
        E = err_dict[key].to(dtype=torch.float32, device=device)
    else:
        if W_dict is None or Wq_dict is None:
            return None
        if key not in W_dict or key not in Wq_dict:
            return None
        W = W_dict[key].to(dtype=torch.float32, device=device)
        Wq = Wq_dict[key].to(dtype=torch.float32, device=device)
        E = W - Wq

    Rw = (E * s[None, :]).contiguous()
    return Rw


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_dir", type=str, required=True)
    ap.add_argument("--calib_s", type=str, required=True)

    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--layer_regex", type=str, default=None, help="regex to filter weight keys")
    ap.add_argument("--max_layers", type=int, default=0, help="0 = no limit")

    # uv fitting controls (match step3_5 느낌)
    ap.add_argument("--uv_outer", type=int, default=3, help="how many (rbar-SVD <-> uv-ALS) rounds")
    ap.add_argument("--uv_iters", type=int, default=2, help="ALS iters inside each round")
    ap.add_argument("--normalize_mode", type=str, default="mean_abs",
                    choices=["none", "mean_abs", "rms", "median_abs"])
    ap.add_argument("--eps", type=float, default=1e-8)

    ap.add_argument("--out_dir", type=str, default="./uv_metrics_out")

    # optional logs
    ap.add_argument("--log_uv_off", type=str, default=None)
    ap.add_argument("--log_uv_on", type=str, default=None)
    ap.add_argument("--log_field", type=str, default="mse_weighted")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Step1 files
    err_path = first_existing(os.path.join(args.step1_dir, "quant_error.pt"))
    w_path   = first_existing(os.path.join(args.step1_dir, "original_weights.pt"))
    wq_path  = first_existing(os.path.join(args.step1_dir, "quantized_weights.pt"))

    if err_path is None and (w_path is None or wq_path is None):
        raise RuntimeError(
            "Step1 산출물이 부족합니다.\n"
            " - quant_error.pt 를 만들었거나 (--save_err)\n"
            " - 또는 original_weights.pt + quantized_weights.pt 를 만들었어야 합니다. (--save_wq)\n"
        )

    calib = load_pt_dict(args.calib_s)
    err_dict = load_pt_dict(err_path) if err_path is not None else None
    W_dict = load_pt_dict(w_path) if (err_dict is None and w_path is not None) else None
    Wq_dict = load_pt_dict(wq_path) if (err_dict is None and wq_path is not None) else None

    # Key set
    if err_dict is not None:
        keys = [k for k in err_dict.keys() if k in calib]
    else:
        keys = [k for k in W_dict.keys() if (k in Wq_dict and k in calib)]

    if args.layer_regex:
        rr = re.compile(args.layer_regex)
        keys = [k for k in keys if rr.search(k)]

    if len(keys) == 0:
        raise RuntimeError("공통 key가 없습니다. calib_s key와 Step1 key가 맞는지 확인하세요.")

    if args.max_layers and args.max_layers > 0:
        keys = keys[: int(args.max_layers)]

    print(f"Found {len(keys)} layers for evaluation ✅")
    rank = int(args.rank)

    rows = []
    for i, key in enumerate(keys):
        Rw = build_Rw_for_key(key, err_dict, W_dict, Wq_dict, calib, device=device)
        if Rw is None:
            continue

        # OFF (AB only)
        pred_off, _, _, _, _ = fit_ab_only(Rw, rank=rank)
        cap_off, nrmse_off, resid_off = captured_energy_and_nrmse(Rw, pred_off)
        cvrow_off, cvcol_off = cv_energy_rows_cols(resid_off)

        # ON (uv+AB step3 style)
        pred_on, _, _, u, v = fit_uvab_step3_style(
            Rw, rank=rank,
            uv_outer=int(args.uv_outer),
            uv_iters=int(args.uv_iters),
            eps=float(args.eps),
            normalize_mode=str(args.normalize_mode),
        )
        cap_on, nrmse_on, resid_on = captured_energy_and_nrmse(Rw, pred_on)
        cvrow_on, cvcol_on = cv_energy_rows_cols(resid_on)

        rows.append({
            "key": key,
            "captured_off": cap_off,
            "captured_on": cap_on,
            "nrmse_off": nrmse_off,
            "nrmse_on": nrmse_on,
            "cv_row_off": cvrow_off,
            "cv_row_on": cvrow_on,
            "cv_col_off": cvcol_off,
            "cv_col_on": cvcol_on,
            "u_mean_abs": float(u.abs().mean().item()),
            "v_mean_abs": float(v.abs().mean().item()),
        })

        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(keys)} ...")

        del Rw, pred_off, resid_off, pred_on, resid_on, u, v
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(rows) == 0:
        raise RuntimeError("계산된 레이어가 없습니다. (필터/파일 확인 필요)")

    # Save per-layer jsonl
    jsonl_path = os.path.join(args.out_dir, "per_layer_metrics.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Aggregate
    def agg(field):
        arr = np.array([r[field] for r in rows], dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    summary = {
        "rank": rank,
        "n_layers": len(rows),
        "uv_outer": int(args.uv_outer),
        "uv_iters": int(args.uv_iters),
        "normalize_mode": str(args.normalize_mode),
        "captured_off": agg("captured_off"),
        "captured_on": agg("captured_on"),
        "nrmse_off": agg("nrmse_off"),
        "nrmse_on": agg("nrmse_on"),
        "cv_row_off": agg("cv_row_off"),
        "cv_row_on": agg("cv_row_on"),
        "cv_col_off": agg("cv_col_off"),
        "cv_col_on": agg("cv_col_on"),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY (mean) ===")
    print(f"CapturedEnergy  OFF={summary['captured_off']['mean']:.6f} | ON={summary['captured_on']['mean']:.6f}")
    print(f"NRMSE          OFF={summary['nrmse_off']['mean']:.6f} | ON={summary['nrmse_on']['mean']:.6f}")
    print(f"CV_row(resid)  OFF={summary['cv_row_off']['mean']:.6f} | ON={summary['cv_row_on']['mean']:.6f}")
    print(f"CV_col(resid)  OFF={summary['cv_col_off']['mean']:.6f} | ON={summary['cv_col_on']['mean']:.6f}")
    print(f"Saved: {jsonl_path}")

    # ---------------------------
    # Plots
    # ---------------------------
    cap_off_m = summary["captured_off"]["mean"]
    cap_on_m  = summary["captured_on"]["mean"]
    nrm_off_m = summary["nrmse_off"]["mean"]
    nrm_on_m  = summary["nrmse_on"]["mean"]
    cvr_off_m = summary["cv_row_off"]["mean"]
    cvr_on_m  = summary["cv_row_on"]["mean"]
    cvc_off_m = summary["cv_col_off"]["mean"]
    cvc_on_m  = summary["cv_col_on"]["mean"]

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(["OFF", "ON"], [cap_off_m, cap_on_m])
    ax1.set_title(f"Captured Energy @ rank {rank} (mean over layers)")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(["OFF", "ON"], [nrm_off_m, nrm_on_m])
    ax2.set_title(f"NRMSE @ rank {rank} (mean over layers)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "energy_nrmse_mean.png"), dpi=300)
    plt.close()

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(["CV_row OFF", "CV_row ON", "CV_col OFF", "CV_col ON"],
           [cvr_off_m, cvr_on_m, cvc_off_m, cvc_on_m])
    ax.set_title("Residual Row/Col Energy CV (mean) — lower is more balanced")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "residual_cv_mean.png"), dpi=300)
    plt.close()

    # Per-layer deltas (ON-OFF)
    dcap = np.array([r["captured_on"] - r["captured_off"] for r in rows], dtype=np.float64)
    dnrm = np.array([r["nrmse_on"] - r["nrmse_off"] for r in rows], dtype=np.float64)
    dcvr = np.array([r["cv_row_on"] - r["cv_row_off"] for r in rows], dtype=np.float64)
    dcvc = np.array([r["cv_col_on"] - r["cv_col_off"] for r in rows], dtype=np.float64)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dcap, label="Δ Captured (ON-OFF)", linewidth=2)
    ax.plot(dnrm, label="Δ NRMSE (ON-OFF)", linewidth=2)
    ax.plot(dcvr, label="Δ CV_row (ON-OFF)", linewidth=2)
    ax.plot(dcvc, label="Δ CV_col (ON-OFF)", linewidth=2)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Per-layer deltas (ON - OFF) — 원하는 방향: Captured↑, NRMSE↓, CV↓")
    ax.set_xlabel("Layer index (filtered order)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "per_layer_deltas.png"), dpi=300)
    plt.close()

    # Optional: best-so-far objective curves from logs
    if args.log_uv_off and args.log_uv_on:
        off_vals = load_objectives_from_log(args.log_uv_off, field=args.log_field)
        on_vals  = load_objectives_from_log(args.log_uv_on,  field=args.log_field)
        off_best = best_so_far_curve(off_vals)
        on_best  = best_so_far_curve(on_vals)
        T = min(len(off_best), len(on_best))
        off_best, on_best = off_best[:T], on_best[:T]

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(off_best, linewidth=2, label="uv OFF best-so-far")
        ax.plot(on_best,  linewidth=2, label="uv ON  best-so-far")
        ax.set_title("Best-so-far Objective Curve (optional)")
        ax.set_xlabel("Iteration / Outer loop")
        ax.set_ylabel("Objective (lower is better)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "best_so_far_objective.png"), dpi=300)
        plt.close()

        print("\n[Best-so-far]")
        print(f"  OFF best: {off_best[-1]:.6f}  (T={T})")
        print(f"  ON  best: {on_best[-1]:.6f}  (T={T})")

    print(f"\n✅ Done. Outputs saved to: {args.out_dir}\n")


if __name__ == "__main__":
    main()