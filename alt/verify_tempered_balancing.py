#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify beta-tempered metric and iterative bi-diagonal balancing
for AB-only low-rank recovery after Step1 quantization.

We evaluate, for each target layer:
  - baseline weighted SVD on M = (W - Wq) * s
  - beta-tempered residual family: M_beta = (W - Wq) * s^beta
  - iterative bi-diagonal balancing on baseline weighted space:
      M_bal = U M V

For each transformed space, we:
  1) fit rank-r truncated SVD on transformed matrix
  2) map the reconstruction back to the original weighted space M
  3) report transformed-space low-rankness and mapped-back weighted loss

Important:
  - beta-tempering changes the metric (non-orthogonal unless beta=1)
  - balancing also changes the metric via diagonal U,V
  - both are nontrivial under AB-only storage constraint

Outputs:
  - out_dir/layerwise_metrics.csv
  - out_dir/summary.json

Example:
CUDA_VISIBLE_DEVICES=1 python verify_tempered_balancing.py \
  --model_id meta-llama/Llama-3.1-8B \
  --step1_dir ./output/llama3_8b_64/step1_quant/2bit \
  --calib_s_path ./output/llama3_8b_64/calib_sqrtdiag.pt \
  --out_dir ./output/llama3_8b_64/verify_tempered_balancing/2bit \
  --rank 64 \
  --max_layers 16 \
  --device cuda \
  --model_device_map auto
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from step_1_quantize import _snapshot_state_to_cpu, is_target_weight  # noqa: E402


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


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


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
    return xq.reshape(o, g * s)[:, :orig_i].contiguous()


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


@torch.no_grad()
def stable_rank(A: torch.Tensor, eps: float = 1e-12) -> float:
    fro2 = float((A * A).sum().item())
    if fro2 <= eps:
        return 0.0
    svals = torch.linalg.svdvals(A)
    spec2 = float((svals[0] ** 2).item()) if svals.numel() > 0 else 0.0
    return float(fro2 / max(spec2, eps))


@torch.no_grad()
def top_energy_ratio(A: torch.Tensor, rank: int, eps: float = 1e-12) -> float:
    svals = torch.linalg.svdvals(A)
    if svals.numel() == 0:
        return 0.0
    r = min(int(rank), int(svals.numel()))
    numer = float((svals[:r] ** 2).sum().item())
    denom = float((svals ** 2).sum().item())
    return float(numer / max(denom, eps))


@torch.no_grad()
def truncated_svd_reconstruct(M: torch.Tensor, rank: int) -> Tuple[torch.Tensor, Dict[str, float]]:
    o, i = M.shape
    r = min(int(rank), o, i)
    if r <= 0:
        zeros = torch.zeros_like(M)
        return zeros, {
            "rank_used": 0.0,
            "evr_at_rank": 0.0,
            "residual_fro": float(torch.linalg.norm(M).item()),
            "residual_loss": float((M * M).sum().item()),
        }

    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    M_hat = (U_r * S_r.unsqueeze(0)) @ Vh_r
    residual = M - M_hat

    denom = float((M * M).sum().item())
    numer = float((S_r * S_r).sum().item())

    return M_hat, {
        "rank_used": float(r),
        "evr_at_rank": float(numer / max(denom, 1e-12)),
        "residual_fro": float(torch.linalg.norm(residual).item()),
        "residual_loss": float((residual * residual).sum().item()),
    }


@torch.no_grad()
def beta_temper_transform(
    R: torch.Tensor,
    s: torch.Tensor,
    beta: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transformed space:
      M_beta = R * s^beta

    Map back to original weighted space M = R * s:
      M_hat_orig = M_hat_beta * s^(1-beta)
    """
    sb = s.clamp_min(float(eps)).pow(float(beta))
    back = s.clamp_min(float(eps)).pow(float(1.0 - beta))
    M_beta = R * sb.unsqueeze(0)
    return M_beta, back


@torch.no_grad()
def bi_balance_transform(
    M: torch.Tensor,
    iters: int,
    alpha: float,
    mode: str,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Build M_bal = U M V with diagonal U,V accumulated iteratively.

    mode:
      - "rms": use sqrt(mean(x^2))
      - "abs": use mean(abs(x))
    alpha:
      scaling strength in (0, 1]
    """
    assert mode in {"rms", "abs"}

    Mt = M.clone()
    u = torch.ones(M.shape[0], device=M.device, dtype=M.dtype)
    v = torch.ones(M.shape[1], device=M.device, dtype=M.dtype)

    for _ in range(int(iters)):
        if mode == "rms":
            row_measure = torch.sqrt(torch.mean(Mt * Mt, dim=1).clamp_min(float(eps)))
        else:
            row_measure = torch.mean(Mt.abs(), dim=1).clamp_min(float(eps))
        ru = row_measure.pow(-float(alpha))
        Mt = ru.unsqueeze(1) * Mt
        u = u * ru

        if mode == "rms":
            col_measure = torch.sqrt(torch.mean(Mt * Mt, dim=0).clamp_min(float(eps)))
        else:
            col_measure = torch.mean(Mt.abs(), dim=0).clamp_min(float(eps))
        cv = col_measure.pow(-float(alpha))
        Mt = Mt * cv.unsqueeze(0)
        v = v * cv

    meta = {
        "u_cond": float((u.max() / u.min().clamp_min(float(eps))).item()),
        "v_cond": float((v.max() / v.min().clamp_min(float(eps))).item()),
        "u_mean": float(u.mean().item()),
        "v_mean": float(v.mean().item()),
    }
    return Mt, u, v, meta


def load_context(args: argparse.Namespace) -> dict:
    step1_dir = Path(args.step1_dir).resolve()
    codebook_path = step1_dir / "codebook.pt"
    qcodes_path = step1_dir / "qcodes.pt"
    meta_path = step1_dir / "meta.pt"
    if not (codebook_path.exists() and qcodes_path.exists() and meta_path.exists()):
        raise FileNotFoundError("step1_dir must contain codebook.pt, qcodes.pt, meta.pt")

    print(f"[Verify] loading step1 artifacts: {step1_dir}")
    codebooks: Dict[str, torch.Tensor] = torch.load(codebook_path, map_location="cpu")
    qcodes: Dict[str, torch.Tensor] = torch.load(qcodes_path, map_location="cpu")
    metas: Dict[str, dict] = torch.load(meta_path, map_location="cpu")

    print(f"[Verify] loading calib_s: {args.calib_s_path}")
    calib_s: Dict[str, dict] = torch.load(args.calib_s_path, map_location="cpu")

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
        f"[Verify] loading original model: {args.model_id} "
        f"(device_map={resolved_model_device_map}, device={device})"
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
        print("[Verify] Detected meta tensors under device_map mode. Re-loading on CPU.")
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
    for key in codebooks.keys():
        if key not in qcodes or key not in metas or key not in calib_s or key not in state:
            continue
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

    print(f"[Verify] matched layers: {len(keys)}")
    return {
        "device": device,
        "codebooks": codebooks,
        "qcodes": qcodes,
        "metas": metas,
        "calib_s": calib_s,
        "state": state,
        "keys": keys,
    }


@torch.no_grad()
def evaluate_layer(
    key: str,
    ctx: dict,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    device = ctx["device"]
    codebooks = ctx["codebooks"]
    qcodes_dict = ctx["qcodes"]
    metas = ctx["metas"]
    calib_s = ctx["calib_s"]
    state = ctx["state"]

    meta = metas[key]
    orig_i = int(tuple(meta["orig_shape"])[1])

    W = state[key].to(torch.float32).to(device)
    s = load_diag_weight(calib_s[key], eps=float(args.eps)).to(device)

    if s.numel() != orig_i:
        raise RuntimeError(f"diag size mismatch for {key}: expected {orig_i}, got {s.numel()}")

    codebook = codebooks[key].to(device=device, dtype=torch.float32)
    qcodes = qcodes_dict[key].to(device=device)
    Wq = dequant_from_codebook_codes(codebook, qcodes, orig_i=orig_i)

    R = W - Wq
    M = R * s.unsqueeze(0)  # original weighted residual

    baseline_hat, baseline_stats = truncated_svd_reconstruct(M, rank=int(args.rank))
    baseline_loss = float(((M - baseline_hat) ** 2).sum().item())
    baseline_sr = stable_rank(M)
    baseline_evr = float(baseline_stats["evr_at_rank"])
    baseline_top1 = top_energy_ratio(M, rank=1)

    out_rows: List[Dict[str, Any]] = []

    def append_row(
        method: str,
        Mt: torch.Tensor,
        Mhat_orig: torch.Tensor,
        extra: Dict[str, Any],
    ) -> None:
        _, trans_stats = truncated_svd_reconstruct(Mt, rank=int(args.rank))
        mapped_resid = M - Mhat_orig
        mapped_loss = float((mapped_resid * mapped_resid).sum().item())
        mapped_fro = float(torch.linalg.norm(mapped_resid).item())

        row = {
            "layer": key,
            "method": method,
            "rank": int(args.rank),
            "shape_rows": int(M.shape[0]),
            "shape_cols": int(M.shape[1]),

            "baseline_stable_rank": float(baseline_sr),
            "baseline_evr_at_rank": float(baseline_evr),
            "baseline_top1_share": float(baseline_top1),
            "baseline_weighted_loss_after_rank": float(baseline_loss),

            "transformed_stable_rank": float(stable_rank(Mt)),
            "transformed_evr_at_rank": float(trans_stats["evr_at_rank"]),
            "transformed_top1_share": float(top_energy_ratio(Mt, rank=1)),
            "transformed_loss_after_rank": float(trans_stats["residual_loss"]),

            "mapped_back_weighted_loss": float(mapped_loss),
            "mapped_back_weighted_residual_fro": float(mapped_fro),
            "gain_vs_baseline": float(baseline_loss - mapped_loss),
            "rel_gain_vs_baseline": float((baseline_loss - mapped_loss) / max(baseline_loss, 1e-12)),
        }
        row.update(extra)
        out_rows.append(row)

    # baseline
    append_row(
        method="baseline",
        Mt=M,
        Mhat_orig=baseline_hat,
        extra={"family": "baseline"},
    )

    # beta-tempered family
    for beta in [float(x) for x in args.beta_list]:
        Mt, back = beta_temper_transform(R, s, beta=beta, eps=float(args.eps))
        Mhat_t, _ = truncated_svd_reconstruct(Mt, rank=int(args.rank))
        Mhat_orig = Mhat_t * back.unsqueeze(0)

        append_row(
            method=f"beta:{beta:g}",
            Mt=Mt,
            Mhat_orig=Mhat_orig,
            extra={
                "family": "beta_temper",
                "beta": float(beta),
            },
        )

    # bi-balance RMS
    for alpha in [float(x) for x in args.alpha_list]:
        Mt, u, v, meta_uv = bi_balance_transform(
            M=M,
            iters=int(args.balance_iters),
            alpha=alpha,
            mode="rms",
            eps=float(args.eps),
        )
        Mhat_t, _ = truncated_svd_reconstruct(Mt, rank=int(args.rank))
        Mhat_orig = (Mhat_t / u.unsqueeze(1)) / v.unsqueeze(0)

        append_row(
            method=f"bi_balance_rms:{alpha:g}",
            Mt=Mt,
            Mhat_orig=Mhat_orig,
            extra={
                "family": "bi_balance_rms",
                "alpha": float(alpha),
                "balance_iters": int(args.balance_iters),
                **meta_uv,
            },
        )

    # bi-balance ABS
    for alpha in [float(x) for x in args.alpha_list]:
        Mt, u, v, meta_uv = bi_balance_transform(
            M=M,
            iters=int(args.balance_iters),
            alpha=alpha,
            mode="abs",
            eps=float(args.eps),
        )
        Mhat_t, _ = truncated_svd_reconstruct(Mt, rank=int(args.rank))
        Mhat_orig = (Mhat_t / u.unsqueeze(1)) / v.unsqueeze(0)

        append_row(
            method=f"bi_balance_abs:{alpha:g}",
            Mt=Mt,
            Mhat_orig=Mhat_orig,
            extra={
                "family": "bi_balance_abs",
                "alpha": float(alpha),
                "balance_iters": int(args.balance_iters),
                **meta_uv,
            },
        )

    del W, s, codebook, qcodes, Wq, R, M, baseline_hat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser("Verify beta-tempered metric and bi-diagonal balancing")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--step1_dir", required=True)
    ap.add_argument("--calib_s_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto")
    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--beta_list", nargs="+", default=["0.0", "0.25", "0.5", "0.75", "1.0"])
    ap.add_argument("--alpha_list", nargs="+", default=["0.25", "0.5", "0.75", "1.0"])
    ap.add_argument("--balance_iters", type=int, default=5)

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = load_context(args)

    rows: List[Dict[str, Any]] = []
    for idx, key in enumerate(ctx["keys"], start=1):
        print(f"[Verify] ({idx}/{len(ctx['keys'])}) layer: {key}")
        rows.extend(evaluate_layer(key=key, ctx=ctx, args=args))

    csv_path = out_dir / "layerwise_metrics.csv"
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def _mean(method: str, field: str) -> float:
        vals = [float(r[field]) for r in rows if r["method"] == method]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    methods = sorted({str(r["method"]) for r in rows})
    per_method = {}
    for m in methods:
        per_method[m] = {
            "num_rows": sum(1 for r in rows if r["method"] == m),
            "mean_transformed_stable_rank": _mean(m, "transformed_stable_rank"),
            "mean_transformed_evr_at_rank": _mean(m, "transformed_evr_at_rank"),
            "mean_mapped_back_weighted_loss": _mean(m, "mapped_back_weighted_loss"),
            "mean_gain_vs_baseline": _mean(m, "gain_vs_baseline"),
            "mean_rel_gain_vs_baseline": _mean(m, "rel_gain_vs_baseline"),
        }

    summary = {
        "model_id": args.model_id,
        "step1_dir": str(Path(args.step1_dir).resolve()),
        "calib_s_path": str(Path(args.calib_s_path).resolve()),
        "out_dir": str(out_dir),
        "rank": int(args.rank),
        "beta_list": [float(x) for x in args.beta_list],
        "alpha_list": [float(x) for x in args.alpha_list],
        "balance_iters": int(args.balance_iters),
        "num_layers": len(ctx["keys"]),
        "num_rows": len(rows),
        "per_method": per_method,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Verify] saved:")
    print(f"  csv: {csv_path}")
    print(f"  summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()