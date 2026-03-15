#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_hrpca_am_v2.py

목적:
- Hessian-weighted residual Rd = (W - Wq) * d 에 대해
  AM-RPCA(Adaptive Metric RPCA) 방식으로 sparse outlier를 분리한 뒤,
  "outlier 제거 후 residual이 실제로 더 low-rank friendly 해졌는지"를 검증한다.

핵심 수정점:
- 기존 코드의 clean_evr = EVR@r(Ld)는 Ld가 정의상 rank<=r 이므로 항상 1에 가까워지는 문제가 있었음.
- 따라서 본 버전은 반드시 cleaned residual:
      R_clean = R - S
      Rd_clean = R_clean * d
  의 스펙트럼을 직접 측정한다.

보고 지표:
1) base_evr@r          = EVR@r(Rd)
2) clean_evr@r         = EVR@r(Rd_clean)
3) evr_gain            = clean - base
4) base_stable_rank    = ||Rd||_F^2 / ||Rd||_2^2
5) clean_stable_rank   = ||Rd_clean||_F^2 / ||Rd_clean||_2^2
6) stable_rank_delta   = clean - base (음수일수록 더 rank-concentrated)
7) base_top1_share     = sigma1^2 / sum sigma_i^2
8) clean_top1_share    = same on Rd_clean
9) top1_share_gain     = clean - base
10) sparse_energy_ratio = ||Sd||_F^2 / ||Rd||_F^2
11) clean_energy_ratio  = ||Rd_clean||_F^2 / ||Rd||_F^2

사용 예:
CUDA_VISIBLE_DEVICES=2 python test/verify_hrpca_am_v2.py \
    --model_id meta-llama/Llama-3.1-8B \
    --step1_dir ./output/llama3_8b_64/step1_quant/2bit \
    --calib_s ./output/llama3_8b_64/calib_sqrtdiag.pt \
    --out_dir ./output/llama3_8b_64/hrpca_am_verify_v2/2bit \
    --rank 64 \
    --sparse_ratio 0.005
"""

import argparse
import gc
import json
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM


# ------------------------------------------------------------
# AM-RPCA
# ------------------------------------------------------------
@torch.no_grad()
def am_rpca_fit(
    R: torch.Tensor,
    d: torch.Tensor,
    rank: int,
    n_iter: int = 20,
    sparse_ratio: float = 0.005,
):
    """
    Hessian-weighted residual space에서 동작하는 안정화된 RPCA.

    입력:
      R: [O, I] residual in original space
      d: [I] sqrt-diag Hessian proxy
    내부:
      Rw = R * d
      Rn = Rw / scale
      Rn = Ln + Sn

    반환:
      L, S: original space tensors such that approximately R ≈ L + S
    """
    D = d.unsqueeze(0)  # [1, I]

    # Weighted space
    R_w = R * D
    scale = R_w.abs().max().clamp_min(1e-12)
    R_n = R_w / scale

    L_n = torch.zeros_like(R_n)
    S_n = torch.zeros_like(R_n)

    k = max(1, int(R_n.numel() * sparse_ratio))

    for _ in range(n_iter):
        # Low-rank update
        temp_L = R_n - S_n
        u, s, vh = torch.linalg.svd(temp_L, full_matrices=False)

        if rank < s.numel():
            s = s.clone()
            s[rank:] = 0

        L_n = (u * s.unsqueeze(0)) @ vh

        # Sparse update
        temp_S = R_n - L_n
        flat = temp_S.reshape(-1)

        _, idx = torch.topk(flat.abs(), k=k, largest=True, sorted=False)

        S_n.zero_()
        S_n.reshape(-1)[idx] = flat[idx]

        recon_err = torch.norm(R_n - L_n - S_n) / torch.norm(R_n).clamp_min(1e-12)
        if recon_err.item() < 1e-6:
            break

    # Back to original space
    L = (L_n * scale) / D.clamp_min(1e-8)
    S = (S_n * scale) / D.clamp_min(1e-8)
    return L, S


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    target_suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "out_proj", "gate_proj", "up_proj", "down_proj"
    }
    return tensor.ndim == 2 and any(x in name for x in target_suffixes)


def dequant_from_codebook_codes(codebook: torch.Tensor, qcodes: torch.Tensor, orig_i: int) -> torch.Tensor:
    """
    codebook: [O, G, Q]
    qcodes:   [O, G, S]
    output:   [O, I]
    """
    o, g, q = codebook.shape
    _, _, s = qcodes.shape

    cb = codebook.reshape(o * g, q)
    idx = qcodes.reshape(o * g, s).long()
    xq = torch.gather(cb, dim=1, index=idx).reshape(o, g, s)
    return xq.reshape(o, g * s)[:, :orig_i]


@torch.no_grad()
def spectral_stats(m: torch.Tensor, rank: int) -> Dict[str, float]:
    """
    행렬 스펙트럼 관련 통계.
    """
    try:
        s = torch.linalg.svdvals(m)
    except Exception:
        return {
            "evr": 0.0,
            "stable_rank": 0.0,
            "top1_share": 0.0,
            "fro_sq": 0.0,
            "spec_sq": 0.0,
        }

    if s.numel() == 0:
        return {
            "evr": 0.0,
            "stable_rank": 0.0,
            "top1_share": 0.0,
            "fro_sq": 0.0,
            "spec_sq": 0.0,
        }

    s2 = s.pow(2)
    fro_sq = s2.sum().item()
    spec_sq = s2[0].item()

    if fro_sq < 1e-15:
        return {
            "evr": 0.0,
            "stable_rank": 0.0,
            "top1_share": 0.0,
            "fro_sq": 0.0,
            "spec_sq": 0.0,
        }

    q = min(rank, s.numel())
    evr = (s2[:q].sum() / s2.sum()).item()
    stable_rank = float(fro_sq / max(spec_sq, 1e-15))
    top1_share = float(spec_sq / fro_sq)

    return {
        "evr": float(evr),
        "stable_rank": stable_rank,
        "top1_share": top1_share,
        "fro_sq": float(fro_sq),
        "spec_sq": float(spec_sq),
    }


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
@torch.no_grad()
def analyze_layer(key: str, ctx: dict, rank: int, sparse_ratio: float):
    device = ctx["device"]

    W = ctx["state"][key].to(device=device, dtype=torch.float32)

    meta = ctx["metas"][key]
    orig_i = int(meta["orig_shape"][1])

    codebook = ctx["codebooks"][key].to(device=device, dtype=torch.float32)
    qcodes = ctx["qcodes"][key].to(device=device)
    Wq = dequant_from_codebook_codes(codebook, qcodes, orig_i)

    d = ctx["calib_s"][key]["s"].to(device=device, dtype=torch.float32)

    # Residuals
    R = W - Wq
    Rd = R * d.unsqueeze(0)

    # RPCA decomposition
    L, S = am_rpca_fit(R=R, d=d, rank=rank, n_iter=25, sparse_ratio=sparse_ratio)

    # Clean residual after removing sparse outliers
    R_clean = R - S
    Rd_clean = R_clean * d.unsqueeze(0)

    Sd = S * d.unsqueeze(0)

    # Spectral stats on the actual cleaned residual
    base_stats = spectral_stats(Rd, rank)
    clean_stats = spectral_stats(Rd_clean, rank)

    sparse_energy_ratio = float(Sd.pow(2).sum().item() / max(base_stats["fro_sq"], 1e-15))
    clean_energy_ratio = float(Rd_clean.pow(2).sum().item() / max(base_stats["fro_sq"], 1e-15))

    # Optional consistency check
    # L should be close to R_clean
    consistency = float(
        torch.norm((L - R_clean) * d.unsqueeze(0)).item()
        / max(torch.norm(Rd_clean).item(), 1e-15)
    )

    return {
        "layer": key,

        "base_evr": base_stats["evr"],
        "clean_evr": clean_stats["evr"],
        "evr_gain": clean_stats["evr"] - base_stats["evr"],

        "base_stable_rank": base_stats["stable_rank"],
        "clean_stable_rank": clean_stats["stable_rank"],
        "stable_rank_delta": clean_stats["stable_rank"] - base_stats["stable_rank"],

        "base_top1_share": base_stats["top1_share"],
        "clean_top1_share": clean_stats["top1_share"],
        "top1_share_gain": clean_stats["top1_share"] - base_stats["top1_share"],

        "sparse_energy_ratio": sparse_energy_ratio,
        "clean_energy_ratio": clean_energy_ratio,

        "weighted_consistency_err": consistency,
    }


def summarize_results(results):
    n = max(len(results), 1)

    def avg(k):
        return sum(r[k] for r in results) / n

    return {
        "num_layers": len(results),
        "avg_base_evr": avg("base_evr"),
        "avg_clean_evr": avg("clean_evr"),
        "avg_evr_gain": avg("evr_gain"),

        "avg_base_stable_rank": avg("base_stable_rank"),
        "avg_clean_stable_rank": avg("clean_stable_rank"),
        "avg_stable_rank_delta": avg("stable_rank_delta"),

        "avg_base_top1_share": avg("base_top1_share"),
        "avg_clean_top1_share": avg("clean_top1_share"),
        "avg_top1_share_gain": avg("top1_share_gain"),

        "avg_sparse_energy_ratio": avg("sparse_energy_ratio"),
        "avg_clean_energy_ratio": avg("clean_energy_ratio"),

        "avg_weighted_consistency_err": avg("weighted_consistency_err"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--step1_dir", required=True)
    parser.add_argument("--calib_s", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--sparse_ratio", type=float, default=0.005)
    parser.add_argument("--max_layers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[AM-RPCA-v2] Loading context and model...")

    ctx = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "codebooks": torch.load(f"{args.step1_dir}/codebook.pt", map_location="cpu"),
        "qcodes": torch.load(f"{args.step1_dir}/qcodes.pt", map_location="cpu"),
        "metas": torch.load(f"{args.step1_dir}/meta.pt", map_location="cpu"),
        "calib_s": torch.load(args.calib_s, map_location="cpu"),
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    ctx["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    del model
    gc.collect()

    target_keys = sorted([
        k for k in ctx["codebooks"].keys()
        if k in ctx["state"] and is_target_weight(k, ctx["state"][k])
    ])[:args.max_layers]

    print()
    print(f"[AM-RPCA-v2] Analysis (rank={args.rank}, sparse_ratio={args.sparse_ratio * 100:.3f}%)")
    print(
        f"{'Layer':<14} | {'BaseEVR':<8} | {'CleanEVR':<8} | {'dEVR':<8} | "
        f"{'BaseSR':<9} | {'CleanSR':<9} | {'dSR':<9} | "
        f"{'BaseTop1':<9} | {'CleanTop1':<10} | {'SparseE':<8}"
    )
    print("-" * 120)

    results = []
    for key in target_keys:
        res = analyze_layer(key, ctx, args.rank, args.sparse_ratio)
        results.append(res)

        layer_name = ".".join(key.split(".")[-3:-1])  # e.g. self_attn.q_proj

        print(
            f"{layer_name:<14} | "
            f"{res['base_evr']:.4f}   | {res['clean_evr']:.4f}   | {res['evr_gain']:+.4f}  | "
            f"{res['base_stable_rank']:.2f}     | {res['clean_stable_rank']:.2f}     | {res['stable_rank_delta']:+.2f}    | "
            f"{res['base_top1_share']:.4f}    | {res['clean_top1_share']:.4f}     | {res['sparse_energy_ratio']:.4f}"
        )

    summary = summarize_results(results)

    with open(out_dir / "am_rpca_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "am_rpca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 120)
    print(f"Average EVR gain           : {summary['avg_evr_gain']:+.4f}")
    print(f"Average stable-rank delta  : {summary['avg_stable_rank_delta']:+.2f}")
    print(f"Average top1-share gain    : {summary['avg_top1_share_gain']:+.4f}")
    print(f"Average sparse energy ratio: {summary['avg_sparse_energy_ratio']:.4f}")
    print(f"Average clean energy ratio : {summary['avg_clean_energy_ratio']:.4f}")
    print(f"Avg consistency error      : {summary['avg_weighted_consistency_err']:.6e}")


if __name__ == "__main__":
    main()