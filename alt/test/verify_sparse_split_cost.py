#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
verify_sparse_split_cost.py

목적:
  Step1 quantization 산출물(codebook/qcodes/meta)과 calib_sqrtdiag.pt를 사용해
  Hessian-shaped residual

      R_d = (W - Wq) * d

  에 대해 sparse+dense split의 reconstruction 성능과 저장 비용(cost)을 함께 검증한다.

비교 모드:
  - none
  - sparse_global_topk
  - sparse_row_topk

각 레이어마다:
  1) 원본 weight W 로드
  2) Step1 산출물로부터 Wq 복원
  3) residual R = W - Wq
  4) shaped residual R_d = R ⊙ d
  5) sparse support를 선택해 R_d = S + D 로 분해
  6) D에만 rank-r truncated SVD
  7) 최종 복원:
         R_d_hat = S + D_r
  8) 성능/비용 지표 계산:
       - weighted_loss_after_rank
       - total_explained_ratio
       - sparse_fraction / sparse_energy_ratio
       - dense_evr_at_rank
       - low-rank params / sparse nnz / total bits
       - cost ratio vs low-rank only baseline

비용 모델:
  low-rank branch:
      A [m,r], B [r,n]
      => num_params = r(m+n)

  sparse branch:
      nnz = |support|
      index bits:
          - global COO: ceil(log2(m*n))
          - row-wise CSR-like simplified: ceil(log2(n))
      value bits:
          - sparse_value_bits (default 16)

  총 비용(bits):
      sparse_bits + lowrank_bits

주의:
  - 이 cost는 "검증용 근사"다.
  - 실제 구현에서는 packing / shared index / block index / quantized value 등으로 더 줄일 수 있다.
  
CUDA_VISIBLE_DEVICES=2 python test/verify_sparse_split_cost.py \
  --model_id meta-llama/Llama-3.1-8B \
  --step1_dir ./output/llama3_8b/step1_quant/2bit \
  --calib_s_path ./output/llama3_8b/calib_sqrtdiag.pt \
  --out_dir ./output/llama3_8b/verify_sparse_split_cost/2bit_attn_small \
  --rank 64 \
  --layer_regex "mlp" \
  --max_layers 8 \
  --device cuda \
  --model_device_map auto
"""

import os
import re
import gc
import csv
import json
import math
import argparse
from typing import Dict, Optional, Tuple, List

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e


TARGET_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "fc1", "fc2",
}

def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )

def module_name_from_weight(full_weight_name: str) -> str:
    return full_weight_name[: -len(".weight")]

def branch_of_layer(name: str) -> str:
    if ".self_attn." in name or ".attention." in name:
        return "attention"
    if ".mlp." in name or ".ffn." in name or ".feed_forward." in name:
        return "mlp"
    return "other"


# -------------------------------------------------
# dequant from step1 artifacts
# -------------------------------------------------
def _dequant_weight_from_codebook_qcodes(
    codebook: torch.Tensor,   # [O,G,L]
    qcodes: torch.Tensor,     # [O,G,S]
    meta: Dict,
) -> torch.Tensor:
    device = codebook.device
    codes = qcodes.to(device=device, dtype=torch.long)
    O, G, _ = codebook.shape
    O2, G2, S = codes.shape
    assert O == O2 and G == G2, f"shape mismatch: codebook={codebook.shape}, qcodes={qcodes.shape}"

    xq = torch.gather(codebook, dim=2, index=codes)  # [O,G,S]
    flat = xq.reshape(O, G * S)

    orig_shape = meta.get("orig_shape", None)
    if orig_shape is None:
        raise KeyError("meta missing orig_shape")
    _, orig_I = orig_shape

    return flat[:, :orig_I].contiguous()


# -------------------------------------------------
# calib loader
# -------------------------------------------------
def _load_calib_sqrtdiag_map(calib_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    payload = torch.load(calib_path, map_location="cpu")
    calib_map = payload.get("cov_ops", payload)
    if not isinstance(calib_map, dict):
        raise TypeError(f"Unsupported calib payload type: {type(calib_map)!r}")

    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, entry in calib_map.items():
        if not isinstance(entry, dict):
            continue

        s = None
        if "s" in entry:
            s = entry["s"].to(torch.float32)
        elif "sqrt" in entry:
            s = entry["sqrt"].to(torch.float32)
        elif "var" in entry:
            s = entry["var"].to(torch.float32).clamp_min(0.0).sqrt()
        elif "inv_s" in entry:
            inv_s = entry["inv_s"].to(torch.float32).clamp_min(1e-12)
            s = torch.reciprocal(inv_s)

        if s is None:
            continue

        out[key] = {"sqrt_diag": s.detach().to(torch.float32).cpu().contiguous()}

    if not out:
        raise KeyError(f"No usable sqrt-diag statistics found in calib file: {calib_path}")
    return out

def _lookup_sqrt_diag(
    calib_map: Dict[str, Dict[str, torch.Tensor]],
    full_weight_name: str,
) -> Optional[torch.Tensor]:
    entry = calib_map.get(full_weight_name)
    if entry is None:
        entry = calib_map.get(module_name_from_weight(full_weight_name))
    if entry is None:
        return None
    return entry.get("sqrt_diag")


# -------------------------------------------------
# model state
# -------------------------------------------------
def _load_model_state_cpu(
    model_id: str,
    revision: Optional[str],
    trust_remote_code: bool,
    load_dtype: Optional[torch.dtype],
    device_map,
) -> Dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=load_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    state = {}
    for k, v in model.state_dict().items():
        if getattr(v, "is_meta", False):
            raise NotImplementedError(f"meta tensor detected in state_dict: {k}")
        state[k] = v.detach().to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return state


# -------------------------------------------------
# linalg helpers
# -------------------------------------------------
@torch.no_grad()
def _stable_rank(A: torch.Tensor, eps: float = 1e-12) -> float:
    fro2 = float((A * A).sum().item())
    if fro2 <= eps:
        return 0.0
    svals = torch.linalg.svdvals(A)
    spec2 = float((svals[0] ** 2).item()) if svals.numel() > 0 else 0.0
    return float(fro2 / max(spec2, eps))

@torch.no_grad()
def _rank_r_svd(A: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    r = max(0, min(int(rank), int(S.numel())))
    return U[:, :r], S[:r], Vh[:r, :]

@torch.no_grad()
def _truncated_svd_reconstruct(A: torch.Tensor, rank: int) -> Tuple[torch.Tensor, float]:
    U, S, Vh = _rank_r_svd(A, rank=rank)
    if S.numel() == 0:
        return torch.zeros_like(A), 0.0
    Ar = (U * S[None, :]) @ Vh
    denom = float((A * A).sum().item())
    numer = float((S * S).sum().item())
    evr = float(numer / max(denom, 1e-12))
    return Ar, evr


# -------------------------------------------------
# sparse split helpers
# -------------------------------------------------
@torch.no_grad()
def _global_topk_mask_abs(X: torch.Tensor, ratio: float) -> torch.Tensor:
    flat = X.abs().flatten()
    n = flat.numel()
    if n == 0 or ratio <= 0:
        return torch.zeros_like(X, dtype=torch.bool)

    k = max(1, int(math.ceil(float(ratio) * n)))
    k = min(k, n)
    topk_idx = torch.topk(flat, k=k, largest=True, sorted=False).indices
    mask = torch.zeros(n, device=X.device, dtype=torch.bool)
    mask[topk_idx] = True
    return mask.view_as(X)

@torch.no_grad()
def _row_topk_mask_abs(X: torch.Tensor, ratio: float) -> torch.Tensor:
    m, n = X.shape
    if ratio <= 0 or m == 0 or n == 0:
        return torch.zeros_like(X, dtype=torch.bool)

    k = max(1, int(math.ceil(float(ratio) * n)))
    k = min(k, n)
    idx = torch.topk(X.abs(), k=k, dim=1, largest=True, sorted=False).indices
    mask = torch.zeros_like(X, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask

@torch.no_grad()
def _split_sparse_dense(
    X: torch.Tensor,
    mode: str,
    ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "none":
        mask = torch.zeros_like(X, dtype=torch.bool)
    elif mode == "sparse_global_topk":
        mask = _global_topk_mask_abs(X, ratio)
    elif mode == "sparse_row_topk":
        mask = _row_topk_mask_abs(X, ratio)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    S = torch.where(mask, X, torch.zeros_like(X))
    D = X - S
    return S, D, mask


# -------------------------------------------------
# cost model
# -------------------------------------------------
def _ceil_log2_int(x: int) -> int:
    if x <= 1:
        return 1
    return int(math.ceil(math.log2(x)))

def _estimate_cost_bits(
    m: int,
    n: int,
    rank: int,
    mode: str,
    sparse_nnz: int,
    dense_value_bits: int,
    sparse_value_bits: int,
) -> Dict[str, float]:
    # low-rank A/B
    lowrank_num_params = int(rank) * (int(m) + int(n))
    lowrank_bits = float(lowrank_num_params * int(dense_value_bits))

    if mode == "none" or sparse_nnz <= 0:
        sparse_index_bits_per_nnz = 0
        sparse_value_bits_total = 0.0
        sparse_index_bits_total = 0.0
    else:
        if mode == "sparse_global_topk":
            sparse_index_bits_per_nnz = _ceil_log2_int(int(m) * int(n))
        elif mode == "sparse_row_topk":
            # simplified row-wise storage: row id implied by per-row selection sweep
            # column index만 저장한다고 가정
            sparse_index_bits_per_nnz = _ceil_log2_int(int(n))
        else:
            sparse_index_bits_per_nnz = _ceil_log2_int(int(m) * int(n))

        sparse_value_bits_total = float(sparse_nnz * int(sparse_value_bits))
        sparse_index_bits_total = float(sparse_nnz * sparse_index_bits_per_nnz)

    sparse_bits = sparse_value_bits_total + sparse_index_bits_total
    total_bits = lowrank_bits + sparse_bits

    # baseline = low-rank only
    baseline_lowrank_bits = float(int(rank) * (int(m) + int(n)) * int(dense_value_bits))

    return {
        "lowrank_num_params": float(lowrank_num_params),
        "lowrank_bits": float(lowrank_bits),
        "sparse_nnz": float(sparse_nnz),
        "sparse_index_bits_per_nnz": float(sparse_index_bits_per_nnz),
        "sparse_value_bits_total": float(sparse_value_bits_total),
        "sparse_index_bits_total": float(sparse_index_bits_total),
        "sparse_bits_total": float(sparse_bits),
        "total_bits": float(total_bits),
        "baseline_lowrank_bits": float(baseline_lowrank_bits),
        "cost_ratio_vs_lowrank_only": float(total_bits / max(baseline_lowrank_bits, 1e-12)),
    }


# -------------------------------------------------
# evaluate one mode
# -------------------------------------------------
@torch.no_grad()
def _evaluate_mode(
    X: torch.Tensor,
    mode: str,
    ratio: float,
    rank: int,
    dense_value_bits: int,
    sparse_value_bits: int,
) -> Dict[str, float]:
    m, n = X.shape
    S, D, mask = _split_sparse_dense(X, mode=mode, ratio=ratio)

    D_r, dense_evr = _truncated_svd_reconstruct(D, rank=rank)
    X_hat = S + D_r

    resid = X - X_hat
    weighted_loss = float((resid * resid).sum().item())
    weighted_fro = float(torch.linalg.norm(resid).item())

    total_energy = float((X * X).sum().item())
    sparse_energy = float((S * S).sum().item())
    dense_energy = float((D * D).sum().item())
    explained_energy = float((X_hat * X_hat).sum().item())

    sparse_nnz = int(mask.sum().item())
    cost = _estimate_cost_bits(
        m=int(m),
        n=int(n),
        rank=int(rank),
        mode=mode,
        sparse_nnz=sparse_nnz,
        dense_value_bits=int(dense_value_bits),
        sparse_value_bits=int(sparse_value_bits),
    )

    result = {
        "dense_evr_at_rank": float(dense_evr),
        "weighted_loss_after_rank": weighted_loss,
        "weighted_residual_fro_after_rank": weighted_fro,
        "total_explained_ratio": float(explained_energy / max(total_energy, 1e-12)),
        "sparse_fraction": float(mask.float().mean().item()),
        "sparse_energy_ratio": float(sparse_energy / max(total_energy, 1e-12)),
        "dense_energy_ratio": float(dense_energy / max(total_energy, 1e-12)),
        "stable_rank_dense": float(_stable_rank(D)),
        "stable_rank_total": float(_stable_rank(X)),
        **cost,
    }
    return result


# -------------------------------------------------
# selection
# -------------------------------------------------
def _select_target_layers(
    state: Dict[str, torch.Tensor],
    layer_regex: Optional[str],
    max_layers: Optional[int],
) -> List[str]:
    layer_re = re.compile(layer_regex) if layer_regex else None
    names = []
    for k, v in state.items():
        if not is_target_weight(k, v):
            continue
        if layer_re and not layer_re.search(k):
            continue
        names.append(k)
    names = sorted(names)
    if max_layers is not None and max_layers > 0:
        names = names[:max_layers]
    return names


# -------------------------------------------------
# main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser("Verify sparse+dense split with storage cost")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--step1_dir", required=True)
    ap.add_argument("--calib_s_path", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto")

    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--layer_regex", type=str, default="self_attn")
    ap.add_argument("--max_layers", type=int, default=None)

    ap.add_argument("--modes", nargs="+",
                    default=["none", "sparse_global_topk", "sparse_row_topk"])
    ap.add_argument(
        "--topk_ratios",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.02, 0.03, 0.05],
        help="0.001 = top 0.1%%"
    )

    ap.add_argument("--dense_value_bits", type=int, default=16,
                    help="A/B 저장 비트수. fp16이면 16")
    ap.add_argument("--sparse_value_bits", type=int, default=16,
                    help="sparse value 저장 비트수. fp16이면 16")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.dtype == "bf16":
        load_dtype = torch.bfloat16
    elif args.dtype in ("fp16", "float16"):
        load_dtype = torch.float16
    elif args.dtype in ("fp32", "float32"):
        load_dtype = None
    else:
        load_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model_device_map = None if str(args.model_device_map).strip().lower() in {"none", "null", ""} else args.model_device_map

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"[Verify-SparseSplitCost] loading step1 artifacts from: {args.step1_dir}")
    codebook_path = os.path.join(args.step1_dir, "codebook.pt")
    qcodes_path = os.path.join(args.step1_dir, "qcodes.pt")
    meta_path = os.path.join(args.step1_dir, "meta.pt")

    codebooks: Dict[str, torch.Tensor] = torch.load(codebook_path, map_location="cpu")
    qcodes_dict: Dict[str, torch.Tensor] = torch.load(qcodes_path, map_location="cpu")
    metas: Dict[str, Dict] = torch.load(meta_path, map_location="cpu")

    print(f"[Verify-SparseSplitCost] loading calib: {args.calib_s_path}")
    calib_map = _load_calib_sqrtdiag_map(args.calib_s_path)

    print(f"[Verify-SparseSplitCost] loading model state: {args.model_id}")
    try:
        state = _load_model_state_cpu(
            model_id=args.model_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            load_dtype=load_dtype,
            device_map=model_device_map,
        )
    except NotImplementedError:
        print("[Verify-SparseSplitCost] meta tensor detected under device_map mode, reloading on CPU only.")
        state = _load_model_state_cpu(
            model_id=args.model_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            load_dtype=load_dtype,
            device_map="cpu",
        )

    target_layers = _select_target_layers(
        state=state,
        layer_regex=args.layer_regex,
        max_layers=args.max_layers,
    )
    print(f"[Verify-SparseSplitCost] selected layers: {len(target_layers)}")

    rows = []

    for idx, full_name in enumerate(tqdm(target_layers, desc="layers")):
        if full_name not in codebooks or full_name not in qcodes_dict or full_name not in metas:
            print(f"[Verify-SparseSplitCost][warn] missing step1 artifact for {full_name}; skipping")
            continue

        W_cpu = state[full_name].to(torch.float32)
        d_cpu = _lookup_sqrt_diag(calib_map, full_name)
        if d_cpu is None:
            print(f"[Verify-SparseSplitCost][warn] missing sqrt diag for {full_name}; skipping")
            continue

        W = W_cpu.to(device=device, dtype=torch.float32)
        codebook = codebooks[full_name].to(device=device, dtype=torch.float32)
        qcodes = qcodes_dict[full_name].to(device=device)
        meta = metas[full_name]

        Wq = _dequant_weight_from_codebook_qcodes(codebook, qcodes, meta).to(torch.float32)

        d = d_cpu.to(device=device, dtype=torch.float32).flatten()
        if d.numel() != W.shape[1]:
            print(
                f"[Verify-SparseSplitCost][warn] sqrt-diag size mismatch for {full_name}: "
                f"W.shape={tuple(W.shape)}, d={tuple(d.shape)}; skipping"
            )
            del W, codebook, qcodes, Wq, d
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        R = W - Wq
        R_d = R * d.view(1, -1)

        base_total_energy = float((R_d * R_d).sum().item())
        base_total_fro = float(torch.linalg.norm(R_d).item())
        base_stable_rank = float(_stable_rank(R_d))
        layer_branch = branch_of_layer(full_name)

        if "none" in args.modes:
            metrics = _evaluate_mode(
                X=R_d,
                mode="none",
                ratio=0.0,
                rank=int(args.rank),
                dense_value_bits=int(args.dense_value_bits),
                sparse_value_bits=int(args.sparse_value_bits),
            )
            rows.append({
                "layer_idx": int(idx),
                "layer_name": full_name,
                "branch": layer_branch,
                "mode": "none",
                "param_name": "none",
                "param_value": 0.0,
                "rank": int(args.rank),
                "shape_rows": int(W.shape[0]),
                "shape_cols": int(W.shape[1]),
                "base_total_energy": base_total_energy,
                "base_total_fro": base_total_fro,
                "base_stable_rank": base_stable_rank,
                **metrics,
                "weighted_loss_ratio": float(metrics["weighted_loss_after_rank"] / max(base_total_energy, 1e-12)),
            })

        for mode in ("sparse_global_topk", "sparse_row_topk"):
            if mode not in args.modes:
                continue
            for ratio in args.topk_ratios:
                metrics = _evaluate_mode(
                    X=R_d,
                    mode=mode,
                    ratio=float(ratio),
                    rank=int(args.rank),
                    dense_value_bits=int(args.dense_value_bits),
                    sparse_value_bits=int(args.sparse_value_bits),
                )
                rows.append({
                    "layer_idx": int(idx),
                    "layer_name": full_name,
                    "branch": layer_branch,
                    "mode": mode,
                    "param_name": "topk_ratio",
                    "param_value": float(ratio),
                    "rank": int(args.rank),
                    "shape_rows": int(W.shape[0]),
                    "shape_cols": int(W.shape[1]),
                    "base_total_energy": base_total_energy,
                    "base_total_fro": base_total_fro,
                    "base_stable_rank": base_stable_rank,
                    **metrics,
                    "weighted_loss_ratio": float(metrics["weighted_loss_after_rank"] / max(base_total_energy, 1e-12)),
                })

        del W_cpu, W, codebook, qcodes, Wq, d, R, R_d
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    csv_path = os.path.join(args.out_dir, "layerwise_metrics.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            wr.writerows(rows)

    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / max(len(vals), 1))

    def _summarize(group_rows: List[Dict]) -> Dict:
        out = {}
        keys = sorted(set((r["mode"], r["param_name"], float(r["param_value"])) for r in group_rows))
        for mode, param_name, param_value in keys:
            subset = [
                r for r in group_rows
                if r["mode"] == mode and r["param_name"] == param_name and float(r["param_value"]) == float(param_value)
            ]
            if not subset:
                continue
            name = mode if mode == "none" else f"{mode}:{param_name}={param_value}"
            out[name] = {
                "num_rows": len(subset),
                "num_layers": len(set(r["layer_name"] for r in subset)),
                "dense_evr_at_rank": _mean([float(r["dense_evr_at_rank"]) for r in subset]),
                "weighted_loss_after_rank": _mean([float(r["weighted_loss_after_rank"]) for r in subset]),
                "weighted_residual_fro_after_rank": _mean([float(r["weighted_residual_fro_after_rank"]) for r in subset]),
                "weighted_loss_ratio": _mean([float(r["weighted_loss_ratio"]) for r in subset]),
                "total_explained_ratio": _mean([float(r["total_explained_ratio"]) for r in subset]),
                "sparse_fraction": _mean([float(r["sparse_fraction"]) for r in subset]),
                "sparse_energy_ratio": _mean([float(r["sparse_energy_ratio"]) for r in subset]),
                "dense_energy_ratio": _mean([float(r["dense_energy_ratio"]) for r in subset]),
                "stable_rank_dense": _mean([float(r["stable_rank_dense"]) for r in subset]),
                "stable_rank_total": _mean([float(r["stable_rank_total"]) for r in subset]),
                "lowrank_bits": _mean([float(r["lowrank_bits"]) for r in subset]),
                "sparse_bits_total": _mean([float(r["sparse_bits_total"]) for r in subset]),
                "total_bits": _mean([float(r["total_bits"]) for r in subset]),
                "cost_ratio_vs_lowrank_only": _mean([float(r["cost_ratio_vs_lowrank_only"]) for r in subset]),
            }
        return out

    summary = {
        "num_rows": len(rows),
        "num_layers": len(set(r["layer_name"] for r in rows)),
        "rank": int(args.rank),
        "dense_value_bits": int(args.dense_value_bits),
        "sparse_value_bits": int(args.sparse_value_bits),
        "overall": _summarize(rows),
        "attention": _summarize([r for r in rows if r["branch"] == "attention"]),
        "mlp": _summarize([r for r in rows if r["branch"] == "mlp"]),
    }

    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[Verify-SparseSplitCost] saved:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")

    for group_name in ("overall", "attention", "mlp"):
        group = summary.get(group_name, {})
        if not group:
            continue
        print(f"\n[{group_name}]")
        items = sorted(
            group.items(),
            key=lambda x: (x[1]["weighted_loss_after_rank"], x[1]["cost_ratio_vs_lowrank_only"])
        )
        for name, item in items[:12]:
            print(
                f"  {name:42s} | loss={item['weighted_loss_after_rank']:.6f} "
                f"| total_expl={item['total_explained_ratio']:.6f} "
                f"| sparse_frac={item['sparse_fraction']:.6f} "
                f"| sparse_energy={item['sparse_energy_ratio']:.6f} "
                f"| cost_ratio={item['cost_ratio_vs_lowrank_only']:.6f}"
            )
        if items:
            print(f"  -> best by (loss, cost): {items[0][0]}")


if __name__ == "__main__":
    main()