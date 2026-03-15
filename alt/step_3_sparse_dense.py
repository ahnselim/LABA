#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alt Step3 Sparse+Dense - build eval-ready `wdq_star` and dense low-rank AB from Step1/Step2 artifacts.

What this script does:
  1. Load Step1 quantization artifacts (`codebook.pt`, `qcodes.pt`, `meta.pt`)
  2. Reconstruct `Wq` for all target layers and save them into `wdq_star.pt`
  3. Load Step2 calibration stats and form the shaped residual:
       R_d = (W - Wq) * s
  4. Apply sparse top-k only on selected branches/layers, and run dense rank-r SVD on every target layer
  5. Keep `wdq_star.pt` as pure Step1 `Wq`
  6. Save sparse correction together with dense rank-r SVD inside `low_rank_ab_sparse.pt`
  7. Emit metadata files (`b_map.json`, `layerwise_metrics.csv`, `summary.json`)

Compatibility note:
  - `wdq_star.pt` is directly usable by `LABA/alt/step4_eval.py`
  - dense AB is saved in the same per-layer `{A, B, meta}` structure expected by Step4
  - sparse correction is stored inside the AB payload, so `Wdq-only` stays true Wq-only
  - `wdq_star.pt` ignores `--layer_regex` and always covers all target layers from Step1/model state
  - `--layer_regex` only gates sparse splitting; layers outside the regex still receive dense weighted SVD/AB

Usage:
CUDA_VISIBLE_DEVICES=3 nohup python step_3_sparse_dense.py \
  --model_id meta-llama/Llama-3.1-8B \
  --step1_dir ./output/llama3_8b_64/step1_quant/2bit \
  --calib_s_path ./output/llama3_8b_64/calib_sqrtdiag.pt \
  --out_dir ./output/llama3_8b_64/step3_sparse_dense/2bit_attn_1p \
  --rank 64 \
  --sparse_ratio 0.01 \
  --sparse_mode sparse_global_topk \
  --sparse_branch attn \
  --device cuda \
  --model_device_map auto > ./logs/sparse_llama3_8b_2bit_1p.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _should_apply_sparse(layer_name: str, sparse_branch: str) -> bool:
    branch = branch_of_layer(layer_name)
    if sparse_branch == "both":
        return branch in {"attention", "mlp"}
    if sparse_branch == "attn":
        return branch == "attention"
    if sparse_branch == "mlp":
        return branch == "mlp"
    if sparse_branch == "none":
        return False
    raise ValueError(f"Unsupported sparse_branch: {sparse_branch}")


def _matches_layer_regex(layer_name: str, layer_re: Optional[re.Pattern[str]]) -> bool:
    if layer_re is None:
        return True
    return bool(layer_re.search(layer_name))


def _dequant_weight_from_codebook_qcodes(
    codebook: torch.Tensor,
    qcodes: torch.Tensor,
    meta: Dict[str, Any],
) -> torch.Tensor:
    codes = qcodes.to(device=codebook.device, dtype=torch.long)
    O, G, _ = codebook.shape
    O2, G2, S = codes.shape
    if O != O2 or G != G2:
        raise ValueError(f"shape mismatch: codebook={tuple(codebook.shape)}, qcodes={tuple(qcodes.shape)}")

    xq = torch.gather(codebook, dim=2, index=codes)
    flat = xq.reshape(O, G * S)

    orig_shape = meta.get("orig_shape")
    if orig_shape is None:
        raise KeyError("meta missing orig_shape")
    _, orig_i = orig_shape
    return flat[:, :orig_i].contiguous()


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
        inv_s = None
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
        if inv_s is None:
            inv_s = torch.reciprocal(s.clamp_min(1e-12))

        out[key] = {
            "sqrt_diag": s.detach().cpu().contiguous(),
            "inv_s": inv_s.detach().cpu().contiguous(),
        }

    if not out:
        raise KeyError(f"No usable sqrt-diag statistics found in calib file: {calib_path}")
    return out


def _lookup_calib_entry(
    calib_map: Dict[str, Dict[str, torch.Tensor]],
    full_weight_name: str,
) -> Optional[Dict[str, torch.Tensor]]:
    entry = calib_map.get(full_weight_name)
    if entry is None:
        entry = calib_map.get(module_name_from_weight(full_weight_name))
    return entry


def _load_model_state_cpu(
    model_id: str,
    revision: Optional[str],
    trust_remote_code: bool,
    load_dtype: Optional[torch.dtype],
    device_map: Any,
) -> Dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=load_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    state: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if getattr(v, "is_meta", False):
            raise NotImplementedError(f"meta tensor detected in state_dict: {k}")
        state[k] = v.detach().to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return state


@torch.no_grad()
def _global_topk_mask_abs(X: torch.Tensor, ratio: float) -> torch.Tensor:
    flat = X.abs().flatten()
    n = flat.numel()
    if n == 0 or ratio <= 0.0:
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
    if ratio <= 0.0 or m == 0 or n == 0:
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
        raise ValueError(f"Unknown sparse mode: {mode}")

    sparse = torch.where(mask, X, torch.zeros_like(X))
    dense = X - sparse
    return sparse, dense, mask


@torch.no_grad()
def _stable_rank(A: torch.Tensor, eps: float = 1e-12) -> float:
    fro2 = float((A * A).sum().item())
    if fro2 <= eps:
        return 0.0
    svals = torch.linalg.svdvals(A)
    spec2 = float((svals[0] ** 2).item()) if svals.numel() > 0 else 0.0
    return float(fro2 / max(spec2, eps))


@torch.no_grad()
def _dense_to_ab(dense_shaped: torch.Tensor, inv_s: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    U, S, Vh = torch.linalg.svd(dense_shaped, full_matrices=False)
    r = max(0, min(int(rank), int(S.numel())))

    if r == 0:
        A = dense_shaped.new_zeros((dense_shaped.shape[0], 0))
        B = dense_shaped.new_zeros((0, dense_shaped.shape[1]))
        residual = dense_shaped
        dense_energy = float((dense_shaped * dense_shaped).sum().item())
        return A, B, {
            "rank_used": 0.0,
            "dense_energy": dense_energy,
            "dense_evr_at_rank": 0.0,
            "dense_weight_residual_fro": float(torch.linalg.norm(residual).item()),
            "dense_weighted_loss_after_rank": dense_energy,
        }

    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    A = (U_r * S_r.unsqueeze(0)).contiguous()
    B = (Vh_r * inv_s.unsqueeze(0)).contiguous()

    dense_hat = (U_r * S_r.unsqueeze(0)) @ Vh_r
    residual = dense_shaped - dense_hat
    denom = float((dense_shaped * dense_shaped).sum().item())
    numer = float((S_r * S_r).sum().item())

    return A, B, {
        "rank_used": float(r),
        "dense_energy": denom,
        "dense_evr_at_rank": float(numer / max(denom, 1e-12)),
        "dense_weight_residual_fro": float(torch.linalg.norm(residual).item()),
        "dense_weighted_loss_after_rank": float((residual * residual).sum().item()),
    }


def _select_target_layers(
    state: Dict[str, torch.Tensor],
    max_layers: int,
) -> List[str]:
    names: List[str] = []
    for k, v in state.items():
        if not is_target_weight(k, v):
            continue
        names.append(k)
    names = sorted(names)
    if max_layers > 0:
        names = names[:max_layers]
    return names


def _torch_dtype_from_arg(name: str) -> Optional[torch.dtype]:
    if name == "bf16":
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    ap = argparse.ArgumentParser("Alt Step3 Sparse+Dense")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--step1_dir", required=True, help="dir containing codebook.pt/qcodes.pt/meta.pt")
    ap.add_argument("--calib_s_path", required=True, help="step_2_calib output calib_sqrtdiag.pt")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto")

    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--sparse_ratio", type=float, default=0.005)
    ap.add_argument(
        "--sparse_mode",
        default="sparse_global_topk",
        choices=["none", "sparse_global_topk", "sparse_row_topk"],
    )
    ap.add_argument(
        "--layer_regex",
        type=str,
        default=None,
        help="Optional regex for layers that receive sparse split; all target layers still get dense SVD/AB",
    )
    ap.add_argument("--max_layers", type=int, default=0)
    ap.add_argument(
        "--sparse_branch",
        type=str,
        default="both",
        choices=["attn", "mlp", "both", "none"],
        help="Which branch receives sparse correction; dense SVD still runs on every processed layer",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    load_dtype = _torch_dtype_from_arg(args.dtype)
    if args.dtype == "auto":
        load_dtype = None if load_dtype == torch.float32 else load_dtype
    elif args.dtype in {"fp32", "float32"}:
        load_dtype = None

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model_device_map = None if str(args.model_device_map).strip().lower() in {"none", "null", ""} else args.model_device_map
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    step1_dir = Path(args.step1_dir).resolve()
    codebooks: Dict[str, torch.Tensor] = torch.load(step1_dir / "codebook.pt", map_location="cpu")
    qcodes_dict: Dict[str, torch.Tensor] = torch.load(step1_dir / "qcodes.pt", map_location="cpu")
    metas: Dict[str, Dict[str, Any]] = torch.load(step1_dir / "meta.pt", map_location="cpu")
    calib_map = _load_calib_sqrtdiag_map(args.calib_s_path)

    print(f"[SparseDense-Step3] loading model state: {args.model_id}")
    try:
        state = _load_model_state_cpu(
            model_id=args.model_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            load_dtype=load_dtype,
            device_map=model_device_map,
        )
    except NotImplementedError:
        print("[SparseDense-Step3] meta tensor detected under device_map mode, reloading on CPU only.")
        state = _load_model_state_cpu(
            model_id=args.model_id,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            load_dtype=load_dtype,
            device_map="cpu",
        )

    wdq_star: Dict[str, torch.Tensor] = {}
    wdq_layers = _select_target_layers(state, 0)
    print(f"[SparseDense-Step3] wdq layers: {len(wdq_layers)}")
    for full_name in tqdm(wdq_layers, desc="wdq layers"):
        if full_name not in codebooks or full_name not in qcodes_dict or full_name not in metas:
            print(f"[SparseDense-Step3][warn] missing step1 artifact for {full_name}; skipping wdq save")
            continue
        codebook = codebooks[full_name].to(device=device, dtype=torch.float32)
        qcodes = qcodes_dict[full_name].to(device=device)
        meta = metas[full_name]
        Wq = _dequant_weight_from_codebook_qcodes(codebook, qcodes, meta).to(torch.float16).cpu().contiguous()
        wdq_star[full_name] = Wq
        del codebook, qcodes, Wq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    target_layers = _select_target_layers(state, int(args.max_layers))
    layer_re = re.compile(args.layer_regex) if args.layer_regex else None
    sparse_candidate_layers = sum(1 for name in target_layers if _matches_layer_regex(name, layer_re))
    print(
        f"[SparseDense-Step3] target layers: {len(target_layers)} | "
        f"sparse_regex_matched={sparse_candidate_layers} | sparse_branch={args.sparse_branch}"
    )

    lowrank_ab_sparse: Dict[str, Dict[str, Any]] = {}
    b_map: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for idx, full_name in enumerate(tqdm(target_layers, desc="layers"), start=1):
        if full_name not in codebooks or full_name not in qcodes_dict or full_name not in metas:
            print(f"[SparseDense-Step3][warn] missing step1 artifact for {full_name}; skipping")
            continue

        calib_entry = _lookup_calib_entry(calib_map, full_name)
        if calib_entry is None:
            print(f"[SparseDense-Step3][warn] missing calib entry for {full_name}; skipping")
            continue

        W_cpu = state[full_name].to(torch.float32)
        codebook = codebooks[full_name].to(device=device, dtype=torch.float32)
        qcodes = qcodes_dict[full_name].to(device=device)
        meta = metas[full_name]

        W = W_cpu.to(device=device, dtype=torch.float32)
        Wq = _dequant_weight_from_codebook_qcodes(codebook, qcodes, meta).to(torch.float32)

        s = calib_entry["sqrt_diag"].to(device=device, dtype=torch.float32).flatten()
        inv_s = calib_entry["inv_s"].to(device=device, dtype=torch.float32).flatten()
        if s.numel() != W.shape[1]:
            print(
                f"[SparseDense-Step3][warn] sqrt-diag size mismatch for {full_name}: "
                f"W.shape={tuple(W.shape)}, s={tuple(s.shape)}; skipping"
            )
            del W_cpu, W, codebook, qcodes, Wq, s, inv_s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        residual = W - Wq
        shaped = residual * s.unsqueeze(0)
        apply_sparse = _matches_layer_regex(full_name, layer_re) and _should_apply_sparse(full_name, str(args.sparse_branch))
        sparse_mode = str(args.sparse_mode) if apply_sparse else "none"
        sparse_ratio = float(args.sparse_ratio) if apply_sparse else 0.0
        sparse_shaped, dense_shaped, mask = _split_sparse_dense(
            shaped,
            mode=sparse_mode,
            ratio=sparse_ratio,
        )

        sparse_weight = sparse_shaped * inv_s.unsqueeze(0)
        sparse_nnz = int(mask.sum().item())
        if sparse_nnz > 0:
            sparse_indices = mask.nonzero(as_tuple=False).transpose(0, 1).contiguous().cpu()
            sparse_values = sparse_weight[mask].to(torch.float16).cpu().contiguous()
            sparse_shape = [int(W.shape[0]), int(W.shape[1])]
        else:
            sparse_indices = torch.zeros((2, 0), dtype=torch.long)
            sparse_values = torch.zeros((0,), dtype=torch.float16)
            sparse_shape = [int(W.shape[0]), int(W.shape[1])]

        A, B, dense_stats = _dense_to_ab(
            dense_shaped=dense_shaped,
            inv_s=inv_s,
            rank=int(args.rank),
        )
        lowrank_ab_sparse[full_name] = {
            "A": A.to(torch.float16).cpu().contiguous(),
            "B": B.to(torch.float16).cpu().contiguous(),
            "sparse_indices": sparse_indices,
            "sparse_values": sparse_values,
            "sparse_shape": sparse_shape,
            "meta": {
                "rank": int(args.rank),
                "rank_used": int(dense_stats["rank_used"]),
                "sparse_ratio": sparse_ratio,
                "sparse_mode": sparse_mode,
                "sparse_branch": str(args.sparse_branch),
                "bits": int(meta.get("bits", -1)),
                "group_size": int(meta.get("group_size", -1)),
                "folded_sparse_into_wdq": False,
                "has_sparse": bool(sparse_nnz > 0),
            },
        }

        sparse_energy = float((sparse_shaped * sparse_shaped).sum().item())
        dense_energy = float((dense_shaped * dense_shaped).sum().item())
        total_energy = float((shaped * shaped).sum().item())
        b_map[full_name] = {
            "module_name": module_name_from_weight(full_name),
            "branch": branch_of_layer(full_name),
            "shape": [int(W.shape[0]), int(W.shape[1])],
            "a_shape": [int(A.shape[0]), int(A.shape[1])],
            "b_shape": [int(B.shape[0]), int(B.shape[1])],
            "rank": int(args.rank),
            "rank_used": int(dense_stats["rank_used"]),
            "sparse_mode": sparse_mode,
            "sparse_ratio": sparse_ratio,
            "sparse_branch": str(args.sparse_branch),
            "sparse_nnz": sparse_nnz,
            "folded_sparse_into_wdq": False,
            "ab_artifact": "low_rank_ab_sparse.pt",
            "wdq_artifact": "wdq_star.pt",
        }

        rows.append({
            "layer_idx": int(idx - 1),
            "layer_name": full_name,
            "branch": branch_of_layer(full_name),
            "shape_rows": int(W.shape[0]),
            "shape_cols": int(W.shape[1]),
            "bits": int(meta.get("bits", -1)),
            "group_size": int(meta.get("group_size", -1)),
            "rank": int(args.rank),
            "rank_used": int(dense_stats["rank_used"]),
            "sparse_mode": sparse_mode,
            "sparse_ratio": sparse_ratio,
            "sparse_branch": str(args.sparse_branch),
            "sparse_nnz": sparse_nnz,
            "sparse_fraction": float(mask.float().mean().item()),
            "sparse_energy_ratio": float(sparse_energy / max(total_energy, 1e-12)),
            "dense_energy_ratio": float(dense_energy / max(total_energy, 1e-12)),
            "dense_evr_at_rank": float(dense_stats["dense_evr_at_rank"]),
            "dense_weighted_loss_after_rank": float(dense_stats["dense_weighted_loss_after_rank"]),
            "dense_weight_residual_fro": float(dense_stats["dense_weight_residual_fro"]),
            "stable_rank_shaped_total": float(_stable_rank(shaped)),
            "stable_rank_shaped_dense": float(_stable_rank(dense_shaped)),
        })

        del W_cpu, W, codebook, qcodes, Wq, s, inv_s, residual, shaped, sparse_shaped, dense_shaped, mask, sparse_weight, A, B
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    torch.save(wdq_star, out_dir / "wdq_star.pt")
    torch.save(lowrank_ab_sparse, out_dir / "low_rank_ab_sparse.pt")

    with open(out_dir / "b_map.json", "w", encoding="utf-8") as f:
        json.dump(b_map, f, ensure_ascii=False, indent=2)

    if rows:
        with open(out_dir / "layerwise_metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _mean(key: str) -> float:
        if not rows:
            return 0.0
        return float(sum(float(r[key]) for r in rows) / len(rows))

    summary = {
        "model_id": args.model_id,
        "step1_dir": str(step1_dir),
        "calib_s_path": str(Path(args.calib_s_path).resolve()),
        "out_dir": str(out_dir),
        "num_layers": len(rows),
        "wdq_num_layers": len(wdq_star),
        "rank": int(args.rank),
        "layer_regex": args.layer_regex,
        "sparse_mode": str(args.sparse_mode),
        "sparse_ratio": float(args.sparse_ratio),
        "sparse_branch": str(args.sparse_branch),
        "wdq_artifact": "wdq_star.pt",
        "ab_artifact": "low_rank_ab_sparse.pt",
        "folded_sparse_into_wdq": False,
        "mean_sparse_fraction": _mean("sparse_fraction"),
        "mean_sparse_energy_ratio": _mean("sparse_energy_ratio"),
        "mean_dense_energy_ratio": _mean("dense_energy_ratio"),
        "mean_dense_evr_at_rank": _mean("dense_evr_at_rank"),
        "mean_dense_weighted_loss_after_rank": _mean("dense_weighted_loss_after_rank"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[SparseDense-Step3] saved:")
    print(f"  wdq*: {out_dir / 'wdq_star.pt'}")
    print(f"  AB*:  {out_dir / 'low_rank_ab_sparse.pt'}")
    print(f"  map:  {out_dir / 'b_map.json'}")
    print(f"  csv:  {out_dir / 'layerwise_metrics.csv'}")
    print(f"  sum:  {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
