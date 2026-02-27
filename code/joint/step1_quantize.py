#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step1_quantize.py  (NEW) — Asymmetric non-uniform (Lloyd-Max) baseline extractor

목표:
  - asym_nonuniform (per-(O,G) Lloyd-Max / 1D k-means)로
    1) codebook (centroids)
    2) qcodes (각 원소의 code index)
    3) (옵션) Wq 및 error
    를 저장해서 Step3에서 Δ(codebook residual) 학습을 쉽게 만들기.

저장(out_dir):
  - codebook.pt : dict["<full>.weight"] = codebook [O,G,L] (fp16/cpu)
  - qcodes.pt   : dict["<full>.weight"] = qcodes   [O,G,S] (uint8/cpu)
  - meta.pt     : dict["<full>.weight"] = meta dict (bits, group_size, orig_shape, pad, ...)
  - (opt) quantized_weights.pt : dict["<full>.weight"] = Wq [O,I] (fp16/cpu)
  - (opt) quant_error.pt       : dict["<full>.weight"] = (W - Wq) (fp32/cpu)

예시:
CUDA_VISIBLE_DEVICES=2 python step1_quantize.py \
  --model_id meta-llama/Llama-3.2-3B \
  --bits 1 --group_size 128 \
  --clip_percentile 0.0 \
  --lloyd_iter 12 \
  --out_dir ./output/1bit_asym_nonuniform_base \
  --save_wq --save_err

CSV로 bit 지정:
CUDA_VISIBLE_DEVICES=0 python step1_quantize.py \
  --model_id meta-llama/Llama-3.2-3B \
  --bit_assign_csv ./artifacts/bitmin/step3/bit_assign.csv \
  --group_size 128 \
  --out_dir ./output/baseline_lloyd_mpq \
  --save_wq

주의:
  - 이 스크립트는 "asym_nonuniform"만 구현 (너의 새 연구 베이스)
  - CUDA packed params는 만들지 않음 (non-uniform이므로)
"""

import os
import re
import gc
import csv
import argparse
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e


# -------------------------------
# Target filter (기존과 동일한 스타일)
# -------------------------------
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


# -------------------------------
# Bit assignment loader (기존과 동일)
# -------------------------------
def load_selected_bits(csv_path: str) -> Dict[str, int]:
    sel: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = (row.get("layer_name") or row.get("module") or row.get("name") or "").strip()
            if not name:
                continue
            b = None
            for key in ("R_int", "selected_bit", "bit"):
                if key in row and str(row[key]).strip() != "":
                    try:
                        b = int(float(row[key]))
                        break
                    except Exception:
                        pass
            if b is None:
                continue
            b = max(1, min(4, b))
            sel[name] = b
    return sel


# -------------------------------
# Group reshape helpers
# -------------------------------
def _to_groups(W: torch.Tensor, group_size: int):
    """
    W: [O,I] -> Wg: [O,G,S], padding on I so that I_pad % S == 0
    returns (Wg, O, G, S, orig_I, pad)
    """
    O, I = W.shape
    pad = (group_size - (I % group_size)) % group_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    O_, I_pad = W.shape
    G = I_pad // group_size
    return W.view(O_, G, group_size), O_, G, group_size, I, pad

def _from_groups(Xg: torch.Tensor, orig_I: int) -> torch.Tensor:
    O_, G, S = Xg.shape
    return Xg.reshape(O_, G * S)[:, :orig_I]


@torch.no_grad()
def _percentile_clip_lastdim(Wg: torch.Tensor, upper_pct: float, lower_pct: float) -> torch.Tensor:
    """
    clip per-group over flattened (O*gs) for each group G
    Wg: [O,G,S]
    """
    assert Wg.ndim == 3
    O, G, gs = Wg.shape
    flat = Wg.permute(1, 0, 2).reshape(G, -1)  # [G, O*gs]
    n = flat.shape[1]
    lo_k = max(1, int((lower_pct / 100.0) * n))
    hi_k = max(1, int((upper_pct / 100.0) * n))
    lo = torch.kthvalue(flat, lo_k, dim=1).values.view(1, G, 1)
    hi = torch.kthvalue(flat, hi_k, dim=1).values.view(1, G, 1)
    return Wg.clamp(min=lo, max=hi)


# -------------------------------
# Lloyd-Max helpers (너 코드 기반)
# -------------------------------
@torch.no_grad()
def _kth_quantiles_lastdim(X_flat: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """
    X_flat: [N,S], probs: [L] in [0,1]
    returns: [N,L] approx-quantiles via kthvalue
    """
    N, S = X_flat.shape
    probs = probs.clamp(0.0, 1.0).to(device=X_flat.device, dtype=torch.float32)
    ks = (probs * (S - 1)).round().to(torch.int64) + 1  # 1..S
    ks = ks.clamp(1, S).tolist()
    outs = [torch.kthvalue(X_flat, k, dim=1).values for k in ks]
    return torch.stack(outs, dim=1)


@torch.no_grad()
def _lloyd_max_codebook_per_group(
    X_flat: torch.Tensor,         # [N,S], N=O*G
    levels: int,                  # L = 2^b
    max_iter: int = 12,
    tol: float = 1e-4,
    chunk_groups: int = 4096,
) -> torch.Tensor:
    """
    Per-row Lloyd-Max / 1D k-means on X_flat [N,S].
    Returns sorted codebook [N,L].
    """
    if X_flat.ndim != 2:
        raise ValueError(f"X_flat must be 2D [N,S], got {tuple(X_flat.shape)}")
    if levels < 1:
        raise ValueError("levels must be >= 1")

    N, S = X_flat.shape
    cg = max(1, int(chunk_groups))
    codebook = torch.empty((N, levels), device=X_flat.device, dtype=X_flat.dtype)

    for start in range(0, N, cg):
        end = min(start + cg, N)
        x = X_flat[start:end]  # [n,S]

        if levels == 1:
            cb = x.mean(dim=1, keepdim=True)
            codebook[start:end] = cb
            continue

        probs = (torch.arange(levels, device=x.device, dtype=torch.float32) + 0.5) / float(levels)
        cb = _kth_quantiles_lastdim(x, probs)          # init via quantiles
        cb, _ = torch.sort(cb, dim=1)

        for _ in range(max(1, int(max_iter))):
            mid = (cb[:, :-1] + cb[:, 1:]) * 0.5       # [n,L-1]
            idx = torch.sum(x.unsqueeze(-1) > mid.unsqueeze(1), dim=-1).to(torch.long)  # [n,S]

            new_cb = cb.clone()
            for k in range(levels):
                mask = idx == k
                cnt = mask.sum(dim=1)                  # [n]
                valid = cnt > 0
                if valid.any():
                    s = (x * mask.to(x.dtype)).sum(dim=1)
                    new_cb[valid, k] = s[valid] / cnt[valid].to(x.dtype)

            new_cb, _ = torch.sort(new_cb, dim=1)
            delta = (new_cb - cb).abs().amax()
            cb = new_cb
            if float(delta.item()) <= float(tol):
                break

        codebook[start:end] = cb

    return codebook


@torch.no_grad()
def _assign_codes_by_midpoints(
    X_flat: torch.Tensor,      # [N,S]
    codebook: torch.Tensor,    # [N,L] (sorted)
    chunk_groups: int = 4096,
) -> torch.Tensor:
    """
    Return integer codes idx in [0..L-1] for each element in X_flat.
    Uses midpoint boundaries between adjacent centroids.
    """
    if X_flat.ndim != 2 or codebook.ndim != 2:
        raise ValueError("X_flat and codebook must be [N,S] and [N,L]")
    N, S = X_flat.shape
    if codebook.shape[0] != N:
        raise ValueError("N mismatch")
    L = codebook.shape[1]
    out = torch.empty((N, S), device=X_flat.device, dtype=torch.int64)

    if L == 1:
        out.zero_()
        return out

    cg = max(1, int(chunk_groups))
    for start in range(0, N, cg):
        end = min(start + cg, N)
        x = X_flat[start:end]             # [n,S]
        cb = codebook[start:end]          # [n,L]
        mid = (cb[:, :-1] + cb[:, 1:]) * 0.5  # [n,L-1]
        idx = torch.sum(x.unsqueeze(-1) > mid.unsqueeze(1), dim=-1).to(torch.long)  # [n,S]
        out[start:end] = idx

    return out


@torch.no_grad()
def _dequant_from_codebook_and_codes(
    codebook: torch.Tensor,    # [N,L]
    codes: torch.Tensor,       # [N,S] int64/uint8
) -> torch.Tensor:
    """
    Return Xq_flat [N,S] using gather.
    """
    if codes.dtype != torch.long:
        codes = codes.to(torch.long)
    rows = torch.arange(codebook.shape[0], device=codebook.device).unsqueeze(1)  # [N,1]
    return codebook[rows, codes]  # [N,S]


@torch.no_grad()
def lloyd_asym_nonuniform_quantize(
    W: torch.Tensor,                  # [O,I] float32 on device
    b: int,
    group_size: int,
    clip_pct: float = 0.0,
    lloyd_iter: int = 12,
    chunk_groups: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Returns:
      - Wq      : [O,I] float32
      - codebook: [O,G,L] float32
      - qcodes  : [O,G,S] uint8
      - meta    : dict (O,G,S,L,orig_I,pad, ...)
    """
    assert b in (1, 2, 3, 4)
    Wg, O, G, S, orig_I, pad = _to_groups(W, group_size)   # [O,G,S]
    if clip_pct and clip_pct > 0:
        X = _percentile_clip_lastdim(Wg, 100.0 - clip_pct, clip_pct)
    else:
        X = Wg

    X_flat = X.reshape(-1, S)          # [N,S], N=O*G
    L = 1 << b

    cb_flat = _lloyd_max_codebook_per_group(
        X_flat, levels=L, max_iter=lloyd_iter, tol=1e-4, chunk_groups=chunk_groups
    )                                 # [N,L]
    codes_flat = _assign_codes_by_midpoints(X_flat, cb_flat, chunk_groups=chunk_groups)  # [N,S]
    Xq_flat = _dequant_from_codebook_and_codes(cb_flat, codes_flat)                      # [N,S]

    Xq = Xq_flat.reshape(O, G, S)
    Wq = _from_groups(Xq, orig_I)      # [O,I]

    codebook = cb_flat.reshape(O, G, L)
    qcodes = codes_flat.reshape(O, G, S).to(torch.uint8)

    meta = {
        "bits": int(b),
        "group_size": int(group_size),
        "levels": int(L),
        "orig_shape": (int(O), int(orig_I)),
        "O": int(O),
        "G": int(G),
        "S": int(S),
        "pad": int(pad),
        "clip_percentile": float(clip_pct),
        "lloyd_iter": int(lloyd_iter),
    }
    return Wq, codebook, qcodes, meta


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser("Step1 (NEW) — Lloyd-Max asym non-uniform extractor")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--bit_assign_csv", default=None)
    ap.add_argument("--bits", type=int, default=4, choices=[1,2,3,4],
                    help="global bit if csv not provided")
    ap.add_argument("--group_size", type=int, default=128)

    ap.add_argument("--clip_percentile", type=float, default=0.0)
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)

    ap.add_argument("--layer_regex", type=str, default=None)

    ap.add_argument("--save_wq", action="store_true", help="also save Wq as quantized_weights.pt")
    ap.add_argument("--save_err", action="store_true", help="also save error (W - Wq) as quant_error.pt")

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dtype == "bf16":
        load_dtype = torch.bfloat16
    elif args.dtype in ("fp16", "float16"):
        load_dtype = torch.float16
    elif args.dtype in ("fp32", "float32"):
        load_dtype = torch.float32
    else:
        load_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                      else torch.float16)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    print(f"[Step1-LLoyd] Loading model: {args.model_id} (load_dtype={load_dtype}, device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    ).to(device)

    if args.bit_assign_csv:
        sel_bits = load_selected_bits(args.bit_assign_csv)
        print(f"[Step1-LLoyd] Loaded bit assignments: {len(sel_bits)} entries.")
    else:
        sel_bits = None

    # CPU state dict로 옮겨서 GPU 메모리 절약
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    layer_re = re.compile(args.layer_regex) if args.layer_regex else None

    codebooks: Dict[str, torch.Tensor] = {}
    qcodes_dict: Dict[str, torch.Tensor] = {}
    metas: Dict[str, dict] = {}
    qweights: Dict[str, torch.Tensor] = {}
    err_dict: Dict[str, torch.Tensor] = {}

    print("[Step1-LLoyd] Extracting (codebook, qcodes) ...")

    for full_name, W_cpu in tqdm(state.items()):
        if not is_target_weight(full_name, W_cpu):
            continue
        if layer_re and not layer_re.search(full_name):
            continue

        # bit 결정
        bit: Optional[int] = None
        if sel_bits is not None:
            mod_name = module_name_from_weight(full_name)
            bit = sel_bits.get(mod_name, sel_bits.get(full_name.replace(".weight", ""), None))
            if bit is None:
                continue
        else:
            bit = int(args.bits)
        bit = max(1, min(4, int(bit)))

        # quantize
        W = W_cpu.to(device=device, dtype=torch.float32)
        Wq, codebook, qcodes, meta = lloyd_asym_nonuniform_quantize(
            W,
            b=bit,
            group_size=int(args.group_size),
            clip_pct=float(args.clip_percentile),
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
        )

        # save cpu
        codebooks[full_name] = codebook.detach().to(torch.float16).cpu()   # [O,G,L]
        qcodes_dict[full_name] = qcodes.detach().cpu()                     # uint8 [O,G,S]
        metas[full_name] = meta

        if args.save_wq:
            qweights[full_name] = Wq.detach().to(torch.float16).cpu()
        if args.save_err:
            # err는 원래 shape [O,I] 기준으로 저장
            Wq_cpu = Wq.detach().to(torch.float32).cpu()
            err_dict[full_name] = W_cpu.to(torch.float32) - Wq_cpu

        del W, Wq, codebook, qcodes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    cb_path = os.path.join(args.out_dir, "codebook.pt")
    qc_path = os.path.join(args.out_dir, "qcodes.pt")
    mt_path = os.path.join(args.out_dir, "meta.pt")
    torch.save(codebooks, cb_path)
    torch.save(qcodes_dict, qc_path)
    torch.save(metas, mt_path)

    if args.save_wq:
        q_path = os.path.join(args.out_dir, "quantized_weights.pt")
        torch.save(qweights, q_path)
    if args.save_err:
        e_path = os.path.join(args.out_dir, "quant_error.pt")
        torch.save(err_dict, e_path)

    print("[Step1-LLoyd] Saved:")
    print(f"  • {cb_path}  ({len(codebooks)} layers)")
    print(f"  • {qc_path}  ({len(qcodes_dict)} layers)")
    print(f"  • {mt_path}  ({len(metas)} layers)")
    if args.save_wq:
        print(f"  • {q_path}  ({len(qweights)} layers)")
    if args.save_err:
        print(f"  • {e_path}  ({len(err_dict)} layers)")


if __name__ == "__main__":
    main()
