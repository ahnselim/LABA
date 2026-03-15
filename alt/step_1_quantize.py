#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixture step0 module: Step0-1 quantization (Lloyd-Max asym non-uniform baseline).

역할:
  - 타깃 weight를 비균일 코드북/코드 형태로 양자화
  - 필요 시 Step0-2 calib(`calib_sqrtdiag.pt`)를 읽어
    diag Hessian 근사 기반 Hessian-weighted Lloyd-Max 수행
  - step3 최적화 입력용 `codebook.pt`, `qcodes.pt`, `meta.pt` 생성
  - 필요 시 `quantized_weights.pt`, `quant_error.pt` 추가 저장

사용 방식:
  - CLI 실행: 이 파일의 `main()`
  - 모듈 사용: 하단 `Step01QuantizeConfig` + `run()`

참고:
  - `LABA/mixture/step0_optimization.py`와의 호환을 위해 래퍼 API를 함께 제공한다.

CUDA_VISIBLE_DEVICES=1 nohup \
python step_1_quantize.py \
  --model_id meta-llama/Llama-3.1-8B \
  --bits 2 \
  --group_size 128 \
  --calib_s_path ./output/llama3_8b_128/calib_sqrtdiag.pt \
  --out_dir ./output/llama3_8b_128/step1_quant/2bit > ./logs/quant_2bit.log 2>&1 &
  
"""

import os
import re
import gc
import csv
import argparse
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
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


def _snapshot_state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        # With accelerate offload, meta tensors may appear in state_dict.
        if getattr(v, "is_meta", False):
            raise NotImplementedError(f"meta tensor in state_dict: {k}")
        state[k] = v.detach().to("cpu")
    return state


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
def _lloyd_centroid_update(
    x: torch.Tensor,
    idx: torch.Tensor,
    levels: int,
    prev_cb: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    new_cb = prev_cb.clone()
    numer = torch.zeros_like(new_cb)
    numer.scatter_add_(1, idx, x)
    count = torch.zeros_like(new_cb)
    count.scatter_add_(1, idx, torch.ones_like(x))
    assigned = count > 0

    if sample_weight is None:
        mean = numer / count.clamp_min(1.0)
        return torch.where(assigned, mean, new_cb)

    weighted_numer = torch.zeros_like(new_cb)
    weighted_numer.scatter_add_(1, idx, x * sample_weight)
    weighted_denom = torch.zeros_like(new_cb)
    weighted_denom.scatter_add_(1, idx, sample_weight)

    weighted_valid = weighted_denom > 0
    if weighted_valid.any():
        weighted_mean = weighted_numer / weighted_denom.clamp_min(1e-12)
        new_cb = torch.where(weighted_valid, weighted_mean, new_cb)

    fallback = assigned & (~weighted_valid)
    if fallback.any():
        plain_mean = numer / count.clamp_min(1.0)
        new_cb = torch.where(fallback, plain_mean, new_cb)
    return new_cb


@torch.no_grad()
def _lloyd_max_codebook_per_group(
    X_flat: torch.Tensor,         # [N,S], N=O*G
    levels: int,                  # L = 2^b
    max_iter: int = 12,
    tol: float = 1e-4,
    chunk_groups: int = 4096,
    sample_weight: Optional[torch.Tensor] = None,
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
    if sample_weight is not None:
        if sample_weight.shape != X_flat.shape:
            raise ValueError(
                f"sample_weight must match X_flat shape {tuple(X_flat.shape)}, "
                f"got {tuple(sample_weight.shape)}"
            )
        sample_weight = sample_weight.to(device=X_flat.device, dtype=X_flat.dtype).clamp_min(0.0)

    for start in range(0, N, cg):
        end = min(start + cg, N)
        x = X_flat[start:end]  # [n,S]
        w = sample_weight[start:end] if sample_weight is not None else None

        if levels == 1:
            if w is None:
                cb = x.mean(dim=1, keepdim=True)
            else:
                denom = w.sum(dim=1, keepdim=True)
                numer = (x * w).sum(dim=1, keepdim=True)
                cb = torch.where(
                    denom > 0,
                    numer / denom.clamp_min(1e-12),
                    x.mean(dim=1, keepdim=True),
                )
            codebook[start:end] = cb
            continue

        probs = (torch.arange(levels, device=x.device, dtype=torch.float32) + 0.5) / float(levels)
        cb = _kth_quantiles_lastdim(x, probs)          # init via quantiles
        cb, _ = torch.sort(cb, dim=1)

        for _ in range(max(1, int(max_iter))):
            mid = (cb[:, :-1] + cb[:, 1:]) * 0.5       # [n,L-1]
            idx = torch.sum(x.unsqueeze(-1) > mid.unsqueeze(1), dim=-1).to(torch.long)  # [n,S]
            new_cb = _lloyd_centroid_update(x, idx, levels, cb, sample_weight=w)
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
    return torch.gather(codebook, dim=1, index=codes)


@torch.no_grad()
def lloyd_asym_nonuniform_quantize(
    W: torch.Tensor,                  # [O,I] float32 on device
    b: int,
    group_size: int,
    clip_pct: float = 0.0,
    lloyd_iter: int = 12,
    chunk_groups: int = 4096,
    hessian_diag: Optional[torch.Tensor] = None,
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
    H_flat = None
    if hessian_diag is not None:
        hdiag = hessian_diag.detach().to(device=W.device, dtype=torch.float32).flatten()
        if hdiag.numel() != orig_I:
            raise ValueError(
                f"hessian_diag size mismatch for weight {tuple(W.shape)}: "
                f"expected {orig_I}, got {hdiag.numel()}"
            )
        if pad:
            hdiag = F.pad(hdiag, (0, pad), value=0.0)
        H_flat = hdiag.view(1, G, S).expand(O, -1, -1).reshape(-1, S).contiguous()

    cb_flat = _lloyd_max_codebook_per_group(
        X_flat,
        levels=L,
        max_iter=lloyd_iter,
        tol=1e-4,
        chunk_groups=chunk_groups,
        sample_weight=H_flat,
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
        "uses_hessian_weighting": bool(H_flat is not None),
    }
    return Wq, codebook, qcodes, meta


def _load_calib_hessian_map(calib_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    payload = torch.load(calib_path, map_location="cpu")
    calib_map = payload.get("cov_ops", payload)
    if not isinstance(calib_map, dict):
        raise TypeError(f"Unsupported calib payload type: {type(calib_map)!r}")

    normalized: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, entry in calib_map.items():
        if not isinstance(entry, dict):
            continue

        hdiag = None
        if "var" in entry:
            hdiag = entry["var"]
        elif "s" in entry:
            s = entry["s"].to(torch.float32)
            hdiag = s * s
        elif "sqrt" in entry:
            s = entry["sqrt"].to(torch.float32)
            hdiag = s * s
        elif "inv_s" in entry:
            inv_s = entry["inv_s"].to(torch.float32).clamp_min(1e-12)
            hdiag = torch.reciprocal(inv_s * inv_s)

        if hdiag is None:
            continue

        normalized[key] = {
            "hessian_diag": hdiag.detach().to(torch.float32).cpu().clamp_min(0.0).contiguous()
        }

    if not normalized:
        raise KeyError(f"No usable Hessian statistics found in calib file: {calib_path}")
    return normalized


def _lookup_hessian_diag(
    calib_hessian: Dict[str, Dict[str, torch.Tensor]],
    full_weight_name: str,
) -> Optional[torch.Tensor]:
    entry = calib_hessian.get(full_weight_name)
    if entry is None:
        entry = calib_hessian.get(module_name_from_weight(full_weight_name))
    if entry is None:
        return None
    return entry.get("hessian_diag")


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
    ap.add_argument("--device_map", default="auto", help='Model load placement: e.g. "auto" or "none"')

    ap.add_argument("--bit_assign_csv", default=None)
    ap.add_argument("--bits", type=int, default=4, choices=[1,2,3,4],
                    help="global bit if csv not provided")
    ap.add_argument("--group_size", type=int, default=128)

    ap.add_argument("--clip_percentile", type=float, default=0.0)
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)
    ap.add_argument(
        "--calib_s_path",
        type=str,
        default=None,
        help="Optional Step0-2 calib_sqrtdiag.pt. If provided, use diag Hessian weighting for Lloyd-Max.",
    )

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

    dm_raw = str(args.device_map).strip().lower()
    resolved_device_map = None if dm_raw in {"", "none", "null"} else args.device_map

    print(
        f"[Step1-LLoyd] Loading model: {args.model_id} "
        f"(load_dtype={load_dtype}, device={device}, device_map={resolved_device_map})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=args.trust_remote_code,
        device_map=resolved_device_map,
        low_cpu_mem_usage=True,
    )
    if resolved_device_map is None:
        model = model.to(device)

    if args.bit_assign_csv:
        sel_bits = load_selected_bits(args.bit_assign_csv)
        print(f"[Step1-LLoyd] Loaded bit assignments: {len(sel_bits)} entries.")
    else:
        sel_bits = None

    calib_hessian = None
    if args.calib_s_path:
        print(f"[Step1-LLoyd] Loading calib Hessian diag: {args.calib_s_path}")
        calib_hessian = _load_calib_hessian_map(args.calib_s_path)
        print(f"[Step1-LLoyd] Loaded Hessian diag entries: {len(calib_hessian)}")

    # CPU state dict로 옮겨서 GPU 메모리 절약
    try:
        state = _snapshot_state_to_cpu(model)
        del model
    except NotImplementedError:
        if resolved_device_map is None:
            raise
        print("[Step1-LLoyd] Detected meta tensors under device_map mode. Re-loading on CPU to build state snapshot.")
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
        hessian_diag = None
        if calib_hessian is not None:
            hessian_diag = _lookup_hessian_diag(calib_hessian, full_name)
            if hessian_diag is None:
                print(f"[Step1-LLoyd][warn] Missing calib Hessian diag for {full_name}; using unweighted Lloyd-Max.")
        Wq, codebook, qcodes, meta = lloyd_asym_nonuniform_quantize(
            W,
            b=bit,
            group_size=int(args.group_size),
            clip_pct=float(args.clip_percentile),
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
            hessian_diag=hessian_diag,
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


# ---------------------------------------------------------------------------
# Compatibility wrapper API for LABA/mixture/step0_optimization.py
# (No embedded source / exec; directly invokes local `main()`.)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Sequence


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
class Step01QuantizeConfig:
    model_id: str
    out_dir: str
    bits: int
    group_size: int
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    device: str = "cuda"
    device_map: str = "auto"
    bit_assign_csv: Optional[str] = None
    clip_percentile: float = 0.0
    lloyd_iter: int = 12
    chunk_groups: int = 4096
    calib_s_path: Optional[str] = None
    layer_regex: Optional[str] = None
    save_wq: bool = False
    save_err: bool = False
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


def expected_outputs(out_dir: str) -> dict:
    p = Path(out_dir)
    return {
        "codebook": p / "codebook.pt",
        "qcodes": p / "qcodes.pt",
        "meta": p / "meta.pt",
        "quantized_weights": p / "quantized_weights.pt",
        "quant_error": p / "quant_error.pt",
    }


def build_command(cfg: Step01QuantizeConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_id",
        str(cfg.model_id),
        "--bits",
        str(int(cfg.bits)),
        "--group_size",
        str(int(cfg.group_size)),
        "--dtype",
        str(cfg.dtype),
        "--device",
        str(cfg.device),
        "--device_map",
        str(cfg.device_map),
        "--clip_percentile",
        str(float(cfg.clip_percentile)),
        "--lloyd_iter",
        str(int(cfg.lloyd_iter)),
        "--chunk_groups",
        str(int(cfg.chunk_groups)),
        "--out_dir",
        str(cfg.out_dir),
    ]
    if cfg.revision:
        cmd += ["--revision", str(cfg.revision)]
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    if cfg.bit_assign_csv:
        cmd += ["--bit_assign_csv", str(cfg.bit_assign_csv)]
    if cfg.calib_s_path:
        cmd += ["--calib_s_path", str(cfg.calib_s_path)]
    if cfg.layer_regex:
        cmd += ["--layer_regex", str(cfg.layer_regex)]
    if cfg.save_wq:
        cmd.append("--save_wq")
    if cfg.save_err:
        cmd.append("--save_err")
    return cmd


def run(cfg: Step01QuantizeConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp