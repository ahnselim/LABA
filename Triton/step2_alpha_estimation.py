#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2 — Layerwise Residual Ratio α Estimation (fixed-rank restore, no grouping)

개요:
 • 레이어별로 비트 b∈{2,3,4}에 대해 양자화 Wq를 만들고,
   E = W - Wq, Ẽ = E · Σ_x^{1/2} (diag-OAS)
   s = svdvals(Ẽ)
   α(b) = (sum_{i>r} s_i^2) / (sum_i s_i^2)
 • 2-bit는 Lloyd-2 대칭({±α,±β}) vs zero-포함({−α,0,+α}) 중
   가중 Frobenius ||(W - Wq) Σ^{1/2}||_F 가 작은 쪽으로 자동 선택.
 • 타깃 모듈: q/k/v/o, gate/up/down, fc1/fc2 (HF LLaMA/Mistral류 호환)

추가 기능 (이 버전):
 • 2-bit 선택 qmode(LLoyd2 vs zero) 전용 CSV 출력 (output_dir 저장)
 • Calibration forward 후 Σ_x^{1/2} 캐시(.pt) 저장/재사용
   (--reuse_calib, --calib_cache_dir)

사용 예:
 CUDA_VISIBLE_DEVICES=0 \
 python step2_alpha_estimation.py \
   --model_id meta-llama/Llama-3.1-8B \
   --bits 2 3 4 --rank 64 --group_size 128 \
   --dataset DKYoon/SlimPajama-6B --split train \
   --nsamples 64 --seqlen 2048 \
   --dtype bf16 --device_map auto \
   --reuse_calib \
   --output_dir ./artifacts_8/bitmin/step2 \
   --calib_cache_dir ./artifacts_8/bitmin

출력:
  output_dir/alpha_layerwise_rank{r}.csv
  output_dir/alpha_2bit_qmode_rank{r}.csv
  output_dir/alpha_2bit_qmode_rank{r}_summary.csv
  calib_cache_dir/calib_oas_sqrtdiag_{model}__{dataset}__{config}__{split}__ns{ns}_L{L}.pt
"""

import os, re, gc, csv, json, argparse
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

# -------------------------------
# HF
# -------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError(
        "transformers가 필요합니다: pip install transformers datasets accelerate"
    ) from e

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

# -------------------------------
# 타깃 Linear 필터 (step3와 동일)
# -------------------------------
TARGET_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj", "fc1", "fc2",
}


def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and "layers" in name
        and name.endswith(".weight")
        and name.split(".")[-2] in TARGET_SUFFIXES
    )


def module_name_from_weight(full_weight_name: str) -> str:
    return full_weight_name[: -len(".weight")]

# -------------------------------
# dtype & seed
# -------------------------------
def pick_dtype(dtype_str: str):
    if dtype_str == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def set_all_seeds(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------
# SlimPajama-friendly dataset loader (streaming 우선)
# -------------------------------
import re as _re


def _canonical_dataset_name(name: str) -> str:
    a = name.strip()
    low = a.lower()
    if low in {"dkyoon/slimpajama-6b", "slimpajama-6b", "dkyoon_slimpajama_6b"}:
        return "DKYoon/SlimPajama-6B"
    if low in {"slimpajama-627b", "cerebras/slimpajama-627b", "slimpajama627b"}:
        return "cerebras/SlimPajama-627B"
    return name


def open_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str = "train",
    streaming: bool = True,
):
    if not HAS_DATASETS:
        raise RuntimeError("datasets 라이브러리가 필요합니다: pip install datasets")
    dataset_name = _canonical_dataset_name(dataset_name)

    if streaming:
        try:
            ds = load_dataset(
                dataset_name, name=dataset_config, split=split, streaming=True
            )
            return ds, dataset_name, dataset_config, True
        except Exception as e:
            msg = str(e)
            if ("available configs" in msg) or ("Config name is missing" in msg):
                m = _re.search(r"\[(.*?)\]", msg, flags=_re.S)
                if m:
                    cands = [
                        c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()
                    ]
                    for cand in cands:
                        try:
                            ds = load_dataset(
                                dataset_name, name=cand, split=split, streaming=True
                            )
                            return ds, dataset_name, cand, True
                        except Exception:
                            pass
    # non-streaming
    try:
        ds = load_dataset(
            dataset_name, name=dataset_config, split=split, streaming=False
        )
        return ds, dataset_name, dataset_config, False
    except Exception as e:
        msg = str(e)
        if ("available configs" in msg) or ("Config name is missing" in msg):
            m = _re.search(r"\[(.*?)\]", msg, flags=_re.S)
            if m:
                cands = [
                    c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()
                ]
                for cand in cands:
                    try:
                        ds = load_dataset(
                            dataset_name, name=cand, split=split, streaming=False
                        )
                        return ds, dataset_name, cand, False
                    except Exception:
                        pass
    raise


def build_calibration_tokens(
    tokenizer,
    nsamples=64,
    seqlen=2048,
    dataset_name="DKYoon/SlimPajama-6B",
    dataset_config=None,
    split="train",
    use_streaming=True,
):
    ds, dataset_name, dataset_config, is_streaming = open_hf_dataset(
        dataset_name, dataset_config, split=split, streaming=use_streaming
    )
    print(
        f"[Step2] Using calibration dataset={dataset_name}, config={dataset_config}, streaming={is_streaming}"
    )

    sample_budget = max(nsamples * 5, nsamples)
    iterator = (
        ds.take(sample_budget)
        if hasattr(ds, "take")
        else ds.select(range(min(sample_budget, len(ds))))
    )

    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_id is None and getattr(tokenizer, "eos_token", None):
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    buf, samples = [], []
    for row in iterator:
        text = None
        for key in ("text", "content", "raw_content"):
            if key in row and isinstance(row[key], str) and row[key].strip():
                text = row[key]
                break
        if text is None:
            for v in row.values():
                if isinstance(v, str) and v.strip():
                    text = v
                    break
        if not text:
            continue

        ids = (
            tokenizer(text, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .tolist()
        )
        if not ids:
            continue
        if eos_id is not None:
            ids.append(eos_id)
        buf.extend(ids)

        while len(buf) >= seqlen and len(samples) < nsamples:
            samples.append(torch.tensor(buf[:seqlen], dtype=torch.long))
            buf = buf[seqlen:]
        if len(samples) >= nsamples:
            break

    if len(samples) >= nsamples:
        pass
    else:
        print(f"[Step2][warn] Collected only {len(samples)}/{nsamples} sequences.")

    return (
        torch.stack(samples, dim=0) if samples else torch.empty(0, seqlen, dtype=torch.long)
    )

# -------------------------------
# Σ_x 대각 OAS (모듈별)
# -------------------------------
@torch.no_grad()
def estimate_diag_cov_oas_per_module(
    model: nn.Module,
    tokenizer,
    device,
    nsamples=64,
    seqlen=2048,
    calib_dataset="DKYoon/SlimPajama-6B",
    calib_config=None,
    split="train",
    use_streaming=True,
    matmul_dtype=torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    model.eval()

    name_to_dim: Dict[str, int] = {}
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    def hook_factory(mod_name: str):
        def hook(module, inp, _out):
            x = (
                inp[0]
                .detach()
                .reshape(-1, inp[0].shape[-1])
                .to(device=device, dtype=matmul_dtype)
            )
            d = x.shape[-1]
            name_to_dim[mod_name] = d
            if mod_name not in stats:
                stats[mod_name] = {
                    "sum": torch.zeros(d, dtype=torch.float64, device="cpu"),
                    "sumsq": torch.zeros(d, dtype=torch.float64, device="cpu"),
                    "n": torch.tensor(0, dtype=torch.long),
                }
            x_cpu = x.to("cpu", dtype=torch.float32)
            stats[mod_name]["sum"] += x_cpu.sum(dim=0, dtype=torch.float64)
            stats[mod_name]["sumsq"] += (x_cpu.to(torch.float64).pow(2)).sum(dim=0)
            stats[mod_name]["n"] += x_cpu.shape[0]

        return hook

    # 모든 Linear에 hook (필터는 계산 단계에서)
    for mn, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(hook_factory(mn)))

    tokens = build_calibration_tokens(
        tokenizer, nsamples, seqlen, calib_dataset, calib_config, split, use_streaming
    )
    if tokens.numel() == 0:
        for h in handles:
            h.remove()
        raise RuntimeError("Calibration tokens unavailable.")

    with torch.no_grad():
        for i in tqdm(range(tokens.shape[0]), desc="Calibration Forward"):
            model(tokens[i: i + 1].to(device))

    for h in handles:
        h.remove()

    ops: Dict[str, Dict[str, torch.Tensor]] = {}
    for mn, st in stats.items():
        n = int(st["n"].item())
        if n <= 1:
            continue
        d = name_to_dim[mn]
        sumv, sumsq = st["sum"], st["sumsq"]
        mean = sumv / n
        ex2 = sumsq / n
        var = torch.clamp(ex2 - mean.pow(2), min=1e-12)

        p = float(d)
        trS = var.sum().item()
        trS2 = (var.pow(2)).sum().item()
        num = (1.0 - 2.0 / p) * trS2 + (trS * trS)
        den = (n + 1.0 - 2.0 / p) * (trS2 - (trS * trS) / p)
        alpha = 1.0 if den <= 0 else max(0.0, min(1.0, num / den))
        mu = trS / p
        sigma_diag = (1.0 - alpha) * var + alpha * mu
        sqrt_diag = torch.sqrt(torch.clamp(sigma_diag, min=1e-12)).to(torch.float32)
        ops[mn] = {"sqrt": sqrt_diag.cpu()}

    print(f"[Step2] Σ_x^{1/2} prepared for {len(ops)} linear modules.")
    return ops

# -------------------------------
# Group reshape helpers
# -------------------------------
def _to_groups(W: torch.Tensor, group_size: int):
    O, I = W.shape
    pad = (group_size - (I % group_size)) % group_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    O_, I_pad = W.shape
    G = I_pad // group_size
    return W.view(O_, G, group_size), O_, G, group_size, I


def _from_groups(Xg: torch.Tensor, orig_I: int) -> torch.Tensor:
    O_, G, S = Xg.shape
    return Xg.reshape(O_, G * S)[:, :orig_I]


@torch.no_grad()
def _percentile_clip_lastdim(
    X: torch.Tensor,
    lo_pct: float,
    hi_pct: float
) -> torch.Tensor:
    lo = torch.quantile(X, lo_pct / 100.0, dim=-1, keepdim=True)
    hi = torch.quantile(X, hi_pct / 100.0, dim=-1, keepdim=True)
    return torch.clamp(X, lo, hi)

# -------------------------------
# Quantization (step3와 동일 로직)
# -------------------------------
@torch.no_grad()
def dequant_2bit_lloyd2_sym(
    W: torch.Tensor,
    group_size: int,
    clip_pct: float = 99.9,
    steps: int = 4
):
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
    Y = X.abs()

    eps = 1e-12
    alpha = torch.quantile(Y, 0.5, dim=-1, keepdim=True).clamp_min(eps)
    beta = torch.quantile(Y, 0.9, dim=-1, keepdim=True)
    beta = torch.maximum(beta, alpha + 1e-6)

    for _ in range(max(1, steps)):
        t = 0.5 * (alpha + beta)
        mask_lo = Y < t
        cnt_lo = mask_lo.sum(dim=-1, keepdim=True).clamp_min(1)
        cnt_hi = (~mask_lo).sum(dim=-1, keepdim=True).clamp_min(1)
        new_alpha = (Y * mask_lo).sum(dim=-1, keepdim=True) / cnt_lo
        new_beta = (Y * (~mask_lo)).sum(dim=-1, keepdim=True) / cnt_hi
        alpha = torch.where(new_alpha > eps, new_alpha, alpha)
        beta = torch.where(new_beta > alpha + 1e-6, new_beta, beta)

    t = 0.5 * (alpha + beta)
    mag = torch.where(Y < t, alpha, beta)
    sgn = torch.sign(X)
    sgn[sgn == 0] = 1.0
    deq = sgn * mag
    return _from_groups(deq, orig_I).to(dtype=W.dtype, device=W.device)


@torch.no_grad()
def dequant_2bit_qtr_zero(
    W: torch.Tensor,
    group_size: int,
    clip_pct: float = 99.9,
    steps: int = 6
):
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
    Y = X.abs()

    eps = 1e-12
    alpha = Y.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    for _ in range(max(1, steps)):
        t = 0.5 * alpha
        mask = Y >= t
        cnt = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        new_alpha = (Y * mask).sum(dim=-1, keepdim=True) / cnt
        alpha = torch.where(new_alpha > eps, new_alpha, alpha)

    mask = (Y >= 0.5 * alpha).float()
    mag = alpha * mask
    sgn = torch.sign(X)
    sgn[sgn == 0] = 1.0
    deq = sgn * mag
    return _from_groups(deq, orig_I).to(dtype=W.dtype, device=W.device)


@torch.no_grad()
def dequant_uniform_asym(
    W: torch.Tensor,
    b: int,
    group_size: int,
    clip_pct: float = 0.0,
    rounding: str = "nearest",
):
    assert b in (1, 2, 3, 4)
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _percentile_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg

    qmax = float(2**b - 1)
    eps = 1e-12
    minv = X.amin(dim=-1)
    maxv = X.amax(dim=-1)
    span = (maxv - minv).clamp_min(eps)

    scale = span / qmax
    zp = (-minv / scale).round()
    qf = X / scale.unsqueeze(-1) + zp.unsqueeze(-1)

    if rounding == "stochastic":
        frac = qf - torch.floor(qf)
        q = torch.floor(qf) + (torch.rand_like(frac) < frac).float()
    else:
        q = torch.round(qf)

    q = torch.clamp(q, 0.0, qmax)
    deq = (q - zp.unsqueeze(-1)) * scale.unsqueeze(-1)

    flat = span <= eps
    if flat.any():
        deq[flat] = minv[flat].unsqueeze(-1)

    return _from_groups(deq, orig_I).to(dtype=W.dtype, device=W.device)

# -------------------------------
# α(b) 측정 루틴
# -------------------------------
@torch.no_grad()
def measure_alpha_for_layer(
    W: torch.Tensor,
    sqrt_diag: torch.Tensor,   # [n]
    bits: List[int],
    rank_r: int,
    group_size: int,
    clip_pct: float,
    lloyd_steps: int,
    qtr_steps: int,
    uniform_rounding: str,
) -> List[Tuple[int, str, float, float, float]]:
    """
    Returns: list of (bit, qmode, Lq, Lres, alpha)
     • Lq   = || (W - Wq) Σ^{1/2} ||_F^2
     • Lres = tail_energy_after_rank_r(Ẽ) = sum_{i>r} s_i^2
     • alpha = Lres / Lq (0..1, Lq=0이면 0)
    """
    device = W.device
    sqrt_diag = sqrt_diag.to(device=device, dtype=torch.float32)
    out: List[Tuple[int, str, float, float, float]] = []

    for b in bits:
        # 1) 양자화 (2bit는 자동 선택)
        if b == 2:
            Wq_l = dequant_2bit_lloyd2_sym(W, group_size, clip_pct, lloyd_steps)
            Wq_z = dequant_2bit_qtr_zero(W, group_size, clip_pct, qtr_steps)

            El = (W - Wq_l) * sqrt_diag.unsqueeze(0)
            Ez = (W - Wq_z) * sqrt_diag.unsqueeze(0)
            score_l = torch.linalg.norm(El).item()
            score_z = torch.linalg.norm(Ez).item()

            if score_z <= score_l:
                Wq, qmode = Wq_z, "2bit_zero"
                E_tilde = Ez
            else:
                Wq, qmode = Wq_l, "2bit_lloyd2"
                E_tilde = El

        elif b in (3, 4):
            Wq = dequant_uniform_asym(
                W, b=b, group_size=group_size, clip_pct=0.0, rounding=uniform_rounding
            )
            E_tilde = (W - Wq) * sqrt_diag.unsqueeze(0)
            qmode = f"{b}bit_uniform"
        else:
            # 미지원 비트는 스킵
            continue

        # 2) SVD값으로 Lq, Lres, alpha
        try:
            s = torch.linalg.svdvals(E_tilde)  # [min(m,n)]
        except RuntimeError:
            s = torch.linalg.svdvals(E_tilde.to("cpu")).to(device)

        s2 = s * s
        Lq = float(s2.sum().item())
        if Lq <= 0:
            out.append((b, qmode, 0.0, 0.0, 0.0))
            continue

        r_eff = min(rank_r, s.numel())
        tail = 0.0 if r_eff >= s.numel() else float(s2[r_eff:].sum().item())
        alpha = max(0.0, min(1.0, tail / Lq))
        out.append((b, qmode, Lq, tail, alpha))

        # 메모리 정리
        del Wq, E_tilde, s
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return out

# -------------------------------
# 유틸: 파일 이름 정리
# -------------------------------
def _safe_name(s: str) -> str:
    s = str(s) if s is not None else "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

# -------------------------------
# 메인
# -------------------------------
def main():
    ap = argparse.ArgumentParser("Step 2 — Layerwise α(b) Estimation (fixed-rank)")

    # 모델/장치
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
    )
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--seed", type=int, default=42)

    # 캘리브레이션 데이터(Σ_x)
    ap.add_argument("--dataset", type=str, default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in ["1", "true", "yes"],
        default=True,
    )
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)

    # Calibration 캐시
    ap.add_argument(
        "--reuse_calib",
        action="store_true",
        help="기존 Σ_x^{1/2} 캐시(.pt) 재사용",
    )
    ap.add_argument("--calib_cache_dir", type=str, default="./artifacts/bitmin")

    # α 측정 설정
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--group_size", type=int, default=128)

    # 2/3/4비트 세부 옵션
    ap.add_argument(
        "--clip_percentile",
        type=float,
        default=99.9,
        help="2bit 권장 99.9; 0이면 off",
    )
    ap.add_argument("--lloyd_steps", type=int, default=4)
    ap.add_argument("--qtr_steps", type=int, default=6)
    ap.add_argument(
        "--uniform_rounding",
        type=str,
        default="nearest",
        choices=["nearest", "stochastic"],
    )

    # 출력
    ap.add_argument("--output_dir", type=str, default="./artifacts/bitmin/step2")

    args = ap.parse_args()
    set_all_seeds(args.seed)
    dtype = pick_dtype(args.dtype)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.calib_cache_dir, exist_ok=True)

    # 모델/토크나이저 로드
    print(f"[Step2] Loading model: {args.model_id} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=dtype if (dtype in (torch.float16, torch.bfloat16)) else None,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code
    )

    # Σ_x^{1/2} (diag OAS) — 캐시 로드/생성
    device0 = next(model.parameters()).device
    safe_model = _safe_name(
        args.model_id if args.revision is None else f"{args.model_id}@{args.revision}"
    )
    safe_dataset = _safe_name(_canonical_dataset_name(args.dataset))
    safe_config = _safe_name(args.dataset_config)
    calib_basename = (
        f"calib_oas_sqrtdiag_{safe_model}__{safe_dataset}__{safe_config}"
        f"__{args.split}__ns{args.nsamples}_L{args.seqlen}.pt"
    )
    calib_path = os.path.join(args.calib_cache_dir, calib_basename)

    if args.reuse_calib and os.path.exists(calib_path):
        print(f"[Step2] Loading cached Σ_x^{1/2} from: {calib_path}")
        payload = torch.load(calib_path, map_location="cpu")
        cov_ops = payload.get("cov_ops", payload)
    else:
        cov_ops = estimate_diag_cov_oas_per_module(
            model,
            tokenizer,
            device0,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            calib_dataset=args.dataset,
            calib_config=args.dataset_config,
            split=args.split,
            use_streaming=bool(args.use_streaming),
            matmul_dtype=torch.float32,
        )
        meta = {
            "model_id": args.model_id,
            "revision": args.revision,
            "dataset": _canonical_dataset_name(args.dataset),
            "dataset_config": args.dataset_config,
            "split": args.split,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
        }
        torch.save({"cov_ops": cov_ops, "meta": meta}, calib_path)
        print(f"[Step2] Saved Σ_x^{1/2} cache: {calib_path}")

    # state_dict를 CPU로 복사 (GPU 메모리 절약)
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    rows_2bit = []  # 2bit qmode 전용 CSV용

    print("[Step2] Measuring α(b) per target linear weight...")
    for full_name, W_cpu in tqdm(state.items()):
        if not is_target_weight(full_name, W_cpu):
            continue
        mod_name = module_name_from_weight(full_name)
        if mod_name not in cov_ops:
            # Σ_x 정보가 없는 모듈은 스킵
            continue

        W = W_cpu.to(
            device0 if torch.cuda.is_available() else "cpu", dtype=torch.float32
        )
        m, n = W.shape
        sqrt_diag = cov_ops[mod_name]["sqrt"]

        results = measure_alpha_for_layer(
            W=W,
            sqrt_diag=sqrt_diag,
            bits=args.bits,
            rank_r=args.rank,
            group_size=args.group_size,
            clip_pct=args.clip_percentile,
            lloyd_steps=args.lloyd_steps,
            qtr_steps=args.qtr_steps,
            uniform_rounding=args.uniform_rounding,
        )

        for b, qmode, Lq, Lres, alpha in results:
            row = {
                "full_name": full_name,
                "module": mod_name,
                "m": m,
                "n": n,
                "bit": b,
                "qmode": qmode,
                "rank": args.rank,
                "group_size": args.group_size,
                "Lq_weighted": f"{Lq:.6e}",
                "Lres_weighted": f"{Lres:.6e}",
                "alpha": f"{alpha:.8f}",
            }
            rows.append(row)
            if b == 2:
                rows_2bit.append({
                    k: row[k] for k in [
                        "full_name", "module", "bit", "qmode",
                        "Lq_weighted", "Lres_weighted", "alpha"
                    ]
                })

        # free
        del W
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # CSV 저장 — 메인
    csv_path = os.path.join(args.output_dir, f"alpha_layerwise_rank{args.rank}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "full_name", "module", "m", "n", "bit", "qmode",
                "rank", "group_size", "Lq_weighted", "Lres_weighted", "alpha",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[Step2] Saved CSV: {csv_path}")

    # CSV 저장 — 2bit qmode 전용
    if rows_2bit:
        qmode_csv_path = os.path.join(
            args.output_dir, f"alpha_2bit_qmode_rank{args.rank}.csv"
        )
        with open(qmode_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "full_name", "module", "bit", "qmode",
                    "Lq_weighted", "Lres_weighted", "alpha"
                ],
            )
            writer.writeheader()
            for r in rows_2bit:
                writer.writerow(r)
        print(f"[Step2] Saved 2-bit qmode CSV: {qmode_csv_path}")

        # 간단 summary (선택)
        counts = Counter([r["qmode"] for r in rows_2bit])
        qmode_summary_path = os.path.join(
            args.output_dir, f"alpha_2bit_qmode_rank{args.rank}_summary.csv"
        )
        with open(qmode_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["qmode", "count"])
            writer.writeheader()
            for qmode, c in sorted(counts.items()):
                writer.writerow({"qmode": qmode, "count": c})
        print(f"[Step2] Saved 2-bit qmode summary: {qmode_summary_path}")

    # 상위/하위 α 레이어 간단 프린트
    try:
        top = sorted(rows, key=lambda r: float(r["alpha"]))[:8]
        bot = sorted(rows, key=lambda r: float(r["alpha"]), reverse=True)[:8]
        print("\n[Step2] Lowest α (best-restorable) examples:")
        for r in top:
            print(f" {r['full_name']:<70s} bit={r['bit']} α={r['alpha']}")
        print("\n[Step2] Highest α (hard-to-restore) examples:")
        for r in bot:
            print(f" {r['full_name']:<70s} bit={r['bit']} α={r['alpha']}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
