#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 2 — Layerwise Residual Ratio α Estimation (fixed-rank restore, no grouping)
(Optimized: keep math/logic identical; reduce CPU<->GPU sync and useless work)

원본 대비 최적화:
- model.state_dict() 전체 CPU 복사 제거 (모델 유지 + 타깃 weight만 직접 접근)
- hook: 모든 Linear가 아니라 타깃 Linear만 등록
- hook에서 x를 매번 CPU로 보내지 않고 GPU에서 sum/sumsq/n 누적 후 마지막에만 CPU로 이동
- calibration tokens를 device로 한 번만 올려서 forward (HtoD 반복 제거)
- empty_cache 남발 제거 (옵션으로만)

※ α 정의/계산식, OAS diag 계산식, 2-bit qmode 선택 로직, SVD 기반 Lq/tail/alpha는 그대로 유지.

python step2_alpha_estimation_optimized.py \
  --model_id meta-llama/Llama-3.2-3B \
  --dataset DKYoon/SlimPajama-6B \
  --nsamples 64 --seqlen 2048 \
  --rank 64 --group_size 128 \
  --output_dir ./artifacts/bitmin/step2 \
  --calib_cache_dir ./artifacts/bitmin

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
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
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
            ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
            return ds, dataset_name, dataset_config, True
        except Exception as e:
            msg = str(e)
            if ("available configs" in msg) or ("Config name is missing" in msg):
                m = _re.search(r"\[(.*?)\]", msg, flags=_re.S)
                if m:
                    cands = [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
                    for cand in cands:
                        try:
                            ds = load_dataset(dataset_name, name=cand, split=split, streaming=True)
                            return ds, dataset_name, cand, True
                        except Exception:
                            pass

    # non-streaming
    try:
        ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=False)
        return ds, dataset_name, dataset_config, False
    except Exception as e:
        msg = str(e)
        if ("available configs" in msg) or ("Config name is missing" in msg):
            m = _re.search(r"\[(.*?)\]", msg, flags=_re.S)
            if m:
                cands = [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
                for cand in cands:
                    try:
                        ds = load_dataset(dataset_name, name=cand, split=split, streaming=False)
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
    print(f"[Step2] Using calibration dataset={dataset_name}, config={dataset_config}, streaming={is_streaming}")

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

        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
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

    if len(samples) < nsamples:
        print(f"[Step2][warn] Collected only {len(samples)}/{nsamples} sequences.")

    return torch.stack(samples, dim=0) if samples else torch.empty(0, seqlen, dtype=torch.long)

# -------------------------------
# Σ_x 대각 OAS (모듈별) - 최적화: 타깃 모듈만 훅 + GPU 누적
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
    calib_batch_size: int = 1,
) -> Dict[str, Dict[str, torch.Tensor]]:
    model.eval()

    # (1) 타깃 모듈 이름 집합을 먼저 구성: named_parameters 기준(원본 is_target_weight와 동일 조건)
    target_modules = set()
    for pname, p in model.named_parameters():
        if is_target_weight(pname, p.detach()):
            target_modules.add(module_name_from_weight(pname))
    if not target_modules:
        print("[Step2][warn] No target modules found by TARGET_SUFFIXES filter.")
        return {}

    name_to_dim: Dict[str, int] = {}
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    # (2) 훅은 타깃 Linear에만 등록
    def hook_factory(mod_name: str):
        def hook(_module, inp, _out):
            x0 = inp[0]
            # x: [B*T, d] (동일)
            x = x0.detach().reshape(-1, x0.shape[-1]).to(dtype=matmul_dtype)
            d = x.shape[-1]
            name_to_dim[mod_name] = d
            if mod_name not in stats:
                # 원본은 CPU float64 누적이었는데, 동기화 제거를 위해 GPU에서 누적 후 마지막에만 CPU로 이동
                stats[mod_name] = {
                    "sum": torch.zeros(d, dtype=torch.float64, device=device),
                    "sumsq": torch.zeros(d, dtype=torch.float64, device=device),
                    "n": torch.zeros((), dtype=torch.long, device=device),
                }
            # sum, sumsq, n 누적 (연산식 동일)
            stats[mod_name]["sum"] += x.sum(dim=0, dtype=torch.float64)
            stats[mod_name]["sumsq"] += (x.to(torch.float64).pow(2)).sum(dim=0)
            stats[mod_name]["n"] += x.shape[0]
        return hook

    # module lookup
    for mn, mod in model.named_modules():
        if mn in target_modules and isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(hook_factory(mn)))

    if not handles:
        print("[Step2][warn] No hooks registered (target modules not found as nn.Linear).")
        return {}

    tokens = build_calibration_tokens(
        tokenizer, nsamples, seqlen, calib_dataset, calib_config, split, use_streaming
    )
    if tokens.numel() == 0:
        for h in handles:
            h.remove()
        raise RuntimeError("Calibration tokens unavailable.")

    # (3) tokens를 한 번에 device로 올려 HtoD 반복 제거
    tokens = tokens.to(device=device, non_blocking=True)

    with torch.no_grad():
        # 원본은 sample-by-sample. 배치도 eval에서 동일하게 동작하지만, 기본은 1 유지(원본 동일성 최대)
        bs = max(1, int(calib_batch_size))
        for i in tqdm(range(0, tokens.shape[0], bs), desc="Calibration Forward"):
            model(tokens[i : i + bs])

    for h in handles:
        h.remove()

    # (4) OAS 계산은 원본과 동일하게 CPU에서 수행(마지막에만 이동)
    ops: Dict[str, Dict[str, torch.Tensor]] = {}
    for mn, st in stats.items():
        n = int(st["n"].detach().cpu().item())
        if n <= 1:
            continue
        d = int(name_to_dim[mn])

        sumv = st["sum"].detach().cpu()
        sumsq = st["sumsq"].detach().cpu()

        mean = sumv / n
        ex2 = sumsq / n
        var = torch.clamp(ex2 - mean.pow(2), min=1e-12)

        # OAS (diag) — 원본 식 그대로
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

    print(f"[Step2] Σ_x^{1/2} prepared for {len(ops)} target linear modules.")
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
def _percentile_clip_lastdim(X: torch.Tensor, lo_pct: float, hi_pct: float) -> torch.Tensor:
    lo = torch.quantile(X, lo_pct / 100.0, dim=-1, keepdim=True)
    hi = torch.quantile(X, hi_pct / 100.0, dim=-1, keepdim=True)
    return torch.clamp(X, lo, hi)

# -------------------------------
# Quantization (step3와 동일 로직) - 그대로
# -------------------------------
@torch.no_grad()
def dequant_2bit_lloyd2_sym(W: torch.Tensor, group_size: int, clip_pct: float = 99.9, steps: int = 4):
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
def dequant_2bit_qtr_zero(W: torch.Tensor, group_size: int, clip_pct: float = 99.9, steps: int = 6):
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
# α(b) 측정 루틴 (연산 동일, empty_cache 최소화)
# -------------------------------
@torch.no_grad()
def measure_alpha_for_layer(
    W: torch.Tensor,
    sqrt_diag: torch.Tensor,  # [n]
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
        if b == 2:
            Wq_l = dequant_2bit_lloyd2_sym(W, group_size, clip_pct, lloyd_steps)
            Wq_z = dequant_2bit_qtr_zero(W, group_size, clip_pct, qtr_steps)

            El = (W - Wq_l) * sqrt_diag.unsqueeze(0)
            Ez = (W - Wq_z) * sqrt_diag.unsqueeze(0)
            score_l = torch.linalg.norm(El).item()
            score_z = torch.linalg.norm(Ez).item()

            if score_z <= score_l:
                qmode = "2bit_zero"
                E_tilde = Ez
                del Wq_l, El
            else:
                qmode = "2bit_lloyd2"
                E_tilde = El
                del Wq_z, Ez

        elif b in (3, 4):
            Wq = dequant_uniform_asym(
                W, b=b, group_size=group_size, clip_pct=0.0, rounding=uniform_rounding
            )
            E_tilde = (W - Wq) * sqrt_diag.unsqueeze(0)
            qmode = f"{b}bit_uniform"
            del Wq
        else:
            continue

        try:
            s = torch.linalg.svdvals(E_tilde)
        except RuntimeError:
            s = torch.linalg.svdvals(E_tilde.to("cpu")).to(device)

        s2 = s * s
        Lq = float(s2.sum().item())
        if Lq <= 0:
            out.append((b, qmode, 0.0, 0.0, 0.0))
            del E_tilde, s, s2
            continue

        r_eff = min(rank_r, s.numel())
        tail = 0.0 if r_eff >= s.numel() else float(s2[r_eff:].sum().item())
        alpha = max(0.0, min(1.0, tail / Lq))
        out.append((b, qmode, Lq, tail, alpha))

        del E_tilde, s, s2

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
    ap = argparse.ArgumentParser("Step 2 — Layerwise α(b) Estimation (fixed-rank) [Optimized]")

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
    ap.add_argument("--reuse_calib", action="store_true",
                    help="기존 Σ_x^{1/2} 캐시(.pt) 재사용")
    ap.add_argument("--calib_cache_dir", type=str, default="./artifacts/bitmin")

    # α 측정 설정
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--group_size", type=int, default=128)

    # 2/3/4비트 세부 옵션
    ap.add_argument("--clip_percentile", type=float, default=99.9,
                    help="2bit 권장 99.9; 0이면 off")
    ap.add_argument("--lloyd_steps", type=int, default=4)
    ap.add_argument("--qtr_steps", type=int, default=6)
    ap.add_argument("--uniform_rounding", type=str, default="nearest",
                    choices=["nearest", "stochastic"])

    # 성능 옵션 (연산은 동일, 캐시/정리만)
    ap.add_argument("--calib_batch_size", type=int, default=1,
                    help="Calibration forward batch size (eval이므로 값만 동일). 기본 1=원본 동일성 최대.")
    ap.add_argument("--keep_calib_on_device", action="store_true",
                    help="sqrt_diag를 최초 접근 시 GPU에 캐시하여 HtoD 반복을 줄임")
    ap.add_argument("--empty_cache_interval", type=int, default=0,
                    help="0이면 empty_cache 미사용. N>0이면 레이어 N개마다 empty_cache 호출")

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
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code
    )

    device0 = next(model.parameters()).device

    # Σ_x^{1/2} (diag OAS) — 캐시 로드/생성
    safe_model = _safe_name(args.model_id if args.revision is None else f"{args.model_id}@{args.revision}")
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
            calib_batch_size=args.calib_batch_size,
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

    # CSV 준비(스트리밍 작성)
    csv_path = os.path.join(args.output_dir, f"alpha_layerwise_rank{args.rank}.csv")
    qmode_csv_path = os.path.join(args.output_dir, f"alpha_2bit_qmode_rank{args.rank}.csv")
    qmode_summary_path = os.path.join(args.output_dir, f"alpha_2bit_qmode_rank{args.rank}_summary.csv")

    fieldnames_main = [
        "full_name", "module", "m", "n", "bit", "qmode", "rank", "group_size",
        "Lq_weighted", "Lres_weighted", "alpha",
    ]
    fieldnames_2bit = ["full_name", "module", "bit", "qmode", "Lq_weighted", "Lres_weighted", "alpha"]

    # 2bit summary 집계용
    qmode_counts = Counter()

    # top/bot 출력용(가벼운 메타만 저장)
    alpha_samples = []  # (alpha_float, full_name, bit, qmode)

    # sqrt_diag device 캐시
    sqrt_cache_device: Dict[str, torch.Tensor] = {}

    def get_sqrt_diag(mod_name: str) -> Optional[torch.Tensor]:
        if mod_name not in cov_ops:
            return None
        s = cov_ops[mod_name]["sqrt"]  # CPU float32
        if not args.keep_calib_on_device:
            return s
        if mod_name in sqrt_cache_device:
            return sqrt_cache_device[mod_name]
        sd = s.to(device0, non_blocking=True)
        sqrt_cache_device[mod_name] = sd
        return sd

    # 타깃 파라미터만 순회 (state_dict 전체 복사 제거)
    print("[Step2] Measuring α(b) per target linear weight...")

    n_targets = 0
    for pname, p in model.named_parameters():
        if is_target_weight(pname, p.detach()):
            n_targets += 1

    if n_targets == 0:
        raise RuntimeError("[Step2] No target weights found. Check TARGET_SUFFIXES / model naming.")

    # main CSV open
    with open(csv_path, "w", newline="", encoding="utf-8") as f_main, \
         open(qmode_csv_path, "w", newline="", encoding="utf-8") as f_2bit:

        writer_main = csv.DictWriter(f_main, fieldnames=fieldnames_main)
        writer_main.writeheader()

        writer_2bit = csv.DictWriter(f_2bit, fieldnames=fieldnames_2bit)
        writer_2bit.writeheader()

        processed = 0
        for full_name, W_param in tqdm(list(model.named_parameters()), desc="Target Weights"):
            if not is_target_weight(full_name, W_param.detach()):
                continue

            mod_name = module_name_from_weight(full_name)
            sqrt_diag = get_sqrt_diag(mod_name)
            if sqrt_diag is None:
                continue

            # W: 기존 코드와 동일하게 float32로 측정
            W = W_param.detach().to(device0, dtype=torch.float32)
            m, n = W.shape

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
                writer_main.writerow(row)

                alpha_samples.append((float(alpha), full_name, int(b), qmode))

                if b == 2:
                    writer_2bit.writerow({
                        "full_name": full_name,
                        "module": mod_name,
                        "bit": b,
                        "qmode": qmode,
                        "Lq_weighted": f"{Lq:.6e}",
                        "Lres_weighted": f"{Lres:.6e}",
                        "alpha": f"{alpha:.8f}",
                    })
                    qmode_counts[qmode] += 1

            processed += 1

            # 메모리 정리(필수 텐서만)
            del W
            if args.empty_cache_interval > 0 and processed % args.empty_cache_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"[Step2] Saved CSV: {csv_path}")
    print(f"[Step2] Saved 2-bit qmode CSV: {qmode_csv_path}")

    # 2-bit qmode summary
    if qmode_counts:
        with open(qmode_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["qmode", "count"])
            writer.writeheader()
            for qmode, c in sorted(qmode_counts.items()):
                writer.writerow({"qmode": qmode, "count": c})
        print(f"[Step2] Saved 2-bit qmode summary: {qmode_summary_path}")

    # 상위/하위 α 레이어 간단 프린트
    try:
        # alpha 낮을수록 best-restorable
        top = sorted(alpha_samples, key=lambda x: x[0])[:8]
        bot = sorted(alpha_samples, key=lambda x: x[0], reverse=True)[:8]

        print("\n[Step2] Lowest α (best-restorable) examples:")
        for a, name, b, qm in top:
            print(f" {name:<70s} bit={b} qmode={qm} α={a:.8f}")

        print("\n[Step2] Highest α (hard-to-restore) examples:")
        for a, name, b, qm in bot:
            print(f" {name:<70s} bit={b} qmode={qm} α={a:.8f}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
