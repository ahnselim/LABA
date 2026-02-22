#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4 (Prebake) — For every target Linear layer, compute and SAVE Wq and (A,B)
for bits ∈ {1,2,3,4} into per-module files.

• Step2에서 저장한 Σ_x^{1/2} 캐시(.pt) 재사용 지원: --reuse_calib --calib_cache_dir
  파일명 규칙:
    calib_oas_sqrtdiag_{model}__{dataset}__{config}__{split}__ns{ns}_L{L}.pt
  (없으면 본 스크립트가 생성해 동일 경로에 저장)

출력 디렉토리 구조 (예):
prebake_root/
  meta.json
  bit1/
    model_layers_0_self_attn_q_proj.pt  # {"module": "...", "full_weight": "...weight", "Wq": Tensor(fp16,CPU), "A": Tensor(fp16,CPU), "B": Tensor(fp16,CPU), "meta": {...}}
  bit2/
  bit3/
  bit4/

사용 예:
CUDA_VISIBLE_DEVICES=2 python step4_prebake_quant_and_ab.py \
  --model_id meta-llama/Llama-3.2-3B \
  --rank_1 256 --rank_2 128 --rank_3 64 --rank_4 32 \
  --group_size_1 16 --group_size_2 32 --group_size_3 64 --group_size_4 128 \
  --dataset DKYoon/SlimPajama-6B --nsamples 64 --seqlen 2048 \
  --dtype bf16 --device cuda \
  --reuse_calib \
  --calib_cache_dir ../artifacts/bitmin \
  --trust_remote_code \
  --out_root ../artifacts/bitmin/prebake_dyn

"""

import os, re, gc, json, argparse, time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------- 대상 Linear ----------
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
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )


def module_name_from_weight(full_weight_name: str) -> str:
    return full_weight_name[: -len(".weight")]


def _gs_for_bit(bit: int, args) -> int:
    """bit별 group_size 선택 (없으면 global group_size fallback)"""
    if bit == 1 and getattr(args, "group_size_1", None) is not None:
        return int(args.group_size_1)
    if bit == 2 and getattr(args, "group_size_2", None) is not None:
        return int(args.group_size_2)
    if bit == 3 and getattr(args, "group_size_3", None) is not None:
        return int(args.group_size_3)
    if bit == 4 and getattr(args, "group_size_4", None) is not None:
        return int(args.group_size_4)
    return int(args.group_size)


def _rank_for_bit(bit: int, args) -> int:
    """bit별 SVD rank 선택 (없으면 global rank fallback)"""
    if bit == 1 and getattr(args, "rank_1", None) is not None:
        return int(args.rank_1)
    if bit == 2 and getattr(args, "rank_2", None) is not None:
        return int(args.rank_2)
    if bit == 3 and getattr(args, "rank_3", None) is not None:
        return int(args.rank_3)
    if bit == 4 and getattr(args, "rank_4", None) is not None:
        return int(args.rank_4)
    return int(args.rank)


def _safe_name(s) -> str:
    """
    Step2와 동일 규칙:
      • None, "", "None" → "none" (소문자)
      • 나머지는 비영숫자/._- 이외 문자를 '_'로 치환
    """
    if s is None:
        s = "none"
    s = str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


# ---------- Dataset helpers ----------
def _canonical_dataset_name(name: str) -> str:
    a = name.strip().lower()
    if a in {"dkyoon/slimpajama-6b", "slimpajama-6b", "dkyoon_slimpajama_6b"}:
        return "DKYoon/SlimPajama-6B"
    if a in {"slimpajama-627b", "cerebras/slimpajama-627b", "slimpajama627b"}:
        return "cerebras/SlimPajama-627B"
    return name


@torch.no_grad()
def build_calibration_tokens(
    tokenizer, nsamples, seqlen, dataset, dataset_config, split, use_streaming=True
):
    dataset = _canonical_dataset_name(dataset)
    ds = load_dataset(
        dataset, name=dataset_config, split=split, streaming=use_streaming
    )
    take = ds.take if hasattr(ds, "take") else None
    iterator = take(max(nsamples * 5, nsamples)) if take else ds
    eos = tokenizer.eos_token_id or tokenizer.pad_token_id
    samples, buf = [], []
    for row in iterator:
        text = None
        for k in ("text", "content", "raw_content"):
            if k in row and isinstance(row[k], str) and row[k].strip():
                text = row[k]
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
        if eos is not None:
            ids.append(eos)
        buf.extend(ids)
        while len(buf) >= seqlen and len(samples) < nsamples:
            samples.append(torch.tensor(buf[:seqlen], dtype=torch.long))
            buf = buf[seqlen:]
            if len(samples) >= nsamples:
                break
        if len(samples) >= nsamples:
            break
    if not samples:
        raise RuntimeError("No calibration tokens collected.")
    return torch.stack(samples, dim=0)


# ---------- Σ_x (diag OAS) ----------
@torch.no_grad()
def estimate_diag_cov_oas_per_module(
    model: nn.Module,
    tokenizer,
    device,
    nsamples=64,
    seqlen=2048,
    dataset="DKYoon/SlimPajama-6B",
    dataset_config=None,
    split="train",
    use_streaming=True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    model.eval()
    name_to_dim: Dict[str, int] = {}
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    def hook_factory(mn: str):
        def hook(module, inp, _out):
            x = (
                inp[0]
                .detach()
                .reshape(-1, inp[0].shape[-1])
                .to(device=device, dtype=torch.float32)
            )
            d = x.shape[-1]
            name_to_dim[mn] = d
            if mn not in stats:
                stats[mn] = {
                    "sum": torch.zeros(d, dtype=torch.float64, device="cpu"),
                    "sumsq": torch.zeros(d, dtype=torch.float64, device="cpu"),
                    "n": torch.tensor(0, dtype=torch.long),
                }
            x_cpu = x.to("cpu", dtype=torch.float32)
            stats[mn]["sum"] += x_cpu.sum(dim=0, dtype=torch.float64)
            stats[mn]["sumsq"] += (x_cpu.to(torch.float64).pow(2)).sum(dim=0)
            stats[mn]["n"] += x_cpu.shape[0]

        return hook

    for mn, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(hook_factory(mn)))
    tokens = build_calibration_tokens(
        tokenizer, nsamples, seqlen, dataset, dataset_config, split, use_streaming
    ).to(device)
    for i in tqdm(range(tokens.shape[0]), desc="[Prebake] Calibration Fwd"):
        model(tokens[i : i + 1])
    for h in handles:
        h.remove()
    ops = {}
    for mn, st in stats.items():
        n = int(st["n"].item())
        if n <= 1 or mn not in name_to_dim:
            continue
        d = name_to_dim[mn]
        sumv, sumsq = st["sum"], st["sumsq"]
        mean = sumv / n
        ex2 = sumsq / n
        var = torch.clamp(ex2 - mean.pow(2), min=1e-12)
        p = float(d)
        trS = var.sum().item()
        trS2 = (var.pow(2)).sum().item()
        den = (n + 1.0 - 2.0 / p) * (trS2 - (trS * trS) / p)
        num = (1.0 - 2.0 / p) * trS2 + (trS * trS)
        alpha = 1.0 if den <= 0 else max(0.0, min(1.0, num / den))
        mu = trS / p
        sigma_diag = (1.0 - alpha) * var + alpha * mu
        sqrt_diag = torch.sqrt(torch.clamp(sigma_diag, min=1e-12)).to(torch.float32)
        inv_sqrt_diag = (1.0 / torch.clamp(sqrt_diag, min=1e-12)).to(torch.float32)
        ops[mn] = {"sqrt": sqrt_diag.cpu(), "inv_sqrt": inv_sqrt_diag.cpu()}
    return ops


# ---------- Group helpers ----------
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
def _pct_clip_lastdim(X: torch.Tensor, lo_pct: float, hi_pct: float) -> torch.Tensor:
    lo = torch.quantile(X, lo_pct / 100.0, dim=-1, keepdim=True)
    hi = torch.quantile(X, hi_pct / 100.0, dim=-1, keepdim=True)
    return torch.clamp(X, lo, hi)


# ---------- Quantization (Step4와 동일 규칙) ----------
@torch.no_grad()
def dequant_1bit_mu_beta(W: torch.Tensor, group_size: int):
    """
    1bit quantization (binary with global mu shift + per-group beta)

    스텝:
      1) beta (per-group scale):
         - W를 group_size=k로 나누고
         - 각 그룹에서 beta = mean(abs(W_ij))
      2) mu (global mean):
         - 전체 weight의 평균 mu = mean(W)
      3) sign:
         - W_center = W - mu
         - W_center < 0 → -1, W_center > 0 → +1 (0은 +1로 취급)
      4) dequant:
         - 각 그룹에 대해 W_hat = sign * beta
    """
    # float32로 변환
    W32 = W.to(torch.float32)

    # (1) beta: 원본 W 기준 그룹별 평균 |W|
    Wg_orig, O_, G, S, orig_I = _to_groups(W32, group_size)
    beta = Wg_orig.abs().mean(dim=-1, keepdim=True)  # [O_, G, 1]

    # (2) mu: 레이어 전체 평균
    mu = W32.mean()

    # (3) sign: (W - mu)를 기준으로 binary 부호 결정
    W_center = W32 - mu
    Wg_center, _, _, _, _ = _to_groups(W_center, group_size)
    sgn = torch.sign(Wg_center)
    sgn[sgn == 0] = 1.0  # 0은 +1 처리

    # (4) dequant: sign * beta
    deq_g = sgn * beta  # [O_, G, S]
    deq = _from_groups(deq_g, orig_I)

    return deq.to(dtype=W.dtype, device=W.device)

@torch.no_grad()
def dequant_2bit_lloyd2_sym(W: torch.Tensor, group_size: int, clip_pct=99.9, steps=4):
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _pct_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
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
def dequant_2bit_qtr_zero(W: torch.Tensor, group_size: int, clip_pct=99.9, steps=6):
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _pct_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
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
    W: torch.Tensor, b: int, group_size: int, clip_pct=0.0, rounding="nearest"
):
    assert b in (3, 4)
    W32 = W.to(torch.float32)
    Wg, O_, G, S, orig_I = _to_groups(W32, group_size)
    X = _pct_clip_lastdim(Wg, 100 - clip_pct, clip_pct) if clip_pct > 0 else Wg
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


# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser(
        "Step 4 Prebake — Save Wq & (A,B) for bits 1/2/3/4 per module"
    )
    # 모델/장치
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["auto", "fp16", "bf16", "fp32", "float16", "bfloat16", "float32"],
    )
    ap.add_argument("--device", default="cuda")

    # 프리베이크 하이퍼 (글로벌 + bit별 override)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--group_size", type=int, default=128)
    # bit별 group_size
    ap.add_argument("--group_size_1", type=int, default=None, help="1-bit group size (override)")
    ap.add_argument("--group_size_2", type=int, default=None, help="2-bit group size (override)")
    ap.add_argument("--group_size_3", type=int, default=None, help="3-bit group size (override)")
    ap.add_argument("--group_size_4", type=int, default=None, help="4-bit group size (override)")
    # bit별 rank
    ap.add_argument("--rank_1", type=int, default=None, help="1-bit SVD rank (override)")
    ap.add_argument("--rank_2", type=int, default=None, help="2-bit SVD rank (override)")
    ap.add_argument("--rank_3", type=int, default=None, help="3-bit SVD rank (override)")
    ap.add_argument("--rank_4", type=int, default=None, help="4-bit SVD rank (override)")

    # Calibration/캐시 (Step2 규칙 재사용)
    ap.add_argument("--dataset", type=str, default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in ["1", "true", "yes"],
        default=True,
    )
    ap.add_argument("--reuse_calib", action="store_true")
    ap.add_argument("--calib_cache_dir", type=str, default="./artifacts/bitmin")

    # 2/3/4bit 세부
    ap.add_argument("--clip_percentile", type=float, default=99.9)
    ap.add_argument("--lloyd_steps", type=int, default=4)
    ap.add_argument("--qtr_steps", type=int, default=6)
    ap.add_argument(
        "--uniform_rounding",
        type=str,
        default="nearest",
        choices=["nearest", "stochastic"],
    )

    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # dtype
    if args.dtype in ("bf16", "bfloat16"):
        load_dtype = torch.bfloat16
    elif args.dtype in ("fp16", "float16"):
        load_dtype = torch.float16
    elif args.dtype in ("fp32", "float32"):
        load_dtype = torch.float32
    else:
        load_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # 모델 로드
    tok = AutoTokenizer.from_pretrained(
        args.model_id, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(
            load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None
        ),
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    ).to(device)

    # ---- Σ_x 캐시 로드/생성 (Step2 규칙) ----
    safe_model = _safe_name(
        args.model_id if args.revision is None else f"{args.model_id}@{args.revision}"
    )
    safe_dataset = _safe_name(_canonical_dataset_name(args.dataset))
    safe_config = _safe_name(args.dataset_config)  # <-- None/""/"None" → "none" 통일
    os.makedirs(args.calib_cache_dir, exist_ok=True)
    calib_basename = f"calib_oas_sqrtdiag_{safe_model}__{safe_dataset}__{safe_config}__{args.split}__ns{args.nsamples}_L{args.seqlen}.pt"
    calib_path = Path(args.calib_cache_dir) / calib_basename

    # 백워드 호환: 과거 "__None__" 파일명도 탐색 (이전 Step4가 'None'으로 저장했을 가능성)
    if not calib_path.exists() and "__none__" in calib_basename:
        alt = Path(args.calib_cache_dir) / calib_basename.replace(
            "__none__", "__None__"
        )
        if alt.exists():
            calib_path = alt

    cov_ops: Dict[str, Dict[str, torch.Tensor]]
    if args.reuse_calib and calib_path.exists():
        print(f"[Prebake] Loading Σ_x cache: {calib_path}")
        payload = torch.load(calib_path, map_location="cpu")
        cov_ops = payload.get("cov_ops", payload)
        # inv_sqrt 미존재 시 동적 생성 후 캐시에 반영
        changed = False
        for k, entry in cov_ops.items():
            if "inv_sqrt" not in entry and "sqrt" in entry:
                s = entry["sqrt"].to(torch.float32)
                entry["inv_sqrt"] = (1.0 / torch.clamp(s, min=1e-12)).to(torch.float32)
                changed = True
        if changed:
            meta = payload.get(
                "meta",
                {
                    "model_id": args.model_id,
                    "revision": args.revision,
                    "dataset": _canonical_dataset_name(args.dataset),
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "nsamples": args.nsamples,
                    "seqlen": args.seqlen,
                },
            )
            torch.save({"cov_ops": cov_ops, "meta": meta}, calib_path)
    else:
        cov_ops = estimate_diag_cov_oas_per_module(
            model,
            tok,
            device,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            use_streaming=bool(args.use_streaming),
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
        print(f"[Prebake] Saved Σ_x cache: {calib_path}")

    # 원 state_dict 백업
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # bit 디렉토리 준비
    for b in (1, 2, 3, 4):
        (out_root / f"bit{b}").mkdir(parents=True, exist_ok=True)

    # -------- per-layer × {1,2,3,4} 저장 --------
    for full_name, W_cpu in tqdm(state.items(), desc="[Prebake] per-layer x {1,2,3,4}"):
        if not (full_name.endswith(".weight") and is_target_weight(full_name, W_cpu)):
            continue
        module = module_name_from_weight(full_weight_name=full_name)
        if module not in cov_ops:  # Σ_x 없는 모듈은 스킵
            continue

        sqrt_diag = cov_ops[module]["sqrt"].to(torch.float32)
        inv_sqrt = cov_ops[module]["inv_sqrt"].to(torch.float32)
        dev0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        W = W_cpu.to(dev0, dtype=torch.float32)
        m, n = W.shape
        sqrt_diag = sqrt_diag.to(dev0)
        inv_sqrt = inv_sqrt.to(dev0)
        def _svd_ab(Wq: torch.Tensor, r_eff_max: int):
            E = W - Wq.to(W.dtype)
            E_t = E * sqrt_diag.unsqueeze(0)
            try:
                U, S, Vh = torch.linalg.svd(E_t, full_matrices=False)
            except RuntimeError:
                U, S, Vh = torch.linalg.svd(E_t.to("cpu"), full_matrices=False)
                U, S, Vh = U.to(W.device), S.to(W.device), Vh.to(W.device)
            r_eff = min(r_eff_max, S.numel())
            if r_eff == 0:
                A = torch.zeros(m, 0, dtype=torch.float16)
                B = torch.zeros(0, n, dtype=torch.float16)
            else:
                U_r, S_r, V_rT = U[:, :r_eff], S[:r_eff], Vh[:r_eff, :]
                S_sqrt = torch.sqrt(torch.clamp(S_r, min=1e-12))
                A = (U_r * S_sqrt.unsqueeze(0)).to(torch.float16).cpu()
                B = (
                    (S_sqrt.unsqueeze(1) * V_rT * inv_sqrt.unsqueeze(0))
                    .to(torch.float16)
                    .cpu()
                )
            return A, B

        # 1-bit (bit별 group_size / rank)
        gs1 = _gs_for_bit(1, args)
        r1 = min(_rank_for_bit(1, args), min(m, n))
        Wq1 = dequant_1bit_mu_beta(W, gs1)
        A1, B1 = _svd_ab(Wq1, r1)
        torch.save(
            {
                "module": module,
                "full_weight": full_name,
                "Wq": Wq1.to(torch.float16).cpu(),
                "A": A1,
                "B": B1,
                "meta": {"bit": 1},
            },
            out_root / "bit1" / f"{_safe_name(module)}.pt",
        )

        # 2-bit (두 후보 중 weighted Frobenius 작은 쪽, bit별 group_size / rank)
        gs2 = _gs_for_bit(2, args)
        r2 = min(_rank_for_bit(2, args), min(m, n))
        Wq_l = dequant_2bit_lloyd2_sym(
            W, gs2, args.clip_percentile, args.lloyd_steps
        )
        Wq_z = dequant_2bit_qtr_zero(
            W, gs2, args.clip_percentile, args.qtr_steps
        )
        El = (W - Wq_l) * sqrt_diag.unsqueeze(0)
        Ez = (W - Wq_z) * sqrt_diag.unsqueeze(0)
        pick = (
            "lloyd2" if torch.linalg.norm(El) <= torch.linalg.norm(Ez) else "qtr_zero"
        )
        Wq2 = Wq_l if pick == "lloyd2" else Wq_z
        A2, B2 = _svd_ab(Wq2, r2)
        torch.save(
            {
                "module": module,
                "full_weight": full_name,
                "Wq": Wq2.to(torch.float16).cpu(),
                "A": A2,
                "B": B2,
                "meta": {"bit": 2, "variant": pick},
            },
            out_root / "bit2" / f"{_safe_name(module)}.pt",
        )

        # 3-bit (bit별 group_size / rank)
        gs3 = _gs_for_bit(3, args)
        r3 = min(_rank_for_bit(3, args), min(m, n))
        Wq3 = dequant_uniform_asym(W, 3, gs3, 0.0, args.uniform_rounding)
        A3, B3 = _svd_ab(Wq3, r3)
        torch.save(
            {
                "module": module,
                "full_weight": full_name,
                "Wq": Wq3.to(torch.float16).cpu(),
                "A": A3,
                "B": B3,
                "meta": {"bit": 3},
            },
            out_root / "bit3" / f"{_safe_name(module)}.pt",
        )

        # 4-bit (bit별 group_size / rank)
        gs4 = _gs_for_bit(4, args)
        r4 = min(_rank_for_bit(4, args), min(m, n))
        Wq4 = dequant_uniform_asym(W, 4, gs4, 0.0, args.uniform_rounding)
        A4, B4 = _svd_ab(Wq4, r4)
        torch.save(
            {
                "module": module,
                "full_weight": full_name,
                "Wq": Wq4.to(torch.float16).cpu(),
                "A": A4,
                "B": B4,
                "meta": {"bit": 4},
            },
            out_root / "bit4" / f"{_safe_name(module)}.pt",
        )

        # free
        del W
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 메타 저장
    meta = {
        "model_id": args.model_id,
        "revision": args.revision,
        "rank": args.rank,
        "group_size": args.group_size,
        "dataset": _canonical_dataset_name(args.dataset),
        "dataset_config": args.dataset_config,
        "split": args.split,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "dtype": args.dtype,
        "created": int(time.time()),
        "note": "Per-module prebaked Wq/A/B for bits 1/2/3/4; Σ_x cache reused if available.",
    }
    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Prebake] DONE. Saved to: {out_root}")


if __name__ == "__main__":
    main()
