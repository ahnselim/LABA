#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STEP 2 - Calibration: Per-layer right-weight diag S = Σ_x^{1/2} (diag) estimation

목적:
1) calibration 데이터로 모델 forward
2) 타깃 Linear layer 입력 activation x의 diag covariance(또는 OAS-shrunk diag) 추정
3) s = sqrt(diag(Σ_x)) 를 저장 (S=diag(s))
4) 또한 inv_s = 1/s 도 같이 저장 (Step2.5/Step3에서 B = Bbar * inv_s 용)

출력 포맷 (추천):
  out_calib_s.pt : dict
    {
      "<module>.weight": {
         "s": Tensor[in_features] float32 (cpu),
         "inv_s": Tensor[in_features] float32 (cpu),
         "mean": Tensor[in_features] float32 (cpu),
         "var": Tensor[in_features] float32 (cpu),   # diag(Σ_x)
         "n": int,                                   # total samples (tokens*batch)
         "shrinkage": float (oas일 때)
         "meta": {...}
      },
      ...
    }

사용 예시:
  
CUDA_VISIBLE_DEVICES=2 python step2_calib_right_weight_diag.py \
  --model_name meta-llama/Llama-3.2-3B \
  --dataset DKYoon/SlimPajama-6B --dataset_config none --split train \
  --seq_len 2048 --nsamples 64 --batch_size 1 \
  --out_calib_s ./output/3bit_asym_nonuniform_base/calib_sqrtdiag.pt \
  --device cuda --device_map auto --cov_mode oas



"""

import os, gc, re, argparse
from typing import Optional
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Target layer filter (Step1과 동일)
# -------------------------
TARGET_KEYWORDS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "fc1", "fc2",
]

def is_target_module(name: str, module: nn.Module) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return any(kw in name for kw in TARGET_KEYWORDS) and ("layers" in name)

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


# -------------------------
# OAS shrinkage (diag-only variant)
# -------------------------
@torch.no_grad()
def oas_shrink_diag(var_diag: torch.Tensor, n_samples: int, eps: float = 1e-12):
    """
    var_diag: [d] sample covariance diagonal (float32, device any)
    n_samples: number of samples used to estimate (>=2)
    Returns:
      shrunk_diag: [d]
      shrinkage: float
    OAS shrinkage (Chen et al.)를 diag에 맞춰 적용:
      S_shrunk = (1-ρ)S + ρ*(tr(S)/d)*I
    여기서 S는 diag만 사용.
    """
    d = var_diag.numel()
    if n_samples <= 1 or d <= 1:
        return var_diag, 0.0

    trS = var_diag.sum()
    trS2 = (var_diag * var_diag).sum()

    # OAS shrinkage coefficient (sklearn 구현식 기반)
    # ρ = min(1, ((1 - 2/p) tr(S^2) + tr(S)^2) / ((n+1 - 2/p)(tr(S^2) - tr(S)^2/p)))
    p = float(d)
    n = float(n_samples)

    num = (1.0 - 2.0 / p) * trS2 + trS * trS
    den = (n + 1.0 - 2.0 / p) * (trS2 - (trS * trS) / p)
    den = torch.clamp(den, min=eps)

    rho = (num / den).item()
    if rho < 0.0:
        rho = 0.0
    if rho > 1.0:
        rho = 1.0

    mu = (trS / p)  # scalar (tensor)
    shrunk = (1.0 - rho) * var_diag + rho * mu
    return shrunk, rho


# -------------------------
# Dataset helpers (SlimPajama streaming 지원)
# -------------------------
def _canonical_dataset_name(name: str) -> str:
    a = name.strip().lower()
    if a in {"dkyoon/slimpajama-6b", "slimpajama-6b", "dkyoon_slimpajama_6b"}:
        return "DKYoon/SlimPajama-6B"
    if a in {"slimpajama-627b", "cerebras/slimpajama-627b", "slimpajama627b"}:
        return "cerebras/SlimPajama-627B"
    return name


def open_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str = "train",
    streaming: bool = True,
):
    if dataset_config is not None:
        dc = str(dataset_config).strip()
        if dc == "" or dc.lower() == "none":
            dataset_config = None
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
                m = re.search(r"\[(.*?)\]", msg, flags=re.S)
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
            raise

    ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=False)
    return ds, dataset_name, dataset_config, False


@torch.no_grad()
def build_calibration_tokens(
    tokenizer,
    nsamples,
    seqlen,
    dataset,
    dataset_config,
    split,
    use_streaming=True,
):
    dataset = _canonical_dataset_name(dataset)
    ds, dataset, dataset_config, is_streaming = open_hf_dataset(
        dataset, dataset_config, split=split, streaming=use_streaming
    )
    print(
        f"[Calib] dataset={dataset}, config={dataset_config}, streaming={is_streaming}"
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


def build_token_batches_from_tokens(tokens: torch.Tensor, batch_size: int):
    """
    tokens: [N, L] int64
    returns: list of (input_ids, attention_mask), nsamples
    """
    nsamples = tokens.shape[0]
    batches = []
    for i in range(0, nsamples, batch_size):
        x = tokens[i : i + batch_size]
        attn = torch.ones_like(x)
        batches.append((x, attn))
    return batches, nsamples


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Step2 - Calibrate diag Σx^{1/2} per layer")
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--out_calib_s", type=str, required=True)

    ap.add_argument("--dataset", type=str, default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--subset", type=str, default=None, help="(deprecated) alias of --dataset_config")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in ["1", "true", "yes"],
        default=True,
    )

    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=128)      # number of sequences
    ap.add_argument("--batch_size", type=int, default=1)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--cov_mode", type=str, default="oas", choices=["var", "oas", "second_moment"])
    ap.add_argument("--eps", type=float, default=1e-8)

    # (Optional) quant bookkeeping
    ap.add_argument("--bits", type=int, default=4, choices=[1,2,3,4])
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--group_size_1", type=int, default=None, help="1-bit group size (override)")
    ap.add_argument("--group_size_2", type=int, default=None, help="2-bit group size (override)")
    ap.add_argument("--group_size_3", type=int, default=None, help="3-bit group size (override)")
    ap.add_argument("--group_size_4", type=int, default=None, help="4-bit group size (override)")

    # 저장 메타
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    group_size = _gs_for_bit(args.bits, args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name} (device_map={args.device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    dataset_config = args.dataset_config if args.dataset_config is not None else args.subset

    tokens = build_calibration_tokens(
        tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seq_len,
        dataset=args.dataset,
        dataset_config=dataset_config,
        split=args.split,
        use_streaming=bool(args.use_streaming),
    )
    batches, nsamples_eff = build_token_batches_from_tokens(tokens, args.batch_size)
    print(f"Built calibration batches: {len(batches)} (nsamples={nsamples_eff}, seq_len={args.seq_len}, batch_size={args.batch_size})")

    # -------------------------
    # Hook stats: sum, sumsq, n
    # -------------------------
    stats = {}  # module_name -> dict(sum, sumsq, n)
    handles = []

    def _ensure_stats(mod_name: str, d: int, device_x: torch.device):
        if mod_name in stats:
            return
        st = {
            "sum": torch.zeros((d,), device=device_x, dtype=torch.float32),
            "sumsq": torch.zeros((d,), device=device_x, dtype=torch.float32),
            "n": 0,
        }
        stats[mod_name] = st

    def make_hook(mod_name):
        def hook_fn(module, inputs, output):
            x = inputs[0]
            # x: [B,L,K] or [B,K]
            if x is None:
                return
            if x.dim() == 2:
                x2d = x
            else:
                x2d = x.reshape(-1, x.shape[-1])  # [N,K]
            x2d = x2d.detach()

            # accumulate on same device as x (likely cuda)
            x2d_f = x2d.to(dtype=torch.float32)
            d = x2d_f.shape[-1]
            _ensure_stats(mod_name, d, x2d_f.device)
            st = stats[mod_name]
            st["sum"] += x2d_f.sum(dim=0)
            st["sumsq"] += (x2d_f * x2d_f).sum(dim=0)
            st["n"] += x2d_f.shape[0]

        return hook_fn

    # register hooks on target modules
    n_targets = 0
    for name, module in model.named_modules():
        if is_target_module(name, module):
            handles.append(module.register_forward_hook(make_hook(name)))
            n_targets += 1
    print(f"Registered hooks on {n_targets} target Linear modules ✅")

    # -------------------------
    # Run calibration forward (PASS 1)
    # -------------------------
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(batches, desc="Calibrating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # cleanup hooks
    for h in handles:
        h.remove()
    handles.clear()

    # -------------------------
    # Build output dict (weight-key aligned)
    # -------------------------
    out = {}
    for mod_name, st in stats.items():
        n = st["n"]
        if n <= 0:
            continue

        mean = st["sum"] / float(n)
        mean_sq = st["sumsq"] / float(n)

        if args.cov_mode == "second_moment":
            # Σx diag을 E[x^2]로 보고 sqrt
            var = mean_sq
            shrink = 0.0
        else:
            # diag covariance: E[x^2] - (E[x])^2
            var = (mean_sq - mean * mean).clamp(min=0.0)
            shrink = 0.0
            if args.cov_mode == "oas":
                var, shrink = oas_shrink_diag(var, n_samples=n)

        s = torch.sqrt(var.clamp(min=args.eps))
        inv_s = 1.0 / s.clamp(min=args.eps)

        weight_key = mod_name + ".weight"  # Step1과 맞추기

        out[weight_key] = {
            "s": s.detach().cpu().to(torch.float32).contiguous(),
            "inv_s": inv_s.detach().cpu().to(torch.float32).contiguous(),
            "mean": mean.detach().cpu().to(torch.float32).contiguous(),
            "var": var.detach().cpu().to(torch.float32).contiguous(),
            "n": int(n),
            "shrinkage": float(shrink),
            "meta": {
                "cov_mode": args.cov_mode,
                "seq_len": int(args.seq_len),
                "nsamples": int(nsamples_eff),
                "batch_size": int(args.batch_size),
                "dataset": _canonical_dataset_name(args.dataset),
                "subset": dataset_config,
                "split": args.split,
                "tag": args.tag,
                "bits": int(args.bits),
                "group_size": int(group_size),
            }
        }

    # save
    os.makedirs(os.path.dirname(args.out_calib_s) or ".", exist_ok=True)
    torch.save(out, args.out_calib_s)

    print("\n✅ COMPLETED: saved calibration right-weight diag")
    print(f"  • out_calib_s: {args.out_calib_s}")
    print(f"  • layers saved: {len(out)}")

    # quick preview
    some = next(iter(out.keys())) if len(out) > 0 else None
    if some is not None:
        v = out[some]
        print(f"\n[Preview] {some}")
        print(f"  s shape   : {tuple(v['s'].shape)}")
        print(f"  var mean  : {v['var'].mean().item():.6f}")
        print(f"  shrinkage : {v['shrinkage']}")


    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
