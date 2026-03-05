#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixture step0 module: Step0-2 calibration for right-weight diagonal statistics.

역할:
  - calibration 데이터로 타깃 Linear 입력 activation 통계 수집
  - per-layer diag covariance 기반 `s`, `inv_s` (및 mean/var) 저장
  - step3/step4에서 사용하는 `calib_sqrtdiag.pt` 생성

사용 방식:
  - CLI 실행: 이 파일의 `main()`
  - 모듈 사용: 하단 `Step02CalibConfig` + `run()`

참고:
  - `LABA/mixture/step0_optimization.py`와의 호환을 위해 래퍼 API를 함께 제공한다.
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
    ap.add_argument("--num_gpus", type=int, default=2, help="device_map=auto일 때 사용할 최대 GPU 개수")
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
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }

    if str(args.device_map).strip().lower() == "auto" and int(args.num_gpus) > 0 and torch.cuda.is_available():
        visible = torch.cuda.device_count()
        use_n = min(int(args.num_gpus), int(visible))
        if use_n > 0:
            max_memory = {}
            for idx in range(use_n):
                total_gib = int(torch.cuda.get_device_properties(idx).total_memory // (1024 ** 3))
                max_memory[idx] = f"{max(1, total_gib - 1)}GiB"
            max_memory["cpu"] = "512GiB"
            model_kwargs["max_memory"] = max_memory
            print(f"[Calib] auto device_map GPU limit: {use_n} (indices: {list(range(use_n))})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
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
class Step02CalibConfig:
    model_name: str
    out_calib_s: str
    bits: int
    group_size: int
    group_size_1: Optional[int] = None
    group_size_2: Optional[int] = None
    group_size_3: Optional[int] = None
    group_size_4: Optional[int] = None
    dataset: str = "DKYoon/SlimPajama-6B"
    dataset_config: Optional[str] = None
    subset: Optional[str] = None
    split: str = "train"
    use_streaming: bool = True
    seq_len: int = 2048
    nsamples: int = 128
    batch_size: int = 1
    device: str = "cuda"
    device_map: str = "auto"
    num_gpus: int = 2
    trust_remote_code: bool = False
    cov_mode: str = "oas"
    eps: float = 1e-8
    tag: str = ""
    seed: int = 42
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


def build_command(cfg: Step02CalibConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_name",
        str(cfg.model_name),
        "--out_calib_s",
        str(cfg.out_calib_s),
        "--bits",
        str(int(cfg.bits)),
        "--group_size",
        str(int(cfg.group_size)),
        "--dataset",
        str(cfg.dataset),
        "--split",
        str(cfg.split),
        "--use_streaming",
        "true" if cfg.use_streaming else "false",
        "--seq_len",
        str(int(cfg.seq_len)),
        "--nsamples",
        str(int(cfg.nsamples)),
        "--batch_size",
        str(int(cfg.batch_size)),
        "--device",
        str(cfg.device),
        "--device_map",
        str(cfg.device_map),
        "--num_gpus",
        str(int(cfg.num_gpus)),
        "--cov_mode",
        str(cfg.cov_mode),
        "--eps",
        str(float(cfg.eps)),
        "--tag",
        str(cfg.tag),
        "--seed",
        str(int(cfg.seed)),
    ]
    if cfg.dataset_config is not None:
        cmd += ["--dataset_config", str(cfg.dataset_config)]
    if cfg.subset is not None:
        cmd += ["--subset", str(cfg.subset)]
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    for bit in (1, 2, 3, 4):
        v = getattr(cfg, f"group_size_{bit}")
        if v is not None:
            cmd += [f"--group_size_{bit}", str(int(v))]
    return cmd


def run(cfg: Step02CalibConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.out_calib_s).parent.mkdir(parents=True, exist_ok=True)
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp
