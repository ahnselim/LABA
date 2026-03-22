#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alternative Step3: LABA-style alternating Lloyd-Max + layerwise low-rank compensation.

Algorithm:
  1) Initialize Wq^(0) with Lloyd-Max quantization.
  2) Fit weighted rank-r low-rank factors on residual (W - Wq) D.
  3) Alternate:
       - AB update:  min || (W - Wq - AB) D ||_F^2
       - Wq update:  min_{Wq in Q} || (W - AB - Wq) D ||_F^2

Outputs:
  - out_dir/wdq_star.pt
  - out_dir/low_rank_ab.pt
  - out_dir/wdq_star_best.pt
  - out_dir/low_rank_ab_best.pt
  - out_dir/metrics.jsonl
  Optional with `--save_all`:
    - out_dir/codebook_star.pt
    - out_dir/qcodes_star.pt
    - out_dir/quant_meta_star.pt
    - out_dir/summary.json

For 1-bit experiments, you can keep the alternating objective unchanged and
add mini calibration NLL checks via `--eval_every` and `--select_metric nll`.

Usage:
  CUDA_VISIBLE_DEVICES=2 nohup python 1bit_test/step3_1bit_logit.py \
    --model_id meta-llama/Llama-3.1-8B \
    --step1_dir ./output/llama3_8b_64/step1_quant/1bit \
    --calib_s ./output/llama3_8b_64/calib_sqrtdiag.pt \
    --out_dir ./output/llama3_8b_64/step3_alt/1bit_nll \
    --qtype hessian-aware \
    --rank_ab 64 \
    --outer_loops 10 \
    --eval_every 1 \
    --best_on_nll \
    --nll_eval_device cuda \
    --nll_nsamples 8 \
    --nll_seq_len 512 > ./logs/logit_llama3_8b.log 2>&1 &

  CUDA_VISIBLE_DEVICES=3 python 1bit_test/step3_1bit_logit.py \
    --model_id Qwen/Qwen3-8B \
    --step1_dir ./output/qwen3_8b_64/step1_quant/1bit \
    --calib_s ./output/qwen3_8b_64/calib_sqrtdiag.pt \
    --out_dir ./output/qwen3_8b_64/step3_alt/1bit_nll \
    --qtype hessian-aware \
    --rank_ab 64 \
    --outer_loops 10 \
    --eval_every 1 \
    --best_on_nll \
    --nll_eval_device cuda \
    --nll_nsamples 8 \
    --nll_seq_len 512 > ./logs/logit_qwen3_8b.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
for _path in (str(_THIS_DIR), str(_PARENT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from step_1_quantize import (  # noqa: E402
    _snapshot_state_to_cpu,
    is_target_weight,
    lloyd_asym_nonuniform_quantize,
)


MODULE_ORDER = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "o_proj": 3,
    "out_proj": 4,
    "gate_proj": 5,
    "up_proj": 6,
    "down_proj": 7,
    "fc1": 8,
    "fc2": 9,
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def extract_block_index(name: str) -> Optional[int]:
    patterns = (
        r"\bmodel\.layers\.(\d+)\.",
        r"\bencoder\.layers\.(\d+)\.",
        r"\blayers\.(\d+)\.",
    )
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def sort_key(name: str) -> Tuple[int, int, str]:
    bidx = extract_block_index(name)
    suffix = name.split(".")[-2] if "." in name else ""
    return (10**9 if bidx is None else bidx, MODULE_ORDER.get(suffix, 10**6), name)


def get_parent_module(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def dequant_from_codebook_codes(
    codebook_ogq: torch.Tensor,
    qcodes_ogs: torch.Tensor,
    orig_i: int,
) -> torch.Tensor:
    o, g, q = codebook_ogq.shape
    _, _, s = qcodes_ogs.shape
    cb = codebook_ogq.reshape(o * g, q)
    idx = qcodes_ogs.reshape(o * g, s).long()
    xq = torch.gather(cb, dim=1, index=idx).reshape(o, g, s)
    return xq.reshape(o, g * s)[:, :orig_i]


@torch.no_grad()
def rank_r_svd(m: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor]:
    o, i = m.shape
    r_eff = min(int(r), o, i)
    if r_eff < 0:
        raise ValueError("rank must be non-negative")
    if r_eff == 0:
        return m.new_zeros((o, 0)), m.new_zeros((0, i))
    try:
        u, s, v = torch.linalg.svd_lowrank(m, q=r_eff, niter=2)
        sroot = torch.sqrt(s.clamp_min(0.0))
        a = u * sroot.unsqueeze(0)
        b = sroot.unsqueeze(1) * v.T
        return a, b
    except Exception:
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        u = u[:, :r_eff]
        s = s[:r_eff]
        vh = vh[:r_eff, :]
        sroot = torch.sqrt(s.clamp_min(0.0))
        a = u * sroot.unsqueeze(0)
        b = sroot.unsqueeze(1) * vh
        return a, b


def load_diag_weight(entry: dict, eps: float) -> torch.Tensor:
    if "s" in entry:
        d = entry["s"].to(torch.float32)
    elif "sqrt" in entry:
        d = entry["sqrt"].to(torch.float32)
    elif "var" in entry:
        d = torch.sqrt(entry["var"].to(torch.float32).clamp_min(0.0))
    elif "inv_s" in entry:
        d = 1.0 / entry["inv_s"].to(torch.float32).clamp_min(float(eps))
    else:
        raise KeyError("calib entry must include one of: s, sqrt, var, inv_s")
    return d.clamp_min(float(eps)).contiguous()


@torch.no_grad()
def weighted_low_rank_fit(
    residual: torch.Tensor,
    diag_weight: torch.Tensor,
    rank: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = diag_weight.to(device=residual.device, dtype=torch.float32)
    residual_bar = residual * d.unsqueeze(0)
    a, b_bar = rank_r_svd(residual_bar, r=int(rank))
    inv_d = 1.0 / d.clamp_min(float(eps))
    b = b_bar * inv_d.unsqueeze(0)
    return a, b


@torch.no_grad()
def weighted_objective(
    w: torch.Tensor,
    wq: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    diag_weight: torch.Tensor,
) -> float:
    d = diag_weight.to(device=w.device, dtype=torch.float32)
    err = (w - wq - (a @ b)) * d.unsqueeze(0)
    return float(torch.mean(err * err).item())


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def should_reuse_step1_init(meta: dict, qtype: str) -> bool:
    weighted = bool(meta.get("uses_hessian_weighting", False))
    if qtype == "hessian-aware":
        return weighted
    return not weighted


def build_step1_svd_baseline(
    w: torch.Tensor,
    d: torch.Tensor,
    codebook: torch.Tensor,
    qcodes: torch.Tensor,
    orig_i: int,
    rank_ab: int,
    eps: float,
) -> Dict[str, torch.Tensor | float]:
    wq = dequant_from_codebook_codes(codebook, qcodes, orig_i=orig_i)
    a, b = weighted_low_rank_fit(w - wq, d, rank=int(rank_ab), eps=float(eps))
    obj = weighted_objective(w, wq, a, b, d)
    return {
        "wq": wq.detach().clone(),
        "A": a.detach().clone(),
        "B": b.detach().clone(),
        "objective": float(obj),
    }


def _torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def nll_selection_enabled(args: argparse.Namespace) -> bool:
    return bool(args.eval_every > 0 or args.select_metric == "nll" or args.best_on_nll)


def _canonical_dataset_config(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return text


def build_mini_eval_batches(
    tokenizer,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    nsamples: int,
    seq_len: int,
    batch_size: int,
    use_streaming: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if load_dataset is None:
        raise RuntimeError("mini NLL eval requires datasets: pip install datasets")

    ds = load_dataset(
        dataset_name,
        name=_canonical_dataset_config(dataset_config),
        split=split,
        streaming=bool(use_streaming),
    )
    iterator = ds.take(max(int(nsamples) * 5, int(nsamples))) if hasattr(ds, "take") else ds

    eos = tokenizer.eos_token_id or tokenizer.pad_token_id
    samples: List[torch.Tensor] = []
    buf: List[int] = []
    for row in iterator:
        text = None
        for key in ("text", "content", "raw_content"):
            if key in row and isinstance(row[key], str) and row[key].strip():
                text = row[key]
                break
        if text is None:
            for value in row.values():
                if isinstance(value, str) and value.strip():
                    text = value
                    break
        if not text:
            continue
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()
        if not ids:
            continue
        if eos is not None:
            ids.append(int(eos))
        buf.extend(ids)
        while len(buf) >= int(seq_len) and len(samples) < int(nsamples):
            samples.append(torch.tensor(buf[:seq_len], dtype=torch.long))
            buf = buf[seq_len:]
            if len(samples) >= int(nsamples):
                break
        if len(samples) >= int(nsamples):
            break

    if not samples:
        raise RuntimeError("No mini NLL calibration samples collected.")

    tokens = torch.stack(samples, dim=0)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    batch_size = max(1, int(batch_size))
    for i in range(0, tokens.shape[0], batch_size):
        x = tokens[i : i + batch_size]
        batches.append((x, torch.ones_like(x)))
    return batches


def init_nll_eval_context(args: argparse.Namespace, device: torch.device) -> Optional[dict]:
    if not nll_selection_enabled(args):
        return None

    print("[Alt-Step3] preparing mini NLL selection context", flush=True)
    tok = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    batches = build_mini_eval_batches(
        tokenizer=tok,
        dataset_name=str(args.nll_dataset),
        dataset_config=args.nll_dataset_config,
        split=str(args.nll_split),
        nsamples=int(args.nll_nsamples),
        seq_len=int(args.nll_seq_len),
        batch_size=int(args.nll_batch_size),
        use_streaming=bool(args.nll_use_streaming),
    )

    load_dtype = _torch_dtype_from_name(args.dtype_w)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=args.trust_remote_code,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    nll_eval_device = torch.device(
        args.nll_eval_device
        if (args.nll_eval_device != "cuda" or torch.cuda.is_available())
        else "cpu"
    )
    if nll_eval_device.type == "cuda":
        model = model.to(nll_eval_device)
    return {"model": model, "tokenizer": tok, "batches": batches}


def should_run_nll_eval(bits: int, args: argparse.Namespace, step: int) -> bool:
    if int(bits) != 1:
        return False
    if not nll_selection_enabled(args):
        return False
    if step == 0:
        return True
    every = int(args.eval_every)
    if every > 0 and step % every == 0:
        return True
    return step == int(args.outer_loops)


def load_context(args: argparse.Namespace) -> dict:
    step1_dir = Path(args.step1_dir).resolve()
    codebook_path = step1_dir / "codebook.pt"
    qcodes_path = step1_dir / "qcodes.pt"
    meta_path = step1_dir / "meta.pt"
    if not (codebook_path.exists() and qcodes_path.exists() and meta_path.exists()):
        raise FileNotFoundError("step1_dir must contain codebook.pt, qcodes.pt, meta.pt")

    print(f"[Alt-Step3] loading step1 artifacts: {step1_dir}", flush=True)
    codebooks: Dict[str, torch.Tensor] = torch.load(codebook_path, map_location="cpu")
    qcodes: Dict[str, torch.Tensor] = torch.load(qcodes_path, map_location="cpu")
    metas: Dict[str, dict] = torch.load(meta_path, map_location="cpu")

    print(f"[Alt-Step3] loading calib_s: {args.calib_s}", flush=True)
    calib_payload = torch.load(args.calib_s, map_location="cpu")
    calib_s: Dict[str, dict] = calib_payload.get("cov_ops", calib_payload)

    load_dtype = _torch_dtype_from_name(args.dtype_w)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dm_raw = str(args.model_device_map).strip().lower()
    resolved_model_device_map = None if dm_raw in {"", "none", "null"} else args.model_device_map

    print(
        f"[Alt-Step3] loading original model: {args.model_id} "
        f"(device_map={resolved_model_device_map}, device={device})",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=args.trust_remote_code,
        device_map=resolved_model_device_map,
        low_cpu_mem_usage=True,
    )
    try:
        state = _snapshot_state_to_cpu(model)
        del model
    except NotImplementedError:
        if resolved_model_device_map is None:
            raise
        print("[Alt-Step3] Detected meta tensors under device_map mode. Re-loading on CPU to build state snapshot.", flush=True)
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
    keys: List[str] = []
    for key in codebooks.keys():
        if key not in qcodes or key not in metas or key not in calib_s or key not in state:
            continue
        if not is_target_weight(key, state[key]):
            continue
        if layer_re and not layer_re.search(key):
            continue
        keys.append(key)

    keys = sorted(keys, key=sort_key)
    if args.max_layers > 0:
        keys = keys[: int(args.max_layers)]
    if not keys:
        raise RuntimeError("No matched layers found.")

    eval_ctx = init_nll_eval_context(args, device=device)

    print(f"[Alt-Step3] matched layers: {len(keys)}", flush=True)
    state = {key: state[key].detach().to(torch.float32).cpu() for key in keys}
    return {
        "device": device,
        "codebooks": codebooks,
        "qcodes": qcodes,
        "metas": metas,
        "calib_s": calib_s,
        "state": state,
        "keys": keys,
        "eval_ctx": eval_ctx,
    }


@torch.no_grad()
def evaluate_global_candidate_nll(
    eval_ctx: dict,
    layer_states: Dict[str, dict],
) -> float:
    model = eval_ctx["model"]
    batches = eval_ctx["batches"]
    total_layers = len(layer_states)

    print(f"[Alt-Step3] NLL eval: patching {total_layers} layers", flush=True)

    total_loss = 0.0
    total_tokens = 0
    patch_iter = tqdm(
        enumerate(layer_states.items(), start=1),
        total=total_layers,
        desc="[Alt-Step3] NLL patch",
        file=sys.stdout,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    for idx, (key, state) in patch_iter:
        module_name = key[:-7]
        parent, attr = get_parent_module(model, module_name)
        layer = getattr(parent, attr)
        if not hasattr(layer, "weight"):
            raise RuntimeError(f"Target layer has no weight: {module_name}")

        patched_weight_cpu = state["wq"].to(dtype=torch.float32, device="cpu")
        patched_weight_cpu = patched_weight_cpu + (
            state["A"].to(dtype=torch.float32, device="cpu") @ state["B"].to(dtype=torch.float32, device="cpu")
        )
        layer.weight.data.copy_(
            patched_weight_cpu.to(device=layer.weight.device, dtype=layer.weight.dtype, non_blocking=True)
        )
        del patched_weight_cpu
        if idx == 1 or idx == total_layers or idx % 16 == 0:
            patch_iter.set_postfix_str(f"{idx}/{total_layers}", refresh=False)

    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        torch.cuda.empty_cache()

    model_device = next(model.parameters()).device
    total_batches = len(batches)
    print(
        f"[Alt-Step3] NLL eval: running {total_batches} batches on {model_device}",
        flush=True,
    )
    batch_iter = tqdm(
        enumerate(batches, start=1),
        total=total_batches,
        desc="[Alt-Step3] NLL batch",
        file=sys.stdout,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    for batch_idx, (input_ids, attention_mask) in batch_iter:
        input_ids = input_ids.to(model_device, non_blocking=True)
        attention_mask = attention_mask.to(model_device, non_blocking=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += int(shift_labels.numel())
        batch_iter.set_postfix_str(
            f"{batch_idx}/{total_batches} avg_loss={total_loss / max(1, total_tokens):.6f}",
            refresh=False,
        )

    nll = total_loss / max(1, total_tokens)
    print(f"[Alt-Step3] NLL eval: done (nll={nll:.6f})", flush=True)
    return nll


def should_run_global_nll_eval(args: argparse.Namespace, step: int, layer_states: Dict[str, dict]) -> bool:
    if not layer_states:
        return False
    if not any(int(state["bits"]) == 1 for state in layer_states.values()):
        return False
    return should_run_nll_eval(bits=1, args=args, step=step)


def initialize_layer_state(
    key: str,
    ctx: dict,
    args: argparse.Namespace,
) -> dict:
    compute_device = ctx["device"]
    codebooks = ctx["codebooks"]
    qcodes_dict = ctx["qcodes"]
    metas = ctx["metas"]
    calib_s = ctx["calib_s"]
    state = ctx["state"]

    meta = metas[key]
    bits = int(meta["bits"])
    gs = int(meta["group_size"])
    orig_i = int(tuple(meta["orig_shape"])[1])
    clip_pct = float(meta.get("clip_percentile", 0.0) if args.clip_percentile is None else args.clip_percentile)

    w_cpu = state[key].to(torch.float32)
    d_cpu = load_diag_weight(calib_s[key], eps=float(args.eps))
    if d_cpu.numel() != orig_i:
        raise RuntimeError(f"diag weight shape mismatch on {key}: expected {orig_i}, got {d_cpu.numel()}")
    if w_cpu.shape[1] != orig_i:
        raise RuntimeError(f"orig_I mismatch on {key}: meta={orig_i}, weight={w_cpu.shape[1]}")

    w = w_cpu.to(compute_device)
    d = d_cpu.to(compute_device)
    quant_uses_hessian = bool(meta.get("uses_hessian_weighting", False))
    quant_qtype = "hessian-aware" if quant_uses_hessian else "plain"
    hdiag = (d * d) if quant_uses_hessian else None
    step1_codebook = codebooks[key].to(device=compute_device, dtype=torch.float32)
    step1_qcodes = qcodes_dict[key].to(device=compute_device)
    step1_baseline = build_step1_svd_baseline(
        w=w,
        d=d,
        codebook=step1_codebook,
        qcodes=step1_qcodes,
        orig_i=orig_i,
        rank_ab=int(args.rank_ab),
        eps=float(args.eps),
    )
    codebook = step1_codebook
    qcodes = step1_qcodes
    wq = step1_baseline["wq"]
    init_source = "step1_artifact"
    quant_meta = dict(meta)
    quant_meta["clip_percentile"] = float(clip_pct)
    quant_meta["qtype"] = str(quant_qtype)
    quant_meta["init_source"] = str(init_source)

    a, b = weighted_low_rank_fit(w - wq, d, rank=int(args.rank_ab), eps=float(args.eps))
    obj = weighted_objective(w, wq, a, b, d)
    baseline_wq_cpu = step1_baseline["wq"].detach().to(torch.float32).cpu()
    baseline_a_cpu = step1_baseline["A"].detach().to(torch.float32).cpu()
    baseline_b_cpu = step1_baseline["B"].detach().to(torch.float32).cpu()
    ret = {
        "key": key,
        "bits": bits,
        "group_size": gs,
        "clip_percentile": float(clip_pct),
        "baseline_wq": baseline_wq_cpu,
        "baseline_A": baseline_a_cpu,
        "baseline_B": baseline_b_cpu,
        "baseline_objective_weighted": float(step1_baseline["objective"]),
        "init_source": str(init_source),
        "wq": wq.detach().to(torch.float32).cpu(),
        "A": a.detach().to(torch.float32).cpu(),
        "B": b.detach().to(torch.float32).cpu(),
        "objective_weighted": float(obj),
        "d": d_cpu.detach().to(torch.float32).cpu(),
        "quant_meta": quant_meta,
    }
    if args.save_all:
        ret["codebook"] = step1_codebook.detach().to(torch.float32).cpu()
        ret["qcodes"] = step1_qcodes.detach().cpu()

    del w
    del d
    del step1_codebook
    del step1_qcodes
    del a
    del b
    del wq
    if hdiag is not None:
        del hdiag
    if compute_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ret


def advance_layer_state(layer_state: dict, original_weight_cpu: torch.Tensor, args: argparse.Namespace) -> None:
    compute_device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    w = original_weight_cpu.to(device=compute_device, dtype=torch.float32)
    d = layer_state["d"].to(device=compute_device, dtype=torch.float32)
    wq = layer_state["wq"].to(device=compute_device, dtype=torch.float32)
    hdiag = (d * d) if str(layer_state["quant_meta"].get("qtype", args.qtype)) == "hessian-aware" else None

    a, b = weighted_low_rank_fit(w - wq, d, rank=int(args.rank_ab), eps=float(args.eps))
    target = w - (a @ b)
    wq, codebook, qcodes, quant_meta = lloyd_asym_nonuniform_quantize(
        target,
        b=int(layer_state["bits"]),
        group_size=int(layer_state["group_size"]),
        clip_pct=float(layer_state["clip_percentile"]),
        lloyd_iter=int(args.lloyd_iter),
        chunk_groups=int(args.chunk_groups),
        hessian_diag=hdiag,
    )
    a, b = weighted_low_rank_fit(w - wq, d, rank=int(args.rank_ab), eps=float(args.eps))
    obj = weighted_objective(w, wq, a, b, d)

    quant_meta = dict(quant_meta)
    quant_meta["qtype"] = str(layer_state["quant_meta"].get("qtype", args.qtype))
    quant_meta["clip_percentile"] = float(layer_state["clip_percentile"])
    quant_meta["init_source"] = str(layer_state["init_source"])

    layer_state["wq"] = wq.detach().to(torch.float32).cpu()
    layer_state["A"] = a.detach().to(torch.float32).cpu()
    layer_state["B"] = b.detach().to(torch.float32).cpu()
    layer_state["objective_weighted"] = float(obj)
    if args.save_all:
        layer_state["codebook"] = codebook.detach().to(torch.float32).cpu()
        layer_state["qcodes"] = qcodes.detach().cpu()
    layer_state["quant_meta"] = quant_meta

    del w
    del d
    del wq
    del a
    del b
    del target
    del codebook
    del qcodes
    if hdiag is not None:
        del hdiag
    if compute_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def clone_snapshot_state(
    layer_states: Dict[str, dict],
    *,
    include_quant_artifacts: bool = False,
) -> Dict[str, dict]:
    snap: Dict[str, dict] = {}
    for key, state in layer_states.items():
        item = {
            "wq": state["wq"].detach().to(torch.float16).cpu(),
            "A": state["A"].detach().to(torch.float16).cpu(),
            "B": state["B"].detach().to(torch.float16).cpu(),
            "objective_weighted": float(state["objective_weighted"]),
            "bits": int(state["bits"]),
            "group_size": int(state["group_size"]),
            "init_source": str(state["init_source"]),
            "quant_meta": dict(state["quant_meta"]),
        }
        if include_quant_artifacts:
            item["codebook"] = state["codebook"].detach().to(torch.float16).cpu()
            item["qcodes"] = state["qcodes"].detach().cpu()
        snap[key] = item
    return snap


def export_artifacts_from_states(
    layer_states: Dict[str, dict],
    args: argparse.Namespace,
    *,
    selected_outer: int,
    selected_metric_name: Optional[str],
    selected_metric_value: Optional[float],
    per_layer_best: Dict[str, dict],
    global_nll_value: Optional[float] = None,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, Dict[str, torch.Tensor]],
    Dict[str, torch.Tensor],
    Dict[str, Dict[str, torch.Tensor]],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, dict],
    List[dict],
]:
    wdq_out: Dict[str, torch.Tensor] = {}
    ab_out: Dict[str, Dict[str, torch.Tensor]] = {}
    wdq_best_out: Dict[str, torch.Tensor] = {}
    ab_best_out: Dict[str, Dict[str, torch.Tensor]] = {}
    codebook_out: Dict[str, torch.Tensor] = {}
    qcodes_out: Dict[str, torch.Tensor] = {}
    quant_meta_out: Dict[str, dict] = {}
    layer_summaries: List[dict] = []

    for key, state in layer_states.items():
        wdq_cpu = state["wq"].detach().to(torch.float16).cpu()
        a_cpu = state["A"].detach().to(torch.float16).cpu()
        b_cpu = state["B"].detach().to(torch.float16).cpu()
        wdq_out[key] = wdq_cpu
        wdq_best_out[key] = wdq_cpu
        ab_out[key] = {
            "A": a_cpu,
            "B": b_cpu,
            "meta": {
                "rank": int(args.rank_ab),
                "bits": int(state["bits"]),
                "group_size": int(state["group_size"]),
                "qtype": str(args.qtype),
                "objective_weighted_final": float(state["objective_weighted"]),
                "nll_mean_final": (None if global_nll_value is None else float(global_nll_value)),
            },
        }
        ab_best_out[key] = {
            "A": a_cpu,
            "B": b_cpu,
            "meta": {
                "rank": int(args.rank_ab),
                "bits": int(state["bits"]),
                "group_size": int(state["group_size"]),
                "qtype": str(args.qtype),
                "best_outer": int(selected_outer),
                "best_metric_name": selected_metric_name,
                "best_metric_value": selected_metric_value,
                "objective_weighted_best": float(per_layer_best[key]["objective"]),
                "objective_weighted_best_outer": int(per_layer_best[key]["outer"]),
            },
        }
        if "codebook" in state:
            codebook_out[key] = state["codebook"].detach().to(torch.float16).cpu()
        if "qcodes" in state:
            qcodes_out[key] = state["qcodes"].detach().cpu()
        quant_meta_out[key] = dict(state["quant_meta"])
        layer_summaries.append(
            {
                "layer": key,
                "bits": int(state["bits"]),
                "group_size": int(state["group_size"]),
                "init_source": str(state["init_source"]),
                "selection_metric": str(args.select_metric),
                "objective_weighted_final": float(state["objective_weighted"]),
                "objective_weighted_best": float(per_layer_best[key]["objective"]),
                "objective_weighted_best_outer": int(per_layer_best[key]["outer"]),
                "nll_mean_final": (None if global_nll_value is None else float(global_nll_value)),
                "best_metric_name": selected_metric_name,
                "best_metric_value": selected_metric_value,
                "best_outer": int(selected_outer),
            }
        )

    return (
        wdq_out,
        ab_out,
        wdq_best_out,
        ab_best_out,
        codebook_out,
        qcodes_out,
        quant_meta_out,
        layer_summaries,
    )

def main() -> None:
    ap = argparse.ArgumentParser("Alt Step3 - Alternating Lloyd-Max + Low-Rank Compensation")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--step1_dir", required=True, help="step_1_quantize output dir")
    ap.add_argument("--calib_s", required=True, help="step_2_calib calib_sqrtdiag.pt")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto", help='Model load placement: e.g. "auto" or "none"')
    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--qtype", default="hessian-aware", choices=["plain", "hessian-aware"])
    ap.add_argument("--rank_ab", type=int, default=64)
    ap.add_argument("--outer_loops", type=int, default=4)
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)
    ap.add_argument("--clip_percentile", type=float, default=None)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=0)
    ap.add_argument("--save_every_layer", action="store_true")
    ap.add_argument("--eval_every", type=int, default=0, help="Run mini NLL eval every K outer steps (0 disables)")
    ap.add_argument("--select_metric", default="objective", choices=["objective", "nll"])
    ap.add_argument("--best_on_nll", action="store_true", help="Alias for --select_metric nll")
    ap.add_argument("--nll_dataset", default="wikitext")
    ap.add_argument("--nll_dataset_config", default="wikitext-2-raw-v1")
    ap.add_argument("--nll_split", default="test")
    ap.add_argument("--nll_nsamples", type=int, default=8)
    ap.add_argument("--nll_seq_len", type=int, default=512)
    ap.add_argument("--nll_batch_size", type=int, default=1)
    ap.add_argument("--nll_use_streaming", action="store_true")
    ap.add_argument("--nll_eval_device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--save_all",
        action="store_true",
        help="Also save codebook/qcodes/quant_meta artifacts and summary.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.best_on_nll:
        args.select_metric = "nll"

    set_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    ctx = load_context(args)
    eval_ctx = ctx.get("eval_ctx")

    def save_outputs(
        current_states: Dict[str, dict],
        best_states: Dict[str, dict],
        per_layer_best: Dict[str, dict],
        best_outer: int,
        best_metric_name: Optional[str],
        best_metric_value: Optional[float],
        final_global_nll: Optional[float],
    ) -> Tuple[List[dict], dict]:
        (
            wdq_out,
            ab_out,
            _,
            _,
            codebook_out,
            qcodes_out,
            quant_meta_out,
            layer_summaries,
        ) = export_artifacts_from_states(
            current_states,
            args,
            selected_outer=best_outer,
            selected_metric_name=best_metric_name,
            selected_metric_value=best_metric_value,
            per_layer_best=per_layer_best,
            global_nll_value=final_global_nll,
        )
        (
            wdq_best_out,
            _,
            _,
            ab_best_out,
            _,
            _,
            _,
            _,
        ) = export_artifacts_from_states(
            best_states,
            args,
            selected_outer=best_outer,
            selected_metric_name=best_metric_name,
            selected_metric_value=best_metric_value,
            per_layer_best=per_layer_best,
            global_nll_value=best_metric_value if best_metric_name == "nll" else None,
        )

        torch.save(wdq_out, out_dir / "wdq_star.pt")
        torch.save(ab_out, out_dir / "low_rank_ab.pt")
        torch.save(wdq_best_out, out_dir / "wdq_star_best.pt")
        torch.save(ab_best_out, out_dir / "low_rank_ab_best.pt")
        if args.save_all:
            torch.save(codebook_out, out_dir / "codebook_star.pt")
            torch.save(qcodes_out, out_dir / "qcodes_star.pt")
            torch.save(quant_meta_out, out_dir / "quant_meta_star.pt")
        return layer_summaries, quant_meta_out

    layer_states: Dict[str, dict] = {}
    per_layer_best: Dict[str, dict] = {}
    baseline_layer_states: Dict[str, dict] = {}
    t0 = time.time()

    for idx, key in enumerate(ctx["keys"], start=1):
        print(f"[Alt-Step3] ({idx}/{len(ctx['keys'])}) init: {key}", flush=True)
        state = initialize_layer_state(key=key, ctx=ctx, args=args)
        baseline_layer_states[key] = {
            "wq": state["baseline_wq"].detach().to(torch.float16).cpu(),
            "A": state["baseline_A"].detach().to(torch.float16).cpu(),
            "B": state["baseline_B"].detach().to(torch.float16).cpu(),
            "objective_weighted": float(state["baseline_objective_weighted"]),
            "bits": int(state["bits"]),
            "group_size": int(state["group_size"]),
            "init_source": str(state["init_source"]),
            "quant_meta": dict(state["quant_meta"]),
        }
        del state["baseline_wq"]
        del state["baseline_A"]
        del state["baseline_B"]
        layer_states[key] = state
        per_layer_best[key] = {
            "objective": float(state["baseline_objective_weighted"]),
            "outer": -1,
        }
        append_jsonl(
            metrics_path,
            {
                "layer": key,
                "outer": -1,
                "phase": "init",
                "objective_weighted": float(state["baseline_objective_weighted"]),
                "bits": int(state["bits"]),
                "group_size": int(state["group_size"]),
                "rank_ab": int(args.rank_ab),
                "qtype": args.qtype,
                "init_source": "step1_artifact",
                "nll_mean": None,
                "selection_metric": str(args.select_metric),
            },
        )

    for stale_key in ("codebooks", "qcodes", "metas", "calib_s"):
        if stale_key in ctx:
            del ctx[stale_key]
    gc.collect()

    global_objective = sum(state["baseline_objective_weighted"] for state in layer_states.values()) / max(1, len(layer_states))
    global_nll = None
    if eval_ctx is not None and should_run_global_nll_eval(args=args, step=0, layer_states=baseline_layer_states):
        print("[Alt-Step3] init: starting global NLL eval", flush=True)
        global_nll = evaluate_global_candidate_nll(eval_ctx=eval_ctx, layer_states=baseline_layer_states)
    append_jsonl(
        metrics_path,
        {
            "layer": "__global__",
            "outer": -1,
            "phase": "init",
            "objective_weighted_mean": float(global_objective),
            "nll_mean": (None if global_nll is None else float(global_nll)),
            "selection_metric": str(args.select_metric),
            "num_layers": len(layer_states),
        },
    )

    best_outer = -1
    best_metric_name = "nll" if args.select_metric == "nll" and global_nll is not None else "objective"
    best_metric_value = float(global_nll) if best_metric_name == "nll" else float(global_objective)
    best_snapshot = dict(baseline_layer_states)
    final_global_nll = global_nll
    global_history = [
        {
            "outer": -1,
            "objective_weighted_mean": float(global_objective),
            "nll_mean": (None if global_nll is None else float(global_nll)),
        }
    ]
    del baseline_layer_states
    gc.collect()

    for outer in range(int(args.outer_loops)):
        print(
            f"[Alt-Step3] outer {outer + 1}/{int(args.outer_loops)}: updating {len(layer_states)} layers",
            flush=True,
        )
        for idx, key in enumerate(ctx["keys"], start=1):
            advance_layer_state(layer_states[key], original_weight_cpu=ctx["state"][key], args=args)
            state = layer_states[key]
            if float(state["objective_weighted"]) < float(per_layer_best[key]["objective"]):
                per_layer_best[key] = {
                    "objective": float(state["objective_weighted"]),
                    "outer": int(outer),
                }
            append_jsonl(
                metrics_path,
                {
                    "layer": key,
                    "outer": int(outer),
                    "phase": "alternating",
                    "objective_weighted": float(state["objective_weighted"]),
                    "bits": int(state["bits"]),
                    "group_size": int(state["group_size"]),
                    "rank_ab": int(args.rank_ab),
                    "qtype": args.qtype,
                    "nll_mean": None,
                    "selection_metric": str(args.select_metric),
                },
            )
            if torch.cuda.is_available() and (idx % 8 == 0 or idx == len(ctx["keys"])):
                torch.cuda.empty_cache()
            if idx % 8 == 0 or idx == len(ctx["keys"]):
                gc.collect()

        global_objective = sum(state["objective_weighted"] for state in layer_states.values()) / max(1, len(layer_states))
        global_nll = None
        step = int(outer) + 1
        if eval_ctx is not None and should_run_global_nll_eval(args=args, step=step, layer_states=layer_states):
            print(f"[Alt-Step3] outer {outer + 1}: starting global NLL eval", flush=True)
            global_nll = evaluate_global_candidate_nll(eval_ctx=eval_ctx, layer_states=layer_states)
            final_global_nll = global_nll

        append_jsonl(
            metrics_path,
            {
                "layer": "__global__",
                "outer": int(outer),
                "phase": "alternating",
                "objective_weighted_mean": float(global_objective),
                "nll_mean": (None if global_nll is None else float(global_nll)),
                "selection_metric": str(args.select_metric),
                "num_layers": len(layer_states),
            },
        )
        global_history.append(
            {
                "outer": int(outer),
                "objective_weighted_mean": float(global_objective),
                "nll_mean": (None if global_nll is None else float(global_nll)),
            }
        )

        current_metric_name = "objective"
        current_metric_value = float(global_objective)
        if args.select_metric == "nll" and global_nll is not None:
            current_metric_name = "nll"
            current_metric_value = float(global_nll)

        if float(current_metric_value) < float(best_metric_value):
            best_outer = int(outer)
            best_metric_name = str(current_metric_name)
            best_metric_value = float(current_metric_value)
            best_snapshot = clone_snapshot_state(layer_states)

        if args.save_every_layer:
            save_outputs(
                current_states=layer_states,
                best_states=best_snapshot,
                per_layer_best=per_layer_best,
                best_outer=best_outer,
                best_metric_name=best_metric_name,
                best_metric_value=best_metric_value,
                final_global_nll=final_global_nll,
            )

    layer_summaries, quant_meta_out = save_outputs(
        current_states=layer_states,
        best_states=best_snapshot,
        per_layer_best=per_layer_best,
        best_outer=best_outer,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        final_global_nll=final_global_nll,
    )

    final_mean = sum(x["objective_weighted_final"] for x in layer_summaries) / max(1, len(layer_summaries))
    best_mean = sum(x["objective_weighted_best"] for x in layer_summaries) / max(1, len(layer_summaries))
    summary = {
        "model_id": args.model_id,
        "revision": args.revision,
        "step1_dir": str(Path(args.step1_dir).resolve()),
        "calib_s": str(Path(args.calib_s).resolve()),
        "out_dir": str(out_dir),
        "qtype": str(args.qtype),
        "rank_ab": int(args.rank_ab),
        "outer_loops": int(args.outer_loops),
        "eval_every": int(args.eval_every),
        "select_metric": str(args.select_metric),
        "lloyd_iter": int(args.lloyd_iter),
        "chunk_groups": int(args.chunk_groups),
        "clip_percentile_override": args.clip_percentile,
        "num_layers": len(layer_summaries),
        "objective_weighted_final_mean": float(final_mean),
        "objective_weighted_best_mean": float(best_mean),
        "global_best_outer": int(best_outer),
        "global_best_metric_name": str(best_metric_name),
        "global_best_metric_value": float(best_metric_value),
        "global_nll_final": (None if final_global_nll is None else float(final_global_nll)),
        "global_history": global_history,
        "elapsed_sec": time.time() - t0,
        "layers": layer_summaries,
    }
    if args.save_all:
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Alt-Step3] saved:", flush=True)
    print(f"  wdq*: {out_dir / 'wdq_star.pt'}", flush=True)
    print(f"  AB*:  {out_dir / 'low_rank_ab.pt'}", flush=True)
    print(f"  best outer: {best_outer} ({best_metric_name}={best_metric_value:.6f})", flush=True)
    if args.save_all:
        print(f"  quant_meta*: {out_dir / 'quant_meta_star.pt'}", flush=True)
        print(f"  summary: {out_dir / 'summary.json'}", flush=True)

    if eval_ctx is not None:
        del eval_ctx["model"]
        del eval_ctx["tokenizer"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


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
class Step03AlternatingConfig:
    model_id: str
    step1_dir: str
    calib_s: str
    out_dir: str
    revision: Optional[str] = None
    trust_remote_code: bool = False
    device: str = "cuda"
    model_device_map: str = "auto"
    dtype_w: str = "fp16"
    qtype: str = "hessian-aware"
    rank_ab: int = 64
    outer_loops: int = 4
    lloyd_iter: int = 12
    chunk_groups: int = 4096
    clip_percentile: Optional[float] = None
    eps: float = 1e-8
    layer_regex: Optional[str] = None
    max_layers: int = 0
    save_every_layer: bool = False
    eval_every: int = 0
    select_metric: str = "objective"
    best_on_nll: bool = False
    nll_dataset: str = "wikitext"
    nll_dataset_config: Optional[str] = "wikitext-2-raw-v1"
    nll_split: str = "test"
    nll_nsamples: int = 8
    nll_seq_len: int = 512
    nll_batch_size: int = 1
    nll_use_streaming: bool = False
    nll_eval_device: str = "cpu"
    seed: int = 42
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


def build_command(cfg: Step03AlternatingConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_id",
        str(cfg.model_id),
        "--step1_dir",
        str(cfg.step1_dir),
        "--calib_s",
        str(cfg.calib_s),
        "--out_dir",
        str(cfg.out_dir),
        "--device",
        str(cfg.device),
        "--model_device_map",
        str(cfg.model_device_map),
        "--dtype_w",
        str(cfg.dtype_w),
        "--qtype",
        str(cfg.qtype),
        "--rank_ab",
        str(int(cfg.rank_ab)),
        "--outer_loops",
        str(int(cfg.outer_loops)),
        "--lloyd_iter",
        str(int(cfg.lloyd_iter)),
        "--chunk_groups",
        str(int(cfg.chunk_groups)),
        "--eps",
        str(float(cfg.eps)),
        "--max_layers",
        str(int(cfg.max_layers)),
        "--eval_every",
        str(int(cfg.eval_every)),
        "--select_metric",
        str(cfg.select_metric),
        "--nll_dataset",
        str(cfg.nll_dataset),
        "--nll_split",
        str(cfg.nll_split),
        "--nll_nsamples",
        str(int(cfg.nll_nsamples)),
        "--nll_seq_len",
        str(int(cfg.nll_seq_len)),
        "--nll_batch_size",
        str(int(cfg.nll_batch_size)),
        "--nll_eval_device",
        str(cfg.nll_eval_device),
        "--seed",
        str(int(cfg.seed)),
    ]
    if cfg.revision:
        cmd += ["--revision", str(cfg.revision)]
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    if cfg.clip_percentile is not None:
        cmd += ["--clip_percentile", str(float(cfg.clip_percentile))]
    if cfg.layer_regex:
        cmd += ["--layer_regex", str(cfg.layer_regex)]
    if cfg.save_every_layer:
        cmd.append("--save_every_layer")
    if cfg.best_on_nll:
        cmd.append("--best_on_nll")
    if cfg.nll_dataset_config is not None:
        cmd += ["--nll_dataset_config", str(cfg.nll_dataset_config)]
    if cfg.nll_use_streaming:
        cmd.append("--nll_use_streaming")
    return cmd


def run(cfg: Step03AlternatingConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp
