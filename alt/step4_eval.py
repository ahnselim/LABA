#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alt Step4 Eval - evaluate `step_3_alternating.py` outputs.

Supports:
  - Wdq-only baseline
  - Wdq + AB correction

Inputs:
  1) `--step3_dir`:
       auto-resolve `wdq_star_best.pt` and `low_rank_ab_best.pt` first,
       then fallback to non-best artifacts, including sparse+dense Step3 exports
  2) explicit artifact paths:
       `--wdq_star_path` and optional `--low_rank_ab_path`

Outputs:
  - console metrics
  - optional JSON summary

CUDA_VISIBLE_DEVICES=2 \
python step4_eval.py \
  --model_name meta-llama/Llama-3.1-8B \
  --wdq_star_path ./output/llama3_8b_64/step3_alt/1bit/wdq_star.pt \
  --low_rank_ab_path ./output/llama3_8b_64/step3_alt/1bit/low_rank_ab.pt \
  --device cuda:0 \
  --compare_wdq_only
  
CUDA_VISIBLE_DEVICES=1 \
python step4_eval.py \
  --model_name Qwen/Qwen3-8B \
  --wdq_star_path ./output/qwen3_8b_64/step3_alt/1bit_10/wdq_star.pt \
  --low_rank_ab_path ./output/qwen3_8b_64/step3_alt/1bit_10/low_rank_ab.pt \
  --device cuda:0 \
  --compare_wdq_only \
  --trust_remote_code

"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_parent_module(model: nn.Module, name: str):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


class AddLowRankCorrection(nn.Module):
    def __init__(self, inner: nn.Module, A: torch.Tensor, B: torch.Tensor, alpha: float = 1.0):
        super().__init__()
        self.inner = inner
        self.register_buffer("A", A.to(torch.float16), persistent=False)
        self.register_buffer("B", B.to(torch.float16), persistent=False)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha == 0.0:
            return z
        r = F.linear(x, self.B)
        corr = F.linear(r, self.A)
        return z.add_(corr, alpha=self.alpha)


class AddLowRankCorrectionFP32(nn.Module):
    def __init__(self, inner: nn.Module, A: torch.Tensor, B: torch.Tensor, alpha: float = 1.0):
        super().__init__()
        self.inner = inner
        self.register_buffer("A", A.to(torch.float16), persistent=False)
        self.register_buffer("B", B.to(torch.float16), persistent=False)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha == 0.0:
            return z
        x32 = x.to(torch.float32)
        r = F.linear(x32, self.B.to(torch.float32))
        corr = F.linear(r, self.A.to(torch.float32))
        return (z.to(torch.float32) + corr * self.alpha).to(x.dtype)


def _sparse_mm_from_coo(
    x: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_values: torch.Tensor,
    sparse_shape: Tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    x_flat = x.reshape(-1, x.shape[-1])
    sp = torch.sparse_coo_tensor(
        sparse_indices,
        sparse_values,
        size=sparse_shape,
        device=x_flat.device,
    ).coalesce()
    corr_t = torch.sparse.mm(sp.to(torch.float32), x_flat.to(torch.float32).transpose(0, 1))
    corr = corr_t.transpose(0, 1).reshape(*x.shape[:-1], sparse_shape[0])
    return corr.to(out_dtype)


class AddSparseAndLowRankCorrection(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        A: torch.Tensor,
        B: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
        sparse_values: Optional[torch.Tensor] = None,
        sparse_shape: Optional[Tuple[int, int]] = None,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.inner = inner
        self.register_buffer("A", A.to(torch.float16), persistent=False)
        self.register_buffer("B", B.to(torch.float16), persistent=False)
        self.alpha = float(alpha)
        if sparse_indices is not None and sparse_values is not None and sparse_shape is not None:
            self.register_buffer("sparse_indices", sparse_indices.to(torch.long), persistent=False)
            self.register_buffer("sparse_values", sparse_values.to(torch.float16), persistent=False)
            self.sparse_shape = tuple(int(x) for x in sparse_shape)
        else:
            self.sparse_indices = None
            self.sparse_values = None
            self.sparse_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha == 0.0:
            return z
        r = F.linear(x, self.B)
        corr = F.linear(r, self.A)
        if self.sparse_indices is not None and self.sparse_values is not None and self.sparse_shape is not None:
            corr = corr + _sparse_mm_from_coo(
                x=x,
                sparse_indices=self.sparse_indices,
                sparse_values=self.sparse_values,
                sparse_shape=self.sparse_shape,
                out_dtype=corr.dtype,
            )
        return z.add_(corr, alpha=self.alpha)


class AddSparseAndLowRankCorrectionFP32(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        A: torch.Tensor,
        B: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
        sparse_values: Optional[torch.Tensor] = None,
        sparse_shape: Optional[Tuple[int, int]] = None,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.inner = inner
        self.register_buffer("A", A.to(torch.float16), persistent=False)
        self.register_buffer("B", B.to(torch.float16), persistent=False)
        self.alpha = float(alpha)
        if sparse_indices is not None and sparse_values is not None and sparse_shape is not None:
            self.register_buffer("sparse_indices", sparse_indices.to(torch.long), persistent=False)
            self.register_buffer("sparse_values", sparse_values.to(torch.float16), persistent=False)
            self.sparse_shape = tuple(int(x) for x in sparse_shape)
        else:
            self.sparse_indices = None
            self.sparse_values = None
            self.sparse_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha == 0.0:
            return z
        x32 = x.to(torch.float32)
        r = F.linear(x32, self.B.to(torch.float32))
        corr = F.linear(r, self.A.to(torch.float32))
        if self.sparse_indices is not None and self.sparse_values is not None and self.sparse_shape is not None:
            corr = corr + _sparse_mm_from_coo(
                x=x32,
                sparse_indices=self.sparse_indices,
                sparse_values=self.sparse_values,
                sparse_shape=self.sparse_shape,
                out_dtype=torch.float32,
            )
        return (z.to(torch.float32) + corr * self.alpha).to(x.dtype)


def _unwrap_base_linear(module: nn.Module) -> nn.Module:
    while isinstance(
        module,
        (
            AddLowRankCorrection,
            AddLowRankCorrectionFP32,
            AddSparseAndLowRankCorrection,
            AddSparseAndLowRankCorrectionFP32,
        ),
    ):
        module = module.inner
    return module


@torch.no_grad()
def apply_wdq_star(model: nn.Module, wdq: Dict[str, torch.Tensor]):
    for wkey, Wdq in wdq.items():
        if not (isinstance(wkey, str) and wkey.endswith(".weight") and getattr(Wdq, "ndim", 0) == 2):
            continue
        module_name = wkey[:-7]
        try:
            parent, attr = get_parent_module(model, module_name)
            current = getattr(parent, attr, None)
        except AttributeError:
            continue
        if current is None:
            continue
        inner = _unwrap_base_linear(current)
        if not hasattr(inner, "weight"):
            continue
        if inner.weight.shape != Wdq.shape:
            continue
        inner.weight.data.copy_(Wdq.to(device=inner.weight.device, dtype=inner.weight.dtype))


@torch.no_grad()
def build_B_from_Bbar(Bbar: torch.Tensor, inv_s: torch.Tensor) -> torch.Tensor:
    return (Bbar.to(torch.float32) * inv_s.to(torch.float32).unsqueeze(0)).to(Bbar.dtype)


@torch.no_grad()
def build_B_from_Bbar_lowrank(Bbar: torch.Tensor, U: torch.Tensor, sqrt_lambda: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    inv_sqrt = 1.0 / sqrt_lambda.to(torch.float32).clamp_min(eps)
    T = (inv_sqrt.unsqueeze(1) * U.to(torch.float32).T).contiguous()
    B = (Bbar.to(torch.float32) @ T).to(Bbar.dtype)
    return B


@torch.no_grad()
def patch_layerwise_ab_from_step2p5(
    model: nn.Module,
    low_rank_ab: Optional[Dict[str, Any]] = None,
    low_rank_abbar: Optional[Dict[str, Any]] = None,
    calib_s: Optional[Dict[str, Any]] = None,
    calib_h_lowrank: Optional[Dict[str, Any]] = None,
    alpha: float = 1.0,
    ab_compute: str = "fp16",
):
    assert (low_rank_ab is not None) or (low_rank_abbar is not None), "Need low_rank_ab or low_rank_abbar"
    if ab_compute not in {"fp16", "fp32"}:
        raise ValueError(f"Unknown ab_compute: {ab_compute} (expected: fp16|fp32)")

    keys = set()
    if low_rank_ab is not None:
        keys |= set(low_rank_ab.keys())
    if low_rank_abbar is not None:
        keys |= set(low_rank_abbar.keys())
    keys = sorted([k for k in keys if isinstance(k, str) and k.endswith(".weight")])

    for wkey in tqdm(keys, desc="Patching AB*"):
        module_name = wkey[:-7]
        A = None
        B = None
        sparse_indices = None
        sparse_values = None
        sparse_shape = None

        if low_rank_ab is not None and wkey in low_rank_ab:
            item = low_rank_ab[wkey]
            if isinstance(item, dict) and ("A" in item) and ("B" in item):
                A, B = item["A"], item["B"]
                if "sparse_indices" in item and "sparse_values" in item and "sparse_shape" in item:
                    sparse_indices = item["sparse_indices"]
                    sparse_values = item["sparse_values"]
                    sparse_shape = tuple(int(v) for v in item["sparse_shape"])
                if ("u" in item) or ("v" in item):
                    if ("u" not in item) or ("v" not in item):
                        continue
                    if calib_s is None or wkey not in calib_s or "inv_s" not in calib_s[wkey]:
                        raise ValueError(
                            f"low_rank_ab[{wkey}] has u,v (Step3 uv-ab) but missing calib_s inv_s. "
                            "Pass --calib_s_path (Step2 calib_sqrtdiag.pt)."
                        )
                    u = item["u"]
                    v = item["v"]
                    inv_s = calib_s[wkey]["inv_s"]
                    A = (u.to(torch.float32).unsqueeze(1) * A.to(torch.float32)).to(A.dtype)
                    B = (B.to(torch.float32) * (v.to(torch.float32) * inv_s.to(torch.float32)).unsqueeze(0)).to(B.dtype)

        if (A is None or B is None) and (low_rank_abbar is not None) and (wkey in low_rank_abbar):
            item = low_rank_abbar[wkey]
            if isinstance(item, dict) and ("A" in item) and ("Bbar" in item):
                A = item["A"]
                Bbar = item["Bbar"]
                meta = item.get("meta", {}) if isinstance(item, dict) else {}
                h_weighting = meta.get("h_weighting", None)
                try:
                    parent0, attr0 = get_parent_module(model, module_name)
                    cur0 = getattr(parent0, attr0, None)
                except AttributeError:
                    continue
                if cur0 is None:
                    continue
                inner0 = _unwrap_base_linear(cur0)
                if not hasattr(inner0, "weight"):
                    continue
                _, I0 = inner0.weight.shape
                if (h_weighting == "diag") or (Bbar.ndim == 2 and Bbar.shape[1] == I0):
                    if calib_s is None or wkey not in calib_s or "inv_s" not in calib_s[wkey]:
                        continue
                    inv_s = calib_s[wkey]["inv_s"]
                    B = build_B_from_Bbar(Bbar, inv_s)
                else:
                    if calib_h_lowrank is None or wkey not in calib_h_lowrank:
                        continue
                    U = calib_h_lowrank[wkey]["U"]
                    sqrt_l = calib_h_lowrank[wkey]["sqrt_lambda"]
                    if U.ndim != 2 or sqrt_l.ndim != 1:
                        continue
                    if U.shape[0] != I0:
                        continue
                    if Bbar.shape[1] != U.shape[1] or sqrt_l.numel() != U.shape[1]:
                        continue
                    B = build_B_from_Bbar_lowrank(Bbar, U, sqrt_l)

        if A is None or B is None:
            continue
        try:
            parent, attr = get_parent_module(model, module_name)
            current = getattr(parent, attr, None)
        except AttributeError:
            continue
        if current is None:
            continue
        inner = _unwrap_base_linear(current)
        if not hasattr(inner, "weight"):
            continue
        O, I = inner.weight.shape
        if A.ndim != 2 or B.ndim != 2:
            continue
        if A.shape[0] != O or B.shape[1] != I or A.shape[1] != B.shape[0]:
            continue

        has_sparse = sparse_indices is not None and sparse_values is not None and sparse_shape is not None
        if has_sparse and tuple(sparse_shape) != (int(O), int(I)):
            continue

        comp_dtype = torch.float32 if ab_compute == "fp32" else inner.weight.dtype
        A_dev = A.to(device=inner.weight.device, dtype=comp_dtype)
        B_dev = B.to(device=inner.weight.device, dtype=comp_dtype)
        delta = A_dev @ B_dev

        if has_sparse:
            sp = torch.sparse_coo_tensor(
                sparse_indices.to(device=inner.weight.device, dtype=torch.long),
                sparse_values.to(device=inner.weight.device, dtype=comp_dtype),
                size=sparse_shape,
                device=inner.weight.device,
            ).coalesce()
            delta = delta + sp.to_dense()

        inner.weight.data.add_(delta.to(dtype=inner.weight.dtype), alpha=float(alpha))
    return model


@torch.no_grad()
def evaluate_ppl_wikitext2(
    model,
    tokenizer,
    device,
    label: str,
    stride: int = 2048,
    max_tokens: int = 0,
):
    model.eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids
    seq_len = input_ids.size(1)
    if max_tokens and max_tokens > 0 and seq_len > int(max_tokens):
        seq_len = int(max_tokens)
        input_ids = input_ids[:, :seq_len].contiguous()

    total_loss, total_tokens = 0.0, 0
    start = time.time()
    loss_fct = nn.CrossEntropyLoss()

    with torch.inference_mode():
        for i in tqdm(range(0, seq_len, stride), desc=f"🚀 PPL {label}"):
            begin, end = i, min(i + stride, seq_len)
            if end - begin <= 1:
                continue
            x = input_ids[:, begin:end].to(device)
            out = model(x, use_cache=False)
            logits = out.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = x[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ntok = shift_labels.numel()
            total_loss += loss.item() * ntok
            total_tokens += ntok

    ppl = math.exp(total_loss / max(1, total_tokens))
    elapsed = time.time() - start
    return ppl, elapsed


def _cuda_sync(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def temp_generation_overrides(model, **overrides):
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        yield
        return
    old_vals = {k: getattr(gen_cfg, k, None) for k in overrides}
    for k, v in overrides.items():
        try:
            setattr(gen_cfg, k, v)
        except Exception:
            pass
    try:
        yield
    finally:
        for k, v in old_vals.items():
            try:
                setattr(gen_cfg, k, v)
            except Exception:
                pass


def _get_sequences_from_generate(output):
    return output.sequences if hasattr(output, "sequences") else output


@torch.no_grad()
def measure_generation_metrics(
    model,
    tokenizer,
    device,
    prompts,
    max_new_tokens=50,
    do_sample=False,
    num_beams=1,
    temperature=1.0,
    top_p=1.0,
    repeats=1,
):
    model.eval()
    pad_id = tokenizer.eos_token_id
    override_kwargs = {"temperature": 1.0, "top_p": 1.0} if not do_sample else {}

    with temp_generation_overrides(model, **override_kwargs):
        try:
            w = tokenizer(prompts[0], return_tensors="pt", truncation=True, max_length=512).to(device)
            if w["input_ids"].dim() == 1:
                w["input_ids"] = w["input_ids"].unsqueeze(0)
            if "attention_mask" in w and w["attention_mask"].dim() == 1:
                w["attention_mask"] = w["attention_mask"].unsqueeze(0)
            _ = model.generate(**w, max_new_tokens=1, use_cache=True)
            _cuda_sync(device)
        except Exception:
            pass

        ttfb_ms, tokps, total_times, gen_ntoks = [], [], [], []
        gen_kwargs = dict(
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=pad_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

        for _ in range(repeats):
            for p in prompts:
                inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(device)
                if inp["input_ids"].dim() == 1:
                    inp["input_ids"] = inp["input_ids"].unsqueeze(0)
                if "attention_mask" in inp and inp["attention_mask"].dim() == 1:
                    inp["attention_mask"] = inp["attention_mask"].unsqueeze(0)

                _cuda_sync(device)
                t0 = perf_counter()
                model.generate(**inp, max_new_tokens=1, **gen_kwargs)
                _cuda_sync(device)
                ttfb_ms.append((perf_counter() - t0) * 1000.0)

                _cuda_sync(device)
                t1 = perf_counter()
                out = model.generate(**inp, max_new_tokens=max_new_tokens, **gen_kwargs)
                _cuda_sync(device)
                tt = perf_counter() - t1

                newt = _get_sequences_from_generate(out).shape[1] - inp["input_ids"].shape[1]
                tokps.append((newt / tt) if tt > 0 else 0.0)
                total_times.append(tt)
                gen_ntoks.append(newt)

    return {
        "ttfb_ms_mean": mean(ttfb_ms) if ttfb_ms else 0.0,
        "ttfb_ms_median": median(ttfb_ms) if ttfb_ms else 0.0,
        "tok_s_mean": mean(tokps) if tokps else 0.0,
        "tok_s_median": median(tokps) if tokps else 0.0,
        "avg_total_time_s": mean(total_times) if total_times else 0.0,
        "avg_new_tokens": mean(gen_ntoks) if gen_ntoks else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Alt Step4 Eval - Wdq* and AB* evaluation")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--step3_dir", default=None, help="Alt step3 output dir")
    ap.add_argument("--wdq_star_path", default=None, help="Explicit wdq_star(.pt) path")
    ap.add_argument("--low_rank_ab_path", default=None, help="Explicit low_rank_ab(.pt) path")
    ap.add_argument("--calib_s_path", default=None, help="Optional calib_s for uv-ab style artifacts")

    ap.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--model_dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--ab_compute", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--ab_alpha", type=float, default=1.0)
    ap.add_argument("--compare_wdq_only", action="store_true", help="Also run Wdq-only baseline before AB")
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--ppl_stride", type=int, default=2048)
    ap.add_argument("--ppl_max_tokens", type=int, default=0)
    ap.add_argument("--gen_max_new_tokens", type=int, default=50)
    ap.add_argument("--gen_repeats", type=int, default=1)
    ap.add_argument("--gen_do_sample", action="store_true")
    ap.add_argument("--gen_num_beams", type=int, default=1)
    ap.add_argument("--gen_temperature", type=float, default=1.0)
    ap.add_argument("--gen_top_p", type=float, default=1.0)
    ap.add_argument("--save_json", default=None)
    return ap.parse_args()


def _torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported model_dtype: {name}")


def _set_ab_alpha(model: torch.nn.Module, alpha: float) -> int:
    count = 0
    for m in model.modules():
        if hasattr(m, "alpha") and hasattr(m, "inner"):
            m.alpha = float(alpha)
            count += 1
    return count


def _resolve_step3_artifacts(
    step3_dir: Path,
) -> Path:
    wdq_path = step3_dir / "wdq_star_best.pt"
    if not wdq_path.exists():
        wdq_path = step3_dir / "wdq_star.pt"

    if not wdq_path.exists():
        raise FileNotFoundError(f"wdq artifact not found under {step3_dir}")
    return wdq_path


def _load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
):
    print(f"📥 Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"📥 Loading base model: {model_name} (dtype={torch_dtype}, device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model = model.to(device)
    return model, tok


def main() -> None:
    args = _parse_args()

    if args.step3_dir is None and args.wdq_star_path is None:
        raise ValueError("Need --step3_dir or --wdq_star_path")

    wdq_path = None
    ab_path = None
    if args.step3_dir is not None:
        step3_dir = Path(args.step3_dir).resolve()
        if not step3_dir.exists():
            raise FileNotFoundError(f"step3_dir not found: {step3_dir}")
        wdq_path = _resolve_step3_artifacts(step3_dir)

    if args.wdq_star_path is not None:
        wdq_path = Path(args.wdq_star_path).resolve()
    if args.low_rank_ab_path is not None:
        ab_path = Path(args.low_rank_ab_path).resolve()

    if wdq_path is None or not wdq_path.exists():
        raise FileNotFoundError(f"wdq_star path not found: {wdq_path}")
    if ab_path is not None and not ab_path.exists():
        raise FileNotFoundError(f"low_rank_ab path not found: {ab_path}")

    device = torch.device(args.device)
    model_dtype = _torch_dtype_from_name(args.model_dtype)

    wdq_star: Dict[str, torch.Tensor] = torch.load(wdq_path, map_location="cpu")

    low_rank_ab: Optional[Dict[str, Any]] = None
    calib_s = None
    if ab_path is not None:
        low_rank_ab = torch.load(ab_path, map_location="cpu")

    if args.calib_s_path is not None:
        calib_path = Path(args.calib_s_path).resolve()
        calib_s = torch.load(calib_path, map_location="cpu")

    if low_rank_ab is not None:
        has_uvab = any(
            isinstance(item, dict) and (("u" in item) or ("v" in item))
            for item in low_rank_ab.values()
        )
        if has_uvab and calib_s is None:
            raise ValueError(
                "Detected uv-ab artifact but --calib_s_path is missing."
            )

    model = None
    tok = None
    results: Dict[str, Any] = {
        "model_name": args.model_name,
        "wdq_star_path": str(wdq_path),
        "low_rank_ab_path": (str(ab_path) if ab_path is not None else None),
        "device": str(device),
        "model_dtype": args.model_dtype,
        "ab_compute": args.ab_compute,
        "ab_alpha": float(args.ab_alpha),
    }

    prompts = [
        "The history of machine learning began",
        "Large language models are useful because",
        "In a transformer layer, attention works by",
    ]

    try:
        model, tok = _load_model_and_tokenizer(
            model_name=args.model_name,
            device=device,
            torch_dtype=model_dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        apply_wdq_star(model, wdq_star)

        if args.compare_wdq_only or low_rank_ab is None:
            print("👾 [Eval] running Wdq-only")
            ppl_wdq, ppl_wdq_sec = evaluate_ppl_wikitext2(
                model,
                tok,
                device,
                label="Wdq*",
                stride=int(args.ppl_stride),
                max_tokens=int(args.ppl_max_tokens),
            )
            results["wdq_only"] = {
                "ppl": float(ppl_wdq),
                "elapsed_sec": float(ppl_wdq_sec),
            }
            print(f"✌ [Eval] Wdq-only done | PPL={ppl_wdq:.4f}")

        if low_rank_ab is not None:
            print("[Eval] applying AB correction")
            model = patch_layerwise_ab_from_step2p5(
                model,
                low_rank_ab=low_rank_ab,
                low_rank_abbar=None,
                calib_s=calib_s,
                calib_h_lowrank=None,
                alpha=float(args.ab_alpha),
                ab_compute=str(args.ab_compute),
            )
            n_wrapped = _set_ab_alpha(model, float(args.ab_alpha))

            print("👾 [Eval] running Wdq+AB")
            ppl_ab, ppl_ab_sec = evaluate_ppl_wikitext2(
                model,
                tok,
                device,
                label=f"Wdq*+AB* (a={args.ab_alpha:g})",
                stride=int(args.ppl_stride),
                max_tokens=int(args.ppl_max_tokens),
            )
            results["wdq_ab"] = {
                "ppl": float(ppl_ab),
                "elapsed_sec": float(ppl_ab_sec),
                "wrapped_modules": int(n_wrapped),
            }
            print(f"✌ [Eval] Wdq+AB done | PPL={ppl_ab:.4f}")

            if "wdq_only" in results:
                results["delta_ppl_wdq_minus_ab"] = (
                    float(results["wdq_only"]["ppl"]) - float(results["wdq_ab"]["ppl"])
                )

        if not args.skip_gen:
            print("[Eval] running generation metrics")
            gen_stats = measure_generation_metrics(
                model,
                tok,
                device,
                prompts=prompts,
                max_new_tokens=int(args.gen_max_new_tokens),
                do_sample=bool(args.gen_do_sample),
                num_beams=int(args.gen_num_beams),
                temperature=float(args.gen_temperature),
                top_p=float(args.gen_top_p),
                repeats=int(args.gen_repeats),
            )
            results["generation"] = gen_stats

    finally:
        del wdq_star, low_rank_ab, calib_s
        if model is not None:
            del model
        if tok is not None:
            del tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_json:
        out_path = Path(args.save_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if "wdq_only" in results:
        print(f"✨ Wdq-only PPL: {results['wdq_only']['ppl']:.4f}")
    if "wdq_ab" in results:
        print(f"✨ Wdq+AB PPL: {results['wdq_ab']['ppl']:.4f}")
    if "delta_ppl_wdq_minus_ab" in results:
        print(f"✨ Delta PPL (Wdq-only - Wdq+AB): {results['delta_ppl_wdq_minus_ab']:+.4f}")


if __name__ == "__main__":
    main()
