#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alternative Step3 for 1-bit LABA: block-wise output distillation with soft 1-bit refinement.

Idea:
  - Keep the 1-bit soft assignment parameterization from `step3_1bit_soft.py`
  - Replace the weight-space weighted MSE objective with transformer-block output
    distillation loss on calibration batches:
        || Block_fp(X) - Block_quant(X) ||^2
  - Optimize all target Linear weights inside the same transformer block jointly
    using cached block inputs / outputs from the current model state

This is intended as a practical proxy for PPL optimization: much lighter than
full-model CE fine-tuning, but more aligned with downstream PPL than pure
weight-space MSE.

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
  Optional with `--save_calib`:
    - out_dir/calib_tokens_ns{N}_sl{L}_bs{B}_{split}.pt

Usage:
  First run: build and save calibration tokens
  CUDA_VISIBLE_DEVICES=0,1,2 nohup python 1bit_test/step3_1bit_soft_blockdistill.py \
    --model_id Qwen/Qwen3-8B \
    --step1_dir ./output/qwen3_8b_64/step1_quant/1bit \
    --calib_s ./output/qwen3_8b_64/calib_sqrtdiag.pt \
    --out_dir ./output/qwen3_8b_64/step3_blockdistill/1bit \
    --qtype hessian-aware \
    --rank_ab 64 \
    --outer_loops 10 \
    --soft_lr 1e-1 \
    --soft_scale_lr 5e-2 \
    --soft_tau 1.0 \
    --calib_dataset DKYoon/SlimPajama-6B \
--calib_nsamples 16 \
--calib_seq_len 128 \
--out_chunk_size 128 \
    --calib_batch_size 1 \
    --save_calib \
    --save_all > ./logs/qwen3_8b_soft_distill.log 2>&1 &

  Reuse saved calibration tokens on later runs
  CUDA_VISIBLE_DEVICES=0,1 nohup python 1bit_test/step3_1bit_soft_blockdistill.py \
    --model_id Qwen/Qwen3-8B \
    --step1_dir ./output/qwen3_8b_64/step1_quant/1bit \
    --calib_s ./output/qwen3_8b_64/calib_sqrtdiag.pt \
    --out_dir ./output/qwen3_8b_64/step3_blockdistill/1bit \
    --qtype hessian-aware \
    --rank_ab 64 \
    --outer_loops 10 \
    --soft_lr 1e-1 \
    --soft_scale_lr 5e-2 \
    --soft_tau 1.0 \
    --calib_nsamples 128 \
    --calib_seq_len 512 \
    --calib_batch_size 1 \
    --reuse_calib \
    --save_all > ./logs/qwen3_8b_soft_distill.log 2>&1 &

  Lower-memory smoke test
  CUDA_VISIBLE_DEVICES=0,1 python 1bit_test/step3_1bit_soft_blockdistill.py \
    --model_id meta-llama/Llama-3.1-8B \
    --step1_dir ./output/llama3_8b_64/step1_quant/1bit \
    --calib_s ./output/llama3_8b_64/calib_sqrtdiag.pt \
    --out_dir ./output/llama3_8b_64/step3_blockdistill/1bit_smoke \
    --qtype hessian-aware \
    --rank_ab 64 \
    --outer_loops 2 \
    --soft_lr 1e-1 \
    --soft_scale_lr 5e-2 \
    --soft_tau 1.0 \
    --calib_dataset DKYoon/SlimPajama-6B \
    --calib_nsamples 16 \
    --calib_seq_len 128 \
    --calib_batch_size 1 \
    --max_layers 8 \
    --save_calib

"""

from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
for _path in (str(_THIS_DIR), str(_PARENT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from step3_1bit_soft import (  # noqa: E402
    MODULE_ORDER,
    _invoke_local_main,
    _snapshot_state_to_cpu,
    append_jsonl,
    dequant_from_codebook_codes,
    extract_block_index,
    load_diag_weight,
    set_seed,
    should_reuse_step1_init,
    sort_key,
    weighted_low_rank_fit,
    weighted_objective,
)
from step_1_quantize import (  # noqa: E402
    is_target_weight,
    lloyd_asym_nonuniform_quantize,
)
from step_2_calib import (  # noqa: E402
    build_calibration_tokens,
    build_token_batches_from_tokens,
)


def get_parent_module(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


class AddLowRankCorrection(nn.Module):
    def __init__(self, inner: nn.Module, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.register_buffer("A", A.to(torch.float16), persistent=False)
        self.register_buffer("B", B.to(torch.float16), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        r = F.linear(x, self.B)
        corr = F.linear(r, self.A)
        return z.add_(corr)


def _unwrap_base_linear(module: nn.Module) -> nn.Module:
    while isinstance(module, AddLowRankCorrection):
        module = module.inner
    return module


def _torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _resolve_model_input_device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _clone_tree_to_cpu(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().to(device="cpu")
    if isinstance(x, dict):
        return {k: _clone_tree_to_cpu(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_clone_tree_to_cpu(v) for v in x)
    if isinstance(x, list):
        return [_clone_tree_to_cpu(v) for v in x]
    return x


def _move_tree_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device=device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move_tree_to_device(v, device) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(_move_tree_to_device(v, device) for v in x)
    if isinstance(x, list):
        return [_move_tree_to_device(v, device) for v in x]
    return x


def _extract_hidden_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output:
        if torch.is_tensor(output[0]):
            return output[0]
    if hasattr(output, "hidden_states") and torch.is_tensor(output.hidden_states):
        return output.hidden_states
    if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
        return output.last_hidden_state
    raise TypeError(f"Unsupported block output type: {type(output)!r}")


def _collect_transformer_blocks(model: nn.Module) -> Dict[int, Tuple[str, nn.Module]]:
    candidates = [
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("model.decoder.layers", getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None)),
        ("transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
        ("gpt_neox.layers", getattr(getattr(model, "gpt_neox", None), "layers", None)),
        ("layers", getattr(model, "layers", None)),
    ]
    for prefix, layers in candidates:
        if layers is None:
            continue
        out: Dict[int, Tuple[str, nn.Module]] = {}
        for idx, block in enumerate(layers):
            out[int(idx)] = (f"{prefix}.{idx}", block)
        if out:
            return out
    raise RuntimeError("Could not locate transformer blocks on the loaded model.")


def _build_linear_soft_params(
    key: str,
    ctx: dict,
    args: argparse.Namespace,
    param_device: torch.device,
) -> dict:
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
    if bits != 1:
        raise ValueError(f"{Path(__file__).name} only supports 1-bit artifacts, but {key} has bits={bits}")

    w_cpu = state[key].to(torch.float32)
    d_cpu = load_diag_weight(calib_s[key], eps=float(args.eps))
    if d_cpu.numel() != orig_i:
        raise RuntimeError(f"diag weight shape mismatch on {key}: expected {orig_i}, got {d_cpu.numel()}")
    if w_cpu.shape[1] != orig_i:
        raise RuntimeError(f"orig_I mismatch on {key}: meta={orig_i}, weight={w_cpu.shape[1]}")

    w = w_cpu
    d = d_cpu
    hdiag = (d * d) if args.qtype == "hessian-aware" else None

    if should_reuse_step1_init(meta, args.qtype):
        codebook = codebooks[key].to(dtype=torch.float32)
        qcodes = qcodes_dict[key].to(dtype=torch.long)
        wq = dequant_from_codebook_codes(codebook, qcodes, orig_i=orig_i)
        init_source = "step1_artifact"
    else:
        wq, codebook, qcodes, _ = lloyd_asym_nonuniform_quantize(
            w,
            b=bits,
            group_size=gs,
            clip_pct=clip_pct,
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
            hessian_diag=hdiag,
        )
        qcodes = qcodes.to(torch.long)
        init_source = "recomputed"

    a, b = weighted_low_rank_fit(w - wq, d, rank=int(args.rank_ab), eps=float(args.eps))

    c0 = codebook[..., 0].unsqueeze(-1)
    c1 = codebook[..., 1].unsqueeze(-1)
    beta_init = 0.5 * (c0 + c1)
    alpha_init = 0.5 * (c0 - c1)
    param_store_device = torch.device("cpu") if bool(args.cpu_offload_soft_params) else torch.device(param_device)

    logits = torch.where(
        qcodes == 0,
        torch.full_like(qcodes, float(args.soft_init_logit), dtype=torch.float32),
        torch.full_like(qcodes, -float(args.soft_init_logit), dtype=torch.float32),
    ).to(device=param_store_device)

    params = {
        "logits": nn.Parameter(logits),
        "alpha": nn.Parameter(alpha_init.clone().to(device=param_store_device)),
        "beta": nn.Parameter(beta_init.clone().to(device=param_store_device)),
    }
    if args.soft_fixed_codebook:
        params["alpha"].requires_grad_(False)
        params["beta"].requires_grad_(False)

    return {
        "key": key,
        "module_name": key[:-7],
        "meta": meta,
        "bits": bits,
        "group_size": gs,
        "orig_i": orig_i,
        "clip_percentile": clip_pct,
        "init_source": init_source,
        "param_device": torch.device(param_device),
        "param_store_device": torch.device(param_store_device),
        "weight_fp_cpu": w_cpu,
        "diag_weight_cpu": d_cpu,
        "A_fixed": a.to(device=param_store_device, dtype=torch.float32),
        "B_fixed": b.to(device=param_store_device, dtype=torch.float32),
        "out_chunk_size": int(args.out_chunk_size),
        "params": params,
        "codebook_init": codebook.to(device=param_store_device, dtype=torch.float32),
        "qcodes_init": qcodes.to(device=param_store_device, dtype=torch.long),
    }


def _materialize_soft_weight(entry: dict, tau: float, runtime_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = entry["params"]["logits"].to(device=runtime_device, dtype=torch.float32)
    alpha = entry["params"]["alpha"].to(device=runtime_device, dtype=torch.float32)
    beta = entry["params"]["beta"].to(device=runtime_device, dtype=torch.float32)
    probs_code0 = torch.sigmoid(logits / float(max(tau, 1e-6)))
    soft_sign = 2.0 * probs_code0 - 1.0
    wq_soft_g = beta + alpha * soft_sign
    wq_soft = wq_soft_g.reshape(entry["weight_fp_cpu"].shape[0], -1)[:, : entry["orig_i"]]
    with torch.no_grad():
        ab = entry["A_fixed"].to(device=runtime_device, dtype=torch.float32) @ entry["B_fixed"].to(
            device=runtime_device, dtype=torch.float32
        )
    w_eff = wq_soft + ab
    return w_eff, wq_soft


def _linear_forward_chunked(entry: dict, x: torch.Tensor, tau: float, out_chunk_size: int) -> torch.Tensor:
    runtime_device = x.device
    out_features = int(entry["weight_fp_cpu"].shape[0])
    rank = int(entry["A_fixed"].shape[1])
    bias_full = None
    outputs: List[torch.Tensor] = []

    b_fixed = entry["B_fixed"].to(device=runtime_device, dtype=torch.float32)
    proj_rank = F.linear(x.to(torch.float32), b_fixed)

    for start in range(0, out_features, int(out_chunk_size)):
        end = min(start + int(out_chunk_size), out_features)
        sl = slice(start, end)

        logits = entry["params"]["logits"][sl].to(device=runtime_device, dtype=torch.float32)
        alpha = entry["params"]["alpha"][sl].to(device=runtime_device, dtype=torch.float32)
        beta = entry["params"]["beta"][sl].to(device=runtime_device, dtype=torch.float32)
        probs_code0 = torch.sigmoid(logits / float(max(tau, 1e-6)))
        soft_sign = 2.0 * probs_code0 - 1.0
        wq_soft_g = beta + alpha * soft_sign
        wq_chunk = wq_soft_g.reshape(end - start, -1)[:, : entry["orig_i"]]

        a_chunk = entry["A_fixed"][sl].to(device=runtime_device, dtype=torch.float32)
        corr_chunk = F.linear(proj_rank, a_chunk)
        base_chunk = F.linear(x.to(torch.float32), wq_chunk)
        out_chunk = base_chunk + corr_chunk

        if bias_full is None:
            bias_full = getattr(entry.get("runtime_module"), "bias", None)
        if bias_full is not None:
            out_chunk = out_chunk + bias_full[sl].to(device=runtime_device, dtype=torch.float32)

        outputs.append(out_chunk.to(dtype=x.dtype))

        del logits, alpha, beta, probs_code0, soft_sign, wq_soft_g, wq_chunk, a_chunk, corr_chunk, base_chunk, out_chunk

    return torch.cat(outputs, dim=-1)


def _project_hard_artifacts(entry: dict, tau: float, eps: float) -> dict:
    with torch.no_grad():
        logits = entry["params"]["logits"]
        alpha = entry["params"]["alpha"]
        beta = entry["params"]["beta"]
        probs_code0 = torch.sigmoid(logits / float(max(tau, 1e-6)))
        qcodes = (probs_code0 < 0.5).to(torch.uint8)
        codebook = torch.stack(
            [
                (beta + alpha).squeeze(-1),
                (beta - alpha).squeeze(-1),
            ],
            dim=-1,
        )
        wq = dequant_from_codebook_codes(codebook, qcodes, orig_i=entry["orig_i"])
        a, b = weighted_low_rank_fit(
            entry["weight_fp_cpu"] - wq.to(device="cpu", dtype=torch.float32),
            entry["diag_weight_cpu"],
            rank=int(entry["A_fixed"].shape[1]),
            eps=float(eps),
        )
        return {
            "wdq": wq.detach(),
            "codebook": codebook.detach(),
            "qcodes": qcodes.detach(),
            "A": a.detach(),
            "B": b.detach(),
        }


def _apply_final_linear_artifacts(model: nn.Module, key: str, wdq: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    module_name = key[:-7]
    parent, attr = get_parent_module(model, module_name)
    current = getattr(parent, attr)
    base = _unwrap_base_linear(current)
    if not hasattr(base, "weight"):
        raise TypeError(f"Target module has no weight: {module_name}")
    base.weight.data.copy_(wdq.to(device=base.weight.device, dtype=base.weight.dtype))
    setattr(parent, attr, AddLowRankCorrection(base, a.to(base.weight.device), b.to(base.weight.device)))


def _capture_block_io(
    model: nn.Module,
    block: nn.Module,
    calib_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    input_device: torch.device,
) -> List[dict]:
    cache: List[dict] = []
    current_args: Optional[Tuple[Any, ...]] = None
    current_kwargs: Optional[Dict[str, Any]] = None

    def pre_hook(_module, args, kwargs):
        nonlocal current_args, current_kwargs
        current_args = _clone_tree_to_cpu(args)
        current_kwargs = _clone_tree_to_cpu(kwargs)

    def fwd_hook(_module, _args, output):
        if current_args is None or current_kwargs is None:
            raise RuntimeError("block cache hook received output before inputs")
        hidden = _extract_hidden_tensor(output).detach().to(device="cpu")
        cache.append({"args": current_args, "kwargs": current_kwargs, "target": hidden})

    handles = [
        block.register_forward_pre_hook(pre_hook, with_kwargs=True),
        block.register_forward_hook(fwd_hook),
    ]
    try:
        with torch.no_grad():
            model.eval()
            for input_ids, attention_mask in calib_batches:
                model(
                    input_ids=input_ids.to(input_device),
                    attention_mask=attention_mask.to(input_device),
                    use_cache=False,
                )
    finally:
        for h in handles:
            h.remove()
    return cache


def _make_linear_override_hook(entry: dict, tau: float):
    def hook(module, inputs, _output):
        x = inputs[0]
        entry["runtime_module"] = module
        chunk_size = int(entry.get("out_chunk_size", 0))
        if chunk_size > 0:
            return _linear_forward_chunked(entry, x, tau=tau, out_chunk_size=chunk_size)
        w_eff, _ = _materialize_soft_weight(entry, tau=tau, runtime_device=x.device)
        bias = getattr(module, "bias", None)
        return F.linear(x, w_eff.to(device=x.device, dtype=x.dtype), bias)

    return hook


def _evaluate_block_distill_loss(
    block: nn.Module,
    block_cache: List[dict],
    block_device: torch.device,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for rec in block_cache:
        args_dev = _move_tree_to_device(rec["args"], block_device)
        kwargs_dev = _move_tree_to_device(rec["kwargs"], block_device)
        target = rec["target"].to(device=block_device, dtype=torch.float32, non_blocking=True)
        out = block(*args_dev, **kwargs_dev)
        pred = _extract_hidden_tensor(out).to(torch.float32)
        losses.append(F.mse_loss(pred, target))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=block_device)


def _calib_cache_path(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir).resolve()
    cache_name = (
        f"calib_tokens_ns{int(args.calib_nsamples)}"
        f"_sl{int(args.calib_seq_len)}"
        f"_bs{int(args.calib_batch_size)}"
        f"_{str(args.calib_split)}.pt"
    )
    return out_dir / cache_name


def _load_or_build_calib_batches(args: argparse.Namespace, tokenizer) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    cache_path = _calib_cache_path(args)
    if bool(args.reuse_calib) and cache_path.exists():
        print(f"[Block-Distill] loading cached calibration tokens: {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu")
        tokens = payload["tokens"] if isinstance(payload, dict) and "tokens" in payload else payload
        if not torch.is_tensor(tokens):
            raise TypeError(f"Cached calibration payload is not a tensor: {type(tokens)!r}")
        calib_batches, _ = build_token_batches_from_tokens(tokens.to(torch.long), int(args.calib_batch_size))
        return calib_batches

    print("[Block-Distill] building calibration tokens from dataset", flush=True)
    tokens = build_calibration_tokens(
        tokenizer,
        nsamples=int(args.calib_nsamples),
        seqlen=int(args.calib_seq_len),
        dataset=args.calib_dataset,
        dataset_config=args.calib_dataset_config,
        split=args.calib_split,
        use_streaming=bool(args.calib_use_streaming),
    )
    if bool(args.save_calib):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "tokens": tokens.cpu(),
                "dataset": str(args.calib_dataset),
                "dataset_config": args.calib_dataset_config,
                "split": str(args.calib_split),
                "nsamples": int(args.calib_nsamples),
                "seq_len": int(args.calib_seq_len),
                "batch_size": int(args.calib_batch_size),
            },
            cache_path,
        )
        print(f"[Block-Distill] saved calibration tokens: {cache_path}", flush=True)
    calib_batches, _ = build_token_batches_from_tokens(tokens, int(args.calib_batch_size))
    return calib_batches


def load_context(args: argparse.Namespace) -> dict:
    step1_dir = Path(args.step1_dir).resolve()
    codebook_path = step1_dir / "codebook.pt"
    qcodes_path = step1_dir / "qcodes.pt"
    meta_path = step1_dir / "meta.pt"
    if not (codebook_path.exists() and qcodes_path.exists() and meta_path.exists()):
        raise FileNotFoundError("step1_dir must contain codebook.pt, qcodes.pt, meta.pt")

    print(f"[Block-Distill] loading step1 artifacts: {step1_dir}", flush=True)
    codebooks: Dict[str, torch.Tensor] = torch.load(codebook_path, map_location="cpu")
    qcodes: Dict[str, torch.Tensor] = torch.load(qcodes_path, map_location="cpu")
    metas: Dict[str, dict] = torch.load(meta_path, map_location="cpu")

    print(f"[Block-Distill] loading calib_s: {args.calib_s}", flush=True)
    calib_s: Dict[str, dict] = torch.load(args.calib_s, map_location="cpu")

    load_dtype = _torch_dtype_from_name(args.dtype_w)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dm_raw = str(args.model_device_map).strip().lower()
    resolved_model_device_map = None if dm_raw in {"", "none", "null"} else args.model_device_map

    print(
        f"[Block-Distill] loading model: {args.model_id} "
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
    if resolved_model_device_map is None:
        model = model.to(device)
    model.eval()
    try:
        state = _snapshot_state_to_cpu(model)
    except NotImplementedError:
        if resolved_model_device_map is None:
            raise
        print("[Block-Distill] Detected meta tensors under device_map mode. Re-loading on CPU for state snapshot.", flush=True)
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
        if str(args.model_device_map).strip().lower() not in {"", "none", "null"}:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                revision=args.revision,
                torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
                trust_remote_code=args.trust_remote_code,
                device_map=resolved_model_device_map,
                low_cpu_mem_usage=True,
            )
        else:
            model = model.to(device)
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib_batches = _load_or_build_calib_batches(args, tokenizer)

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

    blocks = _collect_transformer_blocks(model)
    block_keys: Dict[int, List[str]] = {}
    for key in keys:
        bidx = extract_block_index(key)
        if bidx is None or bidx not in blocks:
            continue
        block_keys.setdefault(int(bidx), []).append(key)
    block_keys = {k: sorted(v, key=sort_key) for k, v in sorted(block_keys.items())}
    if not block_keys:
        raise RuntimeError("No matched transformer blocks found for the selected keys.")

    print(
        f"[Block-Distill] matched layers={len(keys)}, blocks={len(block_keys)}, "
        f"calib_batches={len(calib_batches)}",
        flush=True,
    )
    return {
        "device": device,
        "model": model,
        "model_input_device": _resolve_model_input_device(model),
        "blocks": blocks,
        "block_keys": block_keys,
        "calib_batches": calib_batches,
        "codebooks": codebooks,
        "qcodes": qcodes,
        "metas": metas,
        "calib_s": calib_s,
        "state": state,
        "keys": keys,
    }


def optimize_block(
    block_idx: int,
    block_name: str,
    block: nn.Module,
    layer_keys: List[str],
    ctx: dict,
    args: argparse.Namespace,
    metrics_path: Path,
) -> Dict[str, dict]:
    block_device = next(block.parameters()).device
    block_cache = _capture_block_io(
        model=ctx["model"],
        block=block,
        calib_batches=ctx["calib_batches"],
        input_device=ctx["model_input_device"],
    )
    if not block_cache:
        raise RuntimeError(f"No calibration cache captured for block {block_name}")

    entries = []
    for key in layer_keys:
        module_name = key[:-7]
        parent, attr = get_parent_module(ctx["model"], module_name)
        current = getattr(parent, attr)
        base = _unwrap_base_linear(current)
        if not hasattr(base, "weight"):
            raise TypeError(f"Target module has no weight: {module_name}")
        entries.append(_build_linear_soft_params(key, ctx, args, param_device=base.weight.device))
    module_to_entry = {entry["module_name"]: entry for entry in entries}
    clip_value = float(max(float(args.soft_logit_clip), float(args.soft_init_logit)))

    opt_params: List[dict] = []
    for entry in entries:
        opt_params.append({"params": [entry["params"]["logits"]], "lr": float(args.soft_lr)})
        if entry["params"]["alpha"].requires_grad or entry["params"]["beta"].requires_grad:
            scale_params = [p for p in (entry["params"]["alpha"], entry["params"]["beta"]) if p.requires_grad]
            if scale_params:
                opt_params.append({"params": scale_params, "lr": float(args.soft_scale_lr)})
    optimizer = torch.optim.Adam(opt_params)

    handles = []
    for module_name, entry in module_to_entry.items():
        parent, attr = get_parent_module(ctx["model"], module_name)
        current = getattr(parent, attr)
        base = _unwrap_base_linear(current)
        handles.append(base.register_forward_hook(_make_linear_override_hook(entry, tau=float(args.soft_tau))))

    try:
        init_loss = float(_evaluate_block_distill_loss(block, block_cache, block_device).item())
        best_loss = init_loss
        best_payload = {
            entry["key"]: _project_hard_artifacts(entry, tau=float(args.soft_tau), eps=float(args.eps))
            for entry in entries
        }

        append_jsonl(
            metrics_path,
            {
                "block": block_name,
                "block_idx": int(block_idx),
                "outer": -1,
                "phase": "init",
                "objective_block_mse": float(init_loss),
                "num_layers": len(layer_keys),
                "alternating_stage": "step1_wq_plus_weighted_svd_ab",
            },
        )

        for outer in range(int(args.outer_loops)):
            optimizer.zero_grad(set_to_none=True)
            loss = _evaluate_block_distill_loss(block, block_cache, block_device)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for entry in entries:
                    entry["params"]["logits"].clamp_(-clip_value, clip_value)

            loss_val = float(loss.item())
            append_jsonl(
                metrics_path,
                {
                    "block": block_name,
                    "block_idx": int(block_idx),
                    "outer": int(outer),
                    "phase": "wq_update",
                    "objective_block_mse": float(loss_val),
                    "num_layers": len(layer_keys),
                },
            )
            current_payload = {
                entry["key"]: _project_hard_artifacts(entry, tau=float(args.soft_tau), eps=float(args.eps))
                for entry in entries
            }
            for entry in entries:
                payload = current_payload[entry["key"]]
                entry["A_fixed"] = payload["A"].to(device=entry["param_store_device"], dtype=torch.float32)
                entry["B_fixed"] = payload["B"].to(device=entry["param_store_device"], dtype=torch.float32)
            refit_loss = float(_evaluate_block_distill_loss(block, block_cache, block_device).item())
            append_jsonl(
                metrics_path,
                {
                    "block": block_name,
                    "block_idx": int(block_idx),
                    "outer": int(outer),
                    "phase": "ab_refit",
                    "objective_block_mse": float(refit_loss),
                    "num_layers": len(layer_keys),
                },
            )
            if refit_loss < best_loss:
                best_loss = refit_loss
                best_payload = current_payload
    finally:
        for h in handles:
            h.remove()

    out: Dict[str, dict] = {}
    for entry in entries:
        payload = best_payload[entry["key"]]
        wdq = payload["wdq"]
        a = payload["A"]
        b = payload["B"]
        obj_weighted = weighted_objective(
            entry["weight_fp_cpu"],
            wdq.to(device="cpu", dtype=torch.float32),
            a.to(device="cpu", dtype=torch.float32),
            b.to(device="cpu", dtype=torch.float32),
            entry["diag_weight_cpu"],
        )
        out[entry["key"]] = {
            "wdq": wdq.to(torch.float16).cpu(),
            "low_rank_ab": {
                "A": a.to(torch.float16).cpu(),
                "B": b.to(torch.float16).cpu(),
                "meta": {
                    "rank": int(args.rank_ab),
                    "bits": int(entry["bits"]),
                    "group_size": int(entry["group_size"]),
                    "qtype": str(args.qtype),
                    "objective_weighted_final": float(obj_weighted),
                    "objective_block_mse_best": float(best_loss),
                    "block_name": block_name,
                },
            },
            "wdq_best": wdq.to(torch.float16).cpu(),
            "low_rank_ab_best": {
                "A": a.to(torch.float16).cpu(),
                "B": b.to(torch.float16).cpu(),
                "meta": {
                    "rank": int(args.rank_ab),
                    "bits": int(entry["bits"]),
                    "group_size": int(entry["group_size"]),
                    "qtype": str(args.qtype),
                    "objective_weighted_best": float(obj_weighted),
                    "objective_block_mse_best": float(best_loss),
                    "best_outer": -1,
                    "block_name": block_name,
                },
            },
            "codebook": payload["codebook"].to(torch.float16).cpu(),
            "qcodes": payload["qcodes"].cpu(),
            "quant_meta": {
                "bits": int(entry["bits"]),
                "group_size": int(entry["group_size"]),
                "clip_percentile": float(entry["clip_percentile"]),
                "qtype": str(args.qtype),
                "init_source": str(entry["init_source"]),
                "block_name": block_name,
                "objective_block_mse_best": float(best_loss),
            },
            "summary": {
                "layer": entry["key"],
                "block_name": block_name,
                "bits": int(entry["bits"]),
                "group_size": int(entry["group_size"]),
                "init_source": str(entry["init_source"]),
                "objective_weighted_final": float(obj_weighted),
                "objective_weighted_best": float(obj_weighted),
                "objective_block_mse_best": float(best_loss),
                "best_outer": -1,
            },
        }
        _apply_final_linear_artifacts(ctx["model"], entry["key"], wdq, a, b)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Alt Step3 - 1-bit soft refinement with block output distillation")
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
    ap.add_argument("--outer_loops", type=int, default=10)
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)
    ap.add_argument("--clip_percentile", type=float, default=None)
    ap.add_argument("--soft_lr", type=float, default=1e-1)
    ap.add_argument("--soft_scale_lr", type=float, default=5e-2)
    ap.add_argument("--ab_lr", type=float, default=1e-3)
    ap.add_argument("--soft_tau", type=float, default=1.0)
    ap.add_argument("--soft_init_logit", type=float, default=4.0)
    ap.add_argument("--soft_logit_clip", type=float, default=8.0)
    ap.add_argument("--soft_fixed_codebook", action="store_true")
    ap.add_argument(
        "--cpu_offload_soft_params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep soft Wq parameters and fixed AB on CPU, moving them to the layer GPU on demand",
    )
    ap.add_argument(
        "--out_chunk_size",
        type=int,
        default=512,
        help="Chunk size over output channels for hook-time linear reconstruction; lower is safer for memory",
    )
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=0)
    ap.add_argument("--save_every_layer", action="store_true")
    ap.add_argument("--save_all", action="store_true")

    ap.add_argument("--calib_dataset", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--calib_dataset_config", default=None)
    ap.add_argument("--calib_split", default="train")
    ap.add_argument("--calib_use_streaming", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument("--calib_nsamples", type=int, default=128)
    ap.add_argument("--calib_seq_len", type=int, default=512)
    ap.add_argument("--calib_batch_size", type=int, default=1)
    ap.add_argument("--reuse_calib", action="store_true", help="Reuse cached calibration tokens from out_dir if present")
    ap.add_argument("--save_calib", action="store_true", help="Save calibration tokens to out_dir for later reuse")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    ctx = load_context(args)

    wdq_out: Dict[str, torch.Tensor] = {}
    ab_out: Dict[str, Dict[str, torch.Tensor]] = {}
    wdq_best_out: Dict[str, torch.Tensor] = {}
    ab_best_out: Dict[str, Dict[str, torch.Tensor]] = {}
    codebook_out: Dict[str, torch.Tensor] = {}
    qcodes_out: Dict[str, torch.Tensor] = {}
    quant_meta_out: Dict[str, dict] = {}
    layer_summaries: List[dict] = []

    t0 = time.time()
    for idx, (block_idx, layer_keys) in enumerate(ctx["block_keys"].items(), start=1):
        block_name, block = ctx["blocks"][block_idx]
        print(
            f"[Block-Distill] ({idx}/{len(ctx['block_keys'])}) optimizing block {block_name} "
            f"with {len(layer_keys)} target linears",
            flush=True,
        )
        block_res = optimize_block(
            block_idx=block_idx,
            block_name=block_name,
            block=block,
            layer_keys=layer_keys,
            ctx=ctx,
            args=args,
            metrics_path=metrics_path,
        )
        for key in layer_keys:
            layer_res = block_res[key]
            wdq_out[key] = layer_res["wdq"]
            ab_out[key] = layer_res["low_rank_ab"]
            wdq_best_out[key] = layer_res["wdq_best"]
            ab_best_out[key] = layer_res["low_rank_ab_best"]
            codebook_out[key] = layer_res["codebook"]
            qcodes_out[key] = layer_res["qcodes"]
            quant_meta_out[key] = layer_res["quant_meta"]
            layer_summaries.append(layer_res["summary"])

        if args.save_every_layer:
            torch.save(wdq_out, out_dir / "wdq_star.pt")
            torch.save(ab_out, out_dir / "low_rank_ab.pt")
            torch.save(wdq_best_out, out_dir / "wdq_star_best.pt")
            torch.save(ab_best_out, out_dir / "low_rank_ab_best.pt")
            if args.save_all:
                torch.save(codebook_out, out_dir / "codebook_star.pt")
                torch.save(qcodes_out, out_dir / "qcodes_star.pt")
                torch.save(quant_meta_out, out_dir / "quant_meta_star.pt")

    torch.save(wdq_out, out_dir / "wdq_star.pt")
    torch.save(ab_out, out_dir / "low_rank_ab.pt")
    torch.save(wdq_best_out, out_dir / "wdq_star_best.pt")
    torch.save(ab_best_out, out_dir / "low_rank_ab_best.pt")
    if args.save_all:
        torch.save(codebook_out, out_dir / "codebook_star.pt")
        torch.save(qcodes_out, out_dir / "qcodes_star.pt")
        torch.save(quant_meta_out, out_dir / "quant_meta_star.pt")

    final_mean = sum(x["objective_weighted_final"] for x in layer_summaries) / max(1, len(layer_summaries))
    best_mean = sum(x["objective_weighted_best"] for x in layer_summaries) / max(1, len(layer_summaries))
    block_mse_mean = sum(x["objective_block_mse_best"] for x in layer_summaries) / max(1, len(layer_summaries))
    summary = {
        "model_id": args.model_id,
        "revision": args.revision,
        "step1_dir": str(Path(args.step1_dir).resolve()),
        "calib_s": str(Path(args.calib_s).resolve()),
        "out_dir": str(out_dir),
        "qtype": str(args.qtype),
        "rank_ab": int(args.rank_ab),
        "outer_loops": int(args.outer_loops),
        "lloyd_iter": int(args.lloyd_iter),
        "chunk_groups": int(args.chunk_groups),
        "clip_percentile_override": args.clip_percentile,
        "num_layers": len(layer_summaries),
        "num_blocks": len(ctx["block_keys"]),
        "calib_dataset": str(args.calib_dataset),
        "calib_dataset_config": args.calib_dataset_config,
        "calib_nsamples": int(args.calib_nsamples),
        "calib_seq_len": int(args.calib_seq_len),
        "objective_weighted_final_mean": float(final_mean),
        "objective_weighted_best_mean": float(best_mean),
        "objective_block_mse_best_mean": float(block_mse_mean),
        "elapsed_sec": time.time() - t0,
        "layers": layer_summaries,
    }
    if args.save_all:
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Block-Distill] saved:", flush=True)
    print(f"  wdq*: {out_dir / 'wdq_star.pt'}", flush=True)
    print(f"  AB*:  {out_dir / 'low_rank_ab.pt'}", flush=True)
    if args.save_all:
        print(f"  summary: {out_dir / 'summary.json'}", flush=True)


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
class Step03BlockDistillConfig:
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
    outer_loops: int = 10
    lloyd_iter: int = 12
    chunk_groups: int = 4096
    clip_percentile: Optional[float] = None
    soft_lr: float = 1e-1
    soft_scale_lr: float = 5e-2
    ab_lr: float = 1e-3
    soft_tau: float = 1.0
    soft_init_logit: float = 4.0
    soft_logit_clip: float = 8.0
    soft_fixed_codebook: bool = False
    cpu_offload_soft_params: bool = True
    out_chunk_size: int = 512
    eps: float = 1e-8
    layer_regex: Optional[str] = None
    max_layers: int = 0
    save_every_layer: bool = False
    save_all: bool = False
    calib_dataset: str = "DKYoon/SlimPajama-6B"
    calib_dataset_config: Optional[str] = None
    calib_split: str = "train"
    calib_use_streaming: bool = True
    calib_nsamples: int = 128
    calib_seq_len: int = 512
    calib_batch_size: int = 1
    reuse_calib: bool = False
    save_calib: bool = False
    seed: int = 42
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


def build_command(cfg: Step03BlockDistillConfig) -> List[str]:
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
        "--soft_lr",
        str(float(cfg.soft_lr)),
        "--soft_scale_lr",
        str(float(cfg.soft_scale_lr)),
        "--ab_lr",
        str(float(cfg.ab_lr)),
        "--soft_tau",
        str(float(cfg.soft_tau)),
        "--soft_init_logit",
        str(float(cfg.soft_init_logit)),
        "--soft_logit_clip",
        str(float(cfg.soft_logit_clip)),
        "--out_chunk_size",
        str(int(cfg.out_chunk_size)),
        "--eps",
        str(float(cfg.eps)),
        "--max_layers",
        str(int(cfg.max_layers)),
        "--calib_dataset",
        str(cfg.calib_dataset),
        "--calib_split",
        str(cfg.calib_split),
        "--calib_use_streaming",
        str(bool(cfg.calib_use_streaming)).lower(),
        "--calib_nsamples",
        str(int(cfg.calib_nsamples)),
        "--calib_seq_len",
        str(int(cfg.calib_seq_len)),
        "--calib_batch_size",
        str(int(cfg.calib_batch_size)),
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
    if cfg.soft_fixed_codebook:
        cmd.append("--soft_fixed_codebook")
    cmd.append("--cpu_offload_soft_params" if cfg.cpu_offload_soft_params else "--no-cpu_offload_soft_params")
    if cfg.save_all:
        cmd.append("--save_all")
    if cfg.reuse_calib:
        cmd.append("--reuse_calib")
    if cfg.save_calib:
        cmd.append("--save_calib")
    if cfg.calib_dataset_config is not None:
        cmd += ["--calib_dataset_config", str(cfg.calib_dataset_config)]
    return cmd


def run(cfg: Step03BlockDistillConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp
