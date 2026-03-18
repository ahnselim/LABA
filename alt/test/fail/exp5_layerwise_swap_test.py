#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
exp5_layerwise_swap_test.py

Experiment 5:
  Single-block intervention / swap test.

Goal:
  For one selected block, measure how much loss changes when only that block is
  replaced by:
    - second quant + AB
    - OAS quant + AB

This is the single-block analogue of the layerwise swap test:
  full model
  -> selected block swapped to second
  -> selected block swapped to OAS

Supported block selectors:
  - layer0
  - layer0_attn
  - layer0_mlp
  - model.layers.0.self_attn.q_proj
  - model.layers.0.mlp.up_proj

Example:
CUDA_VISIBLE_DEVICES=1,2 python test/exp5_layerwise_swap_test.py \
  --model_id meta-llama/Llama-3.1-8B \
  --oas_step1_dir ./output/llama3_8b/step1_quant/1bit \
  --oas_low_rank_path ./output/llama3_8b/step3_svd/1bit/low_rank_ab.pt \
  --second_step1_dir ./output/llama3_8b_64/step1_quant/1bit \
  --second_low_rank_path ./output/llama3_8b_64/step3_svd/1bit/low_rank_ab.pt \
  --block layer0 \
  --seq_len 256 \
  --nsamples 4 \
  --batch_size 1 \
  --out_dir ./output/exp5_1bit/layer0
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj", "fc1", "fc2",
}
ATTN_TYPES = {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}
MLP_TYPES = {"gate_proj", "up_proj", "down_proj", "fc1", "fc2"}


def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )


def extract_block_index(name: str) -> Optional[int]:
    for pat in (r"\bmodel\.layers\.(\d+)\.", r"\bencoder\.layers\.(\d+)\.", r"\blayers\.(\d+)\."):
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def module_type_from_key(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(max(b, eps))


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


def load_step1_artifacts(step1_dir: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, dict]]:
    p = Path(step1_dir).resolve()
    codebook_path = p / "codebook.pt"
    qcodes_path = p / "qcodes.pt"
    meta_path = p / "meta.pt"
    if not (codebook_path.exists() and qcodes_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"step1_dir must contain codebook.pt, qcodes.pt, meta.pt: {p}")
    return (
        torch.load(codebook_path, map_location="cpu"),
        torch.load(qcodes_path, map_location="cpu"),
        torch.load(meta_path, map_location="cpu"),
    )


def load_low_rank_ab(low_rank_path: str) -> Dict[str, dict]:
    payload = torch.load(low_rank_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported low_rank_ab payload type: {type(payload)!r}")
    return payload


def reconstruct_ab(entry: dict, out_dim: int, in_dim: int) -> torch.Tensor:
    a = entry.get("A", None)
    b = entry.get("B", None)
    if a is None or b is None:
        raise KeyError("low_rank_ab entry must contain A and B")
    a = a.to(dtype=torch.float32, device="cpu")
    b = b.to(dtype=torch.float32, device="cpu")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be rank-2 tensors")
    if a.shape[0] != out_dim or b.shape[1] != in_dim or a.shape[1] != b.shape[0]:
        raise ValueError(
            f"AB shape mismatch: A={tuple(a.shape)} B={tuple(b.shape)} expected ({out_dim}, r) and (r, {in_dim})"
        )
    return a @ b


def _canonical_dataset_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"dkyoon/slimpajama-6b", "slimpajama-6b", "dkyoon_slimpajama_6b"}:
        return "DKYoon/SlimPajama-6B"
    if lowered in {"slimpajama-627b", "cerebras/slimpajama-627b", "slimpajama627b"}:
        return "cerebras/SlimPajama-627B"
    return name


def open_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    streaming: bool,
):
    return load_dataset(
        _canonical_dataset_name(dataset_name),
        name=dataset_config,
        split=split,
        streaming=streaming,
    )


@torch.no_grad()
def build_calibration_tokens(
    tokenizer,
    nsamples: int,
    seqlen: int,
    dataset: str,
    dataset_config: Optional[str],
    split: str,
    use_streaming: bool,
) -> torch.Tensor:
    ds = open_hf_dataset(dataset, dataset_config, split=split, streaming=use_streaming)
    take = ds.take if hasattr(ds, "take") else None
    iterator = take(max(nsamples * 4, nsamples)) if take else ds

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


def iterate_batches(tokens: torch.Tensor, batch_size: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    for i in range(0, tokens.shape[0], batch_size):
        batch = tokens[i:i + batch_size]
        yield batch, torch.ones_like(batch)


def resolve_model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def resolve_selected_keys(selector: str, available_weight_keys: Iterable[str]) -> List[str]:
    weight_keys = sorted(set(available_weight_keys))
    direct_key = selector if selector.endswith(".weight") else f"{selector}.weight"
    if direct_key in weight_keys:
        return [direct_key]

    m = re.fullmatch(r"layer(\d+)", selector.strip())
    if m:
        layer_idx = int(m.group(1))
        matched = [key for key in weight_keys if extract_block_index(key) == layer_idx]
        if not matched:
            raise RuntimeError(f"No modules matched selector={selector}")
        return matched

    m = re.fullmatch(r"layer(\d+)_(attn|mlp)", selector.strip())
    if not m:
        raise ValueError(f"Unsupported --block selector: {selector}")
    layer_idx = int(m.group(1))
    family = m.group(2)

    matched: List[str] = []
    for key in weight_keys:
        if extract_block_index(key) != layer_idx:
            continue
        mtype = module_type_from_key(key)
        if family == "attn" and mtype in ATTN_TYPES:
            matched.append(key)
        elif family == "mlp" and mtype in MLP_TYPES:
            matched.append(key)
    if not matched:
        raise RuntimeError(f"No modules matched selector={selector}")
    return matched


def load_quantized_weights(
    *,
    state: Dict[str, torch.Tensor],
    step1_dir: str,
    low_rank_path: str,
    selected_keys: List[str],
) -> Dict[str, torch.Tensor]:
    codebooks, qcodes, metas = load_step1_artifacts(step1_dir)
    low_rank_ab = load_low_rank_ab(low_rank_path)
    out: Dict[str, torch.Tensor] = {}
    for key in selected_keys:
        w = state[key].to(torch.float32).cpu()
        meta = metas[key]
        orig_i = int(tuple(meta["orig_shape"])[1])
        wq = dequant_from_codebook_codes(codebooks[key].to(torch.float32), qcodes[key], orig_i=orig_i).cpu()
        ab = reconstruct_ab(low_rank_ab[key], out_dim=int(w.shape[0]), in_dim=int(w.shape[1]))
        out[key] = (wq + ab).to(torch.float32).contiguous()
    return out


def load_model_and_state(args: argparse.Namespace) -> Tuple[nn.Module, Dict[str, torch.Tensor], torch.device]:
    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    model_device_map = None if str(args.model_device_map).lower() in {"", "none", "null"} else args.model_device_map
    load_dtype = dtype_map[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=load_dtype,
        trust_remote_code=args.trust_remote_code,
        device_map=model_device_map,
        low_cpu_mem_usage=True,
    )
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model.eval()
    if model_device_map is None:
        model.to(device)
    if any(getattr(v, "is_meta", False) for v in model.state_dict().values()):
        snapshot_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            torch_dtype=load_dtype,
            trust_remote_code=args.trust_remote_code,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        snapshot_model.eval()
        state = {k: v.detach().to("cpu") for k, v in snapshot_model.state_dict().items()}
        del snapshot_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    return model, state, resolve_model_device(model, device)


def apply_weight_patch(named_modules: Dict[str, nn.Module], patch_map: Dict[str, torch.Tensor]) -> None:
    for weight_key, weight_cpu in patch_map.items():
        module = named_modules[weight_key[:-7]]
        module.weight.data.copy_(weight_cpu.to(device=module.weight.device, dtype=module.weight.dtype))


def percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    xs, _ = torch.sort(values.reshape(-1))
    idx = min(max(int(round(q * (xs.numel() - 1))), 0), xs.numel() - 1)
    return float(xs[idx].item())


def evaluate_patch(
    *,
    model: nn.Module,
    model_device: torch.device,
    tokens: torch.Tensor,
    batch_size: int,
    max_batches: int,
) -> dict:
    token_nlls: List[torch.Tensor] = []
    seq_mean_nlls: List[float] = []

    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(iterate_batches(tokens, batch_size)):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            shift_logits = logits[:, :-1, :].detach().to("cpu", dtype=torch.float32)
            shift_labels = input_ids[:, 1:].detach().to("cpu")
            token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(shift_labels.shape)
            token_nlls.append(token_nll)
            seq_mean_nlls.extend(token_nll.mean(dim=1).tolist())

    all_token_nll = torch.cat(token_nlls, dim=0).reshape(-1).to(torch.float32)
    mean_nll = float(all_token_nll.mean().item())
    ppl = float(math.exp(min(mean_nll, 20.0)))
    return {
        "mean_nll": mean_nll,
        "ppl": ppl,
        "token_nll_p95": percentile(all_token_nll, 0.95),
        "token_nll_p99": percentile(all_token_nll, 0.99),
        "token_nll_max": float(all_token_nll.max().item()) if all_token_nll.numel() > 0 else 0.0,
        "seq_mean_nll_mean": float(sum(seq_mean_nlls) / max(len(seq_mean_nlls), 1)),
        "seq_mean_nll_p95": percentile(torch.tensor(seq_mean_nlls, dtype=torch.float32), 0.95),
        "num_tokens": float(all_token_nll.numel()),
        "num_sequences": float(len(seq_mean_nlls)),
        "token_nll_values": all_token_nll.tolist(),
        "seq_mean_nll_values": seq_mean_nlls,
    }


def main() -> None:
    ap = argparse.ArgumentParser("Experiment 5 - Single-block layerwise swap test")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto", help="Default is auto for multi-GPU loading.")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")

    ap.add_argument("--oas_step1_dir", required=True)
    ap.add_argument("--oas_low_rank_path", required=True)
    ap.add_argument("--second_step1_dir", required=True)
    ap.add_argument("--second_low_rank_path", required=True)

    ap.add_argument("--block", required=True, help="e.g. layer0, layer0_attn, layer0_mlp, or full module name")
    ap.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--use_streaming", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=0, help="0 means use all batches built from nsamples.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model, state, model_device = load_model_and_state(args)
    available_weight_keys = [k for k, v in state.items() if is_target_weight(k, v)]
    selected_keys = resolve_selected_keys(args.block, available_weight_keys)
    named_modules = dict(model.named_modules())

    original_patch = {key: state[key].clone() for key in selected_keys}
    second_patch = load_quantized_weights(
        state=state,
        step1_dir=args.second_step1_dir,
        low_rank_path=args.second_low_rank_path,
        selected_keys=selected_keys,
    )
    oas_patch = load_quantized_weights(
        state=state,
        step1_dir=args.oas_step1_dir,
        low_rank_path=args.oas_low_rank_path,
        selected_keys=selected_keys,
    )

    tokens = build_calibration_tokens(
        tokenizer=tokenizer,
        nsamples=int(args.nsamples),
        seqlen=int(args.seq_len),
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        use_streaming=bool(args.use_streaming),
    )

    apply_weight_patch(named_modules, original_patch)
    full_eval = evaluate_patch(
        model=model,
        model_device=model_device,
        tokens=tokens,
        batch_size=int(args.batch_size),
        max_batches=int(args.max_batches),
    )

    apply_weight_patch(named_modules, second_patch)
    second_eval = evaluate_patch(
        model=model,
        model_device=model_device,
        tokens=tokens,
        batch_size=int(args.batch_size),
        max_batches=int(args.max_batches),
    )

    apply_weight_patch(named_modules, oas_patch)
    oas_eval = evaluate_patch(
        model=model,
        model_device=model_device,
        tokens=tokens,
        batch_size=int(args.batch_size),
        max_batches=int(args.max_batches),
    )

    apply_weight_patch(named_modules, original_patch)

    token_full = torch.tensor(full_eval["token_nll_values"], dtype=torch.float32)
    token_second = torch.tensor(second_eval["token_nll_values"], dtype=torch.float32)
    token_oas = torch.tensor(oas_eval["token_nll_values"], dtype=torch.float32)

    delta_second = token_second - token_full
    delta_oas = token_oas - token_full
    delta_oas_vs_second = token_oas - token_second

    per_module_rows: List[dict] = []
    for key in selected_keys:
        per_module_rows.append({
            "layer": key,
            "block_idx": extract_block_index(key),
            "module_type": module_type_from_key(key),
            "input_dim": int(state[key].shape[1]),
            "output_dim": int(state[key].shape[0]),
        })

    summary = {
        "experiment": "exp5_layerwise_swap_test",
        "model_id": args.model_id,
        "block": args.block,
        "selected_layers": selected_keys,
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "seq_len": int(args.seq_len),
            "nsamples": int(args.nsamples),
            "batch_size": int(args.batch_size),
            "max_batches": int(args.max_batches),
        },
        "full": {
            "mean_nll": full_eval["mean_nll"],
            "ppl": full_eval["ppl"],
            "token_nll_p95": full_eval["token_nll_p95"],
            "token_nll_p99": full_eval["token_nll_p99"],
            "token_nll_max": full_eval["token_nll_max"],
            "seq_mean_nll_mean": full_eval["seq_mean_nll_mean"],
            "seq_mean_nll_p95": full_eval["seq_mean_nll_p95"],
        },
        "second_swap": {
            "mean_nll": second_eval["mean_nll"],
            "ppl": second_eval["ppl"],
            "delta_mean_nll_vs_full": second_eval["mean_nll"] - full_eval["mean_nll"],
            "delta_ppl_vs_full": second_eval["ppl"] - full_eval["ppl"],
            "token_nll_p95": second_eval["token_nll_p95"],
            "token_nll_p99": second_eval["token_nll_p99"],
            "token_nll_max": second_eval["token_nll_max"],
            "delta_token_nll_vs_full_mean": float(delta_second.mean().item()),
            "delta_token_nll_vs_full_p95": percentile(delta_second, 0.95),
            "delta_token_nll_vs_full_p99": percentile(delta_second, 0.99),
        },
        "oas_swap": {
            "mean_nll": oas_eval["mean_nll"],
            "ppl": oas_eval["ppl"],
            "delta_mean_nll_vs_full": oas_eval["mean_nll"] - full_eval["mean_nll"],
            "delta_ppl_vs_full": oas_eval["ppl"] - full_eval["ppl"],
            "token_nll_p95": oas_eval["token_nll_p95"],
            "token_nll_p99": oas_eval["token_nll_p99"],
            "token_nll_max": oas_eval["token_nll_max"],
            "delta_token_nll_vs_full_mean": float(delta_oas.mean().item()),
            "delta_token_nll_vs_full_p95": percentile(delta_oas, 0.95),
            "delta_token_nll_vs_full_p99": percentile(delta_oas, 0.99),
        },
        "oas_minus_second": {
            "delta_mean_nll": oas_eval["mean_nll"] - second_eval["mean_nll"],
            "delta_ppl": oas_eval["ppl"] - second_eval["ppl"],
            "delta_token_nll_mean": float(delta_oas_vs_second.mean().item()),
            "delta_token_nll_p95": percentile(delta_oas_vs_second, 0.95),
            "delta_token_nll_p99": percentile(delta_oas_vs_second, 0.99),
            "delta_token_nll_max": float(delta_oas_vs_second.max().item()) if delta_oas_vs_second.numel() > 0 else 0.0,
            "oas_better_than_second_token_fraction": float((delta_oas_vs_second < 0).to(torch.float32).mean().item()),
        },
        "wins": {
            "mean_nll": {
                "oas": int(oas_eval["mean_nll"] < second_eval["mean_nll"]),
                "second": int(second_eval["mean_nll"] < oas_eval["mean_nll"]),
            },
            "ppl": {
                "oas": int(oas_eval["ppl"] < second_eval["ppl"]),
                "second": int(second_eval["ppl"] < oas_eval["ppl"]),
            },
            "token_nll_p99": {
                "oas": int(oas_eval["token_nll_p99"] < second_eval["token_nll_p99"]),
                "second": int(second_eval["token_nll_p99"] < oas_eval["token_nll_p99"]),
            },
        },
    }

    metrics_path = out_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for row in per_module_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for obj in (model, state, tokens, token_full, token_second, token_oas):
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(json.dumps({
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "num_modules": len(per_module_rows),
        "block": args.block,
    }, indent=2))


if __name__ == "__main__":
    main()
