#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualize_exp1_mean.py

Mean-metric visualization across three metrics:
  1. E[hidden error]
  2. E[KL]
  3. E[Delta NLL]

Layout:
  - 1 row x 2 columns
  - left  : 1bit
  - right : 2bit
  - x-axis: hidden error / logit KL / delta NLL
  - bars  : Second (red) vs OAS (blue)

Example:
CUDA_VISIBLE_DEVICES=2 nohup python test/visualize_exp1_mean.py \
  --model_id meta-llama/Llama-3.1-8B \
  --block full_model \
  --bit1_oas_step1_dir ./output/llama3_8b/step1_quant/1bit \
  --bit1_oas_low_rank_path ./output/llama3_8b/step3_svd/1bit/low_rank_ab.pt \
  --bit1_second_step1_dir ./output/llama3_8b_64/step1_quant/1bit \
  --bit1_second_low_rank_path ./output/llama3_8b_64/step3_svd/1bit/low_rank_ab.pt \
  --bit2_oas_step1_dir ./output/llama3_8b/step1_quant/2bit \
  --bit2_oas_low_rank_path ./output/llama3_8b/step3_svd/2bit/low_rank_ab.pt \
  --bit2_second_step1_dir ./output/llama3_8b_64/step1_quant/2bit \
  --bit2_second_low_rank_path ./output/llama3_8b_64/step3_svd/2bit/low_rank_ab.pt \
  --seq_len 516 \
  --nsamples 16 \
  --batch_size 1 \
  --out_dir ./output/exp1_mean/full_model > ./logs/visualize_mean.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from exp1_after_ab_harmfulness import (
    apply_weight_patch,
    build_calibration_tokens,
    is_target_weight,
    load_model_and_state,
    load_quantized_weights,
    resolve_selected_keys,
)


def resolve_visualization_scope(block: str, available_weight_keys: List[str]) -> tuple[str, List[str]]:
    block_norm = str(block).strip().lower()
    if block_norm in {"all", "full", "full_model", "whole_model", "entire_model"}:
        return "full_model", sorted(set(available_weight_keys))
    return str(block), resolve_selected_keys(block, available_weight_keys)


def build_eval_batches(*, tokens: torch.Tensor, batch_size: int):
    yield from (
        (batch, mask)
        for batch, mask in (
            (tokens[i:i + batch_size], torch.ones_like(tokens[i:i + batch_size]))
            for i in range(0, tokens.shape[0], batch_size)
        )
    )


def evaluate_token_metrics(
    *,
    model,
    model_device: torch.device,
    tokens: torch.Tensor,
    batch_size: int,
    max_batches: int,
    desc: str,
) -> dict:
    token_nlls = []
    logits_list = []
    hidden_list = []
    total_batches = int(math.ceil(tokens.shape[0] / max(batch_size, 1)))
    if max_batches > 0:
        total_batches = min(total_batches, int(max_batches))
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(
            tqdm(
                build_eval_batches(tokens=tokens, batch_size=batch_size),
                total=total_batches,
                desc=desc,
                leave=False,
                dynamic_ncols=True,
            )
        ):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            logits = out.logits
            final_hidden = out.hidden_states[-1]
            shift_logits = logits[:, :-1, :].detach().to("cpu", dtype=torch.float32)
            shift_labels = input_ids[:, 1:].detach().to("cpu")
            token_nll = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(shift_labels.shape)
            token_nlls.append(token_nll.to(torch.float32))
            logits_list.append(shift_logits.contiguous())
            hidden_list.append(final_hidden[:, :-1, :].detach().to("cpu", dtype=torch.float32).contiguous())
    if not token_nlls:
        zero = torch.zeros(0, dtype=torch.float32)
        return {"nll": zero, "logits": zero.reshape(0, 0), "hidden": zero.reshape(0, 0)}
    return {
        "nll": torch.cat(token_nlls, dim=0).reshape(-1).contiguous(),
        "logits": torch.cat(logits_list, dim=0).contiguous(),
        "hidden": torch.cat(hidden_list, dim=0).contiguous(),
    }


def tokenwise_kl(full_logits: torch.Tensor, method_logits: torch.Tensor) -> torch.Tensor:
    if full_logits.shape != method_logits.shape:
        raise RuntimeError(f"logit shape mismatch: full={tuple(full_logits.shape)} method={tuple(method_logits.shape)}")
    full_log_probs = F.log_softmax(full_logits, dim=-1)
    method_log_probs = F.log_softmax(method_logits, dim=-1)
    full_probs = full_log_probs.exp()
    kl = (full_probs * (full_log_probs - method_log_probs)).sum(dim=-1)
    return kl.reshape(-1).to(torch.float32).contiguous()


def tokenwise_hidden_rms(full_hidden: torch.Tensor, method_hidden: torch.Tensor) -> torch.Tensor:
    if full_hidden.shape != method_hidden.shape:
        raise RuntimeError(f"hidden shape mismatch: full={tuple(full_hidden.shape)} method={tuple(method_hidden.shape)}")
    delta = (method_hidden - full_hidden).to(torch.float32)
    rms = torch.sqrt((delta * delta).mean(dim=-1).clamp_min(0.0))
    return rms.reshape(-1).contiguous()


def build_bit_result(
    *,
    label: str,
    model,
    selected_keys: List[str],
    state: Dict[str, torch.Tensor],
    named_modules: Dict[str, torch.nn.Module],
    model_device: torch.device,
    full_metrics: dict,
    tokens: torch.Tensor,
    batch_size: int,
    max_batches: int,
    oas_step1_dir: str,
    oas_low_rank_path: str,
    second_step1_dir: str,
    second_low_rank_path: str,
) -> dict:
    original_patch = {key: state[key] for key in selected_keys}
    oas_patch = load_quantized_weights(
        state=state,
        step1_dir=oas_step1_dir,
        low_rank_path=oas_low_rank_path,
        selected_keys=selected_keys,
    )
    second_patch = load_quantized_weights(
        state=state,
        step1_dir=second_step1_dir,
        low_rank_path=second_low_rank_path,
        selected_keys=selected_keys,
    )

    full_nll = full_metrics["nll"]
    full_logits = full_metrics["logits"]
    full_hidden = full_metrics["hidden"]

    apply_weight_patch(named_modules, oas_patch)
    oas_metrics = evaluate_token_metrics(
        model=model,
        model_device=model_device,
        tokens=tokens,
        batch_size=batch_size,
        max_batches=max_batches,
        desc=f"{label} OAS",
    )

    apply_weight_patch(named_modules, second_patch)
    second_metrics = evaluate_token_metrics(
        model=model,
        model_device=model_device,
        tokens=tokens,
        batch_size=batch_size,
        max_batches=max_batches,
        desc=f"{label} Second",
    )

    apply_weight_patch(named_modules, original_patch)

    delta_oas = (oas_metrics["nll"] - full_nll).reshape(-1).to(torch.float32)
    delta_second = (second_metrics["nll"] - full_nll).reshape(-1).to(torch.float32)
    oas_kl = tokenwise_kl(full_logits, oas_metrics["logits"])
    second_kl = tokenwise_kl(full_logits, second_metrics["logits"])
    oas_hidden_err = tokenwise_hidden_rms(full_hidden, oas_metrics["hidden"])
    second_hidden_err = tokenwise_hidden_rms(full_hidden, second_metrics["hidden"])

    return {
        "label": label,
        "metrics": {
            "hidden error": {
                "oas": float(oas_hidden_err.mean().item()) if oas_hidden_err.numel() > 0 else 0.0,
                "second": float(second_hidden_err.mean().item()) if second_hidden_err.numel() > 0 else 0.0,
            },
            "logit KL": {
                "oas": float(oas_kl.mean().item()) if oas_kl.numel() > 0 else 0.0,
                "second": float(second_kl.mean().item()) if second_kl.numel() > 0 else 0.0,
            },
            "delta NLL": {
                "oas": float(delta_oas.mean().item()) if delta_oas.numel() > 0 else 0.0,
                "second": float(delta_second.mean().item()) if delta_second.numel() > 0 else 0.0,
            },
        },
        "num_tokens": int(full_nll.numel()),
    }


def plot_mean_bar_figure(results: Dict[str, dict], out_path: Path, block_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    colors = {"second": "#d62728", "oas": "#1f77b4"}
    bit_order = [("1bit", "1bit"), ("2bit", "2bit")]
    metric_order = ["hidden error", "logit KL", "delta NLL"]
    metric_labels = ["E[hidden error]", "E[KL]", "E[Delta NLL]"]
    bar_width = 0.34
    x = list(range(len(metric_order)))

    for ax, (bit_key, title) in zip(axes, bit_order):
        bit_data = results[bit_key]["metrics"]
        second_vals = [bit_data[metric]["second"] for metric in metric_order]
        oas_vals = [bit_data[metric]["oas"] for metric in metric_order]

        ax.bar(
            [xi - bar_width / 2 for xi in x],
            second_vals,
            width=bar_width,
            color=colors["second"],
            label="Second",
        )
        ax.bar(
            [xi + bar_width / 2 for xi in x],
            oas_vals,
            width=bar_width,
            color=colors["oas"],
            label="OAS",
        )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylabel("Mean value")

    axes[0].legend(loc="upper left")
    fig.suptitle(f"Mean distortion metrics for {block_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser("Visualize Exp1 mean metrics")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")

    ap.add_argument("--block", required=True, help="One selected block/layer selector, or all/full_model for whole-model evaluation")

    ap.add_argument("--bit1_oas_step1_dir", required=True)
    ap.add_argument("--bit1_oas_low_rank_path", required=True)
    ap.add_argument("--bit1_second_step1_dir", required=True)
    ap.add_argument("--bit1_second_low_rank_path", required=True)

    ap.add_argument("--bit2_oas_step1_dir", required=True)
    ap.add_argument("--bit2_oas_low_rank_path", required=True)
    ap.add_argument("--bit2_second_step1_dir", required=True)
    ap.add_argument("--bit2_second_low_rank_path", required=True)

    ap.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--use_streaming", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--nsamples", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model, state, model_device = load_model_and_state(args)
    available_weight_keys = [k for k, v in state.items() if is_target_weight(k, v)]
    block_label, selected_keys = resolve_visualization_scope(args.block, available_weight_keys)
    named_modules = dict(model.named_modules())

    print("[visualize_exp1_mean] Building calibration tokens...", flush=True)
    tokens = build_calibration_tokens(
        tokenizer=tokenizer,
        nsamples=int(args.nsamples),
        seqlen=int(args.seq_len),
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        use_streaming=bool(args.use_streaming),
    )
    print(f"[visualize_exp1_mean] Collected tokens with shape={tuple(tokens.shape)}", flush=True)

    original_patch = {key: state[key] for key in selected_keys}
    apply_weight_patch(named_modules, original_patch)
    full_metrics = evaluate_token_metrics(
        model=model,
        model_device=model_device,
        tokens=tokens,
        batch_size=int(args.batch_size),
        max_batches=int(args.max_batches),
        desc="FP baseline",
    )

    results = {
        "1bit": build_bit_result(
            label="1bit",
            model=model,
            selected_keys=selected_keys,
            state=state,
            named_modules=named_modules,
            model_device=model_device,
            full_metrics=full_metrics,
            tokens=tokens,
            batch_size=int(args.batch_size),
            max_batches=int(args.max_batches),
            oas_step1_dir=args.bit1_oas_step1_dir,
            oas_low_rank_path=args.bit1_oas_low_rank_path,
            second_step1_dir=args.bit1_second_step1_dir,
            second_low_rank_path=args.bit1_second_low_rank_path,
        ),
        "2bit": build_bit_result(
            label="2bit",
            model=model,
            selected_keys=selected_keys,
            state=state,
            named_modules=named_modules,
            model_device=model_device,
            full_metrics=full_metrics,
            tokens=tokens,
            batch_size=int(args.batch_size),
            max_batches=int(args.max_batches),
            oas_step1_dir=args.bit2_oas_step1_dir,
            oas_low_rank_path=args.bit2_oas_low_rank_path,
            second_step1_dir=args.bit2_second_step1_dir,
            second_low_rank_path=args.bit2_second_low_rank_path,
        ),
    }

    figure_path = out_dir / "mean_metric_bar.png"
    plot_mean_bar_figure(results, out_path=figure_path, block_label=block_label)

    summary = {
        "experiment": "visualize_exp1_mean_bar",
        "model_id": args.model_id,
        "block": args.block,
        "block_label": block_label,
        "selected_layers": selected_keys,
        "results": results,
        "figure_path": str(figure_path),
    }
    summary_path = out_dir / "summary_mean.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for obj in (model, state, tokens):
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(json.dumps({
        "figure_path": str(figure_path),
        "summary_path": str(summary_path),
        "block": args.block,
        "block_label": block_label,
    }, indent=2))


if __name__ == "__main__":
    main()
