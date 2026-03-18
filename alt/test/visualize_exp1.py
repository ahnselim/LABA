#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
visualize_exp1.py

Tail-quantile visualization across three metrics:
  1. final hidden RMS error
  2. logit KL(fp || method)
  3. Delta NLL = NLL_method - NLL_full

Layout:
  - columns : 1bit / 2bit
  - rows    : hidden / KL / Delta NLL

This script uses the same patch/eval setup as exp1_after_ab_harmfulness.py.
It can run on:
  - one selected layer/block
  - the whole target model

Example:
CUDA_VISIBLE_DEVICES=1,2 nohup \
python test/visualize_exp1.py \
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
  --out_dir ./output/exp1_tail_quantile/full_model > ./logs/visualize.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence

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


def parse_quantiles(values: Sequence[str] | None) -> List[float]:
    if not values:
        return [0.90, 0.95, 0.99, 0.995, 0.999]
    out: List[float] = []
    for value in values:
        for piece in str(value).split(","):
            piece = piece.strip()
            if not piece:
                continue
            q = float(piece)
            if q > 1.0:
                q = q / 100.0
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile must be in (0, 1): {piece}")
            out.append(q)
    if not out:
        raise ValueError("No valid quantiles parsed")
    return out


def resolve_visualization_scope(block: str, available_weight_keys: List[str]) -> tuple[str, List[str]]:
    block_norm = str(block).strip().lower()
    if block_norm in {"all", "full", "full_model", "whole_model", "entire_model"}:
        return "full_model", sorted(set(available_weight_keys))
    return str(block), resolve_selected_keys(block, available_weight_keys)


def quantile_value(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    xs, _ = torch.sort(values.reshape(-1))
    idx = min(max(int(round(q * (xs.numel() - 1))), 0), xs.numel() - 1)
    return float(xs[idx].item())


def evaluate_token_metrics(
    *,
    model,
    model_device: torch.device,
    tokens: torch.Tensor,
    batch_size: int,
    max_batches: int,
    desc: str,
) -> dict:
    token_nlls: List[torch.Tensor] = []
    logits_list: List[torch.Tensor] = []
    hidden_list: List[torch.Tensor] = []
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


def build_eval_batches(*, tokens: torch.Tensor, batch_size: int):
    yield from (
        (batch, mask) for batch, mask in (
            (tokens[i:i + batch_size], torch.ones_like(tokens[i:i + batch_size]))
            for i in range(0, tokens.shape[0], batch_size)
        )
    )


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
    quantiles: List[float],
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

    oas_nll = oas_metrics["nll"]
    second_nll = second_metrics["nll"]
    if full_nll.ndim != 1 or oas_nll.ndim != 1 or second_nll.ndim != 1:
        raise RuntimeError(
            f"NLL tensors must be token-wise 1D after flatten: "
            f"full={tuple(full_nll.shape)} oas={tuple(oas_nll.shape)} second={tuple(second_nll.shape)}"
        )
    if not (full_nll.numel() == oas_nll.numel() == second_nll.numel()):
        raise RuntimeError(
            f"NLL length mismatch: full={full_nll.numel()} oas={oas_nll.numel()} second={second_nll.numel()}"
        )

    delta_oas = (oas_nll - full_nll).reshape(-1).to(torch.float32)
    delta_second = (second_nll - full_nll).reshape(-1).to(torch.float32)
    oas_kl = tokenwise_kl(full_logits, oas_metrics["logits"])
    second_kl = tokenwise_kl(full_logits, second_metrics["logits"])
    oas_hidden_err = tokenwise_hidden_rms(full_hidden, oas_metrics["hidden"])
    second_hidden_err = tokenwise_hidden_rms(full_hidden, second_metrics["hidden"])

    return {
        "label": label,
        "hidden_oas_curve": [quantile_value(oas_hidden_err, q) for q in quantiles],
        "hidden_second_curve": [quantile_value(second_hidden_err, q) for q in quantiles],
        "kl_oas_curve": [quantile_value(oas_kl, q) for q in quantiles],
        "kl_second_curve": [quantile_value(second_kl, q) for q in quantiles],
        "nll_oas_curve": [quantile_value(delta_oas, q) for q in quantiles],
        "nll_second_curve": [quantile_value(delta_second, q) for q in quantiles],
        "hidden_oas_mean": float(oas_hidden_err.mean().item()) if oas_hidden_err.numel() > 0 else 0.0,
        "hidden_second_mean": float(second_hidden_err.mean().item()) if second_hidden_err.numel() > 0 else 0.0,
        "kl_oas_mean": float(oas_kl.mean().item()) if oas_kl.numel() > 0 else 0.0,
        "kl_second_mean": float(second_kl.mean().item()) if second_kl.numel() > 0 else 0.0,
        "oas_mean_delta_nll": float(delta_oas.mean().item()) if delta_oas.numel() > 0 else 0.0,
        "second_mean_delta_nll": float(delta_second.mean().item()) if delta_second.numel() > 0 else 0.0,
        "oas_p99_delta_nll": quantile_value(delta_oas, 0.99),
        "second_p99_delta_nll": quantile_value(delta_second, 0.99),
        "oas_num_tokens": int(delta_oas.numel()),
        "second_num_tokens": int(delta_second.numel()),
        "full_nll_shape": list(full_nll.shape),
        "oas_nll_shape": list(oas_nll.shape),
        "second_nll_shape": list(second_nll.shape),
        "full_nll_head10": [float(x) for x in full_nll[:10].tolist()],
        "full_logits_shape": list(full_logits.shape),
        "full_hidden_shape": list(full_hidden.shape),
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


def plot_tail_quantile_figure(results: Dict[str, dict], quantiles: List[float], out_path: Path, block_label: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), sharex="col")
    colors = {"oas": "#1f77b4", "second": "#d62728"}
    bit_order = [("1bit", "1bit"), ("2bit", "2bit")]
    row_specs = [
        ("hidden", "Final hidden RMS error quantile"),
        ("kl", "Logit KL(fp || method) quantile"),
        ("nll", "Delta NLL quantile"),
    ]

    x = list(range(len(quantiles)))
    xlabels = []
    for q in quantiles:
        percent = q * 100.0
        if abs(percent - round(percent)) < 1e-9:
            xlabels.append(f"{int(round(percent))}%")
        else:
            xlabels.append(f"{percent:.1f}%")

    for col_idx, (bit_key, bit_title) in enumerate(bit_order):
        data = results[bit_key]
        for row_idx, (metric_key, row_title) in enumerate(row_specs):
            ax = axes[row_idx, col_idx]
            ax.plot(x, data[f"{metric_key}_oas_curve"], marker="o", linewidth=2.2, color=colors["oas"], label="OAS")
            ax.plot(x, data[f"{metric_key}_second_curve"], marker="o", linewidth=2.2, color=colors["second"], label="Second-moment")
            if row_idx == 0:
                ax.set_title(bit_title)
            if col_idx == 0:
                ax.set_ylabel(row_title)
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel("Upper-tail quantile")
                ax.set_xticks(x)
                ax.set_xticklabels(xlabels)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])
            ax.grid(alpha=0.25)

    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"Tail distortion quantile curves for {block_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser("Visualize Exp1 tail Delta-NLL quantile curves")
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
    ap.add_argument("--quantiles", nargs="*", default=None, help="e.g. 0.9 0.95 0.99 0.995 0.999")
    ap.add_argument("--sanity_repeat_baseline", action="store_true", help="Run the FP baseline twice and report max abs diff.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    quantiles = parse_quantiles(args.quantiles)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model, state, model_device = load_model_and_state(args)
    if model.training:
        raise RuntimeError("Model must be in eval() mode for deterministic Delta-NLL comparison.")
    available_weight_keys = [k for k, v in state.items() if is_target_weight(k, v)]
    block_label, selected_keys = resolve_visualization_scope(args.block, available_weight_keys)
    selected_module_names = [k[:-7] for k in selected_keys]
    named_modules = dict(model.named_modules())

    print("[visualize_exp1] Building calibration tokens...", flush=True)
    tokens = build_calibration_tokens(
        tokenizer=tokenizer,
        nsamples=int(args.nsamples),
        seqlen=int(args.seq_len),
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        use_streaming=bool(args.use_streaming),
    )
    print(f"[visualize_exp1] Collected tokens with shape={tuple(tokens.shape)}", flush=True)

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
    full_nll = full_metrics["nll"]
    if full_nll.ndim != 1:
        raise RuntimeError(f"Expected token-wise flattened NLL to be 1D, got shape={tuple(full_nll.shape)}")

    baseline_repeat_max_abs_diff = 0.0
    if bool(args.sanity_repeat_baseline):
        apply_weight_patch(named_modules, original_patch)
        full_metrics_repeat = evaluate_token_metrics(
            model=model,
            model_device=model_device,
            tokens=tokens,
            batch_size=int(args.batch_size),
            max_batches=int(args.max_batches),
            desc="FP baseline repeat",
        )
        full_nll_repeat = full_metrics_repeat["nll"]
        if full_nll.numel() != full_nll_repeat.numel():
            raise RuntimeError(
                f"Repeated baseline length mismatch: {full_nll.numel()} vs {full_nll_repeat.numel()}"
            )
        baseline_repeat_max_abs_diff = float((full_nll - full_nll_repeat).abs().max().item()) if full_nll.numel() > 0 else 0.0

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
            quantiles=quantiles,
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
            quantiles=quantiles,
        ),
    }

    figure_path = out_dir / "tail_quantile_curve.png"
    plot_tail_quantile_figure(results, quantiles=quantiles, out_path=figure_path, block_label=block_label)

    summary = {
        "experiment": "visualize_exp1_tail_quantile_curve",
        "model_id": args.model_id,
        "block": args.block,
        "block_label": block_label,
        "selected_layers": selected_keys,
        "quantiles": quantiles,
        "sanity": {
            "model_eval_mode": bool(not model.training),
            "same_token_tensor_reused_for_all_methods": True,
            "token_tensor_shape": list(tokens.shape),
            "selected_layer_count": len(selected_keys),
            "full_nll_shape": list(full_nll.shape),
            "full_nll_head10": [float(x) for x in full_nll[:10].tolist()],
            "full_logits_shape": list(full_metrics["logits"].shape),
            "full_hidden_shape": list(full_metrics["hidden"].shape),
            "baseline_repeat_enabled": bool(args.sanity_repeat_baseline),
            "baseline_repeat_max_abs_diff": baseline_repeat_max_abs_diff,
            "nll_is_tokenwise_flattened": bool(full_nll.ndim == 1),
            "hidden_collection_disabled_for_visualization": True,
        },
        "results": results,
        "figure_path": str(figure_path),
    }
    summary_path = out_dir / "summary.json"
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
