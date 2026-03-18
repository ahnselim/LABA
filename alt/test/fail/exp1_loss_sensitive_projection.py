#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
exp1_loss_sensitive_projection.py

Experiment 1:
  Compare OAS vs second residuals on loss-sensitive directions for one selected block.

Core idea:
  1. Run the original model on a small calibration mini-batch.
  2. Collect backward gradients on the selected block's linear modules.
  3. Build sensitive input-side directions from grad_input statistics.
  4. Compare OAS / second residuals on:
       - global Frobenius norm
       - mean gradient direction ||R g||
       - top-k gradient subspace ||R G||_F

Supported block selectors:
  - layer0
  - layer0_attn
  - layer0_mlp
  - model.layers.0.self_attn.q_proj
  - model.layers.0.mlp.up_proj

Example:
CUDA_VISIBLE_DEVICES=1,2 python test/exp1_loss_sensitive_projection.py \
  --model_id meta-llama/Llama-3.1-8B \
  --oas_step1_dir ./output/llama3_8b/step1_quant/1bit \
  --oas_low_rank_path ./output/llama3_8b/step3_svd/1bit/low_rank_ab.pt \
  --second_step1_dir ./output/llama3_8b_64/step1_quant/1bit \
  --second_low_rank_path ./output/llama3_8b_64/step3_svd/1bit/low_rank_ab.pt \
  --block layer0 \
  --dataset DKYoon/SlimPajama-6B \
  --seq_len 256 \
  --nsamples 2 \
  --batch_size 1 \
  --topk 8 \
  --out_dir ./output/exp1_1bit/layer0
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

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


def reconstruct_ab(entry: dict, out_dim: int, in_dim: int) -> Tuple[torch.Tensor, int]:
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
    return a @ b, int(a.shape[1])


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


@dataclass
class GradStats:
    dim: int
    count: int = 0
    grad_sum: Optional[torch.Tensor] = None
    gram: Optional[torch.Tensor] = None

    def update(self, grad_input: torch.Tensor) -> None:
        flat = grad_input.detach().to(dtype=torch.float32, device="cpu").reshape(-1, self.dim)
        if flat.numel() == 0:
            return
        if self.grad_sum is None:
            self.grad_sum = torch.zeros(self.dim, dtype=torch.float32)
        if self.gram is None:
            self.gram = torch.zeros((self.dim, self.dim), dtype=torch.float32)
        self.grad_sum += flat.sum(dim=0)
        self.gram += flat.T @ flat
        self.count += int(flat.shape[0])

    def mean_direction(self) -> torch.Tensor:
        if self.count <= 0 or self.grad_sum is None:
            raise RuntimeError("No gradient samples accumulated.")
        g = self.grad_sum / float(self.count)
        return g / g.norm().clamp_min(1e-12)

    def topk_directions(self, topk: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gram is None or self.count <= 0:
            raise RuntimeError("No gradient covariance accumulated.")
        gram = self.gram / float(max(self.count, 1))
        evals, evecs = torch.linalg.eigh(gram)
        kk = min(int(topk), int(evecs.shape[1]))
        idx = torch.argsort(evals, descending=True)[:kk]
        vals = evals[idx].clamp_min(0.0)
        vecs = evecs[:, idx]
        return vecs.contiguous(), vals.contiguous()


def build_residual_map(
    *,
    state: Dict[str, torch.Tensor],
    step1_dir: str,
    low_rank_path: str,
    selected_keys: List[str],
) -> Dict[str, dict]:
    codebooks, qcodes, metas = load_step1_artifacts(step1_dir)
    low_rank_ab = load_low_rank_ab(low_rank_path)

    out: Dict[str, dict] = {}
    for key in selected_keys:
        if key not in state:
            raise KeyError(f"Missing original weight for {key}")
        if key not in codebooks or key not in qcodes or key not in metas or key not in low_rank_ab:
            raise KeyError(f"Missing step1/step3 artifact for {key}")

        w = state[key].to(torch.float32).cpu()
        meta = metas[key]
        orig_i = int(tuple(meta["orig_shape"])[1])
        wq = dequant_from_codebook_codes(codebooks[key].to(torch.float32), qcodes[key], orig_i=orig_i).cpu()
        ab, rank_used = reconstruct_ab(low_rank_ab[key], out_dim=int(w.shape[0]), in_dim=int(w.shape[1]))
        residual = (w - (wq + ab)).to(torch.float32).contiguous()

        out[key] = {
            "residual": residual,
            "rank_used": rank_used,
            "out_dim": int(w.shape[0]),
            "in_dim": int(w.shape[1]),
        }
    return out


def compute_projection_metrics(
    residual: torch.Tensor,
    g_mean: torch.Tensor,
    g_topk: torch.Tensor,
) -> Dict[str, float]:
    rg = residual @ g_mean
    r_topk = residual @ g_topk
    fro = float(torch.norm(residual, p="fro").item())
    mean_proj = float(torch.norm(rg, p=2).item())
    topk_fro = float(torch.norm(r_topk, p="fro").item())
    topk_share = safe_div(float((r_topk * r_topk).sum().item()), float((residual * residual).sum().item()))
    return {
        "residual_fro": fro,
        "proj_mean_direction_l2": mean_proj,
        "proj_mean_direction_ratio": safe_div(mean_proj, fro),
        "proj_topk_fro": topk_fro,
        "proj_topk_ratio": safe_div(topk_fro, fro),
        "proj_topk_energy_share": topk_share,
    }


def load_model_and_state(args: argparse.Namespace) -> Tuple[nn.Module, Dict[str, torch.Tensor], torch.device]:
    dtype_map = {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
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


def main() -> None:
    ap = argparse.ArgumentParser("Experiment 1 - Loss-sensitive directional projection")
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
    ap.add_argument("--topk", type=int, default=8)

    ap.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--use_streaming", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--nsamples", type=int, default=2)
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
    selected_module_names = [k[:-7] for k in selected_keys]

    named_modules = dict(model.named_modules())
    selected_modules: Dict[str, nn.Module] = {}
    for module_name in selected_module_names:
        if module_name not in named_modules:
            raise KeyError(f"Module not found in model: {module_name}")
        selected_modules[module_name] = named_modules[module_name]

    grad_stats: Dict[str, GradStats] = {}
    hooks = []
    for module_name, module in selected_modules.items():
        weight_key = f"{module_name}.weight"
        in_dim = int(state[weight_key].shape[1])
        grad_stats[weight_key] = GradStats(dim=in_dim)

        def backward_hook(mod, grad_input, grad_output, *, weight_key=weight_key):
            if not grad_input or grad_input[0] is None:
                return
            grad_stats[weight_key].update(grad_input[0])

        hooks.append(module.register_full_backward_hook(backward_hook))

    tokens = build_calibration_tokens(
        tokenizer=tokenizer,
        nsamples=int(args.nsamples),
        seqlen=int(args.seq_len),
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        use_streaming=bool(args.use_streaming),
    )

    for batch_idx, (input_ids, attention_mask) in enumerate(iterate_batches(tokens, int(args.batch_size))):
        if int(args.max_batches) > 0 and batch_idx >= int(args.max_batches):
            break
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, use_cache=False)
        out.loss.backward()

    for hook in hooks:
        hook.remove()

    oas_residuals = build_residual_map(
        state=state,
        step1_dir=args.oas_step1_dir,
        low_rank_path=args.oas_low_rank_path,
        selected_keys=selected_keys,
    )
    second_residuals = build_residual_map(
        state=state,
        step1_dir=args.second_step1_dir,
        low_rank_path=args.second_low_rank_path,
        selected_keys=selected_keys,
    )

    rows: List[dict] = []
    for key in selected_keys:
        g_mean = grad_stats[key].mean_direction()
        g_topk, eigvals = grad_stats[key].topk_directions(int(args.topk))

        oas_metrics = compute_projection_metrics(oas_residuals[key]["residual"], g_mean, g_topk)
        second_metrics = compute_projection_metrics(second_residuals[key]["residual"], g_mean, g_topk)

        row = {
            "layer": key,
            "block_idx": extract_block_index(key),
            "module_type": module_type_from_key(key),
            "input_dim": int(oas_residuals[key]["in_dim"]),
            "output_dim": int(oas_residuals[key]["out_dim"]),
            "rank_oas": int(oas_residuals[key]["rank_used"]),
            "rank_second": int(second_residuals[key]["rank_used"]),
            "grad_token_count": int(grad_stats[key].count),
            "grad_top_eval_sum": float(eigvals.sum().item()),
        }
        for prefix, metrics in (("oas", oas_metrics), ("second", second_metrics)):
            for name, value in metrics.items():
                row[f"{prefix}_{name}"] = float(value)
        row["delta_oas_minus_second_proj_mean_direction_l2"] = (
            row["oas_proj_mean_direction_l2"] - row["second_proj_mean_direction_l2"]
        )
        row["delta_oas_minus_second_proj_topk_fro"] = row["oas_proj_topk_fro"] - row["second_proj_topk_fro"]
        row["delta_oas_minus_second_residual_fro"] = row["oas_residual_fro"] - row["second_residual_fro"]
        rows.append(row)

    def mean_of(name: str) -> float:
        return float(sum(float(row[name]) for row in rows) / max(len(rows), 1))

    summary = {
        "experiment": "exp1_loss_sensitive_projection",
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
        "gradient_stats": {
            "topk": int(args.topk),
            "mean_grad_token_count": mean_of("grad_token_count"),
            "mean_top_eval_sum": mean_of("grad_top_eval_sum"),
        },
        "oas_vs_second": {
            "mean_oas_residual_fro": mean_of("oas_residual_fro"),
            "mean_second_residual_fro": mean_of("second_residual_fro"),
            "mean_oas_proj_mean_direction_l2": mean_of("oas_proj_mean_direction_l2"),
            "mean_second_proj_mean_direction_l2": mean_of("second_proj_mean_direction_l2"),
            "mean_oas_proj_topk_fro": mean_of("oas_proj_topk_fro"),
            "mean_second_proj_topk_fro": mean_of("second_proj_topk_fro"),
            "mean_delta_oas_minus_second_residual_fro": mean_of("delta_oas_minus_second_residual_fro"),
            "mean_delta_oas_minus_second_proj_mean_direction_l2": mean_of("delta_oas_minus_second_proj_mean_direction_l2"),
            "mean_delta_oas_minus_second_proj_topk_fro": mean_of("delta_oas_minus_second_proj_topk_fro"),
        },
        "wins": {
            "global_residual_fro": {
                "oas": sum(1 for row in rows if row["oas_residual_fro"] < row["second_residual_fro"]),
                "second": sum(1 for row in rows if row["second_residual_fro"] < row["oas_residual_fro"]),
            },
            "sensitive_mean_direction": {
                "oas": sum(1 for row in rows if row["oas_proj_mean_direction_l2"] < row["second_proj_mean_direction_l2"]),
                "second": sum(1 for row in rows if row["second_proj_mean_direction_l2"] < row["oas_proj_mean_direction_l2"]),
            },
            "sensitive_topk_subspace": {
                "oas": sum(1 for row in rows if row["oas_proj_topk_fro"] < row["second_proj_topk_fro"]),
                "second": sum(1 for row in rows if row["second_proj_topk_fro"] < row["oas_proj_topk_fro"]),
            },
        },
    }

    metrics_path = out_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for obj in (model, state, tokens, oas_residuals, second_residuals):
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(json.dumps({
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "num_modules": len(rows),
        "block": args.block,
    }, indent=2))


if __name__ == "__main__":
    main()
