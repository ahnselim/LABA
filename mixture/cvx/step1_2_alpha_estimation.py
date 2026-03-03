#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step1-2 (Mixture) — Prebake-aware alpha estimation from step0 outputs.

Differences from `LABA/cvx/step2_alpha_estimation.py`:
  - Does NOT re-quantize using hardcoded quantizers.
  - Uses prebaked per-bit payloads from `step0_optimization.py`:
      `prebake_root/bit{b}/{module}.pt` containing `Wq`, `A`, `B`.
  - Measures alpha with actual prebaked correction:
      Lq   = || (W - Wq) Σ_x^{1/2} ||_F^2
      Lab  = || (W - (Wq + A@B)) Σ_x^{1/2} ||_F^2
      alpha = Lab / Lq

Output CSV includes columns compatible with `step3_bit_optimization.py` (`module`,`bit`,`alpha`).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers datasets accelerate") from e

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


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
        and "layers" in name
        and name.endswith(".weight")
        and name.split(".")[-2] in TARGET_SUFFIXES
    )


def module_name_from_weight(full_weight_name: str) -> str:
    return full_weight_name[: -len(".weight")]


def _safe_name(s: str) -> str:
    s = str(s) if s is not None else "none"
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def pick_dtype(dtype_str: str):
    if dtype_str == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def _to_gib_str(v: float) -> str:
    fv = float(v)
    if fv <= 0:
        raise ValueError(f"Memory cap must be > 0 GiB, got {v}")
    if float(fv).is_integer():
        return f"{int(fv)}GiB"
    return f"{fv:.2f}GiB"


def build_max_memory_map(
    gpu_mem_cap_gib: Optional[float], cpu_mem_cap_gib: Optional[float]
) -> Optional[Dict[object, str]]:
    if gpu_mem_cap_gib is None:
        return None
    if not torch.cuda.is_available():
        return None
    max_memory: Dict[object, str] = {
        i: _to_gib_str(gpu_mem_cap_gib) for i in range(torch.cuda.device_count())
    }
    if cpu_mem_cap_gib is not None:
        max_memory["cpu"] = _to_gib_str(cpu_mem_cap_gib)
    return max_memory


def set_all_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _canonical_dataset_name(name: str) -> str:
    a = name.strip()
    low = a.lower()
    if low in {"dkyoon/slimpajama-6b", "slimpajama-6b", "dkyoon_slimpajama_6b"}:
        return "DKYoon/SlimPajama-6B"
    if low in {"slimpajama-627b", "cerebras/slimpajama-627b", "slimpajama627b"}:
        return "cerebras/SlimPajama-627B"
    return name


def _normalize_dataset_config(dataset_config: Optional[str]) -> Optional[str]:
    if dataset_config is None:
        return None
    dc = str(dataset_config).strip()
    if dc == "" or dc.lower() == "none":
        return None
    return dc


def open_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str = "train",
    streaming: bool = True,
):
    if not HAS_DATASETS:
        raise RuntimeError("datasets 라이브러리가 필요합니다: pip install datasets")
    dataset_name = _canonical_dataset_name(dataset_name)
    dataset_config = _normalize_dataset_config(dataset_config)

    if streaming:
        try:
            ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
            return ds, dataset_name, dataset_config, True
        except Exception as e:
            msg = str(e)
            if ("available configs" in msg) or ("Config name is missing" in msg):
                m = re.search(r"\[(.*?)\]", msg, flags=re.S)
                if m:
                    cands = [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
                    for cand in cands:
                        try:
                            ds = load_dataset(dataset_name, name=cand, split=split, streaming=True)
                            return ds, dataset_name, cand, True
                        except Exception:
                            pass

    try:
        ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=False)
        return ds, dataset_name, dataset_config, False
    except Exception as e:
        msg = str(e)
        if ("available configs" in msg) or ("Config name is missing" in msg):
            m = re.search(r"\[(.*?)\]", msg, flags=re.S)
            if m:
                cands = [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
                for cand in cands:
                    try:
                        ds = load_dataset(dataset_name, name=cand, split=split, streaming=False)
                        return ds, dataset_name, cand, False
                    except Exception:
                        pass
        raise


@torch.no_grad()
def build_calibration_tokens(
    tokenizer,
    nsamples: int,
    seqlen: int,
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    use_streaming: bool,
) -> torch.Tensor:
    ds, dataset_name, dataset_config, is_streaming = open_hf_dataset(
        dataset_name, dataset_config, split=split, streaming=use_streaming
    )
    print(
        f"[Step1-2] Using calibration dataset={dataset_name}, config={dataset_config}, streaming={is_streaming}"
    )

    take = ds.take if hasattr(ds, "take") else None
    iterator = take(max(nsamples * 5, nsamples)) if take else ds

    eos = tokenizer.eos_token_id or tokenizer.pad_token_id
    samples: List[torch.Tensor] = []
    buf: List[int] = []
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
        raise RuntimeError("Calibration tokens unavailable.")
    if len(samples) < nsamples:
        print(f"[Step1-2][warn] Collected only {len(samples)}/{nsamples} sequences.")
    return torch.stack(samples, dim=0)


@torch.no_grad()
def estimate_diag_cov_oas_per_module(
    model: nn.Module,
    tokenizer,
    device,
    nsamples: int,
    seqlen: int,
    calib_dataset: str,
    calib_config: Optional[str],
    split: str,
    use_streaming: bool,
    calib_batch_size: int = 1,
) -> Dict[str, Dict[str, torch.Tensor]]:
    model.eval()

    target_modules = set()
    for pname, p in model.named_parameters():
        if is_target_weight(pname, p.detach()):
            target_modules.add(module_name_from_weight(pname))
    if not target_modules:
        raise RuntimeError("No target modules found for calibration.")

    name_to_dim: Dict[str, int] = {}
    stats: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    def hook_factory(mod_name: str):
        def hook(_module, inp, _out):
            x0 = inp[0]
            x = x0.detach().reshape(-1, x0.shape[-1]).to(dtype=torch.float32)
            d = int(x.shape[-1])
            name_to_dim[mod_name] = d
            if mod_name not in stats:
                stats[mod_name] = {
                    "sum": torch.zeros(d, dtype=torch.float64, device=x.device),
                    "sumsq": torch.zeros(d, dtype=torch.float64, device=x.device),
                    "n": torch.zeros((), dtype=torch.long, device=x.device),
                }
            stats[mod_name]["sum"] += x.sum(dim=0, dtype=torch.float64)
            stats[mod_name]["sumsq"] += (x.to(torch.float64).pow(2)).sum(dim=0)
            stats[mod_name]["n"] += x.shape[0]

        return hook

    for mn, mod in model.named_modules():
        if mn in target_modules and isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(hook_factory(mn)))

    tokens = build_calibration_tokens(
        tokenizer,
        nsamples=nsamples,
        seqlen=seqlen,
        dataset_name=calib_dataset,
        dataset_config=calib_config,
        split=split,
        use_streaming=use_streaming,
    ).to(device=device, non_blocking=True)

    bs = max(1, int(calib_batch_size))
    with torch.no_grad():
        for i in tqdm(range(0, tokens.shape[0], bs), desc="[Step1-2] Calibration Forward"):
            model(tokens[i : i + bs])

    for h in handles:
        h.remove()

    ops: Dict[str, Dict[str, torch.Tensor]] = {}
    for mn, st in stats.items():
        n = int(st["n"].detach().cpu().item())
        if n <= 1:
            continue
        d = int(name_to_dim[mn])
        sumv = st["sum"].detach().cpu()
        sumsq = st["sumsq"].detach().cpu()
        mean = sumv / n
        ex2 = sumsq / n
        var = torch.clamp(ex2 - mean.pow(2), min=1e-12)

        p = float(d)
        trS = var.sum().item()
        trS2 = (var.pow(2)).sum().item()
        num = (1.0 - 2.0 / p) * trS2 + (trS * trS)
        den = (n + 1.0 - 2.0 / p) * (trS2 - (trS * trS) / p)
        rho = 1.0 if den <= 0 else max(0.0, min(1.0, num / den))
        mu = trS / p
        sigma_diag = (1.0 - rho) * var + rho * mu
        sqrt_diag = torch.sqrt(torch.clamp(sigma_diag, min=1e-12)).to(torch.float32)
        ops[mn] = {"sqrt": sqrt_diag.cpu()}

    return ops


def _load_or_build_cov_ops(args, model, tokenizer, device0):
    os.makedirs(args.calib_cache_dir, exist_ok=True)
    safe_model = _safe_name(args.model_id if args.revision is None else f"{args.model_id}@{args.revision}")
    safe_dataset = _safe_name(_canonical_dataset_name(args.dataset))
    safe_config = _safe_name(_normalize_dataset_config(args.dataset_config))
    calib_basename = (
        f"calib_oas_sqrtdiag_{safe_model}__{safe_dataset}__{safe_config}"
        f"__{args.split}__ns{args.nsamples}_L{args.seqlen}.pt"
    )
    calib_path = Path(args.calib_cache_dir) / calib_basename

    if args.reuse_calib and calib_path.exists():
        print(f"[Step1-2] Loading cached Σ_x^1/2: {calib_path}")
        payload = torch.load(calib_path, map_location="cpu")
        cov_ops = payload.get("cov_ops", payload)
        return cov_ops, calib_path, True

    cov_ops = estimate_diag_cov_oas_per_module(
        model=model,
        tokenizer=tokenizer,
        device=device0,
        nsamples=int(args.nsamples),
        seqlen=int(args.seqlen),
        calib_dataset=args.dataset,
        calib_config=args.dataset_config,
        split=args.split,
        use_streaming=bool(args.use_streaming),
        calib_batch_size=int(args.calib_batch_size),
    )
    meta = {
        "model_id": args.model_id,
        "revision": args.revision,
        "dataset": _canonical_dataset_name(args.dataset),
        "dataset_config": args.dataset_config,
        "split": args.split,
        "nsamples": int(args.nsamples),
        "seqlen": int(args.seqlen),
    }
    torch.save({"cov_ops": cov_ops, "meta": meta}, calib_path)
    print(f"[Step1-2] Saved Σ_x^1/2 cache: {calib_path}")
    return cov_ops, calib_path, False


@dataclass
class Step12AlphaPrebakeConfig:
    model_id: str
    prebake_root: str
    output_dir: str
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    device_map: str = "auto"
    seed: int = 42
    dataset: str = "DKYoon/SlimPajama-6B"
    dataset_config: Optional[str] = None
    split: str = "train"
    use_streaming: bool = True
    nsamples: int = 64
    seqlen: int = 2048
    reuse_calib: bool = False
    calib_cache_dir: str = "./artifacts/bitmin"
    bits: Tuple[int, ...] = (1, 2, 3, 4)
    calib_batch_size: int = 1
    keep_calib_on_device: bool = False
    empty_cache_interval: int = 0
    strict_prebake: bool = False
    gpu_mem_cap_gib: Optional[float] = None
    cpu_mem_cap_gib: Optional[float] = None
    offload_folder: Optional[str] = None


def _iter_bits(bits: Iterable[int]) -> List[int]:
    out = []
    seen = set()
    for b in bits:
        bi = int(b)
        if bi < 1 or bi > 4:
            raise ValueError(f"bits must be in [1,4], got {bi}")
        if bi not in seen:
            seen.add(bi)
            out.append(bi)
    return out


@torch.no_grad()
def _measure_alpha_from_prebake(
    W: torch.Tensor,
    sqrt_diag: torch.Tensor,
    Wq: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> Tuple[float, float, float]:
    sqrt_diag = sqrt_diag.to(device=W.device, dtype=torch.float32)
    W = W.to(dtype=torch.float32)
    Wq = Wq.to(device=W.device, dtype=torch.float32)
    A = A.to(device=W.device, dtype=torch.float32)
    B = B.to(device=W.device, dtype=torch.float32)

    eq = (W - Wq) * sqrt_diag.unsqueeze(0)
    Lq = float((eq * eq).sum().item())

    if A.numel() == 0 or B.numel() == 0:
        wab = Wq
    else:
        wab = Wq + (A @ B)
    eab = (W - wab) * sqrt_diag.unsqueeze(0)
    Lab = float((eab * eab).sum().item())

    if Lq <= 0.0:
        return 0.0, 0.0, 0.0
    alpha_raw = Lab / Lq
    alpha = max(0.0, float(alpha_raw))
    return Lq, Lab, alpha if alpha == alpha else 0.0  # NaN-guard


def _materialize_weight_from_module(module: nn.Module, full_name: str, device: torch.device) -> torch.Tensor:
    """
    Read a module weight safely even when `device_map="auto"` leaves it as a meta tensor
    and relies on accelerate CPU offload hooks.
    """
    w = getattr(module, "weight", None)
    if w is None:
        raise RuntimeError(f"Module has no weight for {full_name}")

    hook = getattr(module, "_hf_hook", None)
    used_offload_hook = False

    try:
        if getattr(w, "is_meta", False):
            if hook is None or not hasattr(hook, "pre_forward"):
                raise RuntimeError(
                    f"Weight is meta for {full_name}, but no accelerate hook is attached. "
                    "Try running with --alpha_device_map disabled/non-auto."
                )
            # Trigger on-demand weight load from accelerate offload map.
            hook.pre_forward(module)
            used_offload_hook = True
            w = getattr(module, "weight", None)
            if w is None or getattr(w, "is_meta", False):
                raise RuntimeError(
                    f"Failed to materialize offloaded weight for {full_name} (still meta after pre_forward)."
                )

        return w.detach().to(device, dtype=torch.float32)
    finally:
        if used_offload_hook and hook is not None and hasattr(hook, "post_forward"):
            try:
                hook.post_forward(module, None)
            except Exception:
                # Best-effort cleanup; tensor copy already succeeded or an earlier error will be raised.
                pass


def run(cfg: Step12AlphaPrebakeConfig) -> Dict[str, str]:
    bits = _iter_bits(cfg.bits)
    os.makedirs(cfg.output_dir, exist_ok=True)
    Path(cfg.calib_cache_dir).mkdir(parents=True, exist_ok=True)

    set_all_seeds(int(cfg.seed))
    dtype = pick_dtype(cfg.dtype)

    print(f"[Step1-2] Loading model: {cfg.model_id}")
    model_kwargs = dict(
        revision=cfg.revision,
        torch_dtype=(dtype if dtype in (torch.float16, torch.bfloat16) else None),
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.gpu_mem_cap_gib is not None:
        if str(cfg.device_map).lower() != "auto":
            print(
                f"[Step1-2][warn] gpu_mem_cap_gib requires device_map=auto; current={cfg.device_map}. Ignoring cap."
            )
        else:
            mm = build_max_memory_map(cfg.gpu_mem_cap_gib, cfg.cpu_mem_cap_gib)
            if mm is not None:
                offload_dir = (
                    str(Path(cfg.offload_folder).resolve())
                    if cfg.offload_folder
                    else str((Path(cfg.output_dir) / "_hf_offload_step1_2").resolve())
                )
                Path(offload_dir).mkdir(parents=True, exist_ok=True)
                model_kwargs["max_memory"] = mm
                model_kwargs["offload_folder"] = offload_dir
                model_kwargs["offload_state_dict"] = True
                print(
                    f"[Step1-2] Applying max_memory={mm} with offload_folder={offload_dir}"
                )

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
    model.eval()
    tok = AutoTokenizer.from_pretrained(
        cfg.model_id, use_fast=True, trust_remote_code=cfg.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device0 = next(model.parameters()).device

    class _ArgsObj:
        pass

    args_obj = _ArgsObj()
    for k, v in vars(cfg).items():
        setattr(args_obj, k, v)
    cov_ops, calib_path, reused = _load_or_build_cov_ops(args_obj, model, tok, device0)

    prebake_root = Path(cfg.prebake_root)
    if not prebake_root.exists():
        raise FileNotFoundError(f"prebake_root not found: {prebake_root}")

    csv_path = Path(cfg.output_dir) / "alpha_layerwise_prebake.csv"
    alias_rankvar = Path(cfg.output_dir) / "alpha_layerwise_rankvar.csv"

    fieldnames_main = [
        "full_name",
        "module",
        "m",
        "n",
        "bit",
        "qmode",
        "rank",
        "group_size",
        "Lq_weighted",
        "Lres_weighted",
        "alpha",
        "alpha_mode",
        "prebake_file",
    ]
    sqrt_cache_device: Dict[str, torch.Tensor] = {}

    def get_sqrt_diag(mod_name: str) -> Optional[torch.Tensor]:
        if mod_name not in cov_ops:
            return None
        entry = cov_ops[mod_name]
        if "sqrt" in entry:
            s = entry["sqrt"]
        elif "s" in entry:
            s = entry["s"]
        else:
            return None
        if not cfg.keep_calib_on_device:
            return s
        if mod_name in sqrt_cache_device:
            return sqrt_cache_device[mod_name]
        sd = s.to(device0, non_blocking=True)
        sqrt_cache_device[mod_name] = sd
        return sd

    processed_layers = 0
    rows_written = 0
    missing_prebake = 0
    missing_cov = 0
    mismatched_full_weight = 0
    alpha_samples: List[Tuple[float, str, int, str]] = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f_main:
        writer_main = csv.DictWriter(f_main, fieldnames=fieldnames_main)
        writer_main.writeheader()

        for full_name, W_param in tqdm(model.named_parameters(), desc="[Step1-2] Target Weights"):
            if not is_target_weight(full_name, W_param.detach()):
                continue

            mod_name = module_name_from_weight(full_name)
            sqrt_diag = get_sqrt_diag(mod_name)
            if sqrt_diag is None:
                missing_cov += 1
                continue

            mod = model.get_submodule(mod_name)
            W = _materialize_weight_from_module(mod, full_name, device0)
            m, n = map(int, W.shape)

            for b in bits:
                pb_file = prebake_root / f"bit{b}" / f"{_safe_name(mod_name)}.pt"
                if not pb_file.exists():
                    missing_prebake += 1
                    if cfg.strict_prebake:
                        raise FileNotFoundError(f"Missing prebake file: {pb_file}")
                    continue

                payload = torch.load(pb_file, map_location="cpu")
                pb_full = payload.get("full_weight")
                if pb_full is not None and str(pb_full) != full_name:
                    mismatched_full_weight += 1
                    if cfg.strict_prebake:
                        raise RuntimeError(
                            f"Prebake full_weight mismatch: {pb_file} => {pb_full}, expected {full_name}"
                        )

                Wq = payload["Wq"]
                A = payload["A"]
                B = payload["B"]
                if tuple(Wq.shape) != (m, n):
                    raise RuntimeError(
                        f"Wq shape mismatch for {pb_file}: got {tuple(Wq.shape)} expected {(m, n)}"
                    )
                if A.ndim != 2 or B.ndim != 2 or A.shape[0] != m or B.shape[1] != n or A.shape[1] != B.shape[0]:
                    raise RuntimeError(
                        f"A/B shape mismatch for {pb_file}: A{tuple(A.shape)} B{tuple(B.shape)} expected ({m},r),(r,{n})"
                    )

                Lq, Lab, alpha = _measure_alpha_from_prebake(
                    W=W,
                    sqrt_diag=sqrt_diag,
                    Wq=Wq,
                    A=A,
                    B=B,
                )

                meta = payload.get("meta", {}) or {}
                step3_meta = meta.get("step3_meta", {}) if isinstance(meta, dict) else {}
                qmode = str(meta.get("variant") or meta.get("qmode") or meta.get("source") or "prebake")
                rank = int(A.shape[1])
                group_size = step3_meta.get("group_size", meta.get("group_size", ""))

                row = {
                    "full_name": full_name,
                    "module": mod_name,
                    "m": m,
                    "n": n,
                    "bit": int(b),
                    "qmode": qmode,
                    "rank": rank,
                    "group_size": group_size,
                    "Lq_weighted": f"{Lq:.6e}",
                    "Lres_weighted": f"{Lab:.6e}",
                    "alpha": f"{alpha:.8f}",
                    "alpha_mode": "actual_post_ab_over_quant",
                    "prebake_file": str(pb_file),
                }
                writer_main.writerow(row)
                rows_written += 1
                alpha_samples.append((float(alpha), full_name, int(b), qmode))

            processed_layers += 1
            del W
            if int(cfg.empty_cache_interval) > 0 and processed_layers % int(cfg.empty_cache_interval) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Backward-compatible alias for existing step3 usage patterns.
    try:
        shutil.copy2(csv_path, alias_rankvar)
    except Exception:
        pass

    summary = {
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "prebake_root": str(prebake_root),
        "bits": bits,
        "output_csv": str(csv_path),
        "output_csv_alias_rankvar": str(alias_rankvar),
        "calib_cache_path": str(calib_path),
        "calib_reused": bool(reused),
        "alpha_mode": "actual_post_ab_over_quant",
        "processed_target_layers": int(processed_layers),
        "rows_written": int(rows_written),
        "missing_prebake_files": int(missing_prebake),
        "missing_cov_layers": int(missing_cov),
        "mismatched_full_weight": int(mismatched_full_weight),
    }
    with open(Path(cfg.output_dir) / "alpha_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Step1-2] Saved CSV: {csv_path}")
    print(f"[Step1-2] Saved summary: {Path(cfg.output_dir) / 'alpha_summary.json'}")

    if alpha_samples:
        top = sorted(alpha_samples, key=lambda x: x[0])[:8]
        bot = sorted(alpha_samples, key=lambda x: x[0], reverse=True)[:8]
        print("\n[Step1-2] Lowest alpha examples:")
        for a, name, b, qm in top:
            print(f" {name:<70s} bit={b} qmode={qm} alpha={a:.8f}")
        print("\n[Step1-2] Highest alpha examples:")
        for a, name, b, qm in bot:
            print(f" {name:<70s} bit={b} qmode={qm} alpha={a:.8f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "alpha_csv": str(csv_path),
        "alpha_csv_rankvar_alias": str(alias_rankvar),
        "alpha_summary_json": str(Path(cfg.output_dir) / "alpha_summary.json"),
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> Step12AlphaPrebakeConfig:
    ap = argparse.ArgumentParser("Step1-2 (Mixture) — prebake-aware alpha estimation")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--prebake_root", required=True, help="step0_optimization output root containing bit1..bit4/")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
    )
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default=True,
    )
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--reuse_calib", action="store_true")
    ap.add_argument("--calib_cache_dir", default="./artifacts/bitmin")
    ap.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--calib_batch_size", type=int, default=1)
    ap.add_argument("--keep_calib_on_device", action="store_true")
    ap.add_argument("--empty_cache_interval", type=int, default=0)
    ap.add_argument("--strict_prebake", action="store_true")
    ap.add_argument("--gpu_mem_cap_gib", type=float, default=None)
    ap.add_argument("--cpu_mem_cap_gib", type=float, default=None)
    ap.add_argument("--offload_folder", default=None)
    ns = ap.parse_args(argv)
    ns.bits = tuple(ns.bits)
    return Step12AlphaPrebakeConfig(**vars(ns))


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
