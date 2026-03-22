#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export LABA/alt/step_3_alternating.py outputs into prebake-style bit shards.

Output layout:
  out_root/
    meta.json
    bit1/*.pt
    bit2/*.pt
    bit3/*.pt
    bit4/*.pt
    summaries/bit{b}.json

Example:
CUDA_VISIBLE_DEVICES=2 \
  python utils/prebake_from_alternating.py \
    --out_root ./output/llama_3_8/output_step0_prebake_alt \
    --step3 1=./output/llama3_8b_64/step3_alt/1bit_nll \
    --metric best \
    --step3 4=./output/llama3_8b/step3_alt/4bit_50 

CUDA_VISIBLE_DEVICES=3 \
  python utils/prebake_from_alternating.py \
    --out_root ./output/qwen3_8b_64/output_step0_prebake_alt \
    --step3 1=./output/qwen3_8b_64/step3_1bit_ridge \
    --step3 2=./output/qwen3_8b_64/step3_alt/2bit_100 \
    --step3 3=./output/qwen3_8b_64/step3_alt/3bit_50 \
    --step3 4=./output/qwen3_8b_64/step3_alt/4bit_50 
    
CUDA_VISIBLE_DEVICES=1 \
  python utils/prebake_from_alternating.py \
    --out_root ./output/llama3_8b/output_step0_prebake_alt \
    --step3 4=./output/llama3_8b/step3_alt/4bit_50 
        
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

ARTIFACT_VARIANTS = {
    "plain": {
        "wdq": "wdq_star.pt",
        "ab": "low_rank_ab.pt",
        "source": "step_3_alternating:final",
    },
    "best": {
        "wdq": "wdq_star_best.pt",
        "ab": "low_rank_ab_best.pt",
        "source": "step_3_alternating:best",
    },
}


def _safe_name(s: object) -> str:
    s = "none" if s is None else str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def module_name_from_weight(full_weight_name: str) -> str:
    if full_weight_name.endswith(".weight"):
        return full_weight_name[: -len(".weight")]
    return full_weight_name


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_dir(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)


def _canonical_bits(bits: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for bit in bits:
        b = int(bit)
        if b < 1 or b > 4:
            raise ValueError(f"bits must be within [1,4], got {b}")
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def _infer_bit_from_summary(summary: Optional[dict]) -> Optional[int]:
    if not summary:
        return None
    layers = summary.get("layers")
    if not isinstance(layers, list) or not layers:
        return None
    bits = {int(layer["bits"]) for layer in layers if isinstance(layer, dict) and "bits" in layer}
    if len(bits) == 1:
        return next(iter(bits))
    return None


def _infer_bit_from_quant_meta(quant_meta: Optional[Dict[str, dict]]) -> Optional[int]:
    if not quant_meta:
        return None
    bits = {
        int(meta["bits"])
        for meta in quant_meta.values()
        if isinstance(meta, dict) and meta.get("bits") is not None
    }
    if len(bits) == 1:
        return next(iter(bits))
    return None


def _infer_bit_from_ab_payload(ab_payload: Dict[str, dict]) -> Optional[int]:
    bits = {
        int(rec["meta"]["bits"])
        for rec in ab_payload.values()
        if isinstance(rec, dict)
        and isinstance(rec.get("meta"), dict)
        and rec["meta"].get("bits") is not None
    }
    if len(bits) == 1:
        return next(iter(bits))
    return None


def _resolve_bit(
    explicit_bit: Optional[int],
    *,
    summary: Optional[dict],
    quant_meta: Optional[Dict[str, dict]],
    ab_payload: Dict[str, dict],
    step3_dir: Path,
) -> int:
    if explicit_bit is not None:
        return int(explicit_bit)
    for cand in (
        _infer_bit_from_summary(summary),
        _infer_bit_from_quant_meta(quant_meta),
        _infer_bit_from_ab_payload(ab_payload),
    ):
        if cand is not None:
            return int(cand)
    raise RuntimeError(
        f"Could not infer bit for {step3_dir}. Pass an explicit mapping like --step3 2={step3_dir}."
    )


def _maybe_copy_raw(bit: int, wdq_path: Path, ab_path: Path, out_root: Path, enabled: bool) -> dict:
    if not enabled:
        return {}
    raw_root = out_root / "raw" / f"bit{bit}"
    raw_root.mkdir(parents=True, exist_ok=True)
    wdq_dst = raw_root / wdq_path.name
    ab_dst = raw_root / ab_path.name
    shutil.copy2(wdq_path, wdq_dst)
    shutil.copy2(ab_path, ab_dst)
    return {
        "wdq": str(wdq_dst),
        "ab": str(ab_dst),
    }


@dataclass(frozen=True)
class ExportSpec:
    bit: Optional[int]
    step3_dir: Path


@dataclass
class ExportConfig:
    out_root: str
    step3_specs: Tuple[ExportSpec, ...]
    metric: str = "plain"
    copy_raw: bool = False


def parse_step3_spec(spec: str) -> ExportSpec:
    item = str(spec).strip()
    if not item:
        raise ValueError("empty --step3 entry")
    if "=" not in item:
        return ExportSpec(bit=None, step3_dir=Path(item).resolve())
    left, right = item.split("=", 1)
    bit = int(left.strip())
    return ExportSpec(bit=bit, step3_dir=Path(right.strip()).resolve())


def _flatten_step3_args(step3_args: Sequence[Sequence[str]]) -> Tuple[ExportSpec, ...]:
    specs: List[ExportSpec] = []
    for group in step3_args:
        for item in group:
            specs.append(parse_step3_spec(item))
    return tuple(specs)


@torch.no_grad()
def export_prebake_from_alternating(
    *,
    step3_dir: Path,
    out_root: Path,
    explicit_bit: Optional[int] = None,
    metric: str = "plain",
    copy_raw: bool = False,
) -> dict:
    artifacts = ARTIFACT_VARIANTS[str(metric)]
    wdq_path = step3_dir / artifacts["wdq"]
    ab_path = step3_dir / artifacts["ab"]
    quant_meta_path = step3_dir / "quant_meta_star.pt"
    summary_path = step3_dir / "summary.json"

    if not wdq_path.exists():
        raise FileNotFoundError(f"Missing alternating artifact: {wdq_path}")
    if not ab_path.exists():
        raise FileNotFoundError(f"Missing alternating artifact: {ab_path}")

    wdq_payload: Dict[str, torch.Tensor] = torch.load(wdq_path, map_location="cpu")
    ab_payload: Dict[str, dict] = torch.load(ab_path, map_location="cpu")
    quant_meta = torch.load(quant_meta_path, map_location="cpu") if quant_meta_path.exists() else {}
    summary = _read_json(summary_path)

    bit = _resolve_bit(
        explicit_bit,
        summary=summary,
        quant_meta=quant_meta,
        ab_payload=ab_payload,
        step3_dir=step3_dir,
    )
    _canonical_bits([bit])

    bit_dir = out_root / f"bit{bit}"
    _clean_dir(bit_dir)
    bit_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_missing_ab = 0
    mismatched_shapes = 0
    qtypes = set()
    group_sizes = set()

    for full_name, Wq in wdq_payload.items():
        rec = ab_payload.get(full_name)
        if rec is None:
            skipped_missing_ab += 1
            continue

        A = rec["A"].to(torch.float32).cpu()
        B = rec["B"].to(torch.float32).cpu()
        Wq = Wq.to(torch.float32).cpu()
        if Wq.ndim != 2 or A.ndim != 2 or B.ndim != 2:
            mismatched_shapes += 1
            raise RuntimeError(
                f"Invalid tensor rank for {full_name}: Wq{tuple(Wq.shape)} A{tuple(A.shape)} B{tuple(B.shape)}"
            )
        if A.shape[0] != Wq.shape[0] or B.shape[1] != Wq.shape[1] or A.shape[1] != B.shape[0]:
            mismatched_shapes += 1
            raise RuntimeError(
                f"Shape mismatch for {full_name}: Wq{tuple(Wq.shape)} A{tuple(A.shape)} B{tuple(B.shape)}"
            )

        module = module_name_from_weight(full_name)
        rec_meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        qmeta = quant_meta.get(full_name, {}) if isinstance(quant_meta, dict) else {}
        merged_meta = {}
        if isinstance(qmeta, dict):
            merged_meta.update(qmeta)
        if isinstance(rec_meta, dict):
            merged_meta.update(rec_meta)
        merged_meta["which"] = str(metric)
        merged_meta["step3_dir"] = str(step3_dir)

        qmode = str(merged_meta.get("qtype", "alternating"))
        group_size = merged_meta.get("group_size")
        qtypes.add(qmode)
        if group_size is not None:
            group_sizes.add(int(group_size))

        payload = {
            "module": module,
            "full_weight": full_name,
            "Wq": Wq.to(torch.float16).cpu(),
            "A": A.to(torch.float16).cpu(),
            "B": B.to(torch.float16).cpu(),
            "meta": {
                "bit": int(bit),
                "variant": "alt-step3-alternating",
                "source": artifacts["source"],
                "qmode": qmode,
                "group_size": (None if group_size is None else int(group_size)),
                "step3_meta": merged_meta,
            },
        }
        torch.save(payload, bit_dir / f"{_safe_name(module)}.pt")
        saved += 1

    raw_paths = _maybe_copy_raw(bit, wdq_path, ab_path, out_root, copy_raw)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    result = {
        "bit": int(bit),
        "step3_dir": str(step3_dir),
        "wdq_path": str(wdq_path),
        "ab_path": str(ab_path),
        "quant_meta_path": (str(quant_meta_path) if quant_meta_path.exists() else None),
        "summary_path": (str(summary_path) if summary_path.exists() else None),
        "saved": int(saved),
        "skipped_missing_ab": int(skipped_missing_ab),
        "mismatched_shapes": int(mismatched_shapes),
        "metric": str(metric),
        "copy_raw": bool(copy_raw),
        "raw_paths": raw_paths,
        "qmodes": sorted(qtypes),
        "group_sizes": sorted(group_sizes),
    }
    return result


def run(cfg: ExportConfig) -> Dict[str, str]:
    if not cfg.step3_specs:
        raise ValueError("At least one --step3 entry is required.")
    explicit_bits = [int(spec.bit) for spec in cfg.step3_specs if spec.bit is not None]
    if len(set(explicit_bits)) != len(explicit_bits):
        raise ValueError(f"Duplicated explicit bits in --step3: {explicit_bits}")
    for spec in cfg.step3_specs:
        if not spec.step3_dir.exists():
            raise FileNotFoundError(f"step3_dir not found: {spec.step3_dir}")

    out_root = Path(cfg.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summaries_root = out_root / "summaries"
    summaries_root.mkdir(parents=True, exist_ok=True)

    exported_bits: List[int] = []
    per_bit_results: List[dict] = []
    model_id = None
    revision = None
    step3_mapping: Dict[str, str] = {}

    for spec in cfg.step3_specs:
        summary = _read_json(spec.step3_dir / "summary.json")
        if model_id is None and summary is not None:
            model_id = summary.get("model_id")
        if revision is None and summary is not None:
            revision = summary.get("revision")

        result = export_prebake_from_alternating(
            step3_dir=spec.step3_dir,
            out_root=out_root,
            explicit_bit=spec.bit,
            metric=cfg.metric,
            copy_raw=bool(cfg.copy_raw),
        )
        bit = int(result["bit"])
        if bit in exported_bits:
            raise ValueError(f"Duplicated bit export detected: bit{bit}")
        exported_bits.append(bit)
        per_bit_results.append(result)
        step3_mapping[str(bit)] = str(spec.step3_dir)
        _write_json(summaries_root / f"bit{bit}.json", result)

    exported_bits = _canonical_bits(exported_bits)
    meta = {
        "model_id": model_id,
        "revision": revision,
        "bits": exported_bits,
        "created": int(time.time()),
        "copy_raw": bool(cfg.copy_raw),
        "artifacts": {
            "metric": str(cfg.metric),
            "wdq": ARTIFACT_VARIANTS[str(cfg.metric)]["wdq"],
            "ab": ARTIFACT_VARIANTS[str(cfg.metric)]["ab"],
        },
        "bit_to_step3_dir": step3_mapping,
        "note": "LABA/alt/step_3_alternating outputs exported in step0 prebake-compatible per-module format.",
    }
    _write_json(out_root / "meta.json", meta)

    summary = {
        "out_root": str(out_root),
        "meta_json": str(out_root / "meta.json"),
        "bits": exported_bits,
        "per_bit_summaries": [str(summaries_root / f"bit{bit}.json") for bit in exported_bits],
    }
    _write_json(summaries_root / "summary.json", summary)

    return {
        "meta_json": str(out_root / "meta.json"),
        "summary_json": str(summaries_root / "summary.json"),
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> ExportConfig:
    ap = argparse.ArgumentParser("Export alt/step_3_alternating outputs to prebake-style shards")
    ap.add_argument("--out_root", required=True)
    ap.add_argument(
        "--step3",
        required=True,
        nargs="+",
        action="append",
        help="Repeatable items in the form BIT=/path/to/step3_out or /path/to/step3_out.",
    )
    ap.add_argument(
        "--metric",
        default="plain",
        choices=sorted(ARTIFACT_VARIANTS.keys()),
        help="Which step3 snapshot to export: plain=final artifacts, best=best-selected artifacts.",
    )
    ap.add_argument("--copy_raw", action="store_true")
    ns = ap.parse_args(argv)
    return ExportConfig(
        out_root=str(ns.out_root),
        step3_specs=_flatten_step3_args(ns.step3),
        metric=str(ns.metric),
        copy_raw=bool(ns.copy_raw),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    outs = run(_parse_args(argv))
    print(f"[prebake-from-alt] meta: {outs['meta_json']}")
    print(f"[prebake-from-alt] summary: {outs['summary_json']}")


if __name__ == "__main__":
    main()
