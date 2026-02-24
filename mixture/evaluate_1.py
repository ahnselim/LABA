#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate PPL from `step1_cvx_optimization.py` bit assignments + step0 prebake shards.

Input sources:
  - step1 bit assignment CSV (typically `step1_3/bit_assign.csv`)
  - step0 prebake root (`bit1/.. bit4/..` shard dirs)

Flow:
  1) Read layer -> bit assignment from step1 CSV (`R_int`).
  2) For each assigned layer, load corresponding prebake shard from `bit{R_int}`.
  3) Build:
       - wdq_star   : { "<module>.weight": Wq }
       - low_rank_ab: { "<module>.weight": {"A": A, "B": B} }
  4) Reuse `joint.step4_eval` injection + WikiText-2 PPL evaluation (same as `evaluate_0.py`).

Examples:
CUDA_VISIBLE_DEVICES=3 \
python evaluate_1.py \
  --step1_root ./output/output_step1_cvx \
  --prebake_root ./output/output_step0_prebake \
  --device cuda:0 \
  --compare_wdq_only

CUDA_VISIBLE_DEVICES=1 \
python evaluate_1.py \
  --bit_assign_csv ./output/output_step1_cvx/step1_3/bit_assign.csv \
  --prebake_root ./output/output_step0_prebake \
  --model_name meta-llama/Llama-3.2-3B \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from joint.step4_eval import (  # noqa: E402
    AddLowRankCorrection,
    AddLowRankCorrectionFP32,
    apply_wdq_star,
    evaluate_ppl_wikitext2,
    patch_layerwise_ab_from_step2p5,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "Evaluate PPL from step1 bit assignments (bit_assign.csv) using step0 prebake shards"
    )
    ap.add_argument(
        "--step1_root",
        default=None,
        help="step1_cvx output root (contains meta.json and step1_3/bit_assign.csv)",
    )
    ap.add_argument(
        "--bit_assign_csv",
        default=None,
        help="Path to step1_3/bit_assign.csv (if omitted, infer from --step1_root)",
    )
    ap.add_argument(
        "--prebake_root",
        default=None,
        help="step0 prebake root (contains bit1..bit4). If omitted, infer from step1_root/meta.json:step0_out_root",
    )
    ap.add_argument(
        "--model_name",
        default=None,
        help="HF model id. If omitted, try step1_root/meta.json:model_id then prebake_root/meta.json:model_id",
    )

    ap.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--model_dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--ab_compute", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--ab_alpha", type=float, default=1.0)
    ap.add_argument("--compare_wdq_only", action="store_true", help="Also run Wdq-only baseline (alpha=0)")

    ap.add_argument("--ppl_stride", type=int, default=2048)
    ap.add_argument("--ppl_max_tokens", type=int, default=0)

    ap.add_argument(
        "--max_modules",
        type=int,
        default=0,
        help="Debug: load only first N assigned modules from bit_assign.csv",
    )
    ap.add_argument(
        "--strict_missing",
        action="store_true",
        help="Raise error if any assigned layer shard is missing/invalid",
    )
    ap.add_argument("--save_json", default=None, help="Optional path to save aggregated results JSON")
    return ap.parse_args()


def _safe_name(s) -> str:
    if s is None:
        s = "none"
    s = str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


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
        if isinstance(m, (AddLowRankCorrection, AddLowRankCorrectionFP32)):
            m.alpha = float(alpha)
            count += 1
    return count


def _read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_meta_model_name(prebake_root: Path) -> Optional[str]:
    meta = _read_json_if_exists(prebake_root / "meta.json")
    if not meta:
        return None
    v = meta.get("model_id", None)
    return str(v) if v else None


def _resolve_inputs(
    step1_root_arg: Optional[str],
    bit_assign_csv_arg: Optional[str],
    prebake_root_arg: Optional[str],
    model_name_arg: Optional[str],
) -> Tuple[Path, Path, Optional[str], Optional[dict]]:
    step1_root = Path(step1_root_arg).resolve() if step1_root_arg else None
    step1_meta = _read_json_if_exists(step1_root / "meta.json") if step1_root else None

    if bit_assign_csv_arg:
        bit_assign_csv = Path(bit_assign_csv_arg).resolve()
    elif step1_root:
        bit_assign_csv = (step1_root / "step1_3" / "bit_assign.csv").resolve()
    else:
        raise ValueError("Provide --bit_assign_csv or --step1_root")

    if prebake_root_arg:
        prebake_root = Path(prebake_root_arg).resolve()
    else:
        step0_from_meta = (step1_meta or {}).get("step0_out_root")
        if not step0_from_meta:
            raise ValueError(
                "Provide --prebake_root (or pass --step1_root with meta.json containing `step0_out_root`)."
            )
        prebake_root = Path(step0_from_meta).resolve()

    model_name = model_name_arg
    if not model_name:
        m = (step1_meta or {}).get("model_id")
        if m:
            model_name = str(m)
    if not model_name and prebake_root:
        model_name = _load_meta_model_name(prebake_root)

    return bit_assign_csv, prebake_root, model_name, step1_meta


def _parse_bit_assign_csv(csv_path: Path, max_modules: int = 0) -> Tuple[List[Tuple[str, int]], dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"bit_assign_csv not found: {csv_path}")

    rows: List[Tuple[str, int]] = []
    duplicates = 0
    invalid_rows = 0
    invalid_bits = 0
    seen: Dict[str, int] = {}
    row_count = 0
    layer_col_used = None
    bit_col_used = None

    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        layer_col_candidates = ["layer_name", "module", "name"]
        bit_col_candidates = ["R_int", "bit", "bits"]
        layer_col = next((c for c in layer_col_candidates if c in rdr.fieldnames), None)
        bit_col = next((c for c in bit_col_candidates if c in rdr.fieldnames), None)
        if layer_col is None:
            raise ValueError(
                f"Could not find layer column in {csv_path}. Expected one of {layer_col_candidates}, got {rdr.fieldnames}"
            )
        if bit_col is None:
            raise ValueError(
                f"Could not find bit column in {csv_path}. Expected one of {bit_col_candidates}, got {rdr.fieldnames}"
            )
        layer_col_used = layer_col
        bit_col_used = bit_col

        for row in rdr:
            row_count += 1
            raw_name = (row.get(layer_col) or "").strip()
            raw_bit = (row.get(bit_col) or "").strip()
            if (not raw_name) or (not raw_bit):
                invalid_rows += 1
                continue

            name = raw_name[:-7] if raw_name.endswith(".weight") else raw_name
            try:
                bit = int(float(raw_bit))
            except Exception:
                invalid_bits += 1
                continue
            if bit < 1 or bit > 4:
                invalid_bits += 1
                continue

            if name in seen:
                duplicates += 1
            seen[name] = bit  # last row wins (consistent with csv override behavior)

    items = sorted(seen.items(), key=lambda x: x[0])
    if max_modules and max_modules > 0:
        items = items[: int(max_modules)]

    stats = {
        "csv_path": str(csv_path),
        "row_count": row_count,
        "unique_layers": len(seen),
        "selected_layers": len(items),
        "duplicates": duplicates,
        "invalid_rows": invalid_rows,
        "invalid_bits": invalid_bits,
        "layer_col": layer_col_used,
        "bit_col": bit_col_used,
        "bit_histogram_selected": _bit_histogram([b for _, b in items]),
    }
    return items, stats


def _bit_histogram(bits: List[int]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for b in bits:
        k = str(int(b))
        hist[k] = hist.get(k, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def _build_bit_module_index(bit_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(bit_dir.glob("*.pt")):
        try:
            item = torch.load(p, map_location="cpu")
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        module = item.get("module")
        full_weight = item.get("full_weight")
        if isinstance(module, str) and module:
            out.setdefault(module, p)
        if isinstance(full_weight, str) and full_weight.endswith(".weight"):
            out.setdefault(full_weight[:-7], p)
    return out


def _load_prebake_for_assignment(
    prebake_root: Path,
    assignments: List[Tuple[str, int]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, dict], dict]:
    wdq_star: Dict[str, torch.Tensor] = {}
    low_rank_ab: Dict[str, dict] = {}

    bit_dirs = {b: prebake_root / f"bit{b}" for b in (1, 2, 3, 4)}
    index_cache: Dict[int, Dict[str, Path]] = {}
    missing_bit_dirs = 0
    missing_files = 0
    fallback_hits = 0
    bad_payloads = 0
    shape_mismatch = 0
    loaded = 0

    for layer_name, bit in assignments:
        bit_dir = bit_dirs.get(int(bit))
        if bit_dir is None or (not bit_dir.is_dir()):
            missing_bit_dirs += 1
            continue

        direct_path = bit_dir / f"{_safe_name(layer_name)}.pt"
        file_path = direct_path
        if not file_path.exists():
            if bit not in index_cache:
                index_cache[bit] = _build_bit_module_index(bit_dir)
            file_path = index_cache[bit].get(layer_name)
            if file_path is None:
                file_path = index_cache[bit].get(f"{layer_name}.weight")  # defensive
            if file_path is None:
                missing_files += 1
                continue
            fallback_hits += 1

        try:
            item = torch.load(file_path, map_location="cpu")
        except Exception:
            bad_payloads += 1
            continue
        if not isinstance(item, dict):
            bad_payloads += 1
            continue

        wkey = item.get("full_weight")
        module = item.get("module")
        if not isinstance(wkey, str):
            if isinstance(module, str):
                wkey = f"{module}.weight"
            else:
                wkey = f"{layer_name}.weight"
        if not wkey.endswith(".weight"):
            wkey = f"{wkey}.weight"

        Wq = item.get("Wq")
        A = item.get("A")
        B = item.get("B")
        if not all(isinstance(x, torch.Tensor) for x in (Wq, A, B)):
            bad_payloads += 1
            continue
        if getattr(Wq, "ndim", 0) != 2 or getattr(A, "ndim", 0) != 2 or getattr(B, "ndim", 0) != 2:
            bad_payloads += 1
            continue
        if A.shape[0] != Wq.shape[0] or B.shape[1] != Wq.shape[1] or A.shape[1] != B.shape[0]:
            shape_mismatch += 1
            continue

        wdq_star[wkey] = Wq.contiguous()
        low_rank_ab[wkey] = {"A": A.contiguous(), "B": B.contiguous()}
        loaded += 1

    stats = {
        "assigned_layers": len(assignments),
        "loaded_layers": loaded,
        "wdq_items": len(wdq_star),
        "ab_items": len(low_rank_ab),
        "missing_bit_dirs": missing_bit_dirs,
        "missing_files": missing_files,
        "fallback_hits": fallback_hits,
        "bad_payloads": bad_payloads,
        "shape_mismatch": shape_mismatch,
        "bit_histogram_assigned": _bit_histogram([b for _, b in assignments]),
    }
    return wdq_star, low_rank_ab, stats


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

    bit_assign_csv, prebake_root, model_name, step1_meta = _resolve_inputs(
        step1_root_arg=args.step1_root,
        bit_assign_csv_arg=args.bit_assign_csv,
        prebake_root_arg=args.prebake_root,
        model_name_arg=args.model_name,
    )

    if not bit_assign_csv.exists():
        raise FileNotFoundError(f"bit_assign_csv not found: {bit_assign_csv}")
    if not prebake_root.exists():
        raise FileNotFoundError(f"prebake_root not found: {prebake_root}")
    if not model_name:
        raise ValueError(
            "model_name is required (or provide --step1_root meta.json:model_id / prebake_root/meta.json:model_id)."
        )

    assignments, csv_stats = _parse_bit_assign_csv(bit_assign_csv, max_modules=args.max_modules)
    if not assignments:
        raise ValueError(f"No valid layer assignments found in {bit_assign_csv}")

    device = torch.device(args.device)
    model_dtype = _torch_dtype_from_name(args.model_dtype)

    print("== step1 mixed-bit prebake eval ==")
    print(f"step1_root   : {Path(args.step1_root).resolve() if args.step1_root else '(not set)'}")
    print(f"bit_assign_csv: {bit_assign_csv}")
    print(f"prebake_root : {prebake_root}")
    print(f"model_name   : {model_name}")
    print(f"device       : {device}")
    print(f"csv_stats    : {json.dumps(csv_stats, ensure_ascii=False)}")

    wdq_star, low_rank_ab, load_stats = _load_prebake_for_assignment(prebake_root, assignments)
    print(f"load_stats   : {json.dumps(load_stats, ensure_ascii=False)}")

    missing_total = int(load_stats["assigned_layers"]) - int(load_stats["loaded_layers"])
    if args.strict_missing and missing_total > 0:
        raise RuntimeError(f"Missing/invalid prebake shards for {missing_total} assigned layers (strict mode)")
    if not wdq_star:
        raise RuntimeError("No valid prebake shards were loaded from the step1 assignment.")

    model = None
    tok = None
    result = {
        "bit_assign_csv": str(bit_assign_csv),
        "prebake_root": str(prebake_root),
        "model_name": model_name,
        "csv_stats": csv_stats,
        "load_stats": load_stats,
        "ab_alpha": float(args.ab_alpha),
        "ab_compute": args.ab_compute,
        "ppl": {},
    }

    try:
        model, tok = _load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            torch_dtype=model_dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )

        apply_wdq_star(model, wdq_star)
        model = patch_layerwise_ab_from_step2p5(
            model,
            low_rank_ab=low_rank_ab,
            low_rank_abbar=None,
            calib_s=None,
            calib_h_lowrank=None,
            alpha=float(args.ab_alpha),
            ab_compute=args.ab_compute,
        )

        if args.compare_wdq_only:
            n_wrapped = _set_ab_alpha(model, 0.0)
            print(f"[step1] Wdq-only eval (wrapped_modules={n_wrapped}, alpha=0.0)")
            ppl_wdq, t_wdq = evaluate_ppl_wikitext2(
                model,
                tok,
                device,
                label="step1:Wdq-only",
                stride=int(args.ppl_stride),
                max_tokens=int(args.ppl_max_tokens),
            )
            result["ppl"]["wdq_only"] = {"ppl": float(ppl_wdq), "time_sec": float(t_wdq)}

        n_wrapped = _set_ab_alpha(model, float(args.ab_alpha))
        print(f"[step1] Wdq+AB eval (wrapped_modules={n_wrapped}, alpha={args.ab_alpha})")
        ppl_ab, t_ab = evaluate_ppl_wikitext2(
            model,
            tok,
            device,
            label="step1:Wdq+AB",
            stride=int(args.ppl_stride),
            max_tokens=int(args.ppl_max_tokens),
        )
        result["ppl"]["wdq_ab"] = {"ppl": float(ppl_ab), "time_sec": float(t_ab)}

        if "wdq_only" in result["ppl"]:
            result["ppl"]["delta_wdq_minus_ab"] = float(
                result["ppl"]["wdq_only"]["ppl"] - result["ppl"]["wdq_ab"]["ppl"]
            )

    finally:
        del wdq_star, low_rank_ab
        if model is not None:
            del model
        if tok is not None:
            del tok
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "step1_root": (str(Path(args.step1_root).resolve()) if args.step1_root else None),
        "step1_meta": step1_meta,
        "model_dtype": args.model_dtype,
        "device": str(device),
        "result": result,
    }

    print(f"\n{'=' * 80}")
    print("[summary]")
    ppl_ab = result["ppl"]["wdq_ab"]["ppl"]
    line = f"step1 mixed assignment: Wdq+AB PPL={ppl_ab:.4f}"
    if "wdq_only" in result["ppl"]:
        ppl_wdq = result["ppl"]["wdq_only"]["ppl"]
        dp = result["ppl"]["delta_wdq_minus_ab"]
        line += f" | Wdq-only={ppl_wdq:.4f} | Δ={dp:+.4f}"
    print(line)

    if args.save_json:
        save_path = Path(args.save_json).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[summary] saved: {save_path}")


if __name__ == "__main__":
    main()
