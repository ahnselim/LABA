#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate PPL from `step0_optimization.py` prebake outputs.

Input layout (step0 output):
  out_root/
    meta.json (optional while run is in progress)
    bit1/*.pt
    bit2/*.pt
    ...

Each `bitN/*.pt` file contains:
  {"module", "full_weight", "Wq", "A", "B", "meta"}

This script converts those prebake shards back into:
  - wdq_star dict: { "<module>.weight": Wq }
  - low_rank_ab dict: { "<module>.weight": {"A": A, "B": B} }
and reuses the existing joint eval utilities for injection + PPL.

CUDA_VISIBLE_DEVICES=1 \
python evaluate_0.py \
  --prebake_root ./mistral_7b/output_step0_prebake \
  --model_name mistralai/Mistral-7B-v0.3 \
  --bits 3 \
  --device cuda:0 \
  --compare_wdq_only

"""

from __future__ import annotations

import argparse
import gc
import json
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
    ap = argparse.ArgumentParser("Evaluate PPL from step0 prebake outputs (bitN/*.pt)")
    ap.add_argument("--prebake_root", required=True, help="step0 out_root (contains bit1..bit4)")
    ap.add_argument("--model_name", default=None, help="HF model id. If omitted, try prebake_root/meta.json:model_id")
    ap.add_argument("--bits", type=int, nargs="*", default=None, help="Bits to evaluate (default: auto-discover)")
    ap.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--model_dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--ab_compute", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--ab_alpha", type=float, default=1.0)
    ap.add_argument("--compare_wdq_only", action="store_true", help="Also run Wdq-only baseline (alpha=0)")
    ap.add_argument("--ppl_stride", type=int, default=2048)
    ap.add_argument("--ppl_max_tokens", type=int, default=0)
    ap.add_argument("--max_modules", type=int, default=0, help="Debug: load only first N module shards per bit")
    ap.add_argument("--save_json", default=None, help="Optional path to save aggregated results JSON")
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
        if isinstance(m, (AddLowRankCorrection, AddLowRankCorrectionFP32)):
            m.alpha = float(alpha)
            count += 1
    return count


def _load_meta_model_name(prebake_root: Path) -> Optional[str]:
    meta_path = prebake_root / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = meta.get("model_id", None)
    return str(v) if v else None


def _discover_bit_dirs(prebake_root: Path, bits: Optional[List[int]]) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    cand_bits = bits if bits is not None else [1, 2, 3, 4]
    for bit in cand_bits:
        bit_dir = prebake_root / f"bit{int(bit)}"
        if bit_dir.is_dir():
            out.append((int(bit), bit_dir))
    return out


def _load_prebake_bit(
    bit_dir: Path,
    max_modules: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, dict], dict]:
    files = sorted(bit_dir.glob("*.pt"))
    if max_modules and max_modules > 0:
        files = files[: int(max_modules)]

    wdq_star: Dict[str, torch.Tensor] = {}
    low_rank_ab: Dict[str, dict] = {}
    bad = 0

    for p in files:
        item = torch.load(p, map_location="cpu")
        if not isinstance(item, dict):
            bad += 1
            continue

        wkey = item.get("full_weight", None)
        module = item.get("module", None)
        if not isinstance(wkey, str):
            if isinstance(module, str):
                wkey = f"{module}.weight"
            else:
                bad += 1
                continue
        if not wkey.endswith(".weight"):
            wkey = f"{wkey}.weight"

        Wq = item.get("Wq", None)
        A = item.get("A", None)
        B = item.get("B", None)
        if not all(isinstance(x, torch.Tensor) for x in (Wq, A, B)):
            bad += 1
            continue

        wdq_star[wkey] = Wq.contiguous()
        low_rank_ab[wkey] = {"A": A.contiguous(), "B": B.contiguous()}

    stats = {
        "files_found": len(list(bit_dir.glob('*.pt'))),
        "files_loaded": len(files),
        "wdq_items": len(wdq_star),
        "ab_items": len(low_rank_ab),
        "bad_files": bad,
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

    prebake_root = Path(args.prebake_root).resolve()
    if not prebake_root.exists():
        raise FileNotFoundError(f"prebake_root not found: {prebake_root}")

    model_name = args.model_name or _load_meta_model_name(prebake_root)
    if not model_name:
        raise ValueError(
            "model_name is required (or provide prebake_root/meta.json with `model_id`)."
        )

    bit_dirs = _discover_bit_dirs(prebake_root, args.bits)
    if not bit_dirs:
        raise ValueError(f"No bit directories found under {prebake_root}")

    device = torch.device(args.device)
    model_dtype = _torch_dtype_from_name(args.model_dtype)

    results = []
    print(f"== step0 prebake eval ==")
    print(f"prebake_root: {prebake_root}")
    print(f"model_name  : {model_name}")
    print(f"device      : {device}")
    print(f"bits        : {[b for b, _ in bit_dirs]}")

    for bit, bit_dir in bit_dirs:
        shard_count = len(list(bit_dir.glob("*.pt")))
        if shard_count == 0:
            print(f"\n[bit{bit}] skip (empty dir): {bit_dir}")
            continue

        print(f"\n{'=' * 80}")
        print(f"[bit{bit}] loading prebake shards from {bit_dir} (files={shard_count})")

        wdq_star, low_rank_ab, load_stats = _load_prebake_bit(bit_dir, max_modules=args.max_modules)
        print(f"[bit{bit}] load_stats: {json.dumps(load_stats, ensure_ascii=False)}")
        if not wdq_star:
            print(f"[bit{bit}] skip (no valid shards)")
            continue

        model = None
        tok = None
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

            bit_result = {
                "bit": int(bit),
                "bit_dir": str(bit_dir),
                "load_stats": load_stats,
                "ab_alpha": float(args.ab_alpha),
                "ab_compute": args.ab_compute,
                "ppl": {},
            }

            if args.compare_wdq_only:
                n_wrapped = _set_ab_alpha(model, 0.0)
                print(f"[bit{bit}] Wdq-only eval (wrapped_modules={n_wrapped}, alpha=0.0)")
                ppl_wdq, t_wdq = evaluate_ppl_wikitext2(
                    model,
                    tok,
                    device,
                    label=f"bit{bit}:Wdq-only",
                    stride=int(args.ppl_stride),
                    max_tokens=int(args.ppl_max_tokens),
                )
                bit_result["ppl"]["wdq_only"] = {"ppl": float(ppl_wdq), "time_sec": float(t_wdq)}

            n_wrapped = _set_ab_alpha(model, float(args.ab_alpha))
            print(f"[bit{bit}] Wdq+AB eval (wrapped_modules={n_wrapped}, alpha={args.ab_alpha})")
            ppl_ab, t_ab = evaluate_ppl_wikitext2(
                model,
                tok,
                device,
                label=f"bit{bit}:Wdq+AB",
                stride=int(args.ppl_stride),
                max_tokens=int(args.ppl_max_tokens),
            )
            bit_result["ppl"]["wdq_ab"] = {"ppl": float(ppl_ab), "time_sec": float(t_ab)}

            if "wdq_only" in bit_result["ppl"]:
                bit_result["ppl"]["delta_wdq_minus_ab"] = float(
                    bit_result["ppl"]["wdq_only"]["ppl"] - bit_result["ppl"]["wdq_ab"]["ppl"]
                )

            results.append(bit_result)

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
        "prebake_root": str(prebake_root),
        "model_name": model_name,
        "device": str(device),
        "model_dtype": args.model_dtype,
        "bits_requested": args.bits,
        "results": results,
    }

    print(f"\n{'=' * 80}")
    print("[summary]")
    for r in results:
        bit = r["bit"]
        ppl_ab = r["ppl"]["wdq_ab"]["ppl"]
        line = f"bit{bit}: Wdq+AB PPL={ppl_ab:.4f}"
        if "wdq_only" in r["ppl"]:
            ppl_wdq = r["ppl"]["wdq_only"]["ppl"]
            dp = r["ppl"]["delta_wdq_minus_ab"]
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
