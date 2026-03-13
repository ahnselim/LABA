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
       then fallback to non-best artifacts
  2) explicit artifact paths:
       `--wdq_star_path` and optional `--low_rank_ab_path`

Outputs:
  - console metrics
  - optional JSON summary

CUDA_VISIBLE_DEVICES=1 \
python step4_eval.py \
  --model_name meta-llama/Llama-3.1-8B \
  --wdq_star_path ./output/llama3_8b_128/step3_alt/2bit/wdq_star.pt \
  --low_rank_ab_path ./output/llama3_8b_128/step3_alt/2bit/low_rank_ab.pt \
  --device cuda:0 \
  --compare_wdq_only
  
CUDA_VISIBLE_DEVICES=1 \
python step4_eval.py \
  --model_name Qwen/Qwen3-8B \
  --wdq_star_path ./output/qwen3_8b/step3_alt/2bit_10/wdq_star.pt \
  --low_rank_ab_path ./output/qwen3_8b/step3_alt/2bit_10/low_rank_ab.pt \
  --device cuda:0 \
  --compare_wdq_only

"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
_JOINT_DIR = _THIS_DIR.parent / "code" / "joint"
if str(_JOINT_DIR) not in sys.path:
    sys.path.insert(0, str(_JOINT_DIR))

from step4_eval import (  # type: ignore  # noqa: E402
    AddLowRankCorrection,
    AddLowRankCorrectionFP32,
    apply_wdq_star,
    evaluate_ppl_wikitext2,
    measure_generation_metrics,
    patch_layerwise_ab_from_step2p5,
)


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
        if isinstance(m, (AddLowRankCorrection, AddLowRankCorrectionFP32)):
            m.alpha = float(alpha)
            count += 1
    return count


def _resolve_step3_artifacts(
    step3_dir: Path,
) -> Tuple[Path, Optional[Path]]:
    wdq_path = step3_dir / "wdq_star_best.pt"
    ab_path = step3_dir / "low_rank_ab_best.pt"

    if not wdq_path.exists():
        wdq_path = step3_dir / "wdq_star.pt"
    if not ab_path.exists():
        ab_path = step3_dir / "low_rank_ab.pt"

    if not wdq_path.exists():
        raise FileNotFoundError(f"wdq artifact not found under {step3_dir}")
    if not ab_path.exists():
        ab_path = None
    return wdq_path, ab_path


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
        wdq_path, ab_path = _resolve_step3_artifacts(step3_dir)

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

    print("== alt step4 eval ==")
    print(f"model_name : {args.model_name}")
    print(f"device     : {device}")
    print(f"wdq_path   : {wdq_path}")
    print(f"ab_path    : {ab_path}")

    print(f"📦 Loading Wdq*: {wdq_path}")
    wdq_star: Dict[str, torch.Tensor] = torch.load(wdq_path, map_location="cpu")

    low_rank_ab: Optional[Dict[str, Any]] = None
    calib_s = None
    if ab_path is not None:
        print(f"📦 Loading AB*: {ab_path}")
        low_rank_ab = torch.load(ab_path, map_location="cpu")

    if args.calib_s_path is not None:
        calib_path = Path(args.calib_s_path).resolve()
        print(f"📦 Loading calib_s: {calib_path}")
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
            print("[Eval] Wdq-only baseline")
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

        if low_rank_ab is not None:
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
            print(f"[Eval] Wdq+AB eval (wrapped_modules={n_wrapped}, alpha={args.ab_alpha})")

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

            if "wdq_only" in results:
                results["delta_ppl_wdq_minus_ab"] = (
                    float(results["wdq_only"]["ppl"]) - float(results["wdq_ab"]["ppl"])
                )

        if not args.skip_gen:
            print("[Eval] generation metrics")
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
        print(f"[Eval] saved json: {out_path}")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
