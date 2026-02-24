#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated CVX pipeline for mixture outputs:
  step1_1_sensitivity -> step1_2_alpha_estimation(prebake-aware) -> step1_3_bit_optimization

Inputs:
  - step0_optimization output root (`bit1..bit4/`, `meta.json`)
Outputs:
  - out_root/step1_1/*
  - out_root/step1_2/*
  - out_root/step1_3/*
  - out_root/meta.json
  
CUDA_VISIBLE_DEVICES=3 nohup \
python step1_cvx_optimization.py \
  --model_id meta-llama/Llama-3.2-3B \
  --step0_out_root ./output/output_step0_prebake \
  --out_root ./output/output_step1_cvx \
  --alpha_reuse_calib \
  --bitopt_mode budget \
  --avg_bits 2.5 \
  --trust_remote_code > ./logs/step1.log 2>&1 &

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

if __package__:
    from .cvx.step1_1_sensitivity import Step11SensitivityConfig, run as run_step11
    from .cvx.step1_2_alpha_estimation import Step12AlphaPrebakeConfig, run as run_step12
    from .cvx.step1_3_bit_optimization import Step13BitOptConfig, run as run_step13
else:
    from cvx.step1_1_sensitivity import Step11SensitivityConfig, run as run_step11
    from cvx.step1_2_alpha_estimation import Step12AlphaPrebakeConfig, run as run_step12
    from cvx.step1_3_bit_optimization import Step13BitOptConfig, run as run_step13


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser("Integrated mixture CVX step1 pipeline (1_1 -> 1_2 -> 1_3)")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--step0_out_root", required=True, help="Output root from mixture/step0_optimization.py")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--python_exe", default=sys.executable)

    # Shared dataset defaults (step1_1 + step1_2)
    ap.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default=True,
    )

    # Step1-1 sensitivity args
    ap.add_argument("--sens_dtype", default="auto")
    ap.add_argument("--sens_device_map", default="auto")
    ap.add_argument("--sens_text_file", default=None)
    ap.add_argument("--sens_seq_len", type=int, default=1024)
    ap.add_argument("--sens_stride", type=int, default=None)
    ap.add_argument("--sens_batch_size", type=int, default=2)
    ap.add_argument("--sens_num_batches", type=int, default=50)
    ap.add_argument("--sens_include_lm_head", action="store_true")
    ap.add_argument("--sens_grad_scale", type=float, default=1.0)
    ap.add_argument("--sens_save_json", action="store_true")

    # Step1-2 alpha (prebake-aware) args
    ap.add_argument("--alpha_dtype", default="auto")
    ap.add_argument("--alpha_device_map", default="auto")
    ap.add_argument("--alpha_bits", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--alpha_nsamples", type=int, default=64)
    ap.add_argument("--alpha_seqlen", type=int, default=2048)
    ap.add_argument("--alpha_reuse_calib", action="store_true")
    ap.add_argument("--alpha_calib_cache_dir", default=None)
    ap.add_argument("--alpha_calib_batch_size", type=int, default=1)
    ap.add_argument("--alpha_keep_calib_on_device", action="store_true")
    ap.add_argument("--alpha_empty_cache_interval", type=int, default=0)
    ap.add_argument("--alpha_strict_prebake", action="store_true")

    # Step1-3 bit optimization args
    ap.add_argument("--bitopt_mode", choices=["target", "budget"], default=None)
    ap.add_argument("--alpha_bit", type=int, default=3, choices=[1, 2, 3, 4])
    ap.add_argument("--C_col", default="C_mean_per_batch")
    ap.add_argument("--w_col", default="numel(w_j)")
    ap.add_argument("--target_ratio", type=float, default=0.50)
    ap.add_argument("--target_residual", type=float, default=None)
    ap.add_argument("--avg_bits", type=float, default=None)
    ap.add_argument("--alpha_default", type=float, default=1.0)
    ap.add_argument("--bmin", type=int, default=1)
    ap.add_argument("--bmax", type=int, default=4)
    ap.add_argument("--max_iter", type=int, default=64)
    ap.add_argument("--tol_rel", type=float, default=1e-6)

    args = ap.parse_args()

    step0_root = Path(args.step0_out_root).resolve()
    if not step0_root.exists():
        raise FileNotFoundError(f"step0_out_root not found: {step0_root}")
    if not (step0_root / "bit1").exists():
        raise FileNotFoundError(f"{step0_root} does not look like a step0 prebake root (missing bit1/)")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    d11 = out_root / "step1_1"
    d12 = out_root / "step1_2"
    d13 = out_root / "step1_3"
    d11.mkdir(parents=True, exist_ok=True)
    d12.mkdir(parents=True, exist_ok=True)
    d13.mkdir(parents=True, exist_ok=True)

    calib_cache_dir = (
        str(Path(args.alpha_calib_cache_dir).resolve())
        if args.alpha_calib_cache_dir is not None
        else str((out_root / "calib_cache").resolve())
    )

    t0 = time.time()
    print("[step1-cvx] Running step1_1_sensitivity ...")
    run_step11(
        Step11SensitivityConfig(
            model_id=args.model_id,
            output_dir=str(d11),
            revision=args.revision,
            trust_remote_code=bool(args.trust_remote_code),
            dtype=args.sens_dtype,
            device_map=args.sens_device_map,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            use_streaming=bool(args.use_streaming),
            text_file=args.sens_text_file,
            seq_len=args.sens_seq_len,
            stride=args.sens_stride,
            batch_size=args.sens_batch_size,
            num_batches=args.sens_num_batches,
            include_lm_head=bool(args.sens_include_lm_head),
            grad_scale=args.sens_grad_scale,
            save_json=bool(args.sens_save_json),
            seed=args.seed,
            python_exe=args.python_exe,
        )
    )
    sens_csv = d11 / "layerwise_sensitivity.csv"
    if not sens_csv.exists():
        raise RuntimeError(f"step1_1 output missing: {sens_csv}")

    print("[step1-cvx] Running step1_2_alpha_estimation (prebake-aware) ...")
    alpha_outs = run_step12(
        Step12AlphaPrebakeConfig(
            model_id=args.model_id,
            prebake_root=str(step0_root),
            output_dir=str(d12),
            revision=args.revision,
            trust_remote_code=bool(args.trust_remote_code),
            dtype=args.alpha_dtype,
            device_map=args.alpha_device_map,
            seed=args.seed,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            use_streaming=bool(args.use_streaming),
            nsamples=args.alpha_nsamples,
            seqlen=args.alpha_seqlen,
            reuse_calib=bool(args.alpha_reuse_calib),
            calib_cache_dir=calib_cache_dir,
            bits=tuple(int(b) for b in args.alpha_bits),
            calib_batch_size=args.alpha_calib_batch_size,
            keep_calib_on_device=bool(args.alpha_keep_calib_on_device),
            empty_cache_interval=args.alpha_empty_cache_interval,
            strict_prebake=bool(args.alpha_strict_prebake),
        )
    )
    alpha_csv = Path(alpha_outs["alpha_csv"])

    print("[step1-cvx] Running step1_3_bit_optimization ...")
    run_step13(
        Step13BitOptConfig(
            sens_csv=str(sens_csv),
            alpha_csv=str(alpha_csv),
            alpha_bit=args.alpha_bit,
            C_col=args.C_col,
            w_col=args.w_col,
            mode=args.bitopt_mode,
            target_residual=args.target_residual,
            target_ratio=args.target_ratio,
            avg_bits=args.avg_bits,
            alpha_default=args.alpha_default,
            bmin=args.bmin,
            bmax=args.bmax,
            max_iter=args.max_iter,
            tol_rel=args.tol_rel,
            output_dir=str(d13),
            python_exe=args.python_exe,
        )
    )

    bit_assign_csv = d13 / "bit_assign.csv"
    if not bit_assign_csv.exists():
        raise RuntimeError(f"step1_3 output missing: {bit_assign_csv}")

    meta = {
        "model_id": args.model_id,
        "revision": args.revision,
        "step0_out_root": str(step0_root),
        "out_root": str(out_root),
        "created": int(time.time()),
        "elapsed_sec": time.time() - t0,
        "steps": {
            "step1_1": str(d11),
            "step1_2": str(d12),
            "step1_3": str(d13),
        },
        "artifacts": {
            "sens_csv": str(sens_csv),
            "alpha_csv": str(alpha_csv),
            "bit_assign_csv": str(bit_assign_csv),
            "bit_assign_meta": str(d13 / "bit_assign_meta.txt"),
        },
        "alpha_mode": "actual_post_ab_over_quant (prebake-aware)",
    }
    _write_json(out_root / "meta.json", meta)

    print("[step1-cvx] COMPLETED")
    print(f"  out_root      : {out_root}")
    print(f"  sens_csv      : {sens_csv}")
    print(f"  alpha_csv     : {alpha_csv}")
    print(f"  bit_assign_csv: {bit_assign_csv}")


if __name__ == "__main__":
    main()
