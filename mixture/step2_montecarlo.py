#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated mixture Monte-Carlo pipeline:
  step2_1_warmup -> step2_2_fintune_mixer
  (+ auto-build surrogate static_info when missing)

This wraps the copied mixer scripts under `LABA/mixture/mc` and wires
step2_1 artifacts into step2_2 automatically.

기본 사용 (step2_1 + step2_2 연속 실행):
CUDA_VISIBLE_DEVICES=2 nohup python step2_montecarlo.py \
  --model_id huggyllama/llama-7b \
  --sens_csv ./output_7b/output_step1_cvx/step1_1/layerwise_sensitivity.csv \
  --alpha_csv ./output_7b/output_step1_cvx/step1_2/alpha_layerwise_prebake.csv \
  --prebake_root ./output_7b/output_step0_prebake \
  --target_avg_bits 2.5 \
  --out_root ./output_7b/output_step2_mc \
  --gpu_id 0 \
  --use_round_band \
  --warmup_bits_lo 2.5 --warmup_bits_hi 3.0 \
  --round_quantum 0.1 > ./logs/step2_mc.log 2>&1 &

Warmup(step2_1) 재사용하고 step2_2만 실행 : 
CUDA_VISIBLE_DEVICES=0 python LABA/mixture/step2_montecarlo.py \
  --model_id meta-llama/Llama-3.2-3B \
  --sens_csv ./output/output_step1_cvx/step1_1/layerwise_sensitivity.csv \
  --alpha_csv ./output/output_step1_cvx/step1_2/alpha_layerwise.csv \
  --prebake_root ./output/output_step0_prebake \
  --target_avg_bits 2.6 \
  --out_root ./output/output_step2_mc \
  --skip_step2_1 \
  --resume

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

if __package__:
    from .mc import step2_1_warmup as mod_step21
    from .mc import step2_2_fintune_mixer as mod_step22
    from .dataset import build_static_info_dynamic_alpha as mod_static_info
else:
    from mc import step2_1_warmup as mod_step21
    from mc import step2_2_fintune_mixer as mod_step22
    from dataset import build_static_info_dynamic_alpha as mod_static_info


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _require_exists(path_str: str, label: str) -> Path:
    p = Path(path_str).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _resolve_or_build_surrogate_static_info(
    *,
    requested_path: str,
    out_root: Path,
    sens_csv: Path,
    alpha_csv: Path,
    model_id: str,
    avg_bits_target: float,
) -> Path:
    req = (requested_path or "").strip()
    if req:
        out_json = Path(req).expanduser().resolve()
    else:
        out_json = (out_root / "surrogate_static_info_auto" / "static_info_v3.json").resolve()

    if out_json.exists():
        print(f"[step2-mc] Reuse surrogate_static_info: {out_json}", flush=True)
        return out_json

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_flat = out_json.parent / "alpha_table_flat.csv"
    print(
        "[step2-mc] surrogate_static_info missing -> auto-building "
        f"from sens/alpha CSVs at {out_json}",
        flush=True,
    )
    static_info, flat = mod_static_info.build_static_info(
        sens_csv=str(sens_csv),
        alpha_csv=str(alpha_csv),
        model_id=model_id,
        avg_bits_target=float(avg_bits_target),
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(static_info, f, ensure_ascii=False, indent=2)
    flat.to_csv(out_flat, index=False)
    print(f"[step2-mc] Built surrogate_static_info: {out_json}", flush=True)
    print(f"[step2-mc] Built alpha_table_flat: {out_flat}", flush=True)
    return out_json


def _parser_dest_set(parser: argparse.ArgumentParser) -> set[str]:
    return {a.dest for a in parser._actions if getattr(a, "dest", None)}


def _apply_kv_overrides(ns: argparse.Namespace, parser: argparse.ArgumentParser, items: Iterable[str], label: str) -> None:
    valid = _parser_dest_set(parser)
    for item in items:
        if "=" in item:
            key, raw = item.split("=", 1)
            key = key.strip()
            raw = raw.strip()
            try:
                value = json.loads(raw)
            except Exception:
                value = raw
        else:
            key = item.strip()
            value = True
        if not key:
            continue
        if key not in valid:
            raise ValueError(f"Unknown {label} override key: {key}")
        setattr(ns, key, value)


def _build_ns(parser: argparse.ArgumentParser, required_pairs: dict[str, Any]) -> argparse.Namespace:
    argv: list[str] = []
    for k, v in required_pairs.items():
        argv.extend([f"--{k}", str(v)])
    return parser.parse_args(argv)


def _set_many(ns: argparse.Namespace, values: dict[str, Any]) -> None:
    for k, v in values.items():
        if v is not None:
            setattr(ns, k, v)


def main() -> None:
    ap = argparse.ArgumentParser("Integrated mixture step2 Monte-Carlo pipeline (step2_1 -> step2_2)")

    # Core
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--sens_csv", required=True)
    ap.add_argument("--alpha_csv", required=True)
    ap.add_argument("--prebake_root", required=True)
    ap.add_argument(
        "--surrogate_static_info",
        default="",
        help="Optional. If missing or file does not exist, build automatically from sens/alpha CSVs.",
    )
    ap.add_argument("--target_avg_bits", type=float, required=True)
    ap.add_argument("--out_root", required=True)

    # Common knobs
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha_bit", type=int, default=3)
    ap.add_argument("--alpha_default", type=float, default=1.0)
    ap.add_argument("--bmin", type=int, default=1)
    ap.add_argument("--bmax", type=int, default=4)
    ap.add_argument("--use_round_band", action="store_true")
    ap.add_argument("--round_quantum", type=float, default=0.1)
    ap.add_argument("--eval_seq_len", type=int, default=2048)
    ap.add_argument("--ppl_cache_max", type=int, default=20000)
    ap.add_argument("--beam_size", type=int, default=10)
    ap.add_argument("--expansion_k", type=int, default=20)
    ap.add_argument("--filter_p", type=int, default=80)
    ap.add_argument("--true_eval_topk", type=int, default=10)
    ap.add_argument("--surrogate_batch", type=int, default=1024)
    ap.add_argument("--surrogate_score_cache_max", type=int, default=200000)
    ap.add_argument("--surrogate_device", default="")
    ap.add_argument("--surrogate_trainer_device", default="")

    # Step2_1 warmup
    ap.add_argument("--init_assign_csv", default=None)
    ap.add_argument("--init_avg_bits", type=float, default=2.75)
    ap.add_argument("--warmup_generations", type=int, default=10)
    ap.add_argument("--warmup_bits_lo", type=float, default=2.5)
    ap.add_argument("--warmup_bits_hi", type=float, default=3.0)
    ap.add_argument("--warmup_bits_sampling", default="uniform", choices=["uniform", "grid", "cycle"])
    ap.add_argument("--warmup_bits_grid_step", type=float, default=0.1)
    ap.add_argument("--warmup_core_keep_gens", type=int, default=3)
    ap.add_argument("--reuse_warmup", action="store_true")

    # Step2_1 surrogate train (frequently tuned)
    ap.add_argument("--sur_train_seed", type=int, default=42)
    ap.add_argument("--sur_train_val_ratio", type=float, default=0.2)
    ap.add_argument("--sur_train_pairs_per_gen", type=int, default=2000)
    ap.add_argument("--sur_train_epochs", type=int, default=50)
    ap.add_argument("--sur_train_batch_size", type=int, default=256)
    ap.add_argument("--sur_train_num_workers", type=int, default=2)
    ap.add_argument("--sur_train_lr", type=float, default=3e-4)
    ap.add_argument("--sur_train_weight_decay", type=float, default=0.01)
    ap.add_argument("--sur_train_grad_clip", type=float, default=1.0)
    ap.add_argument("--sur_train_log_interval", type=int, default=200)
    ap.add_argument("--sur_train_topk", type=int, default=10)
    ap.add_argument("--sur_train_eval_batch", type=int, default=512)

    # Step2_2 target finetune
    ap.add_argument("--probe_n", type=int, default=8)
    ap.add_argument("--init_from_warmup_beam", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--generations", type=int, default=0)
    ap.add_argument("--max_generations", type=int, default=0)
    ap.add_argument("--converge_eps", type=float, default=1e-3)
    ap.add_argument("--converge_rel_eps", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--stable_bits_patience", type=int, default=12)
    ap.add_argument("--time_limit_sec", type=int, default=0)
    ap.add_argument("--online_epochs", type=int, default=3)
    ap.add_argument("--online_val_ratio", type=float, default=0.2)
    ap.add_argument("--online_pairs_per_gen", type=int, default=1200)
    ap.add_argument("--online_replay_keep_gens", type=int, default=12)
    ap.add_argument("--online_batch_size", type=int, default=256)
    ap.add_argument("--online_num_workers", type=int, default=2)
    ap.add_argument("--online_eval_topk", type=int, default=10)
    ap.add_argument("--online_eval_batch", type=int, default=512)

    # Orchestration
    ap.add_argument("--step2_1_dirname", default="step2_1")
    ap.add_argument("--step2_2_dirname", default="step2_2")
    ap.add_argument("--skip_step2_1", action="store_true", help="Reuse existing step2_1 outputs in out_root")
    ap.add_argument("--step2_1_set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--step2_2_set", action="append", default=[], metavar="KEY=VALUE")

    args = ap.parse_args()

    sens_csv = _require_exists(args.sens_csv, "sens_csv")
    alpha_csv = _require_exists(args.alpha_csv, "alpha_csv")
    prebake_root = _require_exists(args.prebake_root, "prebake_root")
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    surrogate_static_info = _resolve_or_build_surrogate_static_info(
        requested_path=args.surrogate_static_info,
        out_root=out_root,
        sens_csv=sens_csv,
        alpha_csv=alpha_csv,
        model_id=args.model_id,
        avg_bits_target=float(args.target_avg_bits),
    )
    d21 = out_root / args.step2_1_dirname
    d22 = out_root / args.step2_2_dirname
    d21.mkdir(parents=True, exist_ok=True)
    d22.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res21: Optional[dict[str, Any]] = None

    summary_path = d21 / "v10_1_summary.json"
    if args.skip_step2_1:
        if not summary_path.exists():
            raise FileNotFoundError(f"--skip_step2_1 set but summary missing: {summary_path}")
        print(f"[step2-mc] Skip step2_1, reuse {summary_path}", flush=True)
    else:
        print("[step2-mc] Running step2_1_warmup ...", flush=True)
        p21 = mod_step21.build_arg_parser()
        ns21 = _build_ns(
            p21,
            {
                "model_id": args.model_id,
                "sens_csv": str(sens_csv),
                "alpha_csv": str(alpha_csv),
                "prebake_root": str(prebake_root),
                "output_dir": str(d21),
                "surrogate_static_info": str(surrogate_static_info),
            },
        )
        _set_many(
            ns21,
            {
                "gpu_id": args.gpu_id,
                "seed": args.seed,
                "alpha_bit": args.alpha_bit,
                "alpha_default": args.alpha_default,
                "bmin": args.bmin,
                "bmax": args.bmax,
                "use_round_band": bool(args.use_round_band),
                "round_quantum": args.round_quantum,
                "eval_seq_len": args.eval_seq_len,
                "ppl_cache_max": args.ppl_cache_max,
                "beam_size": args.beam_size,
                "expansion_k": args.expansion_k,
                "filter_p": args.filter_p,
                "true_eval_topk": args.true_eval_topk,
                "surrogate_batch": args.surrogate_batch,
                "surrogate_score_cache_max": args.surrogate_score_cache_max,
                "surrogate_device": args.surrogate_device,
                "surrogate_trainer_device": args.surrogate_trainer_device,
                "init_assign_csv": args.init_assign_csv,
                "init_avg_bits": args.init_avg_bits,
                "warmup_generations": args.warmup_generations,
                "warmup_bits_lo": args.warmup_bits_lo,
                "warmup_bits_hi": args.warmup_bits_hi,
                "warmup_bits_sampling": args.warmup_bits_sampling,
                "warmup_bits_grid_step": args.warmup_bits_grid_step,
                "warmup_core_keep_gens": args.warmup_core_keep_gens,
                "reuse_warmup": bool(args.reuse_warmup),
                "warmup_ckpt_path": str(d21 / "warmup_ckpt.json"),
                "sur_train_seed": args.sur_train_seed,
                "sur_train_val_ratio": args.sur_train_val_ratio,
                "sur_train_pairs_per_gen": args.sur_train_pairs_per_gen,
                "sur_train_epochs": args.sur_train_epochs,
                "sur_train_batch_size": args.sur_train_batch_size,
                "sur_train_num_workers": args.sur_train_num_workers,
                "sur_train_lr": args.sur_train_lr,
                "sur_train_weight_decay": args.sur_train_weight_decay,
                "sur_train_grad_clip": args.sur_train_grad_clip,
                "sur_train_log_interval": args.sur_train_log_interval,
                "sur_train_topk": args.sur_train_topk,
                "sur_train_eval_batch": args.sur_train_eval_batch,
            },
        )
        _apply_kv_overrides(ns21, p21, args.step2_1_set, "step2_1")
        print(
            "[step2-mc] step2_1 config: "
            f"warmup_gens={ns21.warmup_generations}, "
            f"warmup_bits=[{float(ns21.warmup_bits_lo):.3f},{float(ns21.warmup_bits_hi):.3f}]/{ns21.warmup_bits_sampling}, "
            f"sur_train_epochs={ns21.sur_train_epochs}, "
            f"sur_train_pairs_per_gen={ns21.sur_train_pairs_per_gen}, "
            f"sur_train_log_interval={ns21.sur_train_log_interval}",
            flush=True,
        )
        print(
            "[step2-mc] step2_1 will continue with offline surrogate training after warmup true-PPL evals.",
            flush=True,
        )
        res21 = mod_step21.run(ns21)

    if not summary_path.exists():
        raise RuntimeError(f"step2_1 summary missing: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        s21 = json.load(f)

    base_training_csv = _require_exists(s21["training_csv"], "step2_1 training_csv")
    warmup_ckpt_json = _require_exists(s21["warmup_ckpt_path"], "step2_1 warmup_ckpt_path")
    base_surrogate_ckpt = _require_exists(s21["surrogate_best_ckpt"], "step2_1 surrogate_best_ckpt")
    base_surrogate_config = _require_exists(s21["surrogate_config"], "step2_1 surrogate_config")

    print("[step2-mc] Running step2_2_fintune_mixer ...", flush=True)
    p22 = mod_step22.build_arg_parser()
    ns22 = _build_ns(
        p22,
        {
            "model_id": args.model_id,
            "sens_csv": str(sens_csv),
            "alpha_csv": str(alpha_csv),
            "prebake_root": str(prebake_root),
            "output_dir": str(d22),
            "target_avg_bits": args.target_avg_bits,
            "base_training_csv": str(base_training_csv),
            "warmup_ckpt_json": str(warmup_ckpt_json),
            "base_surrogate_ckpt": str(base_surrogate_ckpt),
            "base_surrogate_config": str(base_surrogate_config),
            "surrogate_static_info": str(surrogate_static_info),
        },
    )
    _set_many(
        ns22,
        {
            "gpu_id": args.gpu_id,
            "seed": args.seed,
            "alpha_bit": args.alpha_bit,
            "alpha_default": args.alpha_default,
            "bmin": args.bmin,
            "bmax": args.bmax,
            "use_round_band": bool(args.use_round_band),
            "round_quantum": args.round_quantum,
            "eval_seq_len": args.eval_seq_len,
            "ppl_cache_max": args.ppl_cache_max,
            "beam_size": args.beam_size,
            "expansion_k": args.expansion_k,
            "filter_p": args.filter_p,
            "true_eval_topk": args.true_eval_topk,
            "probe_n": args.probe_n,
            "surrogate_batch": args.surrogate_batch,
            "surrogate_score_cache_max": args.surrogate_score_cache_max,
            "surrogate_device": args.surrogate_device,
            "surrogate_trainer_device": args.surrogate_trainer_device,
            "init_from_warmup_beam": bool(args.init_from_warmup_beam),
            "resume": bool(args.resume),
            "generations": args.generations,
            "max_generations": args.max_generations,
            "converge_eps": args.converge_eps,
            "converge_rel_eps": args.converge_rel_eps,
            "patience": args.patience,
            "stable_bits_patience": args.stable_bits_patience,
            "time_limit_sec": args.time_limit_sec,
            "online_epochs": args.online_epochs,
            "online_val_ratio": args.online_val_ratio,
            "online_pairs_per_gen": args.online_pairs_per_gen,
            "online_replay_keep_gens": args.online_replay_keep_gens,
            "online_batch_size": args.online_batch_size,
            "online_num_workers": args.online_num_workers,
            "online_eval_topk": args.online_eval_topk,
            "online_eval_batch": args.online_eval_batch,
        },
    )
    _apply_kv_overrides(ns22, p22, args.step2_2_set, "step2_2")
    res22 = mod_step22.run(ns22)

    bit_assign_csv = d22 / "bit_assign.csv"
    if not bit_assign_csv.exists():
        raise RuntimeError(f"step2_2 output missing: {bit_assign_csv}")

    meta = {
        "model_id": args.model_id,
        "sens_csv": str(sens_csv),
        "alpha_csv": str(alpha_csv),
        "prebake_root": str(prebake_root),
        "surrogate_static_info": str(surrogate_static_info),
        "target_avg_bits": float(args.target_avg_bits),
        "out_root": str(out_root),
        "created": int(time.time()),
        "elapsed_sec": time.time() - t0,
        "steps": {
            "step2_1": str(d21),
            "step2_2": str(d22),
        },
        "artifacts": {
            "step2_1_summary": str(summary_path),
            "warmup_ckpt_json": str(warmup_ckpt_json),
            "base_training_csv": str(base_training_csv),
            "base_surrogate_ckpt": str(base_surrogate_ckpt),
            "base_surrogate_config": str(base_surrogate_config),
            "bit_assign_csv": str(bit_assign_csv),
            "ppl_curve_csv": str((d22 / "ppl_curve.csv").resolve()),
            "run_ckpt_json": str((d22 / "run_ckpt.json").resolve()),
        },
        "results": {
            "step2_1": res21,
            "step2_2": res22,
        },
    }
    _write_json(out_root / "meta.json", meta)
    print(f"[step2-mc] Done. meta: {out_root / 'meta.json'}", flush=True)


if __name__ == "__main__":
    main()
