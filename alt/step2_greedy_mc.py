#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated mixture greedy-seeded Monte-Carlo pipeline:
  step1_3c_opt(seed) -> step2_1_warmup -> step1_3c_opt(seed,target) -> step2_2_fintune_mixer
  (+ auto-build surrogate static_info when missing)

This wraps the copied mixer scripts under `LABA/mixture/mc` and wires
step2_1 artifacts into step2_2 automatically.
기본값으로 init seed(`init_assign_csv`)를 `cvx/step1_3c_opt.py`로 자동 생성한다.

기본 사용 (step2_1 + step2_2 연속 실행):
CUDA_VISIBLE_DEVICES=3 nohup python step2_greedy_mc.py \
  --model_id meta-llama/Llama-3.1-8B \
  --sens_csv ./output/llama_3_8/output_step1_greedy/step1_1/layerwise_sensitivity.csv \
  --alpha_csv ./output/llama_3_8/output_step1_greedy/step1_2/alpha_layerwise_prebake.csv \
  --prebake_root ./output/llama_3_8/output_step0_prebake_alt \
  --target_avg_bits 2.25 \
  --out_root ./output/llama_3_8/output_step2_mc_greedy/2_25bit \
  --gpu_id 0 \
  --use_round_band \
  --warmup_bits_lo 2.0 --warmup_bits_hi 2.5 \
  --round_quantum 0.01 > ./logs/step2_mc_llama3_8b_225.log 2>&1 &

Warmup(step2_1) 재사용하고 step2_2만 실행 : 
CUDA_VISIBLE_DEVICES=3 nohup python step2_greedy_mc.py \
  --model_id meta-llama/Llama-3.2-3B \
  --sens_csv ./output/llama_3_8/output_step1_greedy/step1_1/layerwise_sensitivity.csv \
  --alpha_csv ./output/llama_3_8/output_step1_greedy/step1_2/alpha_layerwise_prebake.csv \
  --prebake_root ./output/llama_3_8/output_step0_prebake_alt \
  --target_avg_bits 2.5 \
  --out_root ./output/llama_3_8/output_step2_mc_greedy/2_25bit \
  --step2_2_dir ./2_5bit \
    --use_round_band \
  --skip_step2_1 \
  --resume \
  --round_quantum 0.01 > ./logs/step2_mc_llama3_8b_25.log 2>&1 &

step2_1만 실행:
CUDA_VISIBLE_DEVICES=0 python LABA/alt/step2_greedy_mc.py \
  --model_id meta-llama/Llama-3.1-8B \
  --sens_csv ./output/llama_3_8/output_step1_greedy/step1_1/layerwise_sensitivity.csv \
  --alpha_csv ./output/llama_3_8/output_step1_greedy/step1_2/alpha_layerwise_prebake.csv \
  --prebake_root ./output/llama_3_8/output_step0_prebake_alt \
  --target_avg_bits 2.25 \
  --out_root ./output/llama_3_8/output_step2_mc_greedy/2_25bit \
  --gpu_id 0 \
  --use_round_band \
  --warmup_bits_lo 2.0 --warmup_bits_hi 2.5 \
  --round_quantum 0.01 \
  --only_step2_1

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
    from .cvx.step1_3c_opt import Step13COptConfig, run as run_step13
    from .mc import step2_1_warmup as mod_step21
    from .mc import step2_2_fintune_mixer as mod_step22
    from .dataset import build_static_info_dynamic_alpha as mod_static_info
else:
    from cvx.step1_3c_opt import Step13COptConfig, run as run_step13
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


def _resolve_seed_bits(bits_arg: str, bmin: int, bmax: int) -> str:
    bits = (bits_arg or "").strip()
    if bits:
        return bits
    if int(bmin) > int(bmax):
        raise ValueError(f"bmin must be <= bmax, got bmin={bmin}, bmax={bmax}")
    return ",".join(str(b) for b in range(int(bmin), int(bmax) + 1))


def _resolve_or_build_step13_seed(
    *,
    requested_path: Optional[str],
    auto_seed: bool,
    out_root: Path,
    sens_csv: Path,
    alpha_csv: Path,
    init_avg_bits: float,
    bmin: int,
    bmax: int,
    seed_opt_dirname: str,
    seed_opt_avg_bits: Optional[float],
    seed_opt_bits: str,
    seed_opt_init_bit: int,
    seed_opt_c_col: str,
    seed_opt_w_col: str,
    seed_opt_normalize_lres_by_refbit: bool,
    seed_opt_norm_ref_bit: Optional[int],
    seed_opt_norm_eps: float,
    seed_opt_proxy_shape: str,
    seed_opt_marginal_gain_power: float,
    seed_opt_cj_transform: str,
    seed_opt_cj_power: float,
    seed_opt_cj_clip_min: float,
    seed_opt_cj_floor_ratio: float,
    python_exe: str,
) -> Optional[Path]:
    req = (requested_path or "").strip()
    if req:
        p = _require_exists(req, "init_assign_csv")
        print(f"[step2-greedy-mc] Use explicit init_assign_csv: {p}", flush=True)
        return p
    if not auto_seed:
        print("[step2-greedy-mc] Auto seed disabled. step2_1 will use internal seed initializer.", flush=True)
        return None

    seed_dir = (out_root / seed_opt_dirname).resolve()
    seed_csv = seed_dir / "bit_assign.csv"
    if seed_csv.exists():
        print(f"[step2-greedy-mc] Reuse step1_3c seed: {seed_csv}", flush=True)
        return seed_csv

    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_avg_bits = float(seed_opt_avg_bits if seed_opt_avg_bits is not None else init_avg_bits)
    seed_bits = _resolve_seed_bits(seed_opt_bits, bmin=bmin, bmax=bmax)
    print(
        "[step2-greedy-mc] init_assign_csv missing -> build seed via step1_3c_opt "
        f"(avg_bits={seed_avg_bits:.4f}, bits={seed_bits}) at {seed_dir}",
        flush=True,
    )
    run_step13(
        Step13COptConfig(
            sens_csv=str(sens_csv),
            alpha_csv=str(alpha_csv),
            C_col=seed_opt_c_col,
            w_col=seed_opt_w_col,
            bits=seed_bits,
            mode="budget",
            avg_bits=seed_avg_bits,
            init_bit=int(seed_opt_init_bit),
            normalize_lres_by_refbit=bool(seed_opt_normalize_lres_by_refbit),
            norm_ref_bit=seed_opt_norm_ref_bit,
            norm_eps=float(seed_opt_norm_eps),
            proxy_shape=seed_opt_proxy_shape,
            marginal_gain_power=float(seed_opt_marginal_gain_power),
            cj_transform=seed_opt_cj_transform,
            cj_power=float(seed_opt_cj_power),
            cj_clip_min=float(seed_opt_cj_clip_min),
            cj_floor_ratio=float(seed_opt_cj_floor_ratio),
            output_dir=str(seed_dir),
            python_exe=python_exe,
        )
    )
    if not seed_csv.exists():
        raise RuntimeError(f"step1_3c_opt seed output missing: {seed_csv}")
    print(f"[step2-greedy-mc] Built step1_3c seed: {seed_csv}", flush=True)
    return seed_csv


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
    ap = argparse.ArgumentParser(
        "Integrated mixture greedy-seeded step2 Monte-Carlo pipeline "
        "(step1_3c seed -> step2_1 -> step2_2)"
    )

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
    ap.add_argument("--python_exe", default=sys.executable)

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
    ap.add_argument(
        "--auto_seed_with_step1_3c",
        dest="auto_seed_with_step1_3c",
        action="store_true",
        default=True,
        help="If init_assign_csv is not given, auto-generate it via cvx/step1_3c_opt.py.",
    )
    ap.add_argument(
        "--no_auto_seed_with_step1_3c",
        dest="auto_seed_with_step1_3c",
        action="store_false",
        help="Disable auto seed generation and use step2_1 internal seed builder.",
    )
    ap.add_argument("--seed_opt_dirname", default="step1_3c_seed")
    ap.add_argument("--seed_opt_avg_bits", type=float, default=None, help="Default: init_avg_bits")
    ap.add_argument("--seed_opt_bits", default="", help="Default: bmin..bmax")
    ap.add_argument("--seed_opt_init_bit", type=int, default=2)
    ap.add_argument("--seed_opt_C_col", default="C_mean_per_batch")
    ap.add_argument("--seed_opt_w_col", default="numel(w_j)")
    ap.add_argument(
        "--seed_opt_normalize_lres_by_refbit",
        dest="seed_opt_normalize_lres_by_refbit",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no_seed_opt_normalize_lres_by_refbit",
        dest="seed_opt_normalize_lres_by_refbit",
        action="store_false",
    )
    ap.add_argument("--seed_opt_norm_ref_bit", type=int, default=None)
    ap.add_argument("--seed_opt_norm_eps", type=float, default=1e-12)
    ap.add_argument("--seed_opt_proxy_shape", choices=["absolute", "marginal_gain"], default="marginal_gain")
    ap.add_argument("--seed_opt_marginal_gain_power", type=float, default=1.85)
    ap.add_argument("--seed_opt_cj_transform", choices=["none", "sqrt", "log1p_mean", "power"], default="power")
    ap.add_argument("--seed_opt_cj_power", type=float, default=0.7)
    ap.add_argument("--seed_opt_cj_clip_min", type=float, default=1e-12)
    ap.add_argument("--seed_opt_cj_floor_ratio", type=float, default=0.0)

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
    ap.add_argument("--step2_2_init_assign_csv", default=None)
    ap.add_argument("--step2_2_seed_opt_dirname", default="step1_3c_seed_target")
    ap.add_argument("--step2_2_seed_opt_avg_bits", type=float, default=None, help="Default: target_avg_bits")
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
    ap.add_argument("--only_step2_1", action="store_true", help="Run step2_1 only and stop before step2_2")
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
    init_assign_csv_for_step21: Optional[Path] = None
    init_assign_csv_for_step22: Optional[Path] = None
    expected_auto_seed_csv = (out_root / str(args.seed_opt_dirname) / "bit_assign.csv").resolve()
    expected_auto_seed_csv_step22 = (out_root / str(args.step2_2_seed_opt_dirname) / "bit_assign.csv").resolve()

    summary_path = d21 / "v10_1_summary.json"
    if args.skip_step2_1:
        if not summary_path.exists():
            raise FileNotFoundError(f"--skip_step2_1 set but summary missing: {summary_path}")
        print(f"[step2-mc] Skip step2_1, reuse {summary_path}", flush=True)
    else:
        init_assign_csv_for_step21 = _resolve_or_build_step13_seed(
            requested_path=args.init_assign_csv,
            auto_seed=bool(args.auto_seed_with_step1_3c),
            out_root=out_root,
            sens_csv=sens_csv,
            alpha_csv=alpha_csv,
            init_avg_bits=float(args.init_avg_bits),
            bmin=int(args.bmin),
            bmax=int(args.bmax),
            seed_opt_dirname=str(args.seed_opt_dirname),
            seed_opt_avg_bits=args.seed_opt_avg_bits,
            seed_opt_bits=str(args.seed_opt_bits),
            seed_opt_init_bit=int(args.seed_opt_init_bit),
            seed_opt_c_col=str(args.seed_opt_C_col),
            seed_opt_w_col=str(args.seed_opt_w_col),
            seed_opt_normalize_lres_by_refbit=bool(args.seed_opt_normalize_lres_by_refbit),
            seed_opt_norm_ref_bit=args.seed_opt_norm_ref_bit,
            seed_opt_norm_eps=float(args.seed_opt_norm_eps),
            seed_opt_proxy_shape=str(args.seed_opt_proxy_shape),
            seed_opt_marginal_gain_power=float(args.seed_opt_marginal_gain_power),
            seed_opt_cj_transform=str(args.seed_opt_cj_transform),
            seed_opt_cj_power=float(args.seed_opt_cj_power),
            seed_opt_cj_clip_min=float(args.seed_opt_cj_clip_min),
            seed_opt_cj_floor_ratio=float(args.seed_opt_cj_floor_ratio),
            python_exe=str(args.python_exe),
        )
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
                "init_assign_csv": str(init_assign_csv_for_step21) if init_assign_csv_for_step21 else None,
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

    if args.only_step2_1:
        meta = {
            "model_id": args.model_id,
            "sens_csv": str(sens_csv),
            "alpha_csv": str(alpha_csv),
            "prebake_root": str(prebake_root),
            "surrogate_static_info": str(surrogate_static_info),
            "target_avg_bits": float(args.target_avg_bits),
            "init_assign_csv": str(init_assign_csv_for_step21) if init_assign_csv_for_step21 else args.init_assign_csv,
            "step2_2_init_assign_csv": None,
            "auto_seed_with_step1_3c": bool(args.auto_seed_with_step1_3c),
            "out_root": str(out_root),
            "created": int(time.time()),
            "elapsed_sec": time.time() - t0,
            "steps": {
                "step2_1": str(d21),
                "step2_2": None,
            },
            "artifacts": {
                "step2_1_summary": str(summary_path),
                "warmup_ckpt_json": str(warmup_ckpt_json),
                "base_training_csv": str(base_training_csv),
                "base_surrogate_ckpt": str(base_surrogate_ckpt),
                "base_surrogate_config": str(base_surrogate_config),
                "bit_assign_csv": None,
                "step1_3c_seed_csv": (
                    str(init_assign_csv_for_step21)
                    if (
                        init_assign_csv_for_step21 is not None
                        and init_assign_csv_for_step21.resolve() == expected_auto_seed_csv
                    )
                    else None
                ),
                "step1_3c_seed_csv_step2_2": None,
                "ppl_curve_csv": None,
                "run_ckpt_json": None,
            },
            "results": {
                "step2_1": res21,
                "step2_2": None,
            },
        }
        _write_json(out_root / "meta.json", meta)
        print("[step2-mc] only_step2_1 set -> skip step2_2.", flush=True)
        print(f"[step2-mc] Done. meta: {out_root / 'meta.json'}", flush=True)
        return

    init_assign_csv_for_step22 = _resolve_or_build_step13_seed(
        requested_path=args.step2_2_init_assign_csv,
        auto_seed=bool(args.auto_seed_with_step1_3c),
        out_root=out_root,
        sens_csv=sens_csv,
        alpha_csv=alpha_csv,
        init_avg_bits=float(args.target_avg_bits),
        bmin=int(args.bmin),
        bmax=int(args.bmax),
        seed_opt_dirname=str(args.step2_2_seed_opt_dirname),
        seed_opt_avg_bits=args.step2_2_seed_opt_avg_bits,
        seed_opt_bits=str(args.seed_opt_bits),
        seed_opt_init_bit=int(args.seed_opt_init_bit),
        seed_opt_c_col=str(args.seed_opt_C_col),
        seed_opt_w_col=str(args.seed_opt_w_col),
        seed_opt_normalize_lres_by_refbit=bool(args.seed_opt_normalize_lres_by_refbit),
        seed_opt_norm_ref_bit=args.seed_opt_norm_ref_bit,
        seed_opt_norm_eps=float(args.seed_opt_norm_eps),
        seed_opt_proxy_shape=str(args.seed_opt_proxy_shape),
        seed_opt_marginal_gain_power=float(args.seed_opt_marginal_gain_power),
        seed_opt_cj_transform=str(args.seed_opt_cj_transform),
        seed_opt_cj_power=float(args.seed_opt_cj_power),
        seed_opt_cj_clip_min=float(args.seed_opt_cj_clip_min),
        seed_opt_cj_floor_ratio=float(args.seed_opt_cj_floor_ratio),
        python_exe=str(args.python_exe),
    )

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
            "init_assign_csv": str(init_assign_csv_for_step22) if init_assign_csv_for_step22 else None,
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
        "init_assign_csv": str(init_assign_csv_for_step21) if init_assign_csv_for_step21 else args.init_assign_csv,
        "step2_2_init_assign_csv": (
            str(init_assign_csv_for_step22) if init_assign_csv_for_step22 else args.step2_2_init_assign_csv
        ),
        "auto_seed_with_step1_3c": bool(args.auto_seed_with_step1_3c),
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
            "step1_3c_seed_csv": (
                str(init_assign_csv_for_step21)
                if (init_assign_csv_for_step21 is not None and init_assign_csv_for_step21.resolve() == expected_auto_seed_csv)
                else None
            ),
            "step1_3c_seed_csv_step2_2": (
                str(init_assign_csv_for_step22)
                if (
                    init_assign_csv_for_step22 is not None
                    and init_assign_csv_for_step22.resolve() == expected_auto_seed_csv_step22
                )
                else None
            ),
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
