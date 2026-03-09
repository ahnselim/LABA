#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated 1-bit step0 optimization:
  step0_1 (1-bit quantize) -> step0_2 (calib) -> step0_3 (1-bit joint opt)
then export prebake-style artifacts using best wdq/best AB.

Output layout:
  out_root/
    meta.json
    bit1/*.pt
    best_raw/bit1/wdq_star_best.pt
    best_raw/bit1/lowrank_uv_ab_best.pt
    summaries/bit1.json
    summaries/summary.json
    
python step0_opt_1bit.py \
  --model_id mistralai/Mistral-7B-v0.3 \
  --out_root /ssd/ssd4/asl/LABA/mixture/output_step0_1bit \
  --group_size 128 \
  --rank_ab 64 \
  --stage1_max_blocks 8 \
  --stage1_n_trials 20 \
  --nsamples 64 \
  --seq_len 2048 \
  --device cuda \
  --device_map auto \
  --step3_model_device_map none \
  --trust_remote_code

"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


_THIS_DIR = Path(__file__).resolve().parent
_ONEBIT_DIR = _THIS_DIR / "1bit"
for _path in (str(_THIS_DIR), str(_ONEBIT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_1bit_symbols():
    m1 = importlib.import_module("step0_1_quantize")
    m2 = importlib.import_module("step0_2_calib")
    m3 = importlib.import_module("step0_3_BO_opt")
    return (
        m1.Step01QuantizeConfig,
        m1.run,
        m2.Step02CalibConfig,
        m2.run,
        m3.Step03BOConfig,
        m3.run,
    )


(
    Step01QuantizeConfig,
    run_step01,
    Step02CalibConfig,
    run_step02,
    Step03BOConfig,
    run_step03,
) = _load_1bit_symbols()


def _safe_name(s) -> str:
    if s is None:
        s = "none"
    s = str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def module_name_from_weight(full_weight_name: str) -> str:
    if not full_weight_name.endswith(".weight"):
        return full_weight_name
    return full_weight_name[: -len(".weight")]


def _cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _step1_outputs_ready(step1_out_dir: Path) -> bool:
    required = (
        step1_out_dir / "codebook.pt",
        step1_out_dir / "qcodes.pt",
        step1_out_dir / "meta.pt",
    )
    return all(p.exists() for p in required)


def _step2_output_ready(step2_out_path: Path) -> bool:
    return step2_out_path.exists()


def _find_existing_step3_outputs(step3_out_root: Path, study_name: str) -> Optional[Dict[str, Path]]:
    if not step3_out_root.exists():
        return None
    cands = [
        p
        for p in step3_out_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{study_name}_") and (p / "summary.json").exists()
    ]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in cands:
        stage2 = run_dir / "stage2_full"
        wdq = stage2 / "wdq_star_best.pt"
        uvab = stage2 / "lowrank_uv_ab_best.pt"
        if wdq.exists() and uvab.exists():
            return {
                "run_dir": run_dir,
                "stage1_dir": run_dir / "stage1_bo",
                "stage2_dir": stage2,
                "summary": run_dir / "summary.json",
                "summary_stage2": stage2 / "summary_stage2.json",
                "wdq_star_best": wdq,
                "lowrank_uv_ab_best": uvab,
            }
    return None


def _resolve_calib_s_for_export(step2_out_path: Path, step3_outs: Dict[str, Path]) -> Path:
    if step2_out_path.exists():
        return step2_out_path

    summary_stage2 = Path(step3_outs.get("summary_stage2", ""))
    if summary_stage2.exists():
        try:
            payload = json.loads(summary_stage2.read_text(encoding="utf-8"))
            calib_s_path = Path(str(payload.get("calib_s_path", "")))
            if calib_s_path.exists():
                return calib_s_path
        except Exception:
            pass
    raise FileNotFoundError(
        f"calib_s not found for export (expected {step2_out_path} or calib_s_path from {summary_stage2})"
    )


def _ensure_inv_s(entry: dict) -> torch.Tensor:
    if "inv_s" in entry:
        return entry["inv_s"].to(torch.float32)
    if "s" in entry:
        s = entry["s"].to(torch.float32)
        return (1.0 / torch.clamp(s, min=1e-12)).to(torch.float32)
    raise KeyError("calib entry must include 'inv_s' or 's'")


@torch.no_grad()
def export_prebake_from_step3(
    *,
    wdq_path: Path,
    uvab_path: Path,
    calib_s_path: Path,
    out_root: Path,
) -> Dict[str, int]:
    wdq_best: Dict[str, torch.Tensor] = torch.load(wdq_path, map_location="cpu")
    uvab_best: Dict[str, dict] = torch.load(uvab_path, map_location="cpu")
    calib_s: Dict[str, dict] = torch.load(calib_s_path, map_location="cpu")

    bit_dir = out_root / "bit1"
    bit_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_missing_calib = 0
    skipped_missing_uvab = 0

    for full_name, Wq in wdq_best.items():
        if full_name not in uvab_best:
            skipped_missing_uvab += 1
            continue
        if full_name not in calib_s:
            skipped_missing_calib += 1
            continue

        rec = uvab_best[full_name]
        A = rec["A"].to(torch.float32)
        B = rec["B"].to(torch.float32)
        u = rec.get("u", None)
        v = rec.get("v", None)
        inv_s = _ensure_inv_s(calib_s[full_name]).to(torch.float32)

        if u is not None:
            A = A * u.to(torch.float32).view(-1, 1)
        if v is not None:
            B = B * v.to(torch.float32).view(1, -1)
        B = B * inv_s.view(1, -1)

        module = module_name_from_weight(full_name)
        payload = {
            "module": module,
            "full_weight": full_name,
            "Wq": Wq.to(torch.float16).cpu(),
            "A": A.to(torch.float16).cpu(),
            "B": B.to(torch.float16).cpu(),
            "meta": {
                "bit": 1,
                "source": "step0_opt_1bit(step3 best)",
                "uses_calib_inv_s": True,
                "step3_meta": rec.get("meta", {}),
            },
        }
        torch.save(payload, bit_dir / f"{_safe_name(module)}.pt")
        saved += 1

        del A, B, inv_s

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "saved": saved,
        "skipped_missing_calib": skipped_missing_calib,
        "skipped_missing_uvab": skipped_missing_uvab,
    }


def _copy_best_raw(wdq_src: Path, uvab_src: Path, out_root: Path) -> Dict[str, str]:
    dst_dir = out_root / "best_raw" / "bit1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    wdq_dst = dst_dir / "wdq_star_best.pt"
    uvab_dst = dst_dir / "lowrank_uv_ab_best.pt"
    shutil.copy2(wdq_src, wdq_dst)
    shutil.copy2(uvab_src, uvab_dst)
    return {"wdq_star_best": str(wdq_dst), "lowrank_uv_ab_best": str(uvab_dst)}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser("Integrated 1-bit step0 optimization")

    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--python_exe", default=sys.executable)

    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device_map", default="auto", help='Model load placement for step1/2: e.g. "auto" or "none"')
    ap.add_argument(
        "--step3_model_device_map",
        default=None,
        help='Model load placement for step3 only. If omitted, falls back to --device_map.',
    )
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--clip_percentile", type=float, default=0.0)
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)
    ap.add_argument("--layer_regex", default=None)

    ap.add_argument("--dataset", default="DKYoon/SlimPajama-6B")
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in {"1", "true", "yes"},
        default=True,
    )
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--step2_num_gpus", type=int, default=2)
    ap.add_argument("--cov_mode", default="oas", choices=["var", "oas", "second_moment"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--rank_ab", type=int, default=64)
    ap.add_argument("--outer_loops", type=int, default=40)
    ap.add_argument("--delta_steps", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--lr_min_ratio", type=float, default=0.1)
    ap.add_argument("--lr_step_gamma", type=float, default=0.3)
    ap.add_argument("--stage1_max_blocks", type=int, default=8)
    ap.add_argument("--stage1_n_trials", type=int, default=20)
    ap.add_argument("--stage1_timeout_sec", type=int, default=0)
    ap.add_argument("--stage1_storage", default="")
    ap.add_argument("--stage1_save_artifacts", action="store_true")
    ap.add_argument("--prune_min_layers", type=int, default=2)
    ap.add_argument("--prune_warmup_steps", type=int, default=20)
    ap.add_argument("--pruner_startup_trials", type=int, default=5)
    ap.add_argument("--pruner_interval_steps", type=int, default=5)
    ap.add_argument("--study_name", default="step0_opt_1bit")
    ap.add_argument("--run_step4_eval", action="store_true")
    ap.add_argument("--ab_compute", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--ppl_stride", type=int, default=2048)
    ap.add_argument("--ppl_max_tokens", type=int, default=0)

    ap.add_argument("--keep_temps", action="store_true")
    ap.add_argument(
        "--skip_existing_steps",
        dest="skip_existing_steps",
        action="store_true",
        help="Skip step1/2/3 when their outputs already exist.",
    )
    ap.add_argument(
        "--no_skip_existing_steps",
        dest="skip_existing_steps",
        action="store_false",
        help="Always re-run step1/2/3 even if outputs already exist.",
    )
    ap.set_defaults(skip_existing_steps=True)

    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bit1").mkdir(parents=True, exist_ok=True)
    tmp_root = out_root / "_tmp" / "bit1"
    summaries_root = out_root / "summaries"
    summaries_root.mkdir(parents=True, exist_ok=True)

    step1_out_dir = tmp_root / "step1_quant"
    step2_out_path = tmp_root / "step2_calib" / "calib_sqrtdiag.pt"
    step3_out_root = tmp_root / "step3_runs"
    run_started = int(time.time())

    print(f"[step0_1bit] start | group_size={args.group_size} | rank_ab={args.rank_ab}")

    step1_skipped = False
    if args.skip_existing_steps and _step1_outputs_ready(step1_out_dir):
        step1_skipped = True
        print(f"[step0_1bit] step1 skip (found: {step1_out_dir})")
    else:
        run_step01(
            Step01QuantizeConfig(
                model_id=args.model_id,
                revision=args.revision,
                trust_remote_code=bool(args.trust_remote_code),
                dtype=args.dtype,
                device=args.device,
                device_map=args.device_map,
                bits=1,
                group_size=int(args.group_size),
                clip_percentile=args.clip_percentile,
                lloyd_iter=args.lloyd_iter,
                chunk_groups=args.chunk_groups,
                layer_regex=args.layer_regex,
                out_dir=str(step1_out_dir),
                python_exe=args.python_exe,
            )
        )

    step2_skipped = False
    if args.skip_existing_steps and _step2_output_ready(step2_out_path):
        step2_skipped = True
        print(f"[step0_1bit] step2 skip (found: {step2_out_path})")
    else:
        run_step02(
            Step02CalibConfig(
                model_name=args.model_id,
                out_calib_s=str(step2_out_path),
                bits=1,
                group_size=int(args.group_size),
                group_size_1=int(args.group_size),
                dataset=args.dataset,
                dataset_config=args.dataset_config,
                subset=args.subset,
                split=args.split,
                use_streaming=bool(args.use_streaming),
                seq_len=args.seq_len,
                nsamples=args.nsamples,
                batch_size=args.batch_size,
                device=args.device,
                device_map=args.device_map,
                num_gpus=args.step2_num_gpus,
                trust_remote_code=bool(args.trust_remote_code),
                cov_mode=args.cov_mode,
                eps=args.eps,
                tag="step0_opt_1bit",
                seed=args.seed,
                python_exe=args.python_exe,
            )
        )

    step3_model_device_map = args.device_map if args.step3_model_device_map is None else args.step3_model_device_map

    step3_skipped = False
    step3_outs = None
    if args.skip_existing_steps:
        step3_outs = _find_existing_step3_outputs(step3_out_root, args.study_name)
        if step3_outs is not None:
            step3_skipped = True
            print(f"[step0_1bit] step3 skip (reuse run: {step3_outs['run_dir']})")

    if step3_outs is None:
        step3_outs = run_step03(
            Step03BOConfig(
                model_id=args.model_id,
                revision=args.revision,
                trust_remote_code=bool(args.trust_remote_code),
                step1_dir=str(step1_out_dir),
                calib_s=str(step2_out_path),
                out_root=str(step3_out_root),
                device=args.device,
                eval_device=(f"{args.device}:0" if args.device == "cuda" else args.device),
                model_device_map=step3_model_device_map,
                dtype_w=args.dtype_w,
                rank_ab=int(args.rank_ab),
                outer_loops=args.outer_loops,
                delta_steps=args.delta_steps,
                eps=args.eps,
                layer_regex=args.layer_regex,
                log_every=args.log_every,
                lr_min_ratio=args.lr_min_ratio,
                lr_step_gamma=args.lr_step_gamma,
                study_name=args.study_name,
                seed=args.seed,
                stage1_max_blocks=args.stage1_max_blocks,
                stage1_n_trials=args.stage1_n_trials,
                stage1_timeout_sec=args.stage1_timeout_sec,
                stage1_storage=args.stage1_storage,
                stage1_save_artifacts=args.stage1_save_artifacts,
                prune_min_layers=args.prune_min_layers,
                prune_warmup_steps=args.prune_warmup_steps,
                pruner_startup_trials=args.pruner_startup_trials,
                pruner_interval_steps=args.pruner_interval_steps,
                run_step4_eval=args.run_step4_eval,
                ab_compute=args.ab_compute,
                ppl_stride=args.ppl_stride,
                ppl_max_tokens=args.ppl_max_tokens,
                python_exe=args.python_exe,
            )
        )

    raw_paths = _copy_best_raw(
        wdq_src=step3_outs["wdq_star_best"],
        uvab_src=step3_outs["lowrank_uv_ab_best"],
        out_root=out_root,
    )

    calib_s_for_export = _resolve_calib_s_for_export(step2_out_path, step3_outs)
    export_stats = export_prebake_from_step3(
        wdq_path=Path(raw_paths["wdq_star_best"]),
        uvab_path=Path(raw_paths["lowrank_uv_ab_best"]),
        calib_s_path=calib_s_for_export,
        out_root=out_root,
    )

    bit_summary = {
        "bit": 1,
        "group_size": int(args.group_size),
        "rank_ab": int(args.rank_ab),
        "step1_skipped": bool(step1_skipped),
        "step2_skipped": bool(step2_skipped),
        "step3_skipped": bool(step3_skipped),
        "step1_out_dir": str(step1_out_dir),
        "step2_out_calib_s": str(calib_s_for_export),
        "step3_run_dir": str(step3_outs["run_dir"]),
        "step3_stage2_dir": str(step3_outs["stage2_dir"]),
        "best_raw": raw_paths,
        "prebake_export": export_stats,
    }
    _write_json(summaries_root / "bit1.json", bit_summary)

    meta = {
        "model_id": args.model_id,
        "revision": args.revision,
        "bits": [1],
        "group_size": int(args.group_size),
        "rank_ab": int(args.rank_ab),
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "nsamples": int(args.nsamples),
        "seq_len": int(args.seq_len),
        "step2_num_gpus": int(args.step2_num_gpus),
        "dtype": args.dtype,
        "dtype_w": args.dtype_w,
        "device": args.device,
        "device_map": args.device_map,
        "step3_model_device_map": step3_model_device_map,
        "skip_existing_steps": bool(args.skip_existing_steps),
        "created": run_started,
        "note": "Integrated 1-bit step1/2/3 best outputs exported in prebake-compatible per-module format.",
    }
    _write_json(out_root / "meta.json", meta)

    final_summary = {
        "out_root": str(out_root),
        "meta_json": str(out_root / "meta.json"),
        "bits": [1],
        "per_bit_summaries": [str(summaries_root / "bit1.json")],
        "best_raw_root": str(out_root / "best_raw"),
        "kept_temps": bool(args.keep_temps),
    }
    _write_json(summaries_root / "summary.json", final_summary)

    if not args.keep_temps:
        _cleanup_path(step1_out_dir)
        _cleanup_path(step2_out_path)
        _cleanup_path(step3_out_root)
        try:
            (tmp_root / "step2_calib").rmdir()
        except OSError:
            pass
        try:
            tmp_root.rmdir()
        except OSError:
            pass
        tmp_parent = out_root / "_tmp"
        try:
            tmp_parent.rmdir()
        except OSError:
            pass

    print("\n[step0_1bit] COMPLETED")
    print(f"  out_root: {out_root}")
    print(f"  meta    : {out_root / 'meta.json'}")
    print(f"  summary : {summaries_root / 'summary.json'}")


@dataclass
class Step0Opt1bitConfig:
    model_id: str
    out_root: str
    group_size: int
    rank_ab: int
    revision: Optional[str] = None
    trust_remote_code: bool = False
    python_exe: str = sys.executable
    dtype: str = "auto"
    device: str = "cuda"
    device_map: str = "auto"
    step3_model_device_map: Optional[str] = None
    clip_percentile: float = 0.0
    lloyd_iter: int = 12
    chunk_groups: int = 4096
    layer_regex: Optional[str] = None
    dataset: str = "DKYoon/SlimPajama-6B"
    dataset_config: Optional[str] = None
    subset: Optional[str] = None
    split: str = "train"
    use_streaming: bool = True
    seq_len: int = 2048
    nsamples: int = 64
    batch_size: int = 1
    step2_num_gpus: int = 2
    cov_mode: str = "oas"
    eps: float = 1e-8
    seed: int = 42
    dtype_w: str = "fp16"
    outer_loops: int = 40
    delta_steps: int = 8
    log_every: int = 5
    lr_min_ratio: float = 0.1
    lr_step_gamma: float = 0.3
    stage1_max_blocks: int = 8
    stage1_n_trials: int = 20
    stage1_timeout_sec: int = 0
    stage1_storage: str = ""
    stage1_save_artifacts: bool = False
    prune_min_layers: int = 2
    prune_warmup_steps: int = 20
    pruner_startup_trials: int = 5
    pruner_interval_steps: int = 5
    study_name: str = "step0_opt_1bit"
    run_step4_eval: bool = False
    ab_compute: str = "fp16"
    ppl_stride: int = 2048
    ppl_max_tokens: int = 0
    keep_temps: bool = False
    skip_existing_steps: bool = True
    source_script: str = str(Path(__file__).resolve())


def build_command(cfg: Step0Opt1bitConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_id",
        str(cfg.model_id),
        "--out_root",
        str(cfg.out_root),
        "--group_size",
        str(int(cfg.group_size)),
        "--rank_ab",
        str(int(cfg.rank_ab)),
        "--dtype",
        str(cfg.dtype),
        "--device",
        str(cfg.device),
        "--device_map",
        str(cfg.device_map),
        "--clip_percentile",
        str(float(cfg.clip_percentile)),
        "--lloyd_iter",
        str(int(cfg.lloyd_iter)),
        "--chunk_groups",
        str(int(cfg.chunk_groups)),
        "--dataset",
        str(cfg.dataset),
        "--split",
        str(cfg.split),
        "--use_streaming",
        "true" if cfg.use_streaming else "false",
        "--seq_len",
        str(int(cfg.seq_len)),
        "--nsamples",
        str(int(cfg.nsamples)),
        "--batch_size",
        str(int(cfg.batch_size)),
        "--step2_num_gpus",
        str(int(cfg.step2_num_gpus)),
        "--cov_mode",
        str(cfg.cov_mode),
        "--eps",
        str(float(cfg.eps)),
        "--seed",
        str(int(cfg.seed)),
        "--dtype_w",
        str(cfg.dtype_w),
        "--outer_loops",
        str(int(cfg.outer_loops)),
        "--delta_steps",
        str(int(cfg.delta_steps)),
        "--log_every",
        str(int(cfg.log_every)),
        "--lr_min_ratio",
        str(float(cfg.lr_min_ratio)),
        "--lr_step_gamma",
        str(float(cfg.lr_step_gamma)),
        "--stage1_max_blocks",
        str(int(cfg.stage1_max_blocks)),
        "--stage1_n_trials",
        str(int(cfg.stage1_n_trials)),
        "--stage1_timeout_sec",
        str(int(cfg.stage1_timeout_sec)),
        "--stage1_storage",
        str(cfg.stage1_storage),
        "--prune_min_layers",
        str(int(cfg.prune_min_layers)),
        "--prune_warmup_steps",
        str(int(cfg.prune_warmup_steps)),
        "--pruner_startup_trials",
        str(int(cfg.pruner_startup_trials)),
        "--pruner_interval_steps",
        str(int(cfg.pruner_interval_steps)),
        "--study_name",
        str(cfg.study_name),
        "--ab_compute",
        str(cfg.ab_compute),
        "--ppl_stride",
        str(int(cfg.ppl_stride)),
        "--ppl_max_tokens",
        str(int(cfg.ppl_max_tokens)),
    ]
    if cfg.revision:
        cmd += ["--revision", str(cfg.revision)]
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    if cfg.step3_model_device_map is not None:
        cmd += ["--step3_model_device_map", str(cfg.step3_model_device_map)]
    if cfg.layer_regex:
        cmd += ["--layer_regex", str(cfg.layer_regex)]
    if cfg.dataset_config is not None:
        cmd += ["--dataset_config", str(cfg.dataset_config)]
    if cfg.subset is not None:
        cmd += ["--subset", str(cfg.subset)]
    if cfg.stage1_save_artifacts:
        cmd.append("--stage1_save_artifacts")
    if cfg.run_step4_eval:
        cmd.append("--run_step4_eval")
    if cfg.keep_temps:
        cmd.append("--keep_temps")
    if cfg.skip_existing_steps:
        cmd.append("--skip_existing_steps")
    else:
        cmd.append("--no_skip_existing_steps")
    return cmd


def _find_run_artifacts(out_root: Path) -> Dict[str, Path]:
    stage2 = out_root / "best_raw" / "bit1"
    wdq = stage2 / "wdq_star_best.pt"
    ab = stage2 / "lowrank_uv_ab_best.pt"
    if not wdq.exists() or not ab.exists():
        raise FileNotFoundError(f"Missing 1-bit outputs in {stage2}")
    return {
        "out_root": out_root,
        "summary": out_root / "summaries" / "summary.json",
        "summary_bit1": out_root / "summaries" / "bit1.json",
        "meta": out_root / "meta.json",
        "wdq_star_best": wdq,
        "lowrank_uv_ab_best": ab,
    }


def run(cfg: Step0Opt1bitConfig, check: bool = True) -> Dict[str, Path]:
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prev_argv = sys.argv[:]
    exit_code = 0
    argv = build_command(cfg)[2:]
    try:
        sys.argv = [str(Path(__file__).resolve())] + argv
        try:
            main()
        except SystemExit as e:
            code = e.code
            if code is None:
                exit_code = 0
            elif isinstance(code, int):
                exit_code = int(code)
            else:
                print(code, file=sys.stderr)
                exit_code = 1
    finally:
        sys.argv = prev_argv

    if check and exit_code != 0:
        raise RuntimeError(f"step0_opt_1bit failed with exit code {exit_code}")
    return _find_run_artifacts(out_root)


if __name__ == "__main__":
    main()
