#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step0-3 wrapper: run LABA/joint/step3_5_v2.py and resolve final Stage2 artifacts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _default_source_script() -> Path:
    return Path(__file__).resolve().parents[2] / "joint" / "step3_5_v2.py"


@dataclass
class Step03BOConfig:
    model_id: str
    step1_dir: str
    calib_s: str
    out_root: str
    stage1_max_blocks: int
    rank_ab: int
    study_name: str
    revision: Optional[str] = None
    trust_remote_code: bool = False
    device: str = "cuda"
    eval_device: str = "cuda:0"
    dtype_w: str = "fp16"
    outer_loops: int = 40
    delta_steps: int = 8
    eps: float = 1e-8
    layer_regex: Optional[str] = None
    log_every: int = 5
    save_every_layer: bool = False
    lr_min_ratio: float = 0.1
    lr_step_gamma: float = 0.3
    seed: int = 42
    stage1_n_trials: int = 20
    stage1_timeout_sec: int = 0
    stage1_storage: str = ""
    stage1_save_artifacts: bool = False
    prune_min_layers: int = 2
    prune_warmup_steps: int = 20
    pruner_startup_trials: int = 5
    pruner_interval_steps: int = 5
    run_step4_eval: bool = False
    ab_compute: str = "fp16"
    ppl_stride: int = 2048
    ppl_max_tokens: int = 0
    step4_script: Optional[str] = None
    python_exe: str = sys.executable
    source_script: str = str(_default_source_script())


def build_command(cfg: Step03BOConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_id",
        str(cfg.model_id),
        "--step1_dir",
        str(cfg.step1_dir),
        "--calib_s",
        str(cfg.calib_s),
        "--out_root",
        str(cfg.out_root),
        "--device",
        str(cfg.device),
        "--eval_device",
        str(cfg.eval_device),
        "--dtype_w",
        str(cfg.dtype_w),
        "--rank_ab",
        str(int(cfg.rank_ab)),
        "--outer_loops",
        str(int(cfg.outer_loops)),
        "--delta_steps",
        str(int(cfg.delta_steps)),
        "--eps",
        str(float(cfg.eps)),
        "--log_every",
        str(int(cfg.log_every)),
        "--lr_min_ratio",
        str(float(cfg.lr_min_ratio)),
        "--lr_step_gamma",
        str(float(cfg.lr_step_gamma)),
        "--study_name",
        str(cfg.study_name),
        "--seed",
        str(int(cfg.seed)),
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
    if cfg.layer_regex:
        cmd += ["--layer_regex", str(cfg.layer_regex)]
    if cfg.save_every_layer:
        cmd.append("--save_every_layer")
    if cfg.stage1_save_artifacts:
        cmd.append("--stage1_save_artifacts")
    if cfg.run_step4_eval:
        cmd.append("--run_step4_eval")
    if cfg.step4_script:
        cmd += ["--step4_script", str(cfg.step4_script)]
    return cmd


def _find_run_dir(out_root: Path, study_name: str) -> Path:
    cands = [
        p for p in out_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{study_name}_") and (p / "summary.json").exists()
    ]
    if not cands:
        raise FileNotFoundError(f"No completed step3 run dir found under {out_root} for study_name={study_name}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def resolve_outputs(run_dir: Path) -> Dict[str, Path]:
    stage2 = run_dir / "stage2_full"
    wdq = stage2 / "wdq_star_best.pt"
    ab = stage2 / "lowrank_uv_ab_best.pt"
    if not wdq.exists() or not ab.exists():
        raise FileNotFoundError(f"Missing stage2 outputs in {stage2}")
    return {
        "run_dir": run_dir,
        "stage1_dir": run_dir / "stage1_bo",
        "stage2_dir": stage2,
        "summary": run_dir / "summary.json",
        "summary_stage2": stage2 / "summary_stage2.json",
        "wdq_star_best": wdq,
        "lowrank_uv_ab_best": ab,
    }


def run(cfg: Step03BOConfig, check: bool = True) -> Dict[str, Path]:
    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = build_command(cfg)
    subprocess.run(cmd, check=check)
    run_dir = _find_run_dir(out_root, cfg.study_name)
    return resolve_outputs(run_dir)


def _parse_args(argv: Optional[Sequence[str]] = None) -> Step03BOConfig:
    ap = argparse.ArgumentParser("step0_3_BO_opt wrapper")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--step1_dir", required=True)
    ap.add_argument("--calib_s", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--stage1_max_blocks", type=int, required=True)
    ap.add_argument("--rank_ab", type=int, required=True)
    ap.add_argument("--study_name", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_device", default="cuda:0")
    ap.add_argument("--dtype_w", default="fp16")
    ap.add_argument("--outer_loops", type=int, default=40)
    ap.add_argument("--delta_steps", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--layer_regex", default=None)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_every_layer", action="store_true")
    ap.add_argument("--lr_min_ratio", type=float, default=0.1)
    ap.add_argument("--lr_step_gamma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage1_n_trials", type=int, default=20)
    ap.add_argument("--stage1_timeout_sec", type=int, default=0)
    ap.add_argument("--stage1_storage", default="")
    ap.add_argument("--stage1_save_artifacts", action="store_true")
    ap.add_argument("--prune_min_layers", type=int, default=2)
    ap.add_argument("--prune_warmup_steps", type=int, default=20)
    ap.add_argument("--pruner_startup_trials", type=int, default=5)
    ap.add_argument("--pruner_interval_steps", type=int, default=5)
    ap.add_argument("--run_step4_eval", action="store_true")
    ap.add_argument("--ab_compute", default="fp16")
    ap.add_argument("--ppl_stride", type=int, default=2048)
    ap.add_argument("--ppl_max_tokens", type=int, default=0)
    ap.add_argument("--step4_script", default=None)
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--source_script", default=str(_default_source_script()))
    ns = ap.parse_args(argv)
    return Step03BOConfig(**vars(ns))


def main() -> None:
    cfg = _parse_args()
    outs = run(cfg, check=True)
    print(f"[step0_3] run_dir={outs['run_dir']}")
    print(f"[step0_3] wdq_star_best={outs['wdq_star_best']}")
    print(f"[step0_3] lowrank_uv_ab_best={outs['lowrank_uv_ab_best']}")


if __name__ == "__main__":
    main()

