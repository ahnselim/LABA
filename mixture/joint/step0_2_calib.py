#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step0-2 wrapper: run LABA/joint/step2_calib_right_weight_diag.py as a reusable module/CLI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


def _default_source_script() -> Path:
    return Path(__file__).resolve().parents[2] / "joint" / "step2_calib_right_weight_diag.py"


@dataclass
class Step02CalibConfig:
    model_name: str
    out_calib_s: str
    bits: int
    group_size: int
    group_size_1: Optional[int] = None
    group_size_2: Optional[int] = None
    group_size_3: Optional[int] = None
    group_size_4: Optional[int] = None
    dataset: str = "DKYoon/SlimPajama-6B"
    dataset_config: Optional[str] = None
    subset: Optional[str] = None
    split: str = "train"
    use_streaming: bool = True
    seq_len: int = 2048
    nsamples: int = 128
    batch_size: int = 1
    device: str = "cuda"
    device_map: str = "auto"
    trust_remote_code: bool = False
    cov_mode: str = "oas"
    eps: float = 1e-8
    tag: str = ""
    seed: int = 42
    python_exe: str = sys.executable
    source_script: str = str(_default_source_script())


def build_command(cfg: Step02CalibConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_name",
        str(cfg.model_name),
        "--out_calib_s",
        str(cfg.out_calib_s),
        "--bits",
        str(int(cfg.bits)),
        "--group_size",
        str(int(cfg.group_size)),
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
        "--device",
        str(cfg.device),
        "--device_map",
        str(cfg.device_map),
        "--cov_mode",
        str(cfg.cov_mode),
        "--eps",
        str(float(cfg.eps)),
        "--tag",
        str(cfg.tag),
        "--seed",
        str(int(cfg.seed)),
    ]
    if cfg.dataset_config is not None:
        cmd += ["--dataset_config", str(cfg.dataset_config)]
    if cfg.subset is not None:
        cmd += ["--subset", str(cfg.subset)]
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    for bit in (1, 2, 3, 4):
        v = getattr(cfg, f"group_size_{bit}")
        if v is not None:
            cmd += [f"--group_size_{bit}", str(int(v))]
    return cmd


def run(cfg: Step02CalibConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.out_calib_s).parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(cfg)
    return subprocess.run(cmd, check=check)


def _parse_args(argv: Optional[Sequence[str]] = None) -> Step02CalibConfig:
    ap = argparse.ArgumentParser("step0_2_calib wrapper")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out_calib_s", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group_size", type=int, required=True)
    ap.add_argument("--group_size_1", type=int, default=None)
    ap.add_argument("--group_size_2", type=int, default=None)
    ap.add_argument("--group_size_3", type=int, default=None)
    ap.add_argument("--group_size_4", type=int, default=None)
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
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--cov_mode", default="oas")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--source_script", default=str(_default_source_script()))
    ns = ap.parse_args(argv)
    return Step02CalibConfig(**vars(ns))


def main() -> None:
    cfg = _parse_args()
    run(cfg, check=True)


if __name__ == "__main__":
    main()

