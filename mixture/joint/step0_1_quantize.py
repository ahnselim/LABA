#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step0-1 wrapper: run LABA/joint/step1_quantize.py as a reusable module/CLI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


def _default_source_script() -> Path:
    return Path(__file__).resolve().parents[2] / "joint" / "step1_quantize.py"


@dataclass
class Step01QuantizeConfig:
    model_id: str
    out_dir: str
    bits: int
    group_size: int
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    device: str = "cuda"
    bit_assign_csv: Optional[str] = None
    clip_percentile: float = 0.0
    lloyd_iter: int = 12
    chunk_groups: int = 4096
    layer_regex: Optional[str] = None
    save_wq: bool = False
    save_err: bool = False
    python_exe: str = sys.executable
    source_script: str = str(_default_source_script())


def expected_outputs(out_dir: str) -> dict:
    p = Path(out_dir)
    return {
        "codebook": p / "codebook.pt",
        "qcodes": p / "qcodes.pt",
        "meta": p / "meta.pt",
        "quantized_weights": p / "quantized_weights.pt",
        "quant_error": p / "quant_error.pt",
    }


def build_command(cfg: Step01QuantizeConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_id",
        cfg.model_id,
        "--bits",
        str(int(cfg.bits)),
        "--group_size",
        str(int(cfg.group_size)),
        "--dtype",
        str(cfg.dtype),
        "--device",
        str(cfg.device),
        "--clip_percentile",
        str(float(cfg.clip_percentile)),
        "--lloyd_iter",
        str(int(cfg.lloyd_iter)),
        "--chunk_groups",
        str(int(cfg.chunk_groups)),
        "--out_dir",
        str(cfg.out_dir),
    ]
    if cfg.revision:
        cmd += ["--revision", str(cfg.revision)]
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    if cfg.bit_assign_csv:
        cmd += ["--bit_assign_csv", str(cfg.bit_assign_csv)]
    if cfg.layer_regex:
        cmd += ["--layer_regex", str(cfg.layer_regex)]
    if cfg.save_wq:
        cmd.append("--save_wq")
    if cfg.save_err:
        cmd.append("--save_err")
    return cmd


def run(cfg: Step01QuantizeConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cmd = build_command(cfg)
    return subprocess.run(cmd, check=check)


def _parse_args(argv: Optional[Sequence[str]] = None) -> Step01QuantizeConfig:
    ap = argparse.ArgumentParser("step0_1_quantize wrapper")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--group_size", type=int, required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bit_assign_csv", default=None)
    ap.add_argument("--clip_percentile", type=float, default=0.0)
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)
    ap.add_argument("--layer_regex", default=None)
    ap.add_argument("--save_wq", action="store_true")
    ap.add_argument("--save_err", action="store_true")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--source_script", default=str(_default_source_script()))
    ns = ap.parse_args(argv)
    return Step01QuantizeConfig(**vars(ns))


def main() -> None:
    cfg = _parse_args()
    run(cfg, check=True)


if __name__ == "__main__":
    main()

