#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alternative Step3: LABA-style alternating Lloyd-Max + layerwise low-rank compensation.

Algorithm:
  1) Initialize Wq^(0) with Lloyd-Max quantization.
  2) Fit weighted rank-r low-rank factors on residual (W - Wq) D.
  3) Alternate:
       - AB update:  min || (W - Wq - AB) D ||_F^2
       - Wq update:  min_{Wq in Q} || (W - AB - Wq) D ||_F^2

Outputs:
  - out_dir/wdq_star.pt
  - out_dir/low_rank_ab.pt
  - out_dir/wdq_star_best.pt
  - out_dir/low_rank_ab_best.pt
  - out_dir/metrics.jsonl
  Optional with `--save_all`:
    - out_dir/codebook_star.pt
    - out_dir/qcodes_star.pt
    - out_dir/quant_meta_star.pt
    - out_dir/summary.json
 
CUDA_VISIBLE_DEVICES=2 nohup \
python step_3_alternating.py \
  --model_id Qwen/Qwen3-8B \
  --step1_dir ./output/qwen3_8b_64/step1_quant/1bit \
  --calib_s ./output/qwen3_8b_64/calib_sqrtdiag.pt \
  --out_dir ./output/qwen3_8b_64/step3_alt/1bit_3 \
  --qtype hessian-aware \
  --rank_ab 64 \
  --outer_loops 3 > ./logs/qwen3_8b_1bit_3.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=2 nohup \
python step_3_alternating.py \
  --model_id meta-llama/Llama-3.1-8B \
  --step1_dir ./output/llama3_8b_64/step1_quant/1bit \
  --calib_s ./output/llama3_8b_64/calib_sqrtdiag.pt \
  --out_dir ./output/llama3_8b_64/step3_alt/1bit_3 \
  --qtype hessian-aware \
  --rank_ab 64 \
  --outer_loops 3 > ./logs/llama3_8b_1bit_3.log 2>&1 &

"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from step_1_quantize import (  # noqa: E402
    _snapshot_state_to_cpu,
    is_target_weight,
    lloyd_asym_nonuniform_quantize,
)


MODULE_ORDER = {
    "q_proj": 0,
    "k_proj": 1,
    "v_proj": 2,
    "o_proj": 3,
    "out_proj": 4,
    "gate_proj": 5,
    "up_proj": 6,
    "down_proj": 7,
    "fc1": 8,
    "fc2": 9,
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def extract_block_index(name: str) -> Optional[int]:
    patterns = (
        r"\bmodel\.layers\.(\d+)\.",
        r"\bencoder\.layers\.(\d+)\.",
        r"\blayers\.(\d+)\.",
    )
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def sort_key(name: str) -> Tuple[int, int, str]:
    bidx = extract_block_index(name)
    suffix = name.split(".")[-2] if "." in name else ""
    return (10**9 if bidx is None else bidx, MODULE_ORDER.get(suffix, 10**6), name)


def dequant_from_codebook_codes(
    codebook_ogq: torch.Tensor,
    qcodes_ogs: torch.Tensor,
    orig_i: int,
) -> torch.Tensor:
    o, g, q = codebook_ogq.shape
    _, _, s = qcodes_ogs.shape
    cb = codebook_ogq.reshape(o * g, q)
    idx = qcodes_ogs.reshape(o * g, s).long()
    xq = torch.gather(cb, dim=1, index=idx).reshape(o, g, s)
    return xq.reshape(o, g * s)[:, :orig_i]


@torch.no_grad()
def rank_r_svd(m: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor]:
    o, i = m.shape
    r_eff = min(int(r), o, i)
    if r_eff <= 0:
        raise ValueError("rank must be positive")
    try:
        u, s, v = torch.linalg.svd_lowrank(m, q=r_eff, niter=2)
        sroot = torch.sqrt(s.clamp_min(0.0))
        a = u * sroot.unsqueeze(0)
        b = sroot.unsqueeze(1) * v.T
        return a, b
    except Exception:
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        u = u[:, :r_eff]
        s = s[:r_eff]
        vh = vh[:r_eff, :]
        sroot = torch.sqrt(s.clamp_min(0.0))
        a = u * sroot.unsqueeze(0)
        b = sroot.unsqueeze(1) * vh
        return a, b


def load_diag_weight(entry: dict, eps: float) -> torch.Tensor:
    if "s" in entry:
        d = entry["s"].to(torch.float32)
    elif "sqrt" in entry:
        d = entry["sqrt"].to(torch.float32)
    elif "var" in entry:
        d = torch.sqrt(entry["var"].to(torch.float32).clamp_min(0.0))
    elif "inv_s" in entry:
        d = 1.0 / entry["inv_s"].to(torch.float32).clamp_min(float(eps))
    else:
        raise KeyError("calib entry must include one of: s, sqrt, var, inv_s")
    return d.clamp_min(float(eps)).contiguous()


@torch.no_grad()
def weighted_low_rank_fit(
    residual: torch.Tensor,
    diag_weight: torch.Tensor,
    rank: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = diag_weight.to(device=residual.device, dtype=torch.float32)
    residual_bar = residual * d.unsqueeze(0)
    a, b_bar = rank_r_svd(residual_bar, r=int(rank))
    inv_d = 1.0 / d.clamp_min(float(eps))
    b = b_bar * inv_d.unsqueeze(0)
    return a, b


@torch.no_grad()
def weighted_objective(
    w: torch.Tensor,
    wq: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    diag_weight: torch.Tensor,
) -> float:
    d = diag_weight.to(device=w.device, dtype=torch.float32)
    err = (w - wq - (a @ b)) * d.unsqueeze(0)
    return float(torch.mean(err * err).item())


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def should_reuse_step1_init(meta: dict, qtype: str) -> bool:
    weighted = bool(meta.get("uses_hessian_weighting", False))
    if qtype == "hessian-aware":
        return weighted
    return not weighted


def load_context(args: argparse.Namespace) -> dict:
    step1_dir = Path(args.step1_dir).resolve()
    codebook_path = step1_dir / "codebook.pt"
    qcodes_path = step1_dir / "qcodes.pt"
    meta_path = step1_dir / "meta.pt"
    if not (codebook_path.exists() and qcodes_path.exists() and meta_path.exists()):
        raise FileNotFoundError("step1_dir must contain codebook.pt, qcodes.pt, meta.pt")

    print(f"[Alt-Step3] loading step1 artifacts: {step1_dir}", flush=True)
    codebooks: Dict[str, torch.Tensor] = torch.load(codebook_path, map_location="cpu")
    qcodes: Dict[str, torch.Tensor] = torch.load(qcodes_path, map_location="cpu")
    metas: Dict[str, dict] = torch.load(meta_path, map_location="cpu")

    print(f"[Alt-Step3] loading calib_s: {args.calib_s}", flush=True)
    calib_s: Dict[str, dict] = torch.load(args.calib_s, map_location="cpu")

    if args.dtype_w == "fp16":
        load_dtype = torch.float16
    elif args.dtype_w == "bf16":
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dm_raw = str(args.model_device_map).strip().lower()
    resolved_model_device_map = None if dm_raw in {"", "none", "null"} else args.model_device_map

    print(
        f"[Alt-Step3] loading original model: {args.model_id} "
        f"(device_map={resolved_model_device_map}, device={device})"
    , flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=args.trust_remote_code,
        device_map=resolved_model_device_map,
        low_cpu_mem_usage=True,
    )
    try:
        state = _snapshot_state_to_cpu(model)
        del model
    except NotImplementedError:
        if resolved_model_device_map is None:
            raise
        print("[Alt-Step3] Detected meta tensors under device_map mode. Re-loading on CPU to build state snapshot.", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
            trust_remote_code=args.trust_remote_code,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        state = _snapshot_state_to_cpu(model)
        del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    layer_re = re.compile(args.layer_regex) if args.layer_regex else None
    keys: List[str] = []
    for key in codebooks.keys():
        if key not in qcodes or key not in metas or key not in calib_s or key not in state:
            continue
        if not is_target_weight(key, state[key]):
            continue
        if layer_re and not layer_re.search(key):
            continue
        keys.append(key)

    keys = sorted(keys, key=sort_key)
    if args.max_layers > 0:
        keys = keys[: int(args.max_layers)]
    if not keys:
        raise RuntimeError("No matched layers found.")

    print(f"[Alt-Step3] matched layers: {len(keys)}", flush=True)
    return {
        "device": device,
        "codebooks": codebooks,
        "qcodes": qcodes,
        "metas": metas,
        "calib_s": calib_s,
        "state": state,
        "keys": keys,
    }


def optimize_layer(
    key: str,
    ctx: dict,
    args: argparse.Namespace,
    metrics_path: Path,
) -> dict:
    device = ctx["device"]
    codebooks = ctx["codebooks"]
    qcodes_dict = ctx["qcodes"]
    metas = ctx["metas"]
    calib_s = ctx["calib_s"]
    state = ctx["state"]

    meta = metas[key]
    bits = int(meta["bits"])
    gs = int(meta["group_size"])
    orig_i = int(tuple(meta["orig_shape"])[1])
    clip_pct = float(meta.get("clip_percentile", 0.0) if args.clip_percentile is None else args.clip_percentile)

    w_cpu = state[key].to(torch.float32)
    d_cpu = load_diag_weight(calib_s[key], eps=float(args.eps))
    if d_cpu.numel() != orig_i:
        raise RuntimeError(f"diag weight shape mismatch on {key}: expected {orig_i}, got {d_cpu.numel()}")
    if w_cpu.shape[1] != orig_i:
        raise RuntimeError(f"orig_I mismatch on {key}: meta={orig_i}, weight={w_cpu.shape[1]}")

    w = w_cpu.to(device)
    d = d_cpu.to(device)
    hdiag = (d * d) if args.qtype == "hessian-aware" else None

    if should_reuse_step1_init(meta, args.qtype):
        codebook = codebooks[key].to(device=device, dtype=torch.float32)
        qcodes = qcodes_dict[key].to(device=device)
        wq = dequant_from_codebook_codes(codebook, qcodes, orig_i=orig_i)
        init_source = "step1_artifact"
    else:
        wq, codebook, qcodes, _ = lloyd_asym_nonuniform_quantize(
            w,
            b=bits,
            group_size=gs,
            clip_pct=clip_pct,
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
            hessian_diag=hdiag,
        )
        init_source = "recomputed"

    a, b = weighted_low_rank_fit(w - wq, d, rank=int(args.rank_ab), eps=float(args.eps))
    obj = weighted_objective(w, wq, a, b, d)

    best = {
        "objective": float(obj),
        "outer": -1,
        "wdq": wq.detach().clone(),
        "A": a.detach().clone(),
        "B": b.detach().clone(),
    }

    append_jsonl(
        metrics_path,
        {
            "layer": key,
            "outer": -1,
            "phase": "init",
            "objective_weighted": float(obj),
            "bits": bits,
            "group_size": gs,
            "rank_ab": int(args.rank_ab),
            "qtype": args.qtype,
            "init_source": init_source,
        },
    )

    final_quant_meta = {
        "bits": bits,
        "group_size": gs,
        "clip_percentile": float(clip_pct),
        "qtype": str(args.qtype),
        "init_source": str(init_source),
    }

    for outer in range(int(args.outer_loops)):
        a, b = weighted_low_rank_fit(w - wq, d, rank=int(args.rank_ab), eps=float(args.eps))
        target = w - (a @ b)
        wq, codebook, qcodes, quant_meta = lloyd_asym_nonuniform_quantize(
            target,
            b=bits,
            group_size=gs,
            clip_pct=clip_pct,
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
            hessian_diag=hdiag,
        )
        final_quant_meta = dict(quant_meta)
        final_quant_meta["qtype"] = str(args.qtype)
        final_quant_meta["clip_percentile"] = float(clip_pct)
        obj = weighted_objective(w, wq, a, b, d)

        append_jsonl(
            metrics_path,
            {
                "layer": key,
                "outer": int(outer),
                "phase": "alternating",
                "objective_weighted": float(obj),
                "bits": bits,
                "group_size": gs,
                "rank_ab": int(args.rank_ab),
                "qtype": args.qtype,
            },
        )

        if obj < best["objective"]:
            best = {
                "objective": float(obj),
                "outer": int(outer),
                "wdq": wq.detach().clone(),
                "A": a.detach().clone(),
                "B": b.detach().clone(),
            }

    final_obj = weighted_objective(w, wq, a, b, d)
    return {
        "wdq": wq.detach().to(torch.float16).cpu(),
        "low_rank_ab": {
            "A": a.detach().to(torch.float16).cpu(),
            "B": b.detach().to(torch.float16).cpu(),
            "meta": {
                "rank": int(args.rank_ab),
                "bits": bits,
                "group_size": gs,
                "qtype": str(args.qtype),
                "objective_weighted_final": float(final_obj),
            },
        },
        "wdq_best": best["wdq"].detach().to(torch.float16).cpu(),
        "low_rank_ab_best": {
            "A": best["A"].detach().to(torch.float16).cpu(),
            "B": best["B"].detach().to(torch.float16).cpu(),
            "meta": {
                "rank": int(args.rank_ab),
                "bits": bits,
                "group_size": gs,
                "qtype": str(args.qtype),
                "best_outer": int(best["outer"]),
                "objective_weighted_best": float(best["objective"]),
            },
        },
        "codebook": codebook.detach().to(torch.float16).cpu(),
        "qcodes": qcodes.detach().cpu(),
        "quant_meta": final_quant_meta,
        "summary": {
            "layer": key,
            "bits": bits,
            "group_size": gs,
            "init_source": init_source,
            "objective_weighted_final": float(final_obj),
            "objective_weighted_best": float(best["objective"]),
            "best_outer": int(best["outer"]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser("Alt Step3 - Alternating Lloyd-Max + Low-Rank Compensation")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--step1_dir", required=True, help="step_1_quantize output dir")
    ap.add_argument("--calib_s", required=True, help="step_2_calib calib_sqrtdiag.pt")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_device_map", default="auto", help='Model load placement: e.g. "auto" or "none"')
    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--qtype", default="hessian-aware", choices=["plain", "hessian-aware"])
    ap.add_argument("--rank_ab", type=int, default=64)
    ap.add_argument("--outer_loops", type=int, default=4)
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)
    ap.add_argument("--clip_percentile", type=float, default=None)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--max_layers", type=int, default=0)
    ap.add_argument("--save_every_layer", action="store_true")
    ap.add_argument(
        "--save_all",
        action="store_true",
        help="Also save codebook/qcodes/quant_meta artifacts and summary.json",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    ctx = load_context(args)

    wdq_out: Dict[str, torch.Tensor] = {}
    ab_out: Dict[str, Dict[str, torch.Tensor]] = {}
    wdq_best_out: Dict[str, torch.Tensor] = {}
    ab_best_out: Dict[str, Dict[str, torch.Tensor]] = {}
    codebook_out: Dict[str, torch.Tensor] = {}
    qcodes_out: Dict[str, torch.Tensor] = {}
    quant_meta_out: Dict[str, dict] = {}
    layer_summaries: List[dict] = []

    t0 = time.time()
    for idx, key in enumerate(ctx["keys"], start=1):
        print(f"[Alt-Step3] ({idx}/{len(ctx['keys'])}) optimizing: {key}", flush=True)
        layer_res = optimize_layer(key=key, ctx=ctx, args=args, metrics_path=metrics_path)
        wdq_out[key] = layer_res["wdq"]
        ab_out[key] = layer_res["low_rank_ab"]
        wdq_best_out[key] = layer_res["wdq_best"]
        ab_best_out[key] = layer_res["low_rank_ab_best"]
        codebook_out[key] = layer_res["codebook"]
        qcodes_out[key] = layer_res["qcodes"]
        quant_meta_out[key] = layer_res["quant_meta"]
        layer_summaries.append(layer_res["summary"])

        if args.save_every_layer:
            torch.save(wdq_out, out_dir / "wdq_star.pt")
            torch.save(ab_out, out_dir / "low_rank_ab.pt")
            torch.save(wdq_best_out, out_dir / "wdq_star_best.pt")
            torch.save(ab_best_out, out_dir / "low_rank_ab_best.pt")
            if args.save_all:
                torch.save(codebook_out, out_dir / "codebook_star.pt")
                torch.save(qcodes_out, out_dir / "qcodes_star.pt")
                torch.save(quant_meta_out, out_dir / "quant_meta_star.pt")

        if torch.cuda.is_available() and (idx % 8 == 0 or idx == len(ctx["keys"])):
            torch.cuda.empty_cache()
        if idx % 8 == 0 or idx == len(ctx["keys"]):
            gc.collect()

    torch.save(wdq_out, out_dir / "wdq_star.pt")
    torch.save(ab_out, out_dir / "low_rank_ab.pt")
    torch.save(wdq_best_out, out_dir / "wdq_star_best.pt")
    torch.save(ab_best_out, out_dir / "low_rank_ab_best.pt")
    if args.save_all:
        torch.save(codebook_out, out_dir / "codebook_star.pt")
        torch.save(qcodes_out, out_dir / "qcodes_star.pt")
        torch.save(quant_meta_out, out_dir / "quant_meta_star.pt")

    final_mean = sum(x["objective_weighted_final"] for x in layer_summaries) / max(1, len(layer_summaries))
    best_mean = sum(x["objective_weighted_best"] for x in layer_summaries) / max(1, len(layer_summaries))
    summary = {
        "model_id": args.model_id,
        "revision": args.revision,
        "step1_dir": str(Path(args.step1_dir).resolve()),
        "calib_s": str(Path(args.calib_s).resolve()),
        "out_dir": str(out_dir),
        "qtype": str(args.qtype),
        "rank_ab": int(args.rank_ab),
        "outer_loops": int(args.outer_loops),
        "lloyd_iter": int(args.lloyd_iter),
        "chunk_groups": int(args.chunk_groups),
        "clip_percentile_override": args.clip_percentile,
        "num_layers": len(layer_summaries),
        "objective_weighted_final_mean": float(final_mean),
        "objective_weighted_best_mean": float(best_mean),
        "elapsed_sec": time.time() - t0,
        "layers": layer_summaries,
    }
    if args.save_all:
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Alt-Step3] saved:", flush=True)
    print(f"  wdq*: {out_dir / 'wdq_star.pt'}", flush=True)
    print(f"  AB*:  {out_dir / 'low_rank_ab.pt'}", flush=True)
    if args.save_all:
        print(f"  summary: {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()


def _invoke_local_main(argv: Sequence[str]) -> subprocess.CompletedProcess:
    argv = list(argv)
    args = [str(sys.executable), str(Path(__file__).resolve())] + argv
    prev_argv = sys.argv[:]
    exit_code = 0
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
    return subprocess.CompletedProcess(args=args, returncode=int(exit_code))


@dataclass
class Step03AlternatingConfig:
    model_id: str
    step1_dir: str
    calib_s: str
    out_dir: str
    revision: Optional[str] = None
    trust_remote_code: bool = False
    device: str = "cuda"
    model_device_map: str = "auto"
    dtype_w: str = "fp16"
    qtype: str = "hessian-aware"
    rank_ab: int = 64
    outer_loops: int = 4
    lloyd_iter: int = 12
    chunk_groups: int = 4096
    clip_percentile: Optional[float] = None
    eps: float = 1e-8
    layer_regex: Optional[str] = None
    max_layers: int = 0
    save_every_layer: bool = False
    seed: int = 42
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


def build_command(cfg: Step03AlternatingConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_id",
        str(cfg.model_id),
        "--step1_dir",
        str(cfg.step1_dir),
        "--calib_s",
        str(cfg.calib_s),
        "--out_dir",
        str(cfg.out_dir),
        "--device",
        str(cfg.device),
        "--model_device_map",
        str(cfg.model_device_map),
        "--dtype_w",
        str(cfg.dtype_w),
        "--qtype",
        str(cfg.qtype),
        "--rank_ab",
        str(int(cfg.rank_ab)),
        "--outer_loops",
        str(int(cfg.outer_loops)),
        "--lloyd_iter",
        str(int(cfg.lloyd_iter)),
        "--chunk_groups",
        str(int(cfg.chunk_groups)),
        "--eps",
        str(float(cfg.eps)),
        "--max_layers",
        str(int(cfg.max_layers)),
        "--seed",
        str(int(cfg.seed)),
    ]
    if cfg.revision:
        cmd += ["--revision", str(cfg.revision)]
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    if cfg.clip_percentile is not None:
        cmd += ["--clip_percentile", str(float(cfg.clip_percentile))]
    if cfg.layer_regex:
        cmd += ["--layer_regex", str(cfg.layer_regex)]
    if cfg.save_every_layer:
        cmd.append("--save_every_layer")
    return cmd


def run(cfg: Step03AlternatingConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp
