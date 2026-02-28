#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Integrated step0 optimization:
  step0_1 (quantize) -> step0_2 (calib) -> step0_3 (BO opt)
for bits {1,2,3,4}, then export prebake-style artifacts using best wdq/best AB.

Output (step4_prebake-compatible layout):
  out_root/
    meta.json
    bit1/*.pt
    bit2/*.pt
    bit3/*.pt
    bit4/*.pt
    best_raw/bit{b}/wdq_star_best.pt
    best_raw/bit{b}/lowrank_uv_ab_best.pt
    summaries/*.json
  
CUDA_VISIBLE_DEVICES=1 nohup \
python step0_optimization.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --out_root ./output2_7b/output_step0_prebake \
  --group_size_1 128 --group_size_2 128 --group_size_3 128 --group_size_4 128 \
  --rank_ab_1 64 --rank_ab_2 64 --rank_ab_3 64 --rank_ab_4 64 \
  --stage1_max_blocks 8 --stage1_n_trials 20 \
  --nsamples 64 --seq_len 2048 \
  --trust_remote_code > ./logs/step0_2_7b.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup \
python step0_optimization.py \
  --model_id huggyllama/llama-7b \
  --out_root ./output_7b/output_step0_prebake \
  --group_size_4 128 \
  --rank_ab_4 64 \
   --bits 4 \
  --stage1_max_blocks 8 --stage1_n_trials 20 \
  --nsamples 64 --seq_len 2048 \
  --trust_remote_code > ./logs/step0_4bit.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

# Enable local imports when running as script from arbitrary cwd.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

if __package__:
    from .joint.step0_1_quantize import Step01QuantizeConfig, run as run_step01
    from .joint.step0_2_calib import Step02CalibConfig, run as run_step02
    from .joint.step0_3_BO_opt import Step03BOConfig, run as run_step03
else:
    from joint.step0_1_quantize import Step01QuantizeConfig, run as run_step01
    from joint.step0_2_calib import Step02CalibConfig, run as run_step02
    from joint.step0_3_BO_opt import Step03BOConfig, run as run_step03


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


def _pick_bit_override(bit: int, global_value, per_bit: Dict[int, Optional[int]]):
    v = per_bit.get(bit)
    return global_value if v is None else v


def _canonical_bits(bits: Iterable[int]) -> List[int]:
    out = []
    seen = set()
    for b in bits:
        bi = int(b)
        if bi < 1 or bi > 4:
            raise ValueError(f"bits must be within [1,4], got {bi}")
        if bi not in seen:
            seen.add(bi)
            out.append(bi)
    return out


def _cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


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
    bit: int,
    wdq_path: Path,
    uvab_path: Path,
    calib_s_path: Path,
    out_root: Path,
) -> Dict[str, int]:
    wdq_best: Dict[str, torch.Tensor] = torch.load(wdq_path, map_location="cpu")
    uvab_best: Dict[str, dict] = torch.load(uvab_path, map_location="cpu")
    calib_s: Dict[str, dict] = torch.load(calib_s_path, map_location="cpu")

    bit_dir = out_root / f"bit{bit}"
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
        meta_src = rec.get("meta", {})
        payload = {
            "module": module,
            "full_weight": full_name,
            "Wq": Wq.to(torch.float16).cpu(),
            "A": A.to(torch.float16).cpu(),
            "B": B.to(torch.float16).cpu(),
            "meta": {
                "bit": int(bit),
                "source": "step0_optimization(step3_5_v2 best)",
                "uses_calib_inv_s": True,
                "step3_meta": meta_src,
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


def _copy_best_raw(bit: int, wdq_src: Path, uvab_src: Path, out_root: Path) -> Dict[str, str]:
    dst_dir = out_root / "best_raw" / f"bit{bit}"
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
    ap = argparse.ArgumentParser("Integrated step0 optimization (joint step1/2/3 -> prebake-style export)")

    # Core
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out_root", required=True, help="Final output root (prebake-style bit1..bit4 + meta.json)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--bits", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--python_exe", default=sys.executable)

    # Step1 (quantize)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--group_size_1", type=int, default=None)
    ap.add_argument("--group_size_2", type=int, default=None)
    ap.add_argument("--group_size_3", type=int, default=None)
    ap.add_argument("--group_size_4", type=int, default=None)
    ap.add_argument("--clip_percentile", type=float, default=0.0)
    ap.add_argument("--lloyd_iter", type=int, default=12)
    ap.add_argument("--chunk_groups", type=int, default=4096)
    ap.add_argument("--layer_regex", default=None)

    # Step2 (calib)
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
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--cov_mode", default="oas", choices=["var", "oas", "second_moment"])
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=42)

    # Step3 (BO + optimization)
    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--rank_ab", type=int, default=64)
    ap.add_argument("--rank_ab_1", type=int, default=None)
    ap.add_argument("--rank_ab_2", type=int, default=None)
    ap.add_argument("--rank_ab_3", type=int, default=None)
    ap.add_argument("--rank_ab_4", type=int, default=None)
    ap.add_argument("--outer_loops", type=int, default=40)
    ap.add_argument("--delta_steps", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--lr_min_ratio", type=float, default=0.1)
    ap.add_argument("--lr_step_gamma", type=float, default=0.3)
    ap.add_argument("--stage1_max_blocks", type=int, default=8)
    ap.add_argument("--stage1_n_trials", type=int, default=20)
    ap.add_argument("--stage1_timeout_sec", type=int, default=0)
    ap.add_argument("--stage1_storage", default="")
    ap.add_argument("--prune_min_layers", type=int, default=2)
    ap.add_argument("--prune_warmup_steps", type=int, default=20)
    ap.add_argument("--pruner_startup_trials", type=int, default=5)
    ap.add_argument("--pruner_interval_steps", type=int, default=5)
    ap.add_argument("--study_prefix", default="step0_mix")

    # Cleanup behavior
    ap.add_argument("--keep_temps", action="store_true", help="Keep temporary step1/step2/step3 directories")

    args = ap.parse_args()

    bits = _canonical_bits(args.bits)
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for b in bits:
        (out_root / f"bit{b}").mkdir(parents=True, exist_ok=True)
    tmp_root = out_root / "_tmp"
    summaries_root = out_root / "summaries"
    summaries_root.mkdir(parents=True, exist_ok=True)

    per_bit_gs = {1: args.group_size_1, 2: args.group_size_2, 3: args.group_size_3, 4: args.group_size_4}
    per_bit_rank = {1: args.rank_ab_1, 2: args.rank_ab_2, 3: args.rank_ab_3, 4: args.rank_ab_4}

    run_started = int(time.time())
    all_summaries: List[dict] = []

    for bit in bits:
        group_size = int(_pick_bit_override(bit, args.group_size, per_bit_gs))
        rank_ab = int(_pick_bit_override(bit, args.rank_ab, per_bit_rank))

        bit_tmp = tmp_root / f"bit{bit}"
        step1_out_dir = bit_tmp / "step1_quant"
        step2_out_path = bit_tmp / "step2_calib" / "calib_sqrtdiag.pt"
        step3_out_root = bit_tmp / "step3_runs"
        study_name = f"{args.study_prefix}_b{bit}"

        print(f"\n[step0] bit={bit} start | group_size={group_size} | rank_ab={rank_ab}")

        run_step01(
            Step01QuantizeConfig(
                model_id=args.model_id,
                revision=args.revision,
                trust_remote_code=bool(args.trust_remote_code),
                dtype=args.dtype,
                device=args.device,
                bits=int(bit),
                group_size=group_size,
                clip_percentile=args.clip_percentile,
                lloyd_iter=args.lloyd_iter,
                chunk_groups=args.chunk_groups,
                layer_regex=args.layer_regex,
                out_dir=str(step1_out_dir),
                python_exe=args.python_exe,
            )
        )

        run_step02(
            Step02CalibConfig(
                model_name=args.model_id,
                out_calib_s=str(step2_out_path),
                bits=int(bit),
                group_size=int(args.group_size),
                group_size_1=args.group_size_1,
                group_size_2=args.group_size_2,
                group_size_3=args.group_size_3,
                group_size_4=args.group_size_4,
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
                trust_remote_code=bool(args.trust_remote_code),
                cov_mode=args.cov_mode,
                eps=args.eps,
                tag=f"step0_bit{bit}",
                seed=args.seed,
                python_exe=args.python_exe,
            )
        )

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
                dtype_w=args.dtype_w,
                rank_ab=rank_ab,
                outer_loops=args.outer_loops,
                delta_steps=args.delta_steps,
                eps=args.eps,
                layer_regex=args.layer_regex,
                log_every=args.log_every,
                lr_min_ratio=args.lr_min_ratio,
                lr_step_gamma=args.lr_step_gamma,
                study_name=study_name,
                seed=args.seed,
                stage1_max_blocks=args.stage1_max_blocks,
                stage1_n_trials=args.stage1_n_trials,
                stage1_timeout_sec=args.stage1_timeout_sec,
                stage1_storage=args.stage1_storage,
                prune_min_layers=args.prune_min_layers,
                prune_warmup_steps=args.prune_warmup_steps,
                pruner_startup_trials=args.pruner_startup_trials,
                pruner_interval_steps=args.pruner_interval_steps,
                python_exe=args.python_exe,
            )
        )

        raw_paths = _copy_best_raw(
            bit=bit,
            wdq_src=step3_outs["wdq_star_best"],
            uvab_src=step3_outs["lowrank_uv_ab_best"],
            out_root=out_root,
        )

        export_stats = export_prebake_from_step3(
            bit=bit,
            wdq_path=Path(raw_paths["wdq_star_best"]),
            uvab_path=Path(raw_paths["lowrank_uv_ab_best"]),
            calib_s_path=step2_out_path,
            out_root=out_root,
        )

        bit_summary = {
            "bit": int(bit),
            "group_size": group_size,
            "rank_ab": rank_ab,
            "step1_out_dir": str(step1_out_dir),
            "step2_out_calib_s": str(step2_out_path),
            "step3_run_dir": str(step3_outs["run_dir"]),
            "step3_stage2_dir": str(step3_outs["stage2_dir"]),
            "best_raw": raw_paths,
            "prebake_export": export_stats,
        }
        _write_json(summaries_root / f"bit{bit}.json", bit_summary)
        all_summaries.append(bit_summary)
        print(f"[step0] bit={bit} done | prebake saved={export_stats['saved']}")

        if not args.keep_temps:
            _cleanup_path(step1_out_dir)
            _cleanup_path(step2_out_path)
            # step1/step2를 step0_3 이후에 지우고, step3 임시 run도 최종 raw copy 후 정리
            _cleanup_path(step3_out_root)
            if bit_tmp.exists():
                try:
                    bit_tmp.rmdir()
                except OSError:
                    pass

    meta = {
        "model_id": args.model_id,
        "revision": args.revision,
        "bits": bits,
        "group_size": args.group_size,
        "group_size_overrides": {str(k): (None if v is None else int(v)) for k, v in per_bit_gs.items()},
        "rank_ab": int(args.rank_ab),
        "rank_ab_overrides": {str(k): (None if v is None else int(v)) for k, v in per_bit_rank.items()},
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "nsamples": int(args.nsamples),
        "seq_len": int(args.seq_len),
        "dtype": args.dtype,
        "dtype_w": args.dtype_w,
        "device": args.device,
        "created": run_started,
        "note": "Integrated joint step1/2/3 best outputs exported in step4_prebake-compatible per-module format.",
    }
    _write_json(out_root / "meta.json", meta)

    final_summary = {
        "out_root": str(out_root),
        "meta_json": str(out_root / "meta.json"),
        "bits": bits,
        "per_bit_summaries": [str(summaries_root / f"bit{b}.json") for b in bits],
        "best_raw_root": str(out_root / "best_raw"),
        "kept_temps": bool(args.keep_temps),
    }
    _write_json(summaries_root / "summary.json", final_summary)

    if not args.keep_temps and tmp_root.exists():
        try:
            tmp_root.rmdir()
        except OSError:
            pass

    print("\n[step0] COMPLETED")
    print(f"  out_root: {out_root}")
    print(f"  meta    : {out_root / 'meta.json'}")
    print(f"  summary : {summaries_root / 'summary.json'}")


if __name__ == "__main__":
    main()
