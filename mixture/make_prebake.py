#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create prebake layer shards from best_raw artifacts.

Default target layout:
  <prebake_root>/best_raw/bit{bit}/wdq_star_best.pt
  <prebake_root>/best_raw/bit{bit}/lowrank_uv_ab_best.pt
  -> <prebake_root>/bit{bit}/*.pt

If calib_sqrtdiag.pt is available (via --calib_s_path or summaries/bit{bit}.json),
the script applies inv_s scaling to B exactly like step0_optimization export.
If calibration is not found, it still exports using only wdq/uvab.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Dict, Optional

import torch


def _safe_name(s) -> str:
    if s is None:
        s = "none"
    s = str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _module_name_from_weight(full_weight_name: str) -> str:
    if full_weight_name.endswith(".weight"):
        return full_weight_name[: -len(".weight")]
    return full_weight_name


def _ensure_inv_s(entry: dict) -> torch.Tensor:
    if "inv_s" in entry:
        return entry["inv_s"].to(torch.float32)
    if "s" in entry:
        s = entry["s"].to(torch.float32)
        return (1.0 / torch.clamp(s, min=1e-12)).to(torch.float32)
    raise KeyError("calib entry must include 'inv_s' or 's'")


def _resolve_calib_from_summary(prebake_root: Path, bit: int) -> Optional[Path]:
    summary_path = prebake_root / "summaries" / f"bit{bit}.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = summary.get("step2_out_calib_s")
    if not raw:
        return None
    p = Path(str(raw))
    return p if p.exists() else None


@torch.no_grad()
def export_layerwise_prebake(
    *,
    bit: int,
    wdq_path: Path,
    uvab_path: Path,
    out_dir: Path,
    calib_s_path: Optional[Path],
    strict_calib: bool,
    max_layers: int,
    log_every: int,
) -> Dict[str, int]:
    print(f"[load] wdq : {wdq_path}")
    wdq_best: Dict[str, torch.Tensor] = torch.load(wdq_path, map_location="cpu")
    print(f"[load] uvab: {uvab_path}")
    uvab_best: Dict[str, dict] = torch.load(uvab_path, map_location="cpu")

    use_calib = calib_s_path is not None
    calib_s: Optional[Dict[str, dict]] = None
    if use_calib:
        print(f"[load] calib: {calib_s_path}")
        calib_s = torch.load(calib_s_path, map_location="cpu")
    elif strict_calib:
        raise FileNotFoundError("strict_calib=True but calib_s_path is not set/found.")

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_missing_uvab = 0
    skipped_missing_calib = 0
    skipped_bad_uvab = 0

    for full_name, Wq in wdq_best.items():
        if max_layers > 0 and saved >= max_layers:
            break

        rec = uvab_best.get(full_name)
        if rec is None:
            skipped_missing_uvab += 1
            continue

        A = rec.get("A")
        B = rec.get("B")
        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
            skipped_bad_uvab += 1
            continue

        A = A.to(torch.float32)
        B = B.to(torch.float32)
        u = rec.get("u")
        v = rec.get("v")

        if isinstance(u, torch.Tensor):
            A = A * u.to(torch.float32).view(-1, 1)
        if isinstance(v, torch.Tensor):
            B = B * v.to(torch.float32).view(1, -1)

        applied_calib = False
        if use_calib:
            assert calib_s is not None
            centry = calib_s.get(full_name)
            if centry is None:
                if strict_calib:
                    raise KeyError(f"Missing calib entry for: {full_name}")
                skipped_missing_calib += 1
                continue
            inv_s = _ensure_inv_s(centry).to(torch.float32)
            B = B * inv_s.view(1, -1)
            applied_calib = True

        module = _module_name_from_weight(full_name)
        payload = {
            "module": module,
            "full_weight": full_name,
            "Wq": Wq.to(torch.float16).cpu(),
            "A": A.to(torch.float16).cpu(),
            "B": B.to(torch.float16).cpu(),
            "meta": {
                "bit": int(bit),
                "source": "make_prebake.py",
                "uses_calib_inv_s": bool(applied_calib),
                "step3_meta": rec.get("meta", {}),
            },
        }
        torch.save(payload, out_dir / f"{_safe_name(module)}.pt")
        saved += 1

        if log_every > 0 and saved % log_every == 0:
            print(f"[save] {saved} layers")

        del A, B

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "saved": int(saved),
        "skipped_missing_uvab": int(skipped_missing_uvab),
        "skipped_missing_calib": int(skipped_missing_calib),
        "skipped_bad_uvab": int(skipped_bad_uvab),
        "wdq_total": int(len(wdq_best)),
        "uvab_total": int(len(uvab_best)),
        "used_calib": int(1 if use_calib else 0),
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Export layer-wise prebake shards from best_raw/*.pt")
    ap.add_argument(
        "--prebake_root",
        default="/ssd/ssd4/asl/LABA/mixture/output_7b/output_step0_prebake",
        help="Root path that contains best_raw/, bitN/, summaries/",
    )
    ap.add_argument("--bit", type=int, default=1, help="Target bit (default: 1)")
    ap.add_argument("--wdq_path", default=None, help="Override wdq_star_best.pt path")
    ap.add_argument("--uvab_path", default=None, help="Override lowrank_uv_ab_best.pt path")
    ap.add_argument("--calib_s_path", default=None, help="Optional calib_sqrtdiag.pt path")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: <prebake_root>/bit{bit})")
    ap.add_argument(
        "--strict_calib",
        action="store_true",
        help="Require calibration file+entries for every saved layer",
    )
    ap.add_argument("--max_layers", type=int, default=0, help="Debug: save only first N layers (0=all)")
    ap.add_argument("--log_every", type=int, default=32, help="Progress print interval (0=off)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    bit = int(args.bit)
    if bit < 1:
        raise ValueError(f"bit must be >=1, got {bit}")

    prebake_root = Path(args.prebake_root).resolve()
    raw_dir = prebake_root / "best_raw" / f"bit{bit}"

    wdq_path = Path(args.wdq_path).resolve() if args.wdq_path else (raw_dir / "wdq_star_best.pt")
    uvab_path = Path(args.uvab_path).resolve() if args.uvab_path else (raw_dir / "lowrank_uv_ab_best.pt")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (prebake_root / f"bit{bit}")

    calib_s_path: Optional[Path] = None
    if args.calib_s_path:
        c = Path(args.calib_s_path).resolve()
        if c.exists():
            calib_s_path = c
        elif args.strict_calib:
            raise FileNotFoundError(f"calib_s_path not found: {c}")
    else:
        calib_s_path = _resolve_calib_from_summary(prebake_root, bit)

    if not wdq_path.exists():
        raise FileNotFoundError(f"wdq_path not found: {wdq_path}")
    if not uvab_path.exists():
        raise FileNotFoundError(f"uvab_path not found: {uvab_path}")

    print("== make_prebake ==")
    print(f"prebake_root: {prebake_root}")
    print(f"bit         : {bit}")
    print(f"wdq_path    : {wdq_path}")
    print(f"uvab_path   : {uvab_path}")
    print(f"calib_s_path: {calib_s_path}")
    print(f"out_dir     : {out_dir}")

    stats = export_layerwise_prebake(
        bit=bit,
        wdq_path=wdq_path,
        uvab_path=uvab_path,
        out_dir=out_dir,
        calib_s_path=calib_s_path,
        strict_calib=bool(args.strict_calib),
        max_layers=int(args.max_layers),
        log_every=int(args.log_every),
    )
    print("[done] " + json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
