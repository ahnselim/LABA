#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4B (Assemble) — Build {quantized_weights, correction_layerwise, b_ref_map}
from prebaked per-module bit caches.

입력:
  • --prebake_root (필수): step4_prebake_quant_and_ab.py 산출 디렉토리
  • --bit_assign_csv       : CSV (열 이름 중 하나 사용) → ["layer_name" | "module" | "name"], ["R_int" | "selected_bit" | "bit"]
  • --out_dir              : 출력 디렉토리

출력(파일명은 Step4와 동일):
  • out_dir/quantized_weights.pt
  • out_dir/correction_layerwise.pt
  • out_dir/b_ref_map_layerwise.json

CSV 예시:
layer_name,R_int
model.layers.0.self_attn.q_proj,2
model.layers.0.self_attn.k_proj,3
...

python step4b_assemble_from_cache.py \
  --prebake_root ./artifacts/bitmin/prebake_llama2_7b_r64_g128 \
  --bit_assign_csv ./artifacts/bitmin/step3c_run1/round_0/candidate_000.bits.csv \
  --out_dir ./artifacts/bitmin/step3c_run1/round_0/candidate_000/
"""

import os, csv, json, argparse
from pathlib import Path
import torch, re

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))

def load_bits_csv(path: str):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = (row.get("layer_name") or row.get("module") or row.get("name") or "").strip()
            if not name: continue
            b = None
            for key in ("R_int","selected_bit","bit"):
                if key in row and str(row[key]).strip()!="":
                    try: b = int(float(row[key])); break
                    except: pass
            if b is None: continue
            # 1~4bit 허용 (이전 버전은 2~4bit만 허용했음)
            mapping[name] = max(1, min(4, b))
    if not mapping:
        raise ValueError("No rows in bit_assign_csv")
    return mapping

def main():
    ap = argparse.ArgumentParser("Step 4B — Assemble from prebaked caches")
    ap.add_argument("--prebake_root", required=True)
    ap.add_argument("--bit_assign_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    prebake = Path(args.prebake_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    sel = load_bits_csv(args.bit_assign_csv)

    qweights, corrections, bmap = {}, {}, {}
    miss, total = 0, 0

    for module, bit in sel.items():
        total += 1
        safe = _safe_name(module)
        fpath = prebake / f"bit{bit}" / f"{safe}.pt"
        if not fpath.exists():
            miss += 1
            print(f"[assemble][miss] {fpath} not found (module={module}, bit={bit})")
            continue
        obj = torch.load(str(fpath), map_location="cpu")
        full_w = obj["full_weight"]               # "... .weight"
        qweights[full_w] = obj["Wq"]              # fp16/CPU
        corrections[f"{module}.A"] = obj["A"]     # fp16/CPU
        corrections[f"{module}.B"] = obj["B"]     # fp16/CPU
        bmap[full_w] = f"{module}.B"

    torch.save(qweights, out_dir / "quantized_weights.pt")
    torch.save(corrections, out_dir / "correction_layerwise.pt")
    with open(out_dir / "b_ref_map_layerwise.json", "w", encoding="utf-8") as f:
        json.dump(bmap, f, indent=2)

    print(f"[assemble] wrote: {out_dir/'quantized_weights.pt'} ({len(qweights)} tensors)")
    print(f"[assemble] wrote: {out_dir/'correction_layerwise.pt'} (layers={len([k for k in corrections if k.endswith('.A')])})")
    print(f"[assemble] wrote: {out_dir/'b_ref_map_layerwise.json'} ; miss={miss}/{total}")

if __name__ == "__main__":
    main()
