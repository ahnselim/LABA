#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alt Step4 Eval - evaluate `step_3_alternating.py` outputs.

Supports:
  - Wdq-only baseline
  - Wdq + AB correction

Inputs:
  1) `--step3_dir`:
       auto-resolve `wdq_star_best.pt` and `low_rank_ab_best.pt` by default
  2) explicit artifact paths:
       `--wdq_star_path` and optional `--low_rank_ab_path`

Outputs:
  - console metrics
  - optional JSON summary

CUDA_VISIBLE_DEVICES=1,3 \
python step4_eval.py \
  --model_name meta-llama/Llama-3.1-8B \
  --wdq_star_path ./output/llama3_8b/step3_alt/3bit/wdq_star_best.pt \
  --low_rank_ab_path ./output/llama3_8b/step3_alt/3bit/low_rank_ab_best.pt \
  --device_map auto \
  --num_gpus 2 \
  --device cuda:0 \
  --compare_wdq_only

CUDA_VISIBLE_DEVICES=2 \
python step4_eval.py \
  --model_name meta-llama/Llama-3.1-8B \
  --wdq_star_path ./output/llama3_8b/step3_5_three_stage/2bit/wdq_star.pt \
  --low_rank_ab_path ./output/llama3_8b/step3_5_three_stage/2bit/low_rank_ab.pt \
  --device_map auto \
  --device cuda:0 \
  --compare_wdq_only
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_THIS_DIR = Path(__file__).resolve().parent
_JOINT_DIR = _THIS_DIR.parent / "code" / "joint"
if str(_JOINT_DIR) not in sys.path:
    sys.path.insert(0, str(_JOINT_DIR))

from step4_eval import (  # type: ignore  # noqa: E402
    AddLowRankCorrection,
    AddLowRankCorrectionFP32,
    apply_wdq_star,
    evaluate_ppl_wikitext2,
    measure_generation_metrics,
    patch_layerwise_ab_from_step2p5,
)


def torch_load_cpu(path: Path, *, mmap: bool = False):
    load_kwargs = {"map_location": "cpu"}
    if mmap:
        load_kwargs["mmap"] = True
    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        if not mmap:
            raise
    except RuntimeError:
        if not mmap:
            raise
    return torch.load(path, map_location="cpu")


class ShardedArtifactDict(Mapping[str, Any]):
    def __init__(self, root: Path, *, mmap: bool = False):
        self.root = root
        self.mmap = bool(mmap)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"sharded artifact manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("entries", [])
        self._entries = [(str(item["key"]), str(item["file"])) for item in entries]
        self._index = {key: rel for key, rel in self._entries}

    def __getitem__(self, key: str) -> Any:
        if key not in self._index:
            raise KeyError(key)
        obj = torch_load_cpu(self.root / self._index[key], mmap=self.mmap)
        if isinstance(obj, dict) and len(obj) == 1 and key in obj:
            return obj[key]
        return obj

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def items(self):
        for key, _ in self._entries:
            yield key, self[key]


def _resolve_artifact_path(base: Path, stem: str) -> Optional[Path]:
    file_path = base / f"{stem}.pt"
    if file_path.exists():
        return file_path
    dir_path = base / stem
    if dir_path.exists():
        return dir_path
    return None


def _load_artifact(path: Path, *, mmap: bool):
    if path.is_dir():
        return ShardedArtifactDict(path, mmap=mmap)
    return torch_load_cpu(path, mmap=mmap)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Alt Step4 Eval - Wdq* and AB* evaluation")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--step3_dir", default=None, help="Alt step3 output dir")
    ap.add_argument("--use_best", action="store_true", help="With --step3_dir, prefer *_best.pt artifacts")
    ap.add_argument("--wdq_star_path", default=None, help="Explicit wdq_star(.pt) path")
    ap.add_argument("--low_rank_ab_path", default=None, help="Explicit low_rank_ab(.pt) path")
    ap.add_argument("--calib_s_path", default=None, help="Optional calib_s for uv-ab style artifacts")
    ap.add_argument("--artifact_mmap", dest="artifact_mmap", action="store_true")
    ap.add_argument("--no_artifact_mmap", dest="artifact_mmap", action="store_false")

    ap.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    ap.add_argument(
        "--device_map",
        default=("auto" if torch.cuda.device_count() > 1 else "none"),
        help='Model load placement: e.g. "auto" or "none"',
    )
    ap.add_argument("--num_gpus", type=int, default=0, help="device_map=auto일 때 사용할 최대 GPU 개수 (0이면 visible 전부)")
    ap.add_argument(
        "--first_gpu_reserve_gib",
        type=int,
        default=6,
        help="device_map=auto일 때 첫 번째 visible GPU에 남겨둘 GiB 수",
    )
    ap.add_argument(
        "--other_gpu_reserve_gib",
        type=int,
        default=4,
        help="device_map=auto일 때 나머지 GPU에 남겨둘 GiB 수",
    )
    ap.add_argument("--model_dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--ab_compute", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--ab_alpha", type=float, default=1.0)
    ap.add_argument("--compare_wdq_only", action="store_true", help="Also run Wdq-only baseline before AB")
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--ppl_stride", type=int, default=2048)
    ap.add_argument("--ppl_max_tokens", type=int, default=0)
    ap.add_argument("--gen_max_new_tokens", type=int, default=50)
    ap.add_argument("--gen_repeats", type=int, default=1)
    ap.add_argument("--gen_do_sample", action="store_true")
    ap.add_argument("--gen_num_beams", type=int, default=1)
    ap.add_argument("--gen_temperature", type=float, default=1.0)
    ap.add_argument("--gen_top_p", type=float, default=1.0)
    ap.add_argument("--save_json", default=None)
    ap.set_defaults(artifact_mmap=True)
    return ap.parse_args()


def _torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported model_dtype: {name}")


def _set_ab_alpha(model: torch.nn.Module, alpha: float) -> int:
    count = 0
    for m in model.modules():
        if isinstance(m, (AddLowRankCorrection, AddLowRankCorrectionFP32)):
            m.alpha = float(alpha)
            count += 1
    return count


def _resolve_step3_artifacts(
    step3_dir: Path,
    use_best: bool,
) -> Tuple[Path, Optional[Path]]:
    wdq_stem = "wdq_star_best" if use_best else "wdq_star"
    ab_stem = "low_rank_ab_best" if use_best else "low_rank_ab"

    wdq_path = _resolve_artifact_path(step3_dir, wdq_stem)
    ab_path = _resolve_artifact_path(step3_dir, ab_stem)

    if wdq_path is None and use_best:
        wdq_path = _resolve_artifact_path(step3_dir, "wdq_star")
    if ab_path is None and use_best:
        ab_path = _resolve_artifact_path(step3_dir, "low_rank_ab")

    if wdq_path is None:
        raise FileNotFoundError(f"wdq artifact not found under {step3_dir}")
    if ab_path is None:
        ab_path = None
    return wdq_path, ab_path


def _load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    device_map: Optional[str],
    num_gpus: int,
    first_gpu_reserve_gib: int,
    other_gpu_reserve_gib: int,
    torch_dtype: torch.dtype,
    trust_remote_code: bool,
):
    print(f"📥 Loading tokenizer: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"📥 Loading base model: {model_name} (dtype={torch_dtype}, device={device}, device_map={device_map})")
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if device_map is None:
        model_kwargs["device_map"] = "cpu"
    else:
        model_kwargs["device_map"] = device_map

    if device_map == "auto" and torch.cuda.is_available():
        visible = torch.cuda.device_count()
        use_n = visible if int(num_gpus) <= 0 else min(int(num_gpus), int(visible))
        if use_n > 0:
            max_memory = {}
            for idx in range(use_n):
                total_gib = int(torch.cuda.get_device_properties(idx).total_memory // (1024 ** 3))
                reserve_gib = int(first_gpu_reserve_gib) if idx == 0 else int(other_gpu_reserve_gib)
                max_memory[idx] = f"{max(1, total_gib - reserve_gib)}GiB"
            max_memory["cpu"] = "512GiB"
            model_kwargs["max_memory"] = max_memory
            print(f"[Eval] auto device_map GPU limit: {use_n} (indices: {list(range(use_n))})")
            print(f"[Eval] max_memory: {max_memory}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model.eval()
    if device_map is None:
        model = model.to(device)
    return model, tok


def _normalize_device(device_like: Any) -> Optional[torch.device]:
    if isinstance(device_like, torch.device):
        return device_like
    if isinstance(device_like, int):
        return torch.device(f"cuda:{device_like}")
    if isinstance(device_like, str):
        text = device_like.strip()
        if text == "":
            return None
        if text.isdigit():
            return torch.device(f"cuda:{text}")
        if text in {"cpu", "mps"} or text.startswith("cuda:") or text == "cuda":
            return torch.device(text)
    return None


def _infer_eval_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict) and hf_device_map:
        preferred_keys = (
            "model.embed_tokens",
            "model.decoder.embed_tokens",
            "transformer.wte",
            "transformer.word_embeddings",
            "gpt_neox.embed_in",
        )
        for key in preferred_keys:
            if key in hf_device_map:
                dev = _normalize_device(hf_device_map[key])
                if dev is not None:
                    return dev
        for _, value in hf_device_map.items():
            dev = _normalize_device(value)
            if dev is not None:
                return dev
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def main() -> None:
    args = _parse_args()

    if args.step3_dir is None and args.wdq_star_path is None:
        raise ValueError("Need --step3_dir or --wdq_star_path")

    wdq_path = None
    ab_path = None
    if args.step3_dir is not None:
        step3_dir = Path(args.step3_dir).resolve()
        if not step3_dir.exists():
            raise FileNotFoundError(f"step3_dir not found: {step3_dir}")
        wdq_path, ab_path = _resolve_step3_artifacts(step3_dir, use_best=bool(args.use_best))

    if args.wdq_star_path is not None:
        wdq_path = Path(args.wdq_star_path).resolve()
    if args.low_rank_ab_path is not None:
        ab_path = Path(args.low_rank_ab_path).resolve()

    if wdq_path is None or not wdq_path.exists():
        raise FileNotFoundError(f"wdq_star path not found: {wdq_path}")
    if ab_path is not None and not ab_path.exists():
        raise FileNotFoundError(f"low_rank_ab path not found: {ab_path}")

    requested_device = torch.device(args.device)
    dm_raw = str(args.device_map).strip().lower()
    resolved_device_map = None if dm_raw in {"", "none", "null"} else args.device_map
    model_dtype = _torch_dtype_from_name(args.model_dtype)

    print("== alt step4 eval ==")
    print(f"model_name : {args.model_name}")
    print(f"device     : {requested_device}")
    print(f"device_map : {resolved_device_map}")
    print(f"wdq_path   : {wdq_path}")
    print(f"ab_path    : {ab_path}")

    print(f"📦 Loading Wdq*: {wdq_path}")
    wdq_star = _load_artifact(wdq_path, mmap=bool(args.artifact_mmap))

    low_rank_ab: Optional[Mapping[str, Any]] = None
    calib_s = None
    if ab_path is not None:
        print(f"📦 Loading AB*: {ab_path}")
        low_rank_ab = _load_artifact(ab_path, mmap=bool(args.artifact_mmap))

    if args.calib_s_path is not None:
        calib_path = Path(args.calib_s_path).resolve()
        print(f"📦 Loading calib_s: {calib_path}")
        calib_s = torch_load_cpu(calib_path, mmap=bool(args.artifact_mmap))

    if low_rank_ab is not None:
        has_uvab = any(
            isinstance(item, dict) and (("u" in item) or ("v" in item))
            for item in low_rank_ab.values()
        )
        if has_uvab and calib_s is None:
            raise ValueError(
                "Detected uv-ab artifact but --calib_s_path is missing."
            )

    model = None
    tok = None
    results: Dict[str, Any] = {
        "model_name": args.model_name,
        "wdq_star_path": str(wdq_path),
        "low_rank_ab_path": (str(ab_path) if ab_path is not None else None),
        "device": str(requested_device),
        "device_map": resolved_device_map,
        "model_dtype": args.model_dtype,
        "ab_compute": args.ab_compute,
        "ab_alpha": float(args.ab_alpha),
    }

    prompts = [
        "The history of machine learning began",
        "Large language models are useful because",
        "In a transformer layer, attention works by",
    ]

    try:
        model, tok = _load_model_and_tokenizer(
            model_name=args.model_name,
            device=requested_device,
            device_map=resolved_device_map,
            num_gpus=int(args.num_gpus),
            first_gpu_reserve_gib=int(args.first_gpu_reserve_gib),
            other_gpu_reserve_gib=int(args.other_gpu_reserve_gib),
            torch_dtype=model_dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        eval_device = _infer_eval_device(model, requested_device)
        print(f"[Eval] input/eval device: {eval_device}")
        results["eval_device"] = str(eval_device)

        apply_wdq_star(model, wdq_star)

        if args.compare_wdq_only or low_rank_ab is None:
            print("[Eval] Wdq-only baseline")
            ppl_wdq, ppl_wdq_sec = evaluate_ppl_wikitext2(
                model,
                tok,
                eval_device,
                label="Wdq*",
                stride=int(args.ppl_stride),
                max_tokens=int(args.ppl_max_tokens),
            )
            results["wdq_only"] = {
                "ppl": float(ppl_wdq),
                "elapsed_sec": float(ppl_wdq_sec),
            }

        if low_rank_ab is not None:
            model = patch_layerwise_ab_from_step2p5(
                model,
                low_rank_ab=low_rank_ab,
                low_rank_abbar=None,
                calib_s=calib_s,
                calib_h_lowrank=None,
                alpha=float(args.ab_alpha),
                ab_compute=str(args.ab_compute),
            )
            n_wrapped = _set_ab_alpha(model, float(args.ab_alpha))
            print(f"[Eval] Wdq+AB eval (wrapped_modules={n_wrapped}, alpha={args.ab_alpha})")

            ppl_ab, ppl_ab_sec = evaluate_ppl_wikitext2(
                model,
                tok,
                eval_device,
                label=f"Wdq*+AB* (a={args.ab_alpha:g})",
                stride=int(args.ppl_stride),
                max_tokens=int(args.ppl_max_tokens),
            )
            results["wdq_ab"] = {
                "ppl": float(ppl_ab),
                "elapsed_sec": float(ppl_ab_sec),
                "wrapped_modules": int(n_wrapped),
            }

            if "wdq_only" in results:
                results["delta_ppl_wdq_minus_ab"] = (
                    float(results["wdq_only"]["ppl"]) - float(results["wdq_ab"]["ppl"])
                )

        if not args.skip_gen:
            print("[Eval] generation metrics")
            gen_stats = measure_generation_metrics(
                model,
                tok,
                eval_device,
                prompts=prompts,
                max_new_tokens=int(args.gen_max_new_tokens),
                do_sample=bool(args.gen_do_sample),
                num_beams=int(args.gen_num_beams),
                temperature=float(args.gen_temperature),
                top_p=float(args.gen_top_p),
                repeats=int(args.gen_repeats),
            )
            results["generation"] = gen_stats

    finally:
        del wdq_star, low_rank_ab, calib_s
        if model is not None:
            del model
        if tok is not None:
            del tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.save_json:
        out_path = Path(args.save_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[Eval] saved json: {out_path}")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
