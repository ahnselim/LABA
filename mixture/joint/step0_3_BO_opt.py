#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixture step0 module: Step0-3 BO + full optimization runner (delta + uv + ab).

역할:
  - Stage1: 일부 transformer block 대상으로 BO 하이퍼파라미터 탐색
  - Stage2: 전체 타깃 레이어에 best 파라미터 적용
  - 결과물 `wdq_star_best.pt`, `lowrank_uv_ab_best.pt` 생성

사용 방식:
  - CLI 실행: 이 파일의 `main()`
  - 모듈 사용: 하단 `Step03BOConfig` + `run()` (결과 경로 dict 반환)

참고:
  - 기본 `--step4_script` 경로는 같은 디렉터리의 `step4_eval.py`를 사용한다.
  - `LABA/mixture/step0_optimization.py`와의 호환을 위해 래퍼 API를 함께 제공한다.
"""

import argparse
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import torch

try:
    import optuna
    from optuna.trial import TrialState
except Exception as e:
    raise RuntimeError("optuna 필요: pip install optuna") from e

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers 필요: pip install transformers") from e


TARGET_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
}

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


def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )


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
    codebook_ogq: torch.Tensor,   # [O,G,Q]
    qcodes_ogs: torch.Tensor,     # [O,G,S]
    orig_i: int,
) -> torch.Tensor:
    o, g, q = codebook_ogq.shape
    _, _, s = qcodes_ogs.shape
    n = o * g
    cb = codebook_ogq.reshape(n, q)
    idx = qcodes_ogs.reshape(n, s).long()
    xq = torch.gather(cb, dim=1, index=idx)
    xq = xq.reshape(o, g, s)
    wq_pad = xq.reshape(o, g * s)
    return wq_pad[:, :orig_i].contiguous()


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


@torch.no_grad()
def update_uv_closed_form(
    rw: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    uv_iters: int,
    eps: float,
    normalize_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = a @ b
    u = u.clone()
    v = v.clone()
    for _ in range(max(1, int(uv_iters))):
        mv = m * v.unsqueeze(0)
        num_u = (rw * mv).sum(dim=1)
        den_u = (mv * mv).sum(dim=1).clamp_min(eps)
        u = num_u / den_u

        mu = u.unsqueeze(1) * m
        num_v = (rw * mu).sum(dim=0)
        den_v = (mu * mu).sum(dim=0).clamp_min(eps)
        v = num_v / den_v

        if normalize_mode != "none":
            u_abs = u.abs()
            if normalize_mode == "mean_abs":
                alpha = u_abs.mean().clamp_min(eps)
            elif normalize_mode == "rms":
                alpha = torch.sqrt(torch.mean(u * u)).clamp_min(eps)
            elif normalize_mode == "median_abs":
                alpha = torch.median(u_abs).clamp_min(eps)
            else:
                raise ValueError(f"unknown normalize_mode: {normalize_mode}")
            u = u / alpha
            v = v * alpha
    return u, v


def get_lr_scale(outer: int, total_outer: int, schedule: str, min_ratio: float, step_gamma: float) -> float:
    if schedule == "none":
        return 1.0
    if total_outer <= 1:
        return 1.0
    if schedule == "cosine":
        t = float(outer) / float(max(1, total_outer - 1))
        return float(min_ratio) + (1.0 - float(min_ratio)) * 0.5 * (1.0 + math.cos(math.pi * t))
    if schedule == "step":
        return float(step_gamma) if outer >= (total_outer // 2) else 1.0
    raise ValueError(f"unknown lr_schedule: {schedule}")


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def flush_outputs(
    out_dir: Path,
    delta_out: Dict[str, torch.Tensor],
    uvab_out: Dict[str, Dict[str, torch.Tensor]],
    wdq_out: Dict[str, torch.Tensor],
    delta_best_out: Dict[str, torch.Tensor],
    uvab_best_out: Dict[str, Dict[str, torch.Tensor]],
    wdq_best_out: Dict[str, torch.Tensor],
) -> None:
    torch.save(delta_out, out_dir / "delta.pt")
    torch.save(uvab_out, out_dir / "lowrank_uv_ab.pt")
    torch.save(wdq_out, out_dir / "wdq_star.pt")
    torch.save(delta_best_out, out_dir / "delta_best.pt")
    torch.save(uvab_best_out, out_dir / "lowrank_uv_ab_best.pt")
    torch.save(wdq_best_out, out_dir / "wdq_star_best.pt")


def parse_ppl_from_step4_output(text: str) -> Tuple[Optional[float], Optional[float]]:
    ppl_wdq = None
    ppl_ab = None
    for m in re.finditer(r"✅ PPL\(([^)]+)\) = ([0-9]+(?:\.[0-9]+)?)", text):
        label = m.group(1)
        val = float(m.group(2))
        if "+" in label or "AB" in label:
            ppl_ab = val
        else:
            ppl_wdq = val
    return ppl_wdq, ppl_ab


def load_context(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    codebook_path = Path(args.step1_dir) / "codebook.pt"
    qcodes_path = Path(args.step1_dir) / "qcodes.pt"
    meta_path = Path(args.step1_dir) / "meta.pt"
    if not (codebook_path.exists() and qcodes_path.exists() and meta_path.exists()):
        raise FileNotFoundError("step1_dir must contain codebook.pt, qcodes.pt, meta.pt")

    print(f"[V2] loading step1 artifacts: {args.step1_dir}")
    codebooks: Dict[str, torch.Tensor] = torch.load(codebook_path, map_location="cpu")
    qcodes_dict: Dict[str, torch.Tensor] = torch.load(qcodes_path, map_location="cpu")
    metas: Dict[str, dict] = torch.load(meta_path, map_location="cpu")

    print(f"[V2] loading calib_s: {args.calib_s}")
    calib_s: Dict[str, dict] = torch.load(args.calib_s, map_location="cpu")

    if args.dtype_w == "fp16":
        load_dtype = torch.float16
    elif args.dtype_w == "bf16":
        load_dtype = torch.bfloat16
    else:
        load_dtype = torch.float32

    print(f"[V2] loading original model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    )
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_re = re.compile(args.layer_regex) if args.layer_regex else None
    keys = []
    for k in codebooks.keys():
        if k not in qcodes_dict or k not in metas:
            continue
        if k not in calib_s:
            continue
        if k not in state:
            continue
        w_cpu = state[k]
        if not is_target_weight(k, w_cpu):
            continue
        if layer_re and not layer_re.search(k):
            continue
        keys.append(k)

    keys = sorted(keys, key=sort_key)
    if not keys:
        raise RuntimeError("No matched layers found.")

    block_order = []
    seen = set()
    for k in keys:
        bidx = extract_block_index(k)
        if bidx is None:
            continue
        if bidx not in seen:
            seen.add(bidx)
            block_order.append(bidx)

    if not block_order:
        raise RuntimeError("Failed to parse block index from matched keys. Check key naming format.")

    print(f"[V2] matched matrices: {len(keys)}")
    print(f"[V2] matched blocks: {len(block_order)} (first={block_order[0]}, last={block_order[-1]})")
    return {
        "device": device,
        "codebooks": codebooks,
        "qcodes": qcodes_dict,
        "metas": metas,
        "calib_s": calib_s,
        "state": state,
        "keys_all": keys,
        "block_order": block_order,
    }


def select_stage1_keys(keys_all: List[str], stage1_max_blocks: int) -> Tuple[List[str], List[int]]:
    if int(stage1_max_blocks) <= 0:
        raise ValueError("--stage1_max_blocks must be > 0")

    block_order = []
    seen = set()
    for k in keys_all:
        bidx = extract_block_index(k)
        if bidx is None:
            continue
        if bidx not in seen:
            seen.add(bidx)
            block_order.append(bidx)

    if int(stage1_max_blocks) > len(block_order):
        raise ValueError(
            f"--stage1_max_blocks={stage1_max_blocks} exceeds available blocks ({len(block_order)})"
        )

    selected_blocks = block_order[: int(stage1_max_blocks)]
    selected_set = set(selected_blocks)
    stage1_keys = [k for k in keys_all if extract_block_index(k) in selected_set]
    if not stage1_keys:
        raise RuntimeError("Stage1 key selection is empty.")
    return stage1_keys, selected_blocks


def suggest_params(trial: optuna.trial.Trial) -> dict:
    clip_choice = trial.suggest_categorical("clip_norm", [None, 0.5, 1.0, 2.0])
    return {
        "lr_delta": trial.suggest_float("lr_delta", 1e-4, 3e-3, log=True),
        "lam_delta": trial.suggest_float("lam_delta", 1e-6, 1e-2, log=True),
        "uv_iters": trial.suggest_categorical("uv_iters", [1, 2, 3]),
        "clip_norm": 0.0 if clip_choice is None else float(clip_choice),
        "normalize_mode": trial.suggest_categorical("normalize_mode", ["mean_abs", "rms", "median_abs"]),
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["none", "cosine", "step"]),
    }


def normalize_params(params: dict) -> dict:
    clip = params.get("clip_norm", 0.0)
    return {
        "lr_delta": float(params["lr_delta"]),
        "lam_delta": float(params["lam_delta"]),
        "uv_iters": int(params["uv_iters"]),
        "clip_norm": 0.0 if clip is None else float(clip),
        "normalize_mode": str(params["normalize_mode"]),
        "lr_schedule": str(params["lr_schedule"]),
    }


def optimize_over_keys(
    *,
    keys: List[str],
    trial_number: int,
    params: dict,
    ctx: dict,
    args: argparse.Namespace,
    out_dir: Path,
    trial_obj: Optional[optuna.trial.Trial],
    enable_prune: bool,
    save_artifacts: bool,
) -> float:
    device = ctx["device"]
    codebooks = ctx["codebooks"]
    qcodes_dict = ctx["qcodes"]
    metas = ctx["metas"]
    calib_s = ctx["calib_s"]
    state = ctx["state"]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = out_dir / "train_log.jsonl"
    log_f = open(train_log_path, "w", encoding="utf-8")

    delta_out: Dict[str, torch.Tensor] = {}
    uvab_out: Dict[str, Dict[str, torch.Tensor]] = {}
    wdq_out: Dict[str, torch.Tensor] = {}
    delta_best_out: Dict[str, torch.Tensor] = {}
    uvab_best_out: Dict[str, Dict[str, torch.Tensor]] = {}
    wdq_best_out: Dict[str, torch.Tensor] = {}

    best_by_layer: Dict[str, float] = {}
    report_step = 0

    try:
        for li, key in enumerate(keys):
            meta = metas[key]
            bits = int(meta["bits"])
            gs = int(meta["group_size"])
            q_levels = int(meta["levels"])
            orig_i = int(tuple(meta["orig_shape"])[1])

            w_cpu = state[key].to(torch.float32)
            c_cpu = codebooks[key].to(torch.float32)
            codes_cpu = qcodes_dict[key]
            s_cpu = calib_s[key]["s"].to(torch.float32)

            o, i = w_cpu.shape
            if i != orig_i:
                raise RuntimeError(f"shape mismatch on {key}: I={i}, orig_I={orig_i}")

            w = w_cpu.to(device)
            c = c_cpu.to(device)
            codes = codes_cpu.to(device)
            s = s_cpu.to(device)

            delta = torch.nn.Parameter(torch.zeros_like(c, dtype=torch.float32, device=device))
            u = torch.ones((o,), device=device, dtype=torch.float32)
            v = torch.ones((i,), device=device, dtype=torch.float32)
            opt = torch.optim.Adam([delta], lr=float(params["lr_delta"]))

            with torch.no_grad():
                wq0 = dequant_from_codebook_codes(c + delta, codes, orig_i=i)
                e0 = w - wq0
                rw0 = e0 * s.unsqueeze(0)
                a, b = rank_r_svd(rw0, r=int(args.rank_ab))

            wq = e = rw = None
            mse = float("inf")
            dnorm = float("inf")
            best_mse = float("inf")
            best_outer = -1
            best_delta = best_a = best_b = best_u = best_v = best_wq = None

            try:
                for outer in range(int(args.outer_loops)):
                    lr_scale = get_lr_scale(
                        outer=outer,
                        total_outer=int(args.outer_loops),
                        schedule=str(params["lr_schedule"]),
                        min_ratio=float(args.lr_min_ratio),
                        step_gamma=float(args.lr_step_gamma),
                    )
                    lr_cur = float(params["lr_delta"]) * lr_scale
                    for pg in opt.param_groups:
                        pg["lr"] = lr_cur

                    for _ in range(max(1, int(args.delta_steps))):
                        opt.zero_grad(set_to_none=True)
                        wq = dequant_from_codebook_codes(c + delta, codes, orig_i=i)
                        e = w - wq
                        rw = e * s.unsqueeze(0)

                        m = a @ b
                        pred = (u.unsqueeze(1) * m) * v.unsqueeze(0)
                        loss_rec = torch.mean((rw - pred) ** 2)
                        loss_reg = float(params["lam_delta"]) * torch.mean(delta ** 2)
                        loss = loss_rec + loss_reg
                        loss.backward()

                        if float(params["clip_norm"]) > 0.0:
                            torch.nn.utils.clip_grad_norm_([delta], max_norm=float(params["clip_norm"]))
                        opt.step()

                    with torch.no_grad():
                        wq = dequant_from_codebook_codes(c + delta, codes, orig_i=i)
                        e = w - wq
                        rw = e * s.unsqueeze(0)

                        eps = float(args.eps)
                        u_sign = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
                        v_sign = torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
                        u_inv = 1.0 / (u_sign * u.abs().clamp_min(eps))
                        v_inv = 1.0 / (v_sign * v.abs().clamp_min(eps))
                        rbar = (u_inv.unsqueeze(1) * rw) * v_inv.unsqueeze(0)
                        a, b = rank_r_svd(rbar, r=int(args.rank_ab))

                        u, v = update_uv_closed_form(
                            rw=rw,
                            a=a,
                            b=b,
                            u=u,
                            v=v,
                            uv_iters=int(params["uv_iters"]),
                            eps=float(args.eps),
                            normalize_mode=str(params["normalize_mode"]),
                        )

                        m = a @ b
                        pred = (u.unsqueeze(1) * m) * v.unsqueeze(0)
                        mse = torch.mean((rw - pred) ** 2).item()
                        dnorm = torch.mean(delta.detach() ** 2).item()

                        if mse < best_mse:
                            best_mse = float(mse)
                            best_outer = int(outer)
                            if save_artifacts:
                                best_delta = delta.detach().clone()
                                best_a = a.detach().clone()
                                best_b = b.detach().clone()
                                best_u = u.detach().clone()
                                best_v = v.detach().clone()
                                best_wq = wq.detach().clone()

                    if (outer % max(1, int(args.log_every))) == 0 or outer == int(args.outer_loops) - 1:
                        rec = {
                            "trial": int(trial_number),
                            "layer_idx": li,
                            "outer": outer,
                            "key": key,
                            "block_idx": extract_block_index(key),
                            "bits": bits,
                            "group_size": gs,
                            "Q": q_levels,
                            "rank": int(args.rank_ab),
                            "mse_weighted": float(mse),
                            "delta_m2": float(dnorm),
                            "lr_cur": float(lr_cur),
                            "params": {
                                "lr_delta": float(params["lr_delta"]),
                                "lam_delta": float(params["lam_delta"]),
                                "uv_iters": int(params["uv_iters"]),
                                "clip_norm": (
                                    None if float(params["clip_norm"]) <= 0.0 else float(params["clip_norm"])
                                ),
                                "normalize_mode": str(params["normalize_mode"]),
                                "lr_schedule": str(params["lr_schedule"]),
                            },
                        }
                        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        log_f.flush()

                    prev = best_by_layer.get(key, None)
                    if prev is None or best_mse < prev:
                        best_by_layer[key] = float(best_mse)

                    if enable_prune and trial_obj is not None and len(best_by_layer) >= int(args.prune_min_layers):
                        report_step += 1
                        running_obj = float(mean(best_by_layer.values()))
                        trial_obj.report(running_obj, step=report_step)
                        if report_step >= int(args.prune_warmup_steps) and trial_obj.should_prune():
                            raise optuna.TrialPruned(
                                f"pruned at step={report_step}, running_mean_best_mse={running_obj:.6e}"
                            )

                if save_artifacts:
                    with torch.no_grad():
                        wq_final = dequant_from_codebook_codes(c + delta, codes, orig_i=i)
                        rw_final = (w - wq_final) * s.unsqueeze(0)
                        m_final = a @ b
                        pred_final = (u.unsqueeze(1) * m_final) * v.unsqueeze(0)
                        mse_final = torch.mean((rw_final - pred_final) ** 2).item()

                        if best_delta is None:
                            best_mse = float(mse_final)
                            best_outer = -1
                            best_delta = delta.detach().clone()
                            best_a = a.detach().clone()
                            best_b = b.detach().clone()
                            best_u = u.detach().clone()
                            best_v = v.detach().clone()
                            best_wq = wq_final.detach().clone()

                    delta_out[key] = delta.detach().to(torch.float16).cpu()
                    uvab_out[key] = {
                        "A": a.detach().to(torch.float16).cpu(),
                        "B": b.detach().to(torch.float16).cpu(),
                        "u": u.detach().to(torch.float16).cpu(),
                        "v": v.detach().to(torch.float16).cpu(),
                        "meta": {
                            "rank": int(args.rank_ab),
                            "bits": bits,
                            "group_size": gs,
                            "Q": q_levels,
                            "mse_weighted_last": float(mse_final),
                        },
                    }
                    wdq_out[key] = wq_final.detach().to(torch.float16).cpu()

                    delta_best_out[key] = best_delta.detach().to(torch.float16).cpu()
                    uvab_best_out[key] = {
                        "A": best_a.detach().to(torch.float16).cpu(),
                        "B": best_b.detach().to(torch.float16).cpu(),
                        "u": best_u.detach().to(torch.float16).cpu(),
                        "v": best_v.detach().to(torch.float16).cpu(),
                        "meta": {
                            "rank": int(args.rank_ab),
                            "bits": bits,
                            "group_size": gs,
                            "Q": q_levels,
                            "best_outer": int(best_outer),
                            "mse_weighted_best": float(best_mse),
                        },
                    }
                    wdq_best_out[key] = best_wq.detach().to(torch.float16).cpu()

                    if args.save_every_layer:
                        flush_outputs(
                            out_dir=out_dir,
                            delta_out=delta_out,
                            uvab_out=uvab_out,
                            wdq_out=wdq_out,
                            delta_best_out=delta_best_out,
                            uvab_best_out=uvab_best_out,
                            wdq_best_out=wdq_best_out,
                        )
            finally:
                del w, c, codes, s, delta, a, b, u, v
                if wq is not None:
                    del wq
                if e is not None:
                    del e
                if rw is not None:
                    del rw
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        if save_artifacts:
            flush_outputs(
                out_dir=out_dir,
                delta_out=delta_out,
                uvab_out=uvab_out,
                wdq_out=wdq_out,
                delta_best_out=delta_best_out,
                uvab_best_out=uvab_best_out,
                wdq_best_out=wdq_best_out,
            )
    finally:
        log_f.close()

    if not best_by_layer:
        raise RuntimeError("No layer metrics collected")
    return float(mean(best_by_layer.values()))


def run_step4_eval(args: argparse.Namespace, stage2_dir: Path) -> dict:
    cmd = [
        sys.executable,
        args.step4_script,
        "--model_name",
        args.model_id,
        "--wdq_star_path",
        str(stage2_dir / "wdq_star_best.pt"),
        "--low_rank_ab_path",
        str(stage2_dir / "lowrank_uv_ab_best.pt"),
        "--calib_s_path",
        args.calib_s,
        "--device",
        args.eval_device,
        "--ppl_stride",
        str(args.ppl_stride),
        "--ppl_max_tokens",
        str(args.ppl_max_tokens),
        "--ab_compute",
        args.ab_compute,
        "--skip_gen",
    ]
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out_log = stage2_dir / "step4_eval.log"
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(text)
    if proc.returncode != 0:
        raise RuntimeError(f"step4_eval failed (rc={proc.returncode}). check: {out_log}")

    ppl_wdq, ppl_ab = parse_ppl_from_step4_output(text)
    return {
        "status": "complete",
        "ppl_wdq": ppl_wdq,
        "ppl_ab": ppl_ab,
        "eval_sec": elapsed,
        "log_path": str(out_log),
    }


def main() -> None:
    ap = argparse.ArgumentParser("Step3.5 V2 (block-wise BO -> fixed full optimization)")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--step1_dir", required=True)
    ap.add_argument("--calib_s", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--step4_script", default=os.path.join(os.path.dirname(__file__), "step4_eval.py"))

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval_device", default="cuda:0")
    ap.add_argument("--dtype_w", default="fp16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--rank_ab", type=int, default=64)
    ap.add_argument("--outer_loops", type=int, default=40)
    ap.add_argument("--delta_steps", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--layer_regex", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--save_every_layer", action="store_true")

    ap.add_argument("--lr_min_ratio", type=float, default=0.1)
    ap.add_argument("--lr_step_gamma", type=float, default=0.3)

    ap.add_argument("--study_name", default="step3_5_v2")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--stage1_max_blocks", type=int, required=True)
    ap.add_argument("--stage1_n_trials", type=int, default=20)
    ap.add_argument("--stage1_timeout_sec", type=int, default=0)
    ap.add_argument("--stage1_storage", default="", help="optuna storage URL, e.g. sqlite:///study.db")
    ap.add_argument("--stage1_save_artifacts", action="store_true")

    ap.add_argument("--prune_min_layers", type=int, default=2)
    ap.add_argument("--prune_warmup_steps", type=int, default=20)
    ap.add_argument("--pruner_startup_trials", type=int, default=5)
    ap.add_argument("--pruner_interval_steps", type=int, default=5)

    ap.add_argument("--run_step4_eval", action="store_true")
    ap.add_argument("--ab_compute", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--ppl_stride", type=int, default=2048)
    ap.add_argument("--ppl_max_tokens", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    out_root = Path(args.out_root).resolve()
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{args.study_name}_{ts}"
    stage1_dir = run_dir / "stage1_bo"
    stage2_dir = run_dir / "stage2_full"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir.mkdir(parents=True, exist_ok=True)

    print(f"[V2] run_dir: {run_dir}")
    ctx = load_context(args)

    keys_all = ctx["keys_all"]
    stage1_keys, stage1_blocks = select_stage1_keys(keys_all, int(args.stage1_max_blocks))
    print(f"[V2] stage1 blocks: {stage1_blocks}")
    print(f"[V2] stage1 matrices: {len(stage1_keys)}")
    print(f"[V2] stage2 matrices(full): {len(keys_all)}")

    trials_log = stage1_dir / "trials.jsonl"
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(args.pruner_startup_trials),
        n_warmup_steps=int(args.prune_warmup_steps),
        interval_steps=int(args.pruner_interval_steps),
    )
    study = optuna.create_study(
        study_name=f"{args.study_name}_stage1",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=(args.stage1_storage or None),
        load_if_exists=bool(args.stage1_storage),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        t0 = time.time()
        params = normalize_params(suggest_params(trial))
        trial.set_user_attr("resolved_params", params)
        trial_dir = stage1_dir / f"trial_{trial.number:04d}"
        append_jsonl(
            trials_log,
            {
                "trial": int(trial.number),
                "status": "running",
                "params": params,
                "trial_dir": str(trial_dir),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        try:
            obj = optimize_over_keys(
                keys=stage1_keys,
                trial_number=int(trial.number),
                params=params,
                ctx=ctx,
                args=args,
                out_dir=trial_dir,
                trial_obj=trial,
                enable_prune=True,
                save_artifacts=bool(args.stage1_save_artifacts),
            )
        except optuna.TrialPruned as e:
            append_jsonl(
                trials_log,
                {
                    "trial": int(trial.number),
                    "status": "pruned",
                    "params": params,
                    "trial_dir": str(trial_dir),
                    "reason": str(e),
                    "elapsed_sec": time.time() - t0,
                },
            )
            raise
        except Exception as e:
            append_jsonl(
                trials_log,
                {
                    "trial": int(trial.number),
                    "status": "failed",
                    "params": params,
                    "trial_dir": str(trial_dir),
                    "error": str(e),
                    "elapsed_sec": time.time() - t0,
                },
            )
            raise

        append_jsonl(
            trials_log,
            {
                "trial": int(trial.number),
                "status": "complete",
                "params": params,
                "trial_dir": str(trial_dir),
                "objective_mse_mean_best": float(obj),
                "elapsed_sec": time.time() - t0,
            },
        )
        return float(obj)

    study.optimize(
        objective,
        n_trials=int(args.stage1_n_trials),
        timeout=(None if int(args.stage1_timeout_sec) <= 0 else int(args.stage1_timeout_sec)),
        gc_after_trial=True,
    )

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    completed = sorted(completed, key=lambda x: float(x.value))
    if not completed:
        raise RuntimeError("Stage1 finished but no completed trial found.")

    best_trial = completed[0]
    best_params = best_trial.user_attrs.get("resolved_params")
    if best_params is None:
        best_params = normalize_params(best_trial.params)
    else:
        best_params = normalize_params(best_params)

    summary_stage1 = {
        "run_dir": str(run_dir),
        "stage1_dir": str(stage1_dir),
        "study_name": f"{args.study_name}_stage1",
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(completed),
        "stage1_max_blocks": int(args.stage1_max_blocks),
        "stage1_block_indices": stage1_blocks,
        "stage1_matrices": len(stage1_keys),
        "best_trial": int(best_trial.number),
        "best_objective_mse_mean_best": float(best_trial.value),
        "best_params_raw": dict(best_trial.params),
        "best_params": best_params,
    }
    with open(stage1_dir / "summary_stage1.json", "w", encoding="utf-8") as f:
        json.dump(summary_stage1, f, ensure_ascii=False, indent=2)

    print("[V2] stage1 done")
    print(f"  best trial: {best_trial.number}")
    print(f"  best params: {best_params}")

    t_stage2 = time.time()
    stage2_obj = optimize_over_keys(
        keys=keys_all,
        trial_number=-1,
        params=best_params,
        ctx=ctx,
        args=args,
        out_dir=stage2_dir,
        trial_obj=None,
        enable_prune=False,
        save_artifacts=True,
    )

    wdq_best_path = stage2_dir / "wdq_star_best.pt"
    ab_best_path = stage2_dir / "lowrank_uv_ab_best.pt"
    if not (wdq_best_path.exists() and ab_best_path.exists()):
        raise RuntimeError("Stage2 outputs missing. Expected wdq_star_best.pt and lowrank_uv_ab_best.pt")

    summary_stage2 = {
        "run_dir": str(run_dir),
        "stage2_dir": str(stage2_dir),
        "elapsed_sec": time.time() - t_stage2,
        "objective_mse_mean_best": float(stage2_obj),
        "used_params": best_params,
        "num_matrices_full": len(keys_all),
        "wdq_star_best_path": str(wdq_best_path),
        "lowrank_uv_ab_best_path": str(ab_best_path),
        "calib_s_path": args.calib_s,
    }
    with open(stage2_dir / "summary_stage2.json", "w", encoding="utf-8") as f:
        json.dump(summary_stage2, f, ensure_ascii=False, indent=2)

    if args.run_step4_eval:
        eval_rec = run_step4_eval(args, stage2_dir)
        with open(stage2_dir / "step4_eval_summary.json", "w", encoding="utf-8") as f:
            json.dump(eval_rec, f, ensure_ascii=False, indent=2)
        print(f"[V2] step4_eval done: ppl_wdq={eval_rec.get('ppl_wdq')}, ppl_ab={eval_rec.get('ppl_ab')}")

    final_summary = {
        "run_dir": str(run_dir),
        "stage1_summary": str(stage1_dir / "summary_stage1.json"),
        "stage2_summary": str(stage2_dir / "summary_stage2.json"),
        "wdq_star_best_path": str(wdq_best_path),
        "lowrank_uv_ab_best_path": str(ab_best_path),
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print("✅ Step3.5 V2 finished")
    print(f"  run_dir: {run_dir}")
    print(f"  stage1 summary: {stage1_dir / 'summary_stage1.json'}")
    print(f"  stage2 summary: {stage2_dir / 'summary_stage2.json'}")
    print(f"  wdq*: {wdq_best_path}")
    print(f"  uv-ab*: {ab_best_path}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Compatibility wrapper API for LABA/mixture/step0_optimization.py
# (No embedded source / exec; directly invokes local `main()`.)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Sequence


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
    source_script: str = str(Path(__file__).resolve())


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
        p
        for p in out_root.iterdir()
        if p.is_dir() and p.name.startswith(f"{study_name}_") and (p / "summary.json").exists()
    ]
    if not cands:
        raise FileNotFoundError(
            f"No completed step3 run dir found under {out_root} for study_name={study_name}"
        )
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

    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)

    run_dir = _find_run_dir(out_root, cfg.study_name)
    return resolve_outputs(run_dir)
