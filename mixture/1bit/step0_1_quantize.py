#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mixture step0 module: Step0-1 quantization (1-bit group-wise mu/beta baseline).

역할:
  - 타깃 weight를 1-bit group-wise `(mu_{o,g}, beta_{o,g})` 코드북/코드 형태로 양자화
  - step3 최적화 입력용 `codebook.pt`, `qcodes.pt`, `meta.pt` 생성
  - 필요 시 `quantized_weights.pt`, `quant_error.pt` 추가 저장

사용 방식:
  - CLI 실행: 이 파일의 `main()`
  - 모듈 사용: 하단 `Step01QuantizeConfig` + `run()`

참고:
  - `LABA/mixture/step0_optimization.py`와의 호환을 위해 래퍼 API를 함께 제공한다.
  - 현재 이 디렉터리는 1-bit 전용이다.
"""

import os
import re
import gc
import csv
import argparse
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("transformers가 필요합니다: pip install transformers") from e


# -------------------------------
# Target filter (기존과 동일한 스타일)
# -------------------------------
TARGET_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj",
    "fc1", "fc2",
}

def is_target_weight(name: str, tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 2
        and name.endswith(".weight")
        and ("layers" in name or "encoder.layers" in name or "model.layers" in name)
        and name.split(".")[-2] in TARGET_SUFFIXES
    )

def module_name_from_weight(full_weight_name: str) -> str:
    return full_weight_name[: -len(".weight")]


def _snapshot_state_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        # With accelerate offload, meta tensors may appear in state_dict.
        if getattr(v, "is_meta", False):
            raise NotImplementedError(f"meta tensor in state_dict: {k}")
        state[k] = v.detach().to("cpu")
    return state


# -------------------------------
# Bit assignment loader (기존과 동일)
# -------------------------------
def load_selected_bits(csv_path: str) -> Dict[str, int]:
    sel: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            name = (row.get("layer_name") or row.get("module") or row.get("name") or "").strip()
            if not name:
                continue
            b = None
            for key in ("R_int", "selected_bit", "bit"):
                if key in row and str(row[key]).strip() != "":
                    try:
                        b = int(float(row[key]))
                        break
                    except Exception:
                        pass
            if b is None:
                continue
            sel[name] = 1
    return sel


# -------------------------------
# Group reshape helpers
# -------------------------------
def _to_groups(W: torch.Tensor, group_size: int):
    """
    W: [O,I] -> Wg: [O,G,S], padding on I so that I_pad % S == 0
    returns (Wg, O, G, S, orig_I, pad)
    """
    O, I = W.shape
    pad = (group_size - (I % group_size)) % group_size
    if pad:
        W = torch.nn.functional.pad(W, (0, pad))
    O_, I_pad = W.shape
    G = I_pad // group_size
    return W.view(O_, G, group_size), O_, G, group_size, I, pad

def _from_groups(Xg: torch.Tensor, orig_I: int) -> torch.Tensor:
    O_, G, S = Xg.shape
    return Xg.reshape(O_, G * S)[:, :orig_I]


@torch.no_grad()
def _percentile_clip_lastdim(Wg: torch.Tensor, upper_pct: float, lower_pct: float) -> torch.Tensor:
    """
    clip per-group over flattened (O*gs) for each group G
    Wg: [O,G,S]
    """
    assert Wg.ndim == 3
    O, G, gs = Wg.shape
    flat = Wg.permute(1, 0, 2).reshape(G, -1)  # [G, O*gs]
    n = flat.shape[1]
    lo_k = max(1, int((lower_pct / 100.0) * n))
    hi_k = max(1, int((upper_pct / 100.0) * n))
    lo = torch.kthvalue(flat, lo_k, dim=1).values.view(1, G, 1)
    hi = torch.kthvalue(flat, hi_k, dim=1).values.view(1, G, 1)
    return Wg.clamp(min=lo, max=hi)


# -------------------------------
# 1-bit group-wise (mu,beta) quantization
# -------------------------------
@torch.no_grad()
def quantize_1bit_groupwise_mu_beta(
    W: torch.Tensor,                  # [O,I] float32 on device
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    1-bit group-wise quantization with:
      mu_{o,g}   = mean_s W_{o,g,s}
      beta_{o,g} = mean_s |W_{o,g,s} - mu_{o,g}|
      q_{o,g,s}  = 1[W_{o,g,s} >= mu_{o,g}]
      C_{o,g,:}  = [mu_{o,g} - beta_{o,g}, mu_{o,g} + beta_{o,g}]

    Returns:
      - Wq      : [O,I] float32
      - codebook: [O,G,2] float32
      - qcodes  : [O,G,S] uint8
      - meta    : dict (O,G,S,L,orig_I,pad, ...)
    """
    Wg, O, G, S, orig_I, pad = _to_groups(W, group_size)   # [O,G,S]
    mu = Wg.mean(dim=-1, keepdim=True)                     # [O,G,1]
    centered = Wg - mu
    beta = centered.abs().mean(dim=-1, keepdim=True)      # [O,G,1]

    qcodes = (Wg >= mu).to(torch.uint8)                   # [O,G,S]
    sign = qcodes.to(torch.float32).mul_(2.0).sub_(1.0)   # {-1,+1}
    Wqg = mu + beta * sign
    Wq = _from_groups(Wqg, orig_I).contiguous()

    codebook = torch.cat([mu - beta, mu + beta], dim=-1).contiguous()  # [O,G,2]
    meta = {
        "bits": 1,
        "group_size": int(group_size),
        "levels": 2,
        "orig_shape": (int(O), int(orig_I)),
        "O": int(O),
        "G": int(G),
        "S": int(S),
        "pad": int(pad),
        "quant_scheme": "groupwise_mu_beta_1bit",
        "center_note": "mu_{o,g} = mean_s W_{o,g,s}",
        "scale_note": "beta_{o,g} = mean_s |W_{o,g,s} - mu_{o,g}|",
        "codebook_note": "C_{o,g,:} = [mu_{o,g}-beta_{o,g}, mu_{o,g}+beta_{o,g}]",
        "qcode_note": "q_{o,g,s} = 1[W_{o,g,s} >= mu_{o,g}]",
    }
    return Wq, codebook, qcodes, meta


@torch.no_grad()
def lloyd_asym_nonuniform_quantize(
    W: torch.Tensor,
    b: int,
    group_size: int,
    clip_pct: float = 0.0,
    lloyd_iter: int = 12,
    chunk_groups: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Compatibility wrapper.
    The 1bit directory now uses a fixed group-wise `(mu,beta)` quantizer.
    """
    del clip_pct, lloyd_iter, chunk_groups
    if int(b) != 1:
        raise ValueError("LABA/mixture/1bit/step0_1_quantize.py now supports only bits=1.")
    return quantize_1bit_groupwise_mu_beta(W, group_size=group_size)


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser("Step1 (1-bit) - group-wise mu/beta extractor")
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device_map", default="auto", help='Model load placement: e.g. "auto" or "none"')

    ap.add_argument("--bit_assign_csv", default=None)
    ap.add_argument("--bits", type=int, default=1, choices=[1],
                    help="fixed to 1-bit in this directory")
    ap.add_argument("--group_size", type=int, default=128)

    ap.add_argument("--clip_percentile", type=float, default=0.0,
                    help="unused compatibility flag for the fixed 1-bit quantizer")
    ap.add_argument("--lloyd_iter", type=int, default=12,
                    help="unused compatibility flag for the fixed 1-bit quantizer")
    ap.add_argument("--chunk_groups", type=int, default=4096,
                    help="unused compatibility flag for the fixed 1-bit quantizer")

    ap.add_argument("--layer_regex", type=str, default=None)

    ap.add_argument("--save_wq", action="store_true", help="also save Wq as quantized_weights.pt")
    ap.add_argument("--save_err", action="store_true", help="also save error (W - Wq) as quant_error.pt")

    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dtype == "bf16":
        load_dtype = torch.bfloat16
    elif args.dtype in ("fp16", "float16"):
        load_dtype = torch.float16
    elif args.dtype in ("fp32", "float32"):
        load_dtype = torch.float32
    else:
        load_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                      else torch.float16)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    dm_raw = str(args.device_map).strip().lower()
    resolved_device_map = None if dm_raw in {"", "none", "null"} else args.device_map

    print(
        f"[Step1-1bit] Loading model: {args.model_id} "
        f"(load_dtype={load_dtype}, device={device}, device_map={resolved_device_map})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=args.trust_remote_code,
        device_map=resolved_device_map,
        low_cpu_mem_usage=True,
    )
    if resolved_device_map is None:
        model = model.to(device)

    if args.bit_assign_csv:
        sel_bits = load_selected_bits(args.bit_assign_csv)
        print(f"[Step1-1bit] Loaded selection CSV: {len(sel_bits)} entries (bit values ignored; only layer selection is used).")
    else:
        sel_bits = None

    # CPU state dict로 옮겨서 GPU 메모리 절약
    try:
        state = _snapshot_state_to_cpu(model)
        del model
    except NotImplementedError:
        if resolved_device_map is None:
            raise
        print("[Step1-1bit] Detected meta tensors under device_map mode. Re-loading on CPU to build state snapshot.")
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

    codebooks: Dict[str, torch.Tensor] = {}
    qcodes_dict: Dict[str, torch.Tensor] = {}
    metas: Dict[str, dict] = {}
    qweights: Dict[str, torch.Tensor] = {}
    err_dict: Dict[str, torch.Tensor] = {}

    if float(args.clip_percentile) != 0.0:
        print("[Step1-1bit] clip_percentile is ignored for the fixed group-wise (mu,beta) quantizer.")
    if int(args.lloyd_iter) != 12 or int(args.chunk_groups) != 4096:
        print("[Step1-1bit] lloyd_iter/chunk_groups are ignored for the fixed group-wise (mu,beta) quantizer.")

    print("[Step1-1bit] Extracting (codebook, qcodes) ...")

    for full_name, W_cpu in tqdm(state.items()):
        if not is_target_weight(full_name, W_cpu):
            continue
        if layer_re and not layer_re.search(full_name):
            continue

        # bit 결정
        if sel_bits is not None:
            mod_name = module_name_from_weight(full_name)
            selected = sel_bits.get(mod_name, sel_bits.get(full_name.replace(".weight", ""), None))
            if selected is None:
                continue

        # quantize
        W = W_cpu.to(device=device, dtype=torch.float32)
        Wq, codebook, qcodes, meta = lloyd_asym_nonuniform_quantize(
            W,
            b=1,
            group_size=int(args.group_size),
            clip_pct=float(args.clip_percentile),
            lloyd_iter=int(args.lloyd_iter),
            chunk_groups=int(args.chunk_groups),
        )

        # save cpu
        codebooks[full_name] = codebook.detach().to(torch.float16).cpu()   # [O,G,L]
        qcodes_dict[full_name] = qcodes.detach().cpu()                     # uint8 [O,G,S]
        metas[full_name] = meta

        if args.save_wq:
            qweights[full_name] = Wq.detach().to(torch.float16).cpu()
        if args.save_err:
            # err는 원래 shape [O,I] 기준으로 저장
            Wq_cpu = Wq.detach().to(torch.float32).cpu()
            err_dict[full_name] = W_cpu.to(torch.float32) - Wq_cpu

        del W, Wq, codebook, qcodes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    cb_path = os.path.join(args.out_dir, "codebook.pt")
    qc_path = os.path.join(args.out_dir, "qcodes.pt")
    mt_path = os.path.join(args.out_dir, "meta.pt")
    torch.save(codebooks, cb_path)
    torch.save(qcodes_dict, qc_path)
    torch.save(metas, mt_path)

    if args.save_wq:
        q_path = os.path.join(args.out_dir, "quantized_weights.pt")
        torch.save(qweights, q_path)
    if args.save_err:
        e_path = os.path.join(args.out_dir, "quant_error.pt")
        torch.save(err_dict, e_path)

    print("[Step1-1bit] Saved:")
    print(f"  • {cb_path}  ({len(codebooks)} layers)")
    print(f"  • {qc_path}  ({len(qcodes_dict)} layers)")
    print(f"  • {mt_path}  ({len(metas)} layers)")
    if args.save_wq:
        print(f"  • {q_path}  ({len(qweights)} layers)")
    if args.save_err:
        print(f"  • {e_path}  ({len(err_dict)} layers)")


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
from typing import List, Optional, Sequence


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
class Step01QuantizeConfig:
    model_id: str
    out_dir: str
    bits: int
    group_size: int
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    device: str = "cuda"
    device_map: str = "auto"
    bit_assign_csv: Optional[str] = None
    clip_percentile: float = 0.0
    lloyd_iter: int = 12
    chunk_groups: int = 4096
    layer_regex: Optional[str] = None
    save_wq: bool = False
    save_err: bool = False
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


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
    if int(cfg.bits) != 1:
        raise ValueError("Step01QuantizeConfig.bits must be 1 for LABA/mixture/1bit.")
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_id",
        str(cfg.model_id),
        "--bits",
        str(int(cfg.bits)),
        "--group_size",
        str(int(cfg.group_size)),
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
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp
