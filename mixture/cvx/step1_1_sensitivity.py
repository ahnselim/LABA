#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 1 — Layerwise Sensitivity C_j Estimation (no grouping, SlimPajama-friendly)

변경 요약:
 • 기본 데이터셋을 DKYoon/SlimPajama-6B 로 설정(인터넷 필요).
 • step3와 유사한 견고한 로더: open_hf_dataset()로 streaming 우선 + 자동 config fallbacks.
 • 필요한 총 토큰 수만큼만 1D 버퍼를 구축(build_tokens_from_hfds)하여 메모리 안전.
 • 기존 --text_file 로컬 텍스트 파일 경로를 제공하면 HF 없이도 동작.

예시:
 CUDA_VISIBLE_DEVICES=1 \
 python step1_sensitivity.py \
  --model_id meta-llama/Llama-3.2-3B \
  --dataset DKYoon/SlimPajama-6B \
  --seq_len 1024 --batch_size 2 --num_batches 50 \
  --dtype bf16 --device_map auto \
  --output_dir ../artifacts/bitmin/step1

옵션:
 • --dataset cerebras/SlimPajama-627B 로도 사용 가능
 • --dataset_config 필요 시 지정
 • --use_streaming false 로 비스트리밍 강제
"""

import os, json, math, argparse, random
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import torch
import torch.nn as nn

# Skip flash-attn autoload by default, its binary commonly mismatches local CUDA.
if "FLASH_ATTENTION_FORCE_DISABLE" not in os.environ:
    os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import set_seed as hf_set_seed
    from transformers.utils import import_utils as _tf_import_utils
except Exception as e:
    raise RuntimeError(
        "transformers 가 필요합니다: pip install transformers datasets accelerate"
    ) from e


def _disable_flash_attn_package():
    sentinel = "_flash_attn_guard_installed"
    if getattr(_tf_import_utils, sentinel, False):
        return
    orig = _tf_import_utils._is_package_available

    def _patched(package: str):
        if package == "flash_attn":
            return False
        return orig(package)

    _tf_import_utils._is_package_available = _patched
    setattr(_tf_import_utils, sentinel, True)


if os.environ.get("FLASH_ATTENTION_FORCE_DISABLE", "1") == "1":
    _disable_flash_attn_package()

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

# -----------------------
# Utils
# -----------------------


def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def pick_dtype(dtype_str: str):
    if dtype_str == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def _to_gib_str(v: float) -> str:
    fv = float(v)
    if fv <= 0:
        raise ValueError(f"Memory cap must be > 0 GiB, got {v}")
    if float(fv).is_integer():
        return f"{int(fv)}GiB"
    return f"{fv:.2f}GiB"


def build_max_memory_map(
    gpu_mem_cap_gib: Optional[float], cpu_mem_cap_gib: Optional[float]
) -> Optional[Dict[object, str]]:
    if gpu_mem_cap_gib is None:
        return None
    if not torch.cuda.is_available():
        return None
    max_memory: Dict[object, str] = {
        i: _to_gib_str(gpu_mem_cap_gib) for i in range(torch.cuda.device_count())
    }
    if cpu_mem_cap_gib is not None:
        max_memory["cpu"] = _to_gib_str(cpu_mem_cap_gib)
    return max_memory


def yield_windows(
    input_ids: torch.Tensor,
    seq_len: int,
    stride: int,
    batch_size: int,
    max_batches: Optional[int] = None,
) -> Iterable[torch.Tensor]:
    """1D token stream → (B,L) 윈도우 배치들."""
    assert input_ids.ndim == 1
    T = input_ids.numel()
    ptr = 0
    batches = 0
    buf = []
    while ptr + seq_len <= T:
        buf.append(input_ids[ptr : ptr + seq_len])
        ptr += stride
        if len(buf) == batch_size:
            yield torch.stack(buf, dim=0)  # (B, L)
            buf = []
            batches += 1
            if (max_batches is not None) and (batches >= max_batches):
                return
    if buf:
        yield torch.stack(buf, dim=0)


def tokens_needed_for_windows(
    seq_len: int, stride: int, batch_size: int, num_batches: int
) -> int:
    """필요한 총 토큰 수 계산: 첫 윈도우 L + 이후 (N*B-1)개 윈도우마다 stride씩 추가."""
    n_windows = batch_size * num_batches
    if n_windows <= 0:
        return 0
    return seq_len + max(0, n_windows - 1) * stride


# -----------------------
# HF dataset helpers (SlimPajama-friendly)
# -----------------------
import re as _re


def _canonical_dataset_name(name: str) -> str:
    """널리 쓰는 별칭 → 정식 경로 보정."""
    a = name.strip()
    low = a.lower()
    # DKYoon 6B
    if low in {"dkyoon/slimpajama-6b", "slimpajama-6b", "dkyoon_slimpajama_6b"}:
        return "DKYoon/SlimPajama-6B"
    # Cerebras 627B
    if low in {"slimpajama-627b", "cerebras/slimpajama-627b", "slimpajama627b"}:
        return "cerebras/SlimPajama-627B"
    return name


def open_hf_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str = "train",
    streaming: bool = True,
):
    """streaming 우선, config 자동 탐색 fallback."""
    if not HAS_DATASETS:
        raise RuntimeError("datasets 라이브러리가 필요합니다: pip install datasets")
    dataset_name = _canonical_dataset_name(dataset_name)

    # streaming 시도
    if streaming:
        try:
            ds = load_dataset(
                dataset_name, name=dataset_config, split=split, streaming=True
            )
            return ds, dataset_name, dataset_config, True
        except Exception as e:
            msg = str(e)
            # config 자동 탐색
            if (
                ("available configs" in msg)
                or ("Available configs" in msg)
                or ("Config name is missing" in msg)
            ):
                m = _re.search(r"\[(.*?)\]", msg, flags=_re.S)
                if m:
                    raw = m.group(1)
                    cands = [
                        c.strip().strip("'\"") for c in raw.split(",") if c.strip()
                    ]
                    for cand in cands:
                        try:
                            ds = load_dataset(
                                dataset_name, name=cand, split=split, streaming=True
                            )
                            return ds, dataset_name, cand, True
                        except Exception:
                            pass
        # fallthrough to non-streaming

    # non-streaming 시도
    try:
        ds = load_dataset(
            dataset_name, name=dataset_config, split=split, streaming=False
        )
        return ds, dataset_name, dataset_config, False
    except Exception as e:
        msg = str(e)
        if (
            ("available configs" in msg)
            or ("Available configs" in msg)
            or ("Config name is missing" in msg)
        ):
            m = _re.search(r"\[(.*?)\]", msg, flags=_re.S)
            if m:
                raw = m.group(1)
                cands = [c.strip().strip("'\"") for c in raw.split(",") if c.strip()]
                for cand in cands:
                    try:
                        ds = load_dataset(
                            dataset_name, name=cand, split=split, streaming=False
                        )
                        return ds, dataset_name, cand, False
                    except Exception:
                        pass
    raise


def build_tokens_from_hfds(
    tokenizer,
    dataset: str,
    dataset_config: Optional[str],
    split: str,
    need_tokens: int,
    use_streaming: bool = True,
    text_keys: Tuple[str, ...] = ("text", "content", "raw_content"),
    add_eos: bool = True,
) -> torch.Tensor:
    """필요한 길이의 1D 토큰 스트림 구축(메모리 안전, streaming 우선)."""
    ds, dataset, dataset_config, is_streaming = open_hf_dataset(
        dataset, dataset_config, split, streaming=use_streaming
    )
    print(
        f"[Step1] Using dataset={dataset}, config={dataset_config}, streaming={is_streaming}"
    )
    buf: List[int] = []
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    if eos_id is None and getattr(tokenizer, "eos_token", None):
        eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    # streaming과 비-스트리밍 모두 iterator처럼 다룸
    if hasattr(ds, "take"):  # streaming Dataset
        it = ds  # 무한대는 아니므로 충분히 순회
    else:  # Arrow Dataset
        it = (ds[i] for i in range(len(ds)))

    for row in it:
        txt = None
        # 기본 text key 탐색
        for k in text_keys:
            if k in row and isinstance(row[k], str) and row[k].strip():
                txt = row[k]
                break
        # fallback: 첫 번째 str 필드
        if txt is None:
            for v in row.values():
                if isinstance(v, str) and v.strip():
                    txt = v
                    break
        if not txt:
            continue

        ids = tokenizer(txt, return_tensors=None, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        buf.extend(ids)
        if add_eos and (eos_id is not None):
            buf.append(int(eos_id))
        if len(buf) >= need_tokens:
            break

    if len(buf) < need_tokens:
        print(
            f"[Step1][warn] Collected {len(buf)}/{need_tokens} tokens. Windows may be fewer than requested."
        )
    return torch.tensor(buf[:need_tokens], dtype=torch.long)


def load_text_tokens_from_file(
    text_file: str, tokenizer, need_tokens: Optional[int] = None, add_eos: bool = True
) -> torch.Tensor:
    with open(text_file, "r", encoding="utf-8") as f:
        txt = f.read()
    ids = tokenizer(txt, return_tensors=None, add_special_tokens=False)["input_ids"]
    if add_eos and tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if need_tokens is not None and len(ids) < need_tokens:
        print(f"[Step1][warn] file tokens {len(ids)} < need {need_tokens}.")
    return torch.tensor(
        ids[:need_tokens] if need_tokens is not None else ids, dtype=torch.long
    )


def iter_linear_weight_modules(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    out = []
    for name, mod in model.named_modules():
        if (
            isinstance(mod, nn.Linear)
            and hasattr(mod, "weight")
            and mod.weight is not None
        ):
            out.append((name, mod))
    return out


def is_lm_head_name(name: str) -> bool:
    lname = name.lower()
    return ("lm_head" in lname) or lname.endswith("lm_head")


# -----------------------
# Core
# -----------------------


def estimate_layerwise_sensitivity(
    model: nn.Module,
    input_ids_1d: torch.Tensor,
    seq_len: int,
    stride: int,
    batch_size: int,
    num_batches: int,
    dtype: torch.dtype,
    include_lm_head: bool = False,
    grad_scale: float = 1.0,
) -> Dict[str, dict]:
    """C_j ≈ mean_batch ||grad(W_j)||_F^2 (경험적 Fisher/헤시안 대각 근사)"""
    model.eval()
    torch.set_grad_enabled(True)

    use_autocast = (
        dtype in (torch.bfloat16, torch.float16)
    ) and torch.cuda.is_available()

    sens_sum: Dict[str, float] = {}
    numel_map: Dict[str, int] = {}
    tracked: List[Tuple[str, nn.Linear]] = []

    for name, mod in iter_linear_weight_modules(model):
        if (not include_lm_head) and is_lm_head_name(name):
            continue
        tracked.append((name, mod))
        sens_sum[name] = 0.0
        numel_map[name] = mod.weight.numel()
        mod.weight.requires_grad_(True)

    batches = 0
    for batch in yield_windows(
        input_ids=input_ids_1d,
        seq_len=seq_len,
        stride=stride,
        batch_size=batch_size,
        max_batches=num_batches,
    ):
        device = next(model.parameters()).device
        batch = batch.to(device)
        labels = batch.clone()
        model.zero_grad(set_to_none=True)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(input_ids=batch, labels=labels)
                loss = out.loss
        else:
            out = model(input_ids=batch, labels=labels)
            loss = out.loss

        (loss * grad_scale).backward()

        for name, mod in tracked:
            g = mod.weight.grad
            if g is None:
                continue
            sens_sum[name] += float(g.pow(2).sum().item())

        batches += 1

    if batches == 0:
        raise RuntimeError(
            "수집된 배치가 없습니다. (dataset/token budget/stride/seq_len 설정 확인)"
        )

    results = {}
    for name in sens_sum:
        C_sum = sens_sum[name]
        C_mean_per_batch = C_sum / batches
        results[name] = {
            "C_sum": C_sum,
            "C_mean_per_batch": C_mean_per_batch,
            "C_per_param": C_mean_per_batch / max(1, numel_map[name]),
            "numel": int(numel_map[name]),
            "batches": int(batches),
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 1 — Layerwise Sensitivity C_j Estimation (SlimPajama-friendly)"
    )

    # 모델
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--gpu_mem_cap_gib", type=float, default=None)
    parser.add_argument("--cpu_mem_cap_gib", type=float, default=None)
    parser.add_argument("--offload_folder", type=str, default=None)

    # 데이터 소스 (기본: DKYoon/SlimPajama-6B)
    parser.add_argument(
        "--dataset",
        type=str,
        default="DKYoon/SlimPajama-6B",
        help="예: DKYoon/SlimPajama-6B 또는 cerebras/SlimPajama-627B 등",
    )
    parser.add_argument(
        "--dataset_config", type=str, default=None, help="필요시 HF config name"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--use_streaming",
        type=lambda x: str(x).lower() in ["1", "true", "yes"],
        default=True,
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="로컬 텍스트(.txt) 경로 (지정 시 HF 무시)",
    )

    # 배치/시퀀스
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument(
        "--stride", type=int, default=None, help="기본 seq_len (non-overlap)"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=50)

    # 측정 옵션
    parser.add_argument("--include_lm_head", action="store_true")
    parser.add_argument("--grad_scale", type=float, default=1.0)

    # 출력
    parser.add_argument("--output_dir", type=str, default="./artifacts/bitmin/step1")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_all_seeds(args.seed)
    dtype = pick_dtype(args.dtype)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 토크나이저 & 모델
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        revision=args.revision,
        torch_dtype=dtype if (dtype in (torch.float16, torch.bfloat16)) else None,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gpu_mem_cap_gib is not None:
        if str(args.device_map).lower() != "auto":
            print(
                f"[Step1][warn] gpu_mem_cap_gib requires device_map=auto; current={args.device_map}. Ignoring cap."
            )
        else:
            mm = build_max_memory_map(args.gpu_mem_cap_gib, args.cpu_mem_cap_gib)
            if mm is not None:
                offload_dir = (
                    str(Path(args.offload_folder).resolve())
                    if args.offload_folder
                    else str((Path(args.output_dir) / "_hf_offload_step1_1").resolve())
                )
                Path(offload_dir).mkdir(parents=True, exist_ok=True)
                model_kwargs["max_memory"] = mm
                model_kwargs["offload_folder"] = offload_dir
                model_kwargs["offload_state_dict"] = True
                print(
                    f"[Step1] Applying max_memory={mm} with offload_folder={offload_dir}"
                )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # 토큰 예산 계산
    if args.stride is None:
        args.stride = args.seq_len
    need_tokens = tokens_needed_for_windows(
        args.seq_len, args.stride, args.batch_size, args.num_batches
    )
    if need_tokens <= 0:
        raise ValueError(
            "need_tokens<=0. seq_len/stride/batch_size/num_batches를 확인하세요."
        )
    print(
        f"[Step1] Token budget (needed): {need_tokens} (seq_len={args.seq_len}, stride={args.stride}, "
        f"batch_size={args.batch_size}, num_batches={args.num_batches})"
    )

    # 1D 토큰 스트림 구축
    if args.text_file is not None:
        input_ids_1d = load_text_tokens_from_file(
            args.text_file, tokenizer, need_tokens=need_tokens, add_eos=True
        )
        print(
            f"[Step1] Loaded local text file: {args.text_file} → tokens={input_ids_1d.numel()}"
        )
    else:
        input_ids_1d = build_tokens_from_hfds(
            tokenizer=tokenizer,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            split=args.split,
            need_tokens=need_tokens,
            use_streaming=bool(args.use_streaming),
        )
        print(
            f"[Step1] Built tokens from HF dataset: {args.dataset} (config={args.dataset_config}) "
            f"→ tokens={input_ids_1d.numel()}"
        )

    # 민감도 추정
    results = estimate_layerwise_sensitivity(
        model=model,
        input_ids_1d=input_ids_1d,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        dtype=dtype,
        include_lm_head=args.include_lm_head,
        grad_scale=args.grad_scale,
    )

    # 저장
    import csv

    csv_path = os.path.join(args.output_dir, "layerwise_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer_name",
                "numel(w_j)",
                "C_sum",
                "C_mean_per_batch",
                "C_per_param",
                "batches",
            ]
        )
        for name, rec in results.items():
            writer.writerow(
                [
                    name,
                    rec["numel"],
                    f"{rec['C_sum']:.6e}",
                    f"{rec['C_mean_per_batch']:.6e}",
                    f"{rec['C_per_param']:.6e}",
                    rec["batches"],
                ]
            )
    print(f"[Step1] Saved CSV: {csv_path}")

    if args.save_json:
        json_path = os.path.join(args.output_dir, "layerwise_sensitivity.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Step1] Saved JSON: {json_path}")

    # 요약 출력(상위 10개)
    topk = sorted(
        results.items(), key=lambda kv: kv[1]["C_mean_per_batch"], reverse=True
    )[:10]
    print("\n[Step1] Top-10 layers by C_mean_per_batch:")
    for name, rec in topk:
        print(f" {name:60s} C_mean={rec['C_mean_per_batch']:.3e} numel={rec['numel']}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Compatibility wrapper API for LABA/mixture/step1_cvx_optimization.py
# (No embedded source / exec; directly invokes local `main()`.)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
import subprocess
import sys
from typing import Sequence


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
class Step11SensitivityConfig:
    model_id: str
    output_dir: str
    revision: Optional[str] = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    device_map: str = "auto"
    dataset: str = "DKYoon/SlimPajama-6B"
    dataset_config: Optional[str] = None
    split: str = "train"
    use_streaming: bool = True
    text_file: Optional[str] = None
    seq_len: int = 1024
    stride: Optional[int] = None
    batch_size: int = 2
    num_batches: int = 50
    include_lm_head: bool = False
    grad_scale: float = 1.0
    save_json: bool = False
    seed: int = 42
    gpu_mem_cap_gib: Optional[float] = None
    cpu_mem_cap_gib: Optional[float] = None
    offload_folder: Optional[str] = None
    python_exe: str = sys.executable
    source_script: str = str(Path(__file__).resolve())


def build_command(cfg: Step11SensitivityConfig) -> List[str]:
    cmd: List[str] = [
        str(cfg.python_exe),
        str(cfg.source_script),
        "--model_id",
        str(cfg.model_id),
        "--output_dir",
        str(cfg.output_dir),
        "--dtype",
        str(cfg.dtype),
        "--device_map",
        str(cfg.device_map),
        "--dataset",
        str(cfg.dataset),
        "--split",
        str(cfg.split),
        "--use_streaming",
        "true" if cfg.use_streaming else "false",
        "--seq_len",
        str(int(cfg.seq_len)),
        "--batch_size",
        str(int(cfg.batch_size)),
        "--num_batches",
        str(int(cfg.num_batches)),
        "--grad_scale",
        str(float(cfg.grad_scale)),
        "--seed",
        str(int(cfg.seed)),
    ]
    if cfg.revision:
        cmd += ["--revision", str(cfg.revision)]
    if cfg.trust_remote_code:
        cmd.append("--trust_remote_code")
    if cfg.dataset_config is not None:
        cmd += ["--dataset_config", str(cfg.dataset_config)]
    if cfg.text_file is not None:
        cmd += ["--text_file", str(cfg.text_file)]
    if cfg.stride is not None:
        cmd += ["--stride", str(int(cfg.stride))]
    if cfg.include_lm_head:
        cmd.append("--include_lm_head")
    if cfg.save_json:
        cmd.append("--save_json")
    if cfg.gpu_mem_cap_gib is not None:
        cmd += ["--gpu_mem_cap_gib", str(float(cfg.gpu_mem_cap_gib))]
    if cfg.cpu_mem_cap_gib is not None:
        cmd += ["--cpu_mem_cap_gib", str(float(cfg.cpu_mem_cap_gib))]
    if cfg.offload_folder is not None:
        cmd += ["--offload_folder", str(cfg.offload_folder)]
    return cmd


def run(cfg: Step11SensitivityConfig, check: bool = True) -> subprocess.CompletedProcess:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    cp = _invoke_local_main(build_command(cfg)[2:])
    if check and cp.returncode != 0:
        raise subprocess.CalledProcessError(cp.returncode, cp.args)
    return cp
