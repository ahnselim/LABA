#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 — Inject Wq + Layerwise A/B, then evaluate PPL (robust matching)

입력:
 • quantized_weights.pt         (dict["<module>.weight"] = Wq, fp16/cpu)
 • correction_layerwise.pt      (dict["<module>.A"], dict["<module>.B"] OR "...weight.A/B")
 • b_ref_map_layerwise.json     (선택: dict["<module>.weight"] = "..." ; 값은 무시하고 키만 필터로 사용)

동작:
 1) HF 모델 로드(fp16) → 각 Linear의 .weight에 Wq copy_
 2) correction 파일에서 (A,B) 쌍을 직접 수집 → 모델에 존재하는 Linear만 AddABCorrection으로 래핑
    - b_ref_map이 주어지면 그 키 집합으로 대상 모듈을 필터링 가능
    - 이미 AddABCorrection인 경우 inner를 풀고 새 (A,B)로 재래핑(이중 래핑 방지)
 3) WikiText-2(raw) test로 PPL 2종 측정: Quant-only(α=0), Quant+AB(α=--alpha_svd)

사용 예:
CUDA_VISIBLE_DEVICES=0 \
python step5_inject_and_eval.py \
  --model_name meta-llama/Llama-3.2-3B \
  --quantized_weights_path ../artifacts/montecarlo/step4_budget/quantized_weights.pt \
  --correction_path ../artifacts/montecarlo/step4_budget/correction_layerwise.pt \
  --bmap_path ../artifacts/montecarlo/step4_budget/b_ref_map_layerwise.json \
  --alpha_svd 1.0 \
  --device cuda:0 \
  --trust_remote_code \
  --eval_seq_len 2048
"""

import os
import json
import math
import argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("pip install transformers datasets accelerate") from e


# -----------------------------
# Utils
# -----------------------------
def get_parent_module(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    """'model.layers.0.self_attn.q_proj' → (parent_module, 'q_proj')"""
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def infer_device_dtype(module: nn.Module):
    for t in list(module.parameters()) + list(module.buffers()):
        return t.device, t.dtype
    return (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
        torch.float16,
    )


# -----------------------------
# AB 래퍼
# -----------------------------
class AddABCorrection(nn.Module):
    """z = inner(x) + alpha * A ( B x )
    - A: [out, r], B: [r, in]
    """

    def __init__(
        self,
        inner: nn.Module,
        A_q: torch.Tensor,
        B_q: torch.Tensor,
        alpha_svd: float = 1.0,
    ):
        super().__init__()
        self.inner = inner
        self.alpha_svd = alpha_svd
        dev, dt = infer_device_dtype(inner)
        self.register_buffer(
            "A_q", A_q.to(device=dev, dtype=dt, copy=True), persistent=False
        )
        self.register_buffer(
            "B_q", B_q.to(device=dev, dtype=dt, copy=True), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inner(x)
        if self.alpha_svd == 0.0:
            return z
        A = self.A_q if self.A_q.dtype == x.dtype else self.A_q.to(dtype=x.dtype)
        B = self.B_q if self.B_q.dtype == x.dtype else self.B_q.to(dtype=x.dtype)
        rfeat = F.linear(x, B)  # [*, r]
        corr = F.linear(rfeat, A)  # [*, out]
        return z.add_(corr, alpha=self.alpha_svd)


# -----------------------------
# Quant weight 주입
# -----------------------------
@torch.no_grad()
def apply_quantized_weights(model: nn.Module, qweights: Dict[str, torch.Tensor]):
    injected, missing, mismatch = 0, 0, 0
    for wkey, Wq in qweights.items():
        if not (
            isinstance(wkey, str)
            and wkey.endswith(".weight")
            and getattr(Wq, "ndim", 0) == 2
        ):
            continue

        module_name = wkey[:-7]  # strip ".weight"
        try:
            parent, attr = get_parent_module(model, module_name)
        except AttributeError:
            missing += 1
            continue

        mod = getattr(parent, attr, None)
        # 이미 래핑되어 있으면 inner의 weight에 주입
        if isinstance(mod, AddABCorrection):
            inner = mod.inner
        else:
            inner = mod

        if inner is None or not hasattr(inner, "weight"):
            missing += 1
            continue
        if inner.weight.data.shape != Wq.shape:
            mismatch += 1
            continue

        inner.weight.data.copy_(
            Wq.to(device=inner.weight.device, dtype=inner.weight.dtype)
        )
        injected += 1

    print(f"[Wq] injected={injected}, missing={missing}, shape_mismatch={mismatch}")


# -----------------------------
# AB 키 정규화 & 페어 수집
# -----------------------------
def _normalize_key(k: str) -> str:
    # '...weight.A' → '...A', '...weight.B' → '...B' 로 통일
    if k.endswith(".weight.A"):
        return k[:-8] + ".A"
    if k.endswith(".weight.B"):
        return k[:-8] + ".B"
    return k


def collect_ab_pairs(
    correction_tensors: Dict[str, torch.Tensor],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """correction_tensors 안의 키들을 스캔하여
    base = '<module full name>' (예: 'model.layers.0.self_attn.q_proj') 에 대해
    (A,B) 쌍을 구성해 반환.
    """
    norm2orig = {}
    for k in correction_tensors.keys():
        norm2orig[_normalize_key(k)] = k

    # base 후보
    bases = set()
    for nk in norm2orig.keys():
        if nk.endswith(".A") or nk.endswith(".B"):
            bases.add(nk[:-2])

    pairs = {}
    for base in bases:
        ak = base + ".A"
        bk = base + ".B"
        ak_orig = norm2orig.get(ak)
        bk_orig = norm2orig.get(bk)
        if ak_orig in correction_tensors and bk_orig in correction_tensors:
            pairs[base] = (correction_tensors[ak_orig], correction_tensors[bk_orig])
    return pairs


# -----------------------------
# AB 패치 (bmap은 필터로만 사용)
# -----------------------------
def patch_ab_from_corrections(
    model: nn.Module,
    correction_tensors: Dict[str, torch.Tensor],
    bmap_keys_weight: set,
    alpha_svd: float = 1.0,
) -> int:
    """correction에서 수집한 (A,B) 쌍을 모델에 적용.
    bmap_keys_weight가 비어있지 않으면, 해당 키('...weight')의 모듈만 대상으로 제한.
    반환: 패치된 모듈 수
    """
    pairs = collect_ab_pairs(correction_tensors)
    if not pairs:
        print("[AB] no (A,B) pairs found in correction file.")
        return 0

    # bmap 필터링: '...weight' → '...' 로 변환한 set
    target_modules = set(pairs.keys())
    if bmap_keys_weight:
        filter_modules = set(
            [k[:-7] for k in bmap_keys_weight if k.endswith(".weight")]
        )
        target_modules = target_modules & filter_modules
        print(
            f"[AB] filtering by bmap: {len(filter_modules)} keys → {len(target_modules)} targets"
        )

    patched, missing_module = 0, 0
    examples = []
    for base in sorted(target_modules):
        try:
            parent, attr = get_parent_module(model, base)
        except AttributeError:
            missing_module += 1
            continue

        inner = getattr(parent, attr, None)
        if inner is None:
            missing_module += 1
            continue

        # 이미 AddABCorrection이면 inner를 풀고 교체(이중 래핑 방지)
        if isinstance(inner, AddABCorrection):
            inner = inner.inner

        if not isinstance(inner, nn.Linear):
            missing_module += 1
            continue

        A, B = pairs[base]
        setattr(parent, attr, AddABCorrection(inner, A, B, alpha_svd=alpha_svd))
        patched += 1
        if len(examples) < 3:
            examples.append((base, tuple(A.shape), tuple(B.shape)))

    print(f"[AB] patched={patched}, missing_module={missing_module}")
    if examples:
        print("[AB] examples:", examples)
    return patched


# -----------------------------
# PPL 평가
# -----------------------------
@torch.no_grad()
def evaluate_ppl(
    model: nn.Module,
    tokenizer,
    device,
    tag="Eval",
    eval_seq_len=2048,
    dataset="wikitext",
    config="wikitext-2-raw-v1",
):
    print(f"\n--- PPL Eval: {tag} on {dataset}/{config} ---")
    model.eval()
    ds = load_dataset(dataset, config, split="test")
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)

    total_nll, total_tok = 0.0, 0
    for i in tqdm(range(0, input_ids.size(1), eval_seq_len), desc="Eval"):
        begin, end = i, min(i + eval_seq_len, input_ids.size(1))
        if end - begin <= 1:
            continue
        x = input_ids[:, begin:end]
        y = x
        out = model(x)
        logits = out.logits  # [B, T, V]
        loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            y[..., 1:].contiguous().view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tok += end - begin - 1

    ppl = math.exp(total_nll / max(1, total_tok))
    print(f"PPL({tag}) = {ppl:.4f}")
    return ppl


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        "Step 5 — Inject Wq + Layerwise AB and evaluate PPL (robust)"
    )
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--quantized_weights_path", required=True)
    ap.add_argument("--correction_path", required=True)
    ap.add_argument(
        "--bmap_path", default=None, help="optional; used only as a filter by keys"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--alpha_svd", type=float, default=1.0)
    ap.add_argument(
        "--tf32", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True
    )
    ap.add_argument("--eval_seq_len", type=int, default=2048)
    ap.add_argument("--eval_dataset", type=str, default="wikitext")
    ap.add_argument("--eval_config", type=str, default="wikitext-2-raw-v1")
    args = ap.parse_args()

    if args.tf32:
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)

    # 1) 모델 로드
    print(f"[Load] model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10**9  # 경고 억제(길이 제한)

    # 2) 양자화 가중치 주입
    print(f"[Load] Wq: {args.quantized_weights_path}")
    try:
        qweights = torch.load(args.quantized_weights_path, map_location="cpu")
    except TypeError:
        qweights = torch.load(args.quantized_weights_path, map_location="cpu")
    apply_quantized_weights(model, qweights)

    # 3) AB 로드 및 패치
    print(f"[Load] corrections: {args.correction_path}")
    try:
        correction = torch.load(args.correction_path, map_location="cpu")
    except TypeError:
        correction = torch.load(args.correction_path, map_location="cpu")

    bmap_keys = set()
    if args.bmap_path and os.path.exists(args.bmap_path):
        with open(args.bmap_path, "r") as f:
            bmap = json.load(f)
        # 값은 무시, 키만 필터로 사용
        bmap_keys = set(bmap.keys())
        print(f"[bmap] keys loaded: {len(bmap_keys)}")

    patched = patch_ab_from_corrections(
        model, correction, bmap_keys, alpha_svd=args.alpha_svd
    )
    if patched == 0 and len(correction) > 0:
        print("[AB][warn] patched=0. Retrying WITHOUT bmap filtering...")
        patched = patch_ab_from_corrections(
            model, correction, set(), alpha_svd=args.alpha_svd
        )

    # 4) 평가
    # 4-1) Quant-only
    for m in model.modules():
        if isinstance(m, AddABCorrection):
            m.alpha_svd = 0.0
    ppl_q = evaluate_ppl(
        model,
        tokenizer,
        device,
        tag="Quant-only (α=0)",
        eval_seq_len=args.eval_seq_len,
        dataset=args.eval_dataset,
        config=args.eval_config,
    )

    # 4-2) Quant+AB
    for m in model.modules():
        if isinstance(m, AddABCorrection):
            m.alpha_svd = args.alpha_svd
    ppl_ab = evaluate_ppl(
        model,
        tokenizer,
        device,
        tag=f"Quant+AB (α={args.alpha_svd})",
        eval_seq_len=args.eval_seq_len,
        dataset=args.eval_dataset,
        config=args.eval_config,
    )

    print("\n===== SUMMARY =====")
    print(f"Quant-only PPL : {ppl_q:.4f}")
    print(f"Quant+AB  PPL  : {ppl_ab:.4f}")


if __name__ == "__main__":
    main()
