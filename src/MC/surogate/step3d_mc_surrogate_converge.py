#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3b — Monte-Carlo Beam Search (Surrogate-guided) + True-PPL evaluate only top-k
===============================================================================

요구사항 반영:
- neighbor 생성(generate_random_neighbor)과 budget projection(project_to_weighted_budget)은
  "validate/기존 step3b" 그대로 유지.
- Filter는 proxy L(b) 대신, 학습된 surrogate가 예측한 ppl(score; 낮을수록 좋음)로 수행.
- 매 generation마다:
  Expand -> surrogate로 all_candidates 스코어링 -> top filter_p 선정 ->
  그 중 top beam_size(=top10)만 true PPL 측정 -> beam 갱신 -> 수렴까지 반복.

주의:
- surrogate 출력은 "낮을수록 좋다"로 가정. (validate에서 사용한 방식과 동일)
- surrogate가 proxy 입력을 쓰는 경우, proxy_loss(L(b))를 proxy feature로 넣어줌.

Usage:
CUDA_VISIBLE_DEVICES=0 nohup \
python montecarlo/step3d_mc_surrogate_converge.py \
  --model_id meta-llama/Llama-3.2-3B \
  --gpu_id 0 \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --prebake_root ../artifacts/montecarlo/prebake \
  --init_assign_csv ../artifacts/montecarlo/step3b_budget/bit_assign.csv \
  --avg_bits 2.50 \
  --beam_size 10 \
  --expansion_k 20 \
  --filter_p 80 \
  --surrogate_ckpt ../artifacts/surrogate_checkpoint_brp/best.pt \
  --sur_batch 512 \
  --eval_seq_len 2048 \
  --converge_eps 1e-3 --converge_rel_eps 1e-3 \
  --patience 12 --stable_bits_patience 12 \
  --ppl_cache_max 20000 \
  --output_dir ../artifacts/bitmin/step3b_surrogate_mc_converge > surr_ppl.log 2>&1 &
"""

import os, gc, csv, json, math, random, argparse, re, time, sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict
from functools import reduce
import builtins

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

print = lambda *a, **k: builtins.print(*a, flush=True, **k)

# -------------------------------------------------------------------
# Repo path injection (원본 step3b와 동일한 방식 유지)
# -------------------------------------------------------------------
_FILE_PATH = Path(__file__).resolve()
_PARENTS = _FILE_PATH.parents
_SRC_ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _FILE_PATH.parent
_REPO_ROOT = _PARENTS[2] if len(_PARENTS) > 2 else _SRC_ROOT
_WORKSPACE_ROOT = _PARENTS[3] if len(_PARENTS) > 3 else _REPO_ROOT
for _path in (_SRC_ROOT, _REPO_ROOT, _WORKSPACE_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

# -------------------------------------------------------------------
# Step3 bit optimization utils (원본 step3b의 import 유지)
# -------------------------------------------------------------------
try:
    from RAQ.proxy_codes.step3_bit_optimization import (
        load_sensitivity_csv,
        load_alpha_csv,
        solve_mu_for_budget,
        greedy_integer_refine_budget,
    )
except ImportError:
    print("오류: RAQ.proxy_codes.step3_bit_optimization import 실패")
    raise


# =========================================================
# Helpers (원본 step3b에서 가져온 것 + 그대로 유지)
# =========================================================
def _safe_name(s) -> str:
    if s is None:
        s = "none"
    s = str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


@torch.no_grad()
def run_live_ppl_eval(model, eval_input_ids, seq_len=2048) -> float:
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, eval_input_ids.size(1), seq_len):
        begin, end = i, min(i + seq_len, eval_input_ids.size(1))
        if end - begin <= 1:
            continue
        x = eval_input_ids[:, begin:end]
        y = x
        out = model(x)
        logits = out.logits
        loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            y[..., 1:].contiguous().view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tok += end - begin - 1
    if total_tok == 0:
        return 0.0
    return math.exp(total_nll / total_tok)


def load_seed_from_csv(csv_path: str) -> Dict[str, int]:
    seed_map = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("layer_name")
                r_int = row.get("R_int")
                if name and r_int:
                    try:
                        seed_map[name] = int(float(r_int))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[경고] 시드 CSV 로드 실패: {e}")
    return seed_map


def build_c_prime_map(sens_csv, alpha_csv, alpha_bit=3, alpha_default=1.0):
    names, C, w = load_sensitivity_csv(sens_csv, "C_mean_per_batch", "numel(w_j)")
    amap = load_alpha_csv(alpha_csv, target_bit=alpha_bit) if alpha_csv else {}
    C_prime_map, W_map = {}, {}
    for i, name in enumerate(names):
        alpha = float(amap.get(name, alpha_default))
        C_prime_map[name] = max(0.0, C[i] * alpha)
        W_map[name] = int(w[i])
    return C_prime_map, W_map


def calculate_proxy_loss(bit_assignment, C_prime_map, bmin=2):
    total_loss = 0.0
    for name, cp in C_prime_map.items():
        bit = bit_assignment.get(name, bmin)
        total_loss += cp * (2.0 ** (-2.0 * bit))
    return float(total_loss)


def weighted_sum_bits(b_assign: Dict[str, int], W_map: Dict[str, int]) -> int:
    s = 0
    for n, b in b_assign.items():
        if n in W_map:
            s += int(W_map[n]) * int(b)
    return int(s)


def gcd_list(int_list: List[int]) -> int:
    return (
        reduce(math.gcd, int_list)
        if len(int_list) > 1
        else (int_list[0] if int_list else 1)
    )


def target_weighted_sum(avg_bits: float, W_map: Dict[str, int]) -> int:
    sum_w = sum(int(w) for w in W_map.values())
    g = gcd_list([int(w) for w in W_map.values()]) if W_map else 1
    raw = avg_bits * sum_w
    snapped = int(round(raw / g) * g)
    return snapped


def ensure_complete_assignment(
    b_assign: Dict[str, int], layer_names: List[str], bmin: int
) -> Dict[str, int]:
    out = dict(b_assign)
    for n in layer_names:
        if n not in out:
            out[n] = bmin
    return out


def project_to_weighted_budget(
    b_assign: Dict[str, int],
    W_map: Dict[str, int],
    C_prime_map: Dict[str, float],
    B_target: int,
    bmin: int,
    bmax: int,
    max_steps=None,
) -> Dict[str, int]:
    """
    validate_surrogate_top10_before_step3b.py와 동일한 greedy projection.

    - S_cur > B_target: b > bmin인 항목들 중 (증가손실/절감비용)이 가장 작은 것부터 내림
    - S_cur < B_target: b < bmax인 항목들 중 (감소이득/증가비용)이 가장 큰 것부터 올림
    """
    b = dict(b_assign)
    names = [n for n in b.keys() if n in W_map]
    if not names:
        return b

    S_cur = sum(int(W_map[n]) * int(b[n]) for n in names)

    if S_cur > B_target:
        while S_cur > B_target + 1e-9:
            candidates = [n for n in names if b[n] > bmin]
            if not candidates:
                break
            best_node = None
            min_score = float("inf")
            for n in candidates:
                w_val = float(W_map[n])
                cp_val = float(C_prime_map.get(n, 0.0))
                delta_harm = cp_val * (
                    (2.0 ** (-2.0 * (b[n] - 1))) - (2.0 ** (-2.0 * b[n]))
                )
                score = delta_harm / (w_val + 1e-30)
                if score < min_score:
                    min_score = score
                    best_node = n
            if best_node is None:
                break
            b[best_node] -= 1
            S_cur -= int(W_map[best_node])

    elif S_cur < B_target - 1e-9:
        while S_cur < B_target - 1e-9:
            candidates = [n for n in names if b[n] < bmax]
            if not candidates:
                break
            best_node = None
            max_score = -float("inf")
            for n in candidates:
                w_val = float(W_map[n])
                cp_val = float(C_prime_map.get(n, 0.0))
                delta_gain = cp_val * (
                    (2.0 ** (-2.0 * b[n])) - (2.0 ** (-2.0 * (b[n] + 1)))
                )
                score = delta_gain / (w_val + 1e-30)
                if score > max_score:
                    max_score = score
                    best_node = n
            if best_node is None:
                break
            b[best_node] += 1
            S_cur += int(W_map[best_node])

    return b


def get_initial_seed(C_prime_map, W_map, avg_bits, bmin=2, bmax=4) -> Dict[str, int]:
    names = [n for n in C_prime_map.keys() if n in W_map]
    Cp_arr = np.array([C_prime_map[n] for n in names], dtype=np.float64)
    w_arr = np.array([W_map[n] for n in names], dtype=np.float64)
    if w_arr.sum() == 0:
        print("[경고] 초기 시드 생성 실패: 유효한 가중치 맵이 없습니다.")
        return {}
    B_target = target_weighted_sum(float(avg_bits), {n: int(W_map[n]) for n in names})
    mu, R_cont, R_clamped, S_cont, L_cont, info = solve_mu_for_budget(
        Cp_arr, w_arr, float(B_target), bmin, bmax
    )
    b_init = np.clip(np.floor(R_clamped + 1e-12), bmin, bmax).astype(np.int64)
    b_int, L_int, S_int = greedy_integer_refine_budget(
        Cp_arr, w_arr, b_init, float(B_target), bmin, bmax
    )
    b_seed = {names[i]: int(b_int[i]) for i in range(len(names))}
    return project_to_weighted_budget(
        b_seed,
        {n: int(W_map[n]) for n in names},
        C_prime_map,
        int(B_target),
        bmin,
        bmax,
    )


def generate_random_neighbor(
    b_assign: Dict[str, int], layer_names: List[str], bmin=2, bmax=4
) -> Dict[str, int]:
    new_b = b_assign.copy()
    c_up = [n for n in layer_names if new_b.get(n, bmin) < bmax]
    c_down = [n for n in layer_names if new_b.get(n, bmin) > bmin]
    if not c_up or not c_down:
        return None
    j = random.choice(c_up)
    k = random.choice(c_down)
    if j == k:
        return None
    new_b[j] = new_b.get(j, bmin) + 1
    new_b[k] = new_b.get(k, bmin) - 1
    return new_b


# =========================================================
# PPL Evaluator (원본 step3b 유지)
# =========================================================
class PplEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        original_state_dict,
        eval_input_ids,
        prebake_root,
        eval_seq_len,
        cache_maxsize: int = 5000,
    ):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.original_state_dict = original_state_dict
        self.eval_input_ids = eval_input_ids.to(self.device)
        self.prebake_root = Path(prebake_root)
        self.eval_seq_len = int(eval_seq_len)
        self.cache_maxsize = int(cache_maxsize)
        self.ppl_cache = OrderedDict()

        self.cache_hits = 0
        self.cache_misses = 0

        self.target_layers = []
        bit2_dir = self.prebake_root / "bit2"
        if not bit2_dir.exists():
            raise FileNotFoundError(f"Pre-bake 디렉토리를 찾을 수 없습니다: {bit2_dir}")
        print(f"[PplEvaluator] {bit2_dir} 스캔하여 대상 레이어 찾는 중...")
        for f in bit2_dir.glob("*.pt"):
            try:
                payload = torch.load(f, map_location="cpu")
                module_name = payload.get("module")
                if module_name and f"{module_name}.weight" in original_state_dict:
                    self.target_layers.append(module_name)
                del payload
            except Exception as e:
                print(f"[경고] {f} 로드 실패: {e}")
        self.target_layers = sorted(list(set(self.target_layers)))
        if not self.target_layers:
            raise ValueError(
                f"Pre-bake 디렉토리({bit2_dir})에서 유효한 레이어를 찾지 못했습니다."
            )
        print(f"[PplEvaluator] 탐색 대상 레이어 {len(self.target_layers)}개 확인.")

    def _get_module(self, name):
        mod = self.model
        try:
            for p in name.split("."):
                mod = getattr(mod, p)
            return mod
        except AttributeError:
            return None

    @torch.no_grad()
    def evaluate(self, bit_assignment: Dict[str, int]) -> float:
        b_tuple = tuple(sorted(bit_assignment.items()))
        if b_tuple in self.ppl_cache:
            self.cache_hits += 1
            val = self.ppl_cache.pop(b_tuple)
            self.ppl_cache[b_tuple] = val
            return float(val)

        self.cache_misses += 1

        try:
            for layer_name in self.target_layers:
                bit = bit_assignment.get(layer_name)
                module = self._get_module(layer_name)
                if bit is None or module is None:
                    continue
                safe_name = _safe_name(layer_name)
                file_path = self.prebake_root / f"bit{bit}" / f"{safe_name}.pt"
                if not file_path.exists():
                    continue
                payload = torch.load(file_path, map_location=self.device)
                Wq, A, B = payload["Wq"], payload["A"], payload["B"]
                compute_dtype = Wq.dtype
                W_eff = Wq + (A.to(compute_dtype) @ B.to(compute_dtype))
                module.weight.data.copy_(W_eff.to(module.weight.dtype))
                del payload, Wq, A, B, W_eff
        except Exception as e:
            print(f"!! PPL 평가 중 오류: {e}")
            self.restore_original_weights()
            gc.collect()
            torch.cuda.empty_cache()
            return float("inf")

        ppl = run_live_ppl_eval(self.model, self.eval_input_ids, self.eval_seq_len)
        self.restore_original_weights()

        self.ppl_cache[b_tuple] = float(ppl)
        if len(self.ppl_cache) > self.cache_maxsize:
            self.ppl_cache.popitem(last=False)
        gc.collect()
        torch.cuda.empty_cache()
        return float(ppl)

    @torch.no_grad()
    def restore_original_weights(self):
        for layer_name in self.target_layers:
            module = self._get_module(layer_name)
            if module is None:
                continue
            w_name = f"{layer_name}.weight"
            orig_w = self.original_state_dict[w_name]
            module.weight.data.copy_(
                orig_w.to(device=module.weight.device, dtype=module.weight.dtype)
            )


# =========================================================
# Surrogate model (validate에서 사용한 형태)
# =========================================================
class BitSequenceEncoder(nn.Module):
    def __init__(
        self,
        L: int,
        d_model=128,
        bit_emb_dim=32,
        nhead=4,
        nlayers=2,
        ff_dim=256,
        dropout=0.1,
        use_proxy=True,
    ):
        super().__init__()
        self.L = int(L)
        self.use_proxy = bool(use_proxy)

        self.bit_emb = nn.Embedding(3, int(bit_emb_dim))  # bits {2,3,4} -> idx {0,1,2}
        self.num_proj = nn.Linear(2, int(bit_emb_dim))
        self.in_proj = nn.Linear(int(bit_emb_dim) * 2, int(d_model))
        self.pos_emb = nn.Embedding(self.L, int(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(nlayers))
        self.dropout = nn.Dropout(float(dropout))

        head_in = int(d_model) + (1 if self.use_proxy else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), 1),
        )

    def forward(
        self,
        bits: torch.Tensor,
        C_log: torch.Tensor,
        W_log: torch.Tensor,
        proxy: torch.Tensor = None,
    ) -> torch.Tensor:
        B, L = bits.shape
        assert L == self.L

        bit_idx = torch.clamp(bits - 2, 0, 2)
        e_bit = self.bit_emb(bit_idx)

        C = C_log.view(1, L, 1).expand(B, L, 1)
        W = W_log.view(1, L, 1).expand(B, L, 1)
        x_num = torch.cat([C, W], dim=-1)
        e_num = self.num_proj(x_num)

        x = torch.cat([e_bit, e_num], dim=-1)
        x = self.in_proj(x)

        pos = torch.arange(L, device=bits.device).view(1, L)
        x = x + self.pos_emb(pos)

        x = self.encoder(self.dropout(x))
        x_pool = x.mean(dim=1)

        if self.use_proxy:
            if proxy is None:
                proxy = torch.zeros((B, 1), device=bits.device, dtype=x_pool.dtype)
            h = torch.cat([x_pool, proxy], dim=-1)
        else:
            h = x_pool

        return self.head(h)


class BRPPairwiseSurrogate(nn.Module):
    def __init__(self, encoder: BitSequenceEncoder, tau_pair: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.tau_pair = float(tau_pair)

    @torch.no_grad()
    def score_single(self, bits, C_log, W_log, proxy=None):
        # "낮을수록 좋다" score (학습/검증에서 통일)
        return self.encoder(bits, C_log=C_log, W_log=W_log, proxy=proxy)


def load_surrogate(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    hp = cfg.get("hparams", {})

    layer_names = cfg.get("layer_names", None)
    if layer_names is None:
        raise RuntimeError("Checkpoint missing config.layer_names")

    L = int(cfg.get("L", len(layer_names)))
    norm = cfg.get("norm", {})
    C_mu, C_sd = float(norm["C_log_mu"]), float(norm["C_log_sd"])
    W_mu, W_sd = float(norm["W_log_mu"]), float(norm["W_log_sd"])

    d_model = int(hp.get("d_model", 128))
    bit_emb_dim = int(hp.get("bit_emb_dim", 32))
    nhead = int(hp.get("nhead", 4))
    nlayers = int(hp.get("nlayers", 2))
    ff_dim = int(hp.get("ff_dim", 256))
    dropout = float(hp.get("dropout", 0.1))
    use_proxy = not bool(hp.get("no_proxy", False))
    tau_pair = float(hp.get("tau_pair", 1.0))

    enc = BitSequenceEncoder(
        L=L,
        d_model=d_model,
        bit_emb_dim=bit_emb_dim,
        nhead=nhead,
        nlayers=nlayers,
        ff_dim=ff_dim,
        dropout=dropout,
        use_proxy=use_proxy,
    )
    model = BRPPairwiseSurrogate(enc, tau_pair=tau_pair)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()

    return model, layer_names, (C_mu, C_sd, W_mu, W_sd), use_proxy


def bitdict_to_bitvec(
    b_dict: Dict[str, int], layer_order: List[str], bmin: int
) -> np.ndarray:
    out = np.empty((len(layer_order),), dtype=np.int64)
    for i, ln in enumerate(layer_order):
        out[i] = int(b_dict.get(ln, bmin))
    return out


@torch.no_grad()
def surrogate_score_candidates(
    surrogate: BRPPairwiseSurrogate,
    candidates: List[Dict[str, int]],
    proxy_losses: List[float],
    layer_order: List[str],
    C_log_t: torch.Tensor,
    W_log_t: torch.Tensor,
    use_proxy: bool,
    bmin: int,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    bits = np.stack(
        [bitdict_to_bitvec(b, layer_order, bmin=bmin) for b in candidates], axis=0
    )
    scores = np.zeros((bits.shape[0],), dtype=np.float64)

    for i in range(0, bits.shape[0], batch_size):
        j = min(bits.shape[0], i + batch_size)
        b = torch.tensor(bits[i:j], dtype=torch.long, device=device)
        if use_proxy:
            p = torch.tensor(
                np.array(proxy_losses[i:j], dtype=np.float32).reshape(-1, 1),
                device=device,
            )
        else:
            p = None
        s = surrogate.score_single(b, C_log=C_log_t, W_log=W_log_t, proxy=p).squeeze(-1)
        scores[i:j] = s.detach().cpu().numpy().astype(np.float64)
    return scores


# =========================================================
# Main
# =========================================================
def main(args):
    if args.beam_size <= 0:
        raise ValueError("beam_size must be > 0")
    if args.filter_p < args.beam_size:
        raise ValueError("filter_p should be >= beam_size (top-k true ppl eval size)")

    # seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Phase 1: Load model/tokenizer/dataset
    print("--- 🔬 Phase 1: 기반 데이터 준비 ---")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    original_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    print("PPL 평가용 데이터셋 로드 (wikitext-2-raw-v1)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    eval_input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids

    print("프록시(C', W) 로드...")
    C_prime_map, W_map_all = build_c_prime_map(
        args.sens_csv, args.alpha_csv, args.alpha_bit, args.alpha_default
    )

    # Phase 2: PPL evaluator
    print("--- 🌱 Phase 2: 평가기 초기화 ---")
    ppl_eval = PplEvaluator(
        model,
        tokenizer,
        original_state_dict,
        eval_input_ids,
        args.prebake_root,
        args.eval_seq_len,
        cache_maxsize=args.ppl_cache_max,
    )
    target_layers_list = list(ppl_eval.target_layers)
    W_map = {k: int(W_map_all[k]) for k in target_layers_list if k in W_map_all}
    C_prime_filtered = {
        k: float(C_prime_map[k]) for k in target_layers_list if k in C_prime_map
    }

    sum_w = sum(W_map.values())
    B_target = target_weighted_sum(args.avg_bits, W_map)
    print(f"[Budget] target Σw·b={B_target}, Σw={sum_w}, wavg≈{B_target/sum_w:.6f}")

    proxy_loss_calc = lambda b: calculate_proxy_loss(b, C_prime_filtered, args.bmin)

    # Phase 2.5: Load surrogate
    print("--- 🧠 Phase 2.5: surrogate 로드 ---")
    surrogate, sur_layer_order, (C_mu, C_sd, W_mu, W_sd), use_proxy = load_surrogate(
        args.surrogate_ckpt, device=device
    )

    # static features normalization (surrogate layer order 기준)
    C_log = np.zeros((len(sur_layer_order),), dtype=np.float32)
    W_log = np.zeros((len(sur_layer_order),), dtype=np.float32)
    for i, ln in enumerate(sur_layer_order):
        cp = float(C_prime_filtered.get(ln, 0.0))
        w = float(W_map.get(ln, 1.0))
        C_log[i] = math.log(max(cp, 1e-30))
        W_log[i] = math.log(max(w, 1.0))
    C_log_n = (C_log - C_mu) / (C_sd + 1e-12)
    W_log_n = (W_log - W_mu) / (W_sd + 1e-12)
    C_log_t = torch.tensor(C_log_n, dtype=torch.float32, device=device)
    W_log_t = torch.tensor(W_log_n, dtype=torch.float32, device=device)

    # Phase 3: Seed
    print("--- 🌱 Phase 3: 시드 생성 ---")
    b_seed = get_initial_seed(
        C_prime_filtered, W_map, args.avg_bits, args.bmin, args.bmax
    )
    if args.init_assign_csv and os.path.exists(args.init_assign_csv):
        print(f"--- 덮어쓰기: {args.init_assign_csv}에서 시드 로드 ---")
        seed_csv = load_seed_from_csv(args.init_assign_csv)
        if seed_csv:
            b_seed.update(seed_csv)

    b_seed = ensure_complete_assignment(b_seed, target_layers_list, args.bmin)
    b_seed = project_to_weighted_budget(
        b_seed, W_map, C_prime_filtered, B_target, args.bmin, args.bmax
    )

    # Gen0 initial candidate pool (원본과 동일한 방식 유지)
    initial_candidates = {}
    b_seed_proj = project_to_weighted_budget(
        b_seed, W_map, C_prime_filtered, B_target, args.bmin, args.bmax
    )
    initial_candidates[tuple(sorted(b_seed_proj.items()))] = proxy_loss_calc(
        b_seed_proj
    )

    for _ in range(args.beam_size * args.expansion_k):
        neighbor = generate_random_neighbor(
            b_seed, target_layers_list, args.bmin, args.bmax
        )
        if neighbor:
            neighbor = ensure_complete_assignment(
                neighbor, target_layers_list, args.bmin
            )
            neighbor = project_to_weighted_budget(
                neighbor, W_map, C_prime_filtered, B_target, args.bmin, args.bmax
            )
            initial_candidates[tuple(sorted(neighbor.items()))] = proxy_loss_calc(
                neighbor
            )

    # surrogate로 initial pool 스코어링 -> top filter_p -> 그 중 top beam_size만 true ppl
    init_list = [(dict(bt), l) for bt, l in initial_candidates.items()]
    init_bits = [b for (b, _) in init_list]
    init_proxy = [float(l) for (_, l) in init_list]
    init_scores = surrogate_score_candidates(
        surrogate,
        init_bits,
        init_proxy,
        sur_layer_order,
        C_log_t,
        W_log_t,
        use_proxy,
        args.bmin,
        device,
        batch_size=args.sur_batch,
    )

    idx_sorted = np.argsort(init_scores)  # 낮을수록 좋음
    topP_idx = idx_sorted[: min(args.filter_p, len(idx_sorted))]
    # topP에서 다시 topk true ppl eval (k=beam_size)
    topK_idx = topP_idx[: args.beam_size]

    beam = []
    print(f"--- 초기 True PPL 평가 (surrogate top-{args.beam_size} only) ---")
    for i in tqdm(topK_idx.tolist(), desc="Gen0 true_ppl(topK)"):
        b = init_bits[int(i)]
        l = init_proxy[int(i)]
        ppl = ppl_eval.evaluate(b)
        beam.append((ppl, float(init_scores[int(i)]), l, b))
    beam.sort(key=lambda x: x[0])
    print(
        f"[Gen0] best_true_ppl={beam[0][0]:.4f} | best_pred={beam[0][1]:.6f} | wavg={weighted_sum_bits(beam[0][3], W_map)/sum_w:.6f}"
    )

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    curve_path = os.path.join(args.output_dir, "surrogate_mc_curve.csv")
    new_file = not os.path.exists(curve_path)
    fcurve = open(curve_path, "a", newline="", encoding="utf-8")
    wcurve = csv.writer(fcurve)
    if new_file:
        wcurve.writerow(
            [
                "generation",
                "num_candidates",
                "filter_p",
                "eval_k",
                "best_true_ppl",
                "best_pred_score",
                "best_proxy_L",
                "weighted_avg_bits",
                "global_best_true_ppl",
                "cache_hits",
                "cache_misses",
                "timestamp",
                "note",
            ]
        )
        fcurve.flush()

    start_ts = time.time()
    gen = 0
    no_improve = 0
    stable_bits = 0
    global_best_true = beam[0][0]
    prev_best_true = beam[0][0]
    prev_best_bits = beam[0][3]

    # generations >0이면 상한으로 사용 (원본 방식 유지)
    max_g = (
        args.max_generations
        if args.max_generations > 0
        else (args.generations if args.generations > 0 else 0)
    )

    def time_over():
        return (args.time_limit_sec > 0) and (
            time.time() - start_ts >= args.time_limit_sec
        )

    # Main loop
    print("--- 🎲 Phase 4: Surrogate-guided MC Beam Search (converge) ---")
    while True:
        # Expand
        all_candidates = set()
        for _, _, _, b_assign in beam:
            all_candidates.add(tuple(sorted(b_assign.items())))
            for _ in range(args.expansion_k):
                neighbor = generate_random_neighbor(
                    b_assign, target_layers_list, args.bmin, args.bmax
                )
                if neighbor:
                    neighbor = ensure_complete_assignment(
                        neighbor, target_layers_list, args.bmin
                    )
                    neighbor = project_to_weighted_budget(
                        neighbor,
                        W_map,
                        C_prime_filtered,
                        B_target,
                        args.bmin,
                        args.bmax,
                    )
                    all_candidates.add(tuple(sorted(neighbor.items())))

        cand_bits: List[Dict[str, int]] = []
        cand_proxy: List[float] = []
        for bt in all_candidates:
            b = dict(bt)
            b = ensure_complete_assignment(b, target_layers_list, args.bmin)
            # 안전: 다시 projection
            b = project_to_weighted_budget(
                b, W_map, C_prime_filtered, B_target, args.bmin, args.bmax
            )
            cand_bits.append(b)
            cand_proxy.append(proxy_loss_calc(b))

        # Filter by surrogate (top filter_p)
        sur_scores = surrogate_score_candidates(
            surrogate,
            cand_bits,
            cand_proxy,
            sur_layer_order,
            C_log_t,
            W_log_t,
            use_proxy,
            args.bmin,
            device,
            batch_size=args.sur_batch,
        )
        order = np.argsort(sur_scores)  # 낮을수록 좋음
        P = min(args.filter_p, len(order))
        finalists_idx = order[:P]

        # Evaluate only topK among finalists (top beam_size)
        K = min(args.beam_size, P)
        eval_idx = finalists_idx[:K]

        new_beam_candidates = []
        for i in tqdm(
            eval_idx.tolist(), desc=f"G-{gen} true_ppl(topK={K})", leave=False
        ):
            b = cand_bits[int(i)]
            ppl = ppl_eval.evaluate(b)
            new_beam_candidates.append(
                (ppl, float(sur_scores[int(i)]), float(cand_proxy[int(i)]), b)
            )

        # Select beam by true ppl (size=beam_size)
        new_beam_candidates.sort(key=lambda x: x[0])
        beam = new_beam_candidates[: args.beam_size]

        best_true, best_pred, best_L, best_bits = beam[0]
        global_best_true = min(global_best_true, best_true)

        S = weighted_sum_bits(best_bits, W_map)
        wavg = S / sum_w if sum_w > 0 else 0.0

        abs_gain = prev_best_true - best_true
        rel_gain = (
            (abs_gain / prev_best_true)
            if (math.isfinite(prev_best_true) and prev_best_true > 0)
            else float("inf")
        )
        improved = (abs_gain > args.converge_eps) or (rel_gain > args.converge_rel_eps)

        note = "improved" if improved else "no-improve"
        print(
            f"--- G-{gen} 완료 | Best TRUE: {best_true:.6f} | Best PRED: {best_pred:.6f} "
            f"(L={best_L:.2e}) | w-avg={wavg:.6f} | global_best_true={global_best_true:.6f} | {note} "
            f"| cand={len(all_candidates)} P={P} evalK={K} cache(h/m)={ppl_eval.cache_hits}/{ppl_eval.cache_misses}"
        )

        wcurve.writerow(
            [
                gen,
                len(all_candidates),
                P,
                K,
                f"{best_true:.6f}",
                f"{best_pred:.8f}",
                f"{best_L:.6e}",
                f"{wavg:.6f}",
                f"{global_best_true:.6f}",
                ppl_eval.cache_hits,
                ppl_eval.cache_misses,
                int(time.time()),
                note,
            ]
        )
        fcurve.flush()

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        if best_bits == prev_best_bits:
            stable_bits += 1
        else:
            stable_bits = 0

        prev_best_true = min(prev_best_true, best_true)
        prev_best_bits = best_bits

        stop_reasons = []
        if args.patience > 0 and no_improve >= args.patience:
            stop_reasons.append(f"no-improve≥{args.patience}")
        if args.stable_bits_patience > 0 and stable_bits >= args.stable_bits_patience:
            stop_reasons.append(f"stable-bits≥{args.stable_bits_patience}")
        if time_over():
            stop_reasons.append("time-limit")
        if max_g > 0 and gen + 1 >= max_g:
            stop_reasons.append("max-generations")

        if stop_reasons:
            print(f"✔️ 수렴/중단: {', '.join(stop_reasons)}")
            break

        gen += 1

    # Save final assignment
    print("\n--- 🏆 Phase 5: 탐색 완료 ---")
    best_true, best_pred, best_L, best_b = beam[0]
    print(f"최종 Best TRUE PPL: {best_true:.6f} | Best PRED: {best_pred:.6f}")
    out_csv = os.path.join(args.output_dir, "bit_assign.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_name", "R_int"])
        for name, bit in sorted(best_b.items()):
            writer.writerow([name, bit])
    print(f"최종 할당 저장: {out_csv}")

    cache_path = os.path.join(args.output_dir, "ppl_cache.json")
    str_cache = {str(k): v for k, v in ppl_eval.ppl_cache.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(str_cache, f, indent=2)
    print(f"PPL 캐시 저장: {cache_path}")

    try:
        fcurve.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Step 3b - Surrogate-guided Monte-Carlo (Prebaked, converge)"
    )

    # paths
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--sens_csv", type=str, required=True)
    parser.add_argument("--alpha_csv", type=str, required=True)
    parser.add_argument("--prebake_root", type=str, required=True)
    parser.add_argument("--init_assign_csv", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="./artifacts/bitmin/step3b_surrogate_mc"
    )

    # device
    parser.add_argument("--gpu_id", type=int, default=0)

    # budget / bits
    parser.add_argument("--avg_bits", type=float, default=2.5)
    parser.add_argument("--alpha_bit", type=int, default=3)
    parser.add_argument("--alpha_default", type=float, default=1.0)
    parser.add_argument("--bmin", type=int, default=2)
    parser.add_argument("--bmax", type=int, default=4)

    # surrogate
    parser.add_argument("--surrogate_ckpt", type=str, required=True)
    parser.add_argument("--sur_batch", type=int, default=512)

    # ppl eval
    parser.add_argument("--eval_seq_len", type=int, default=2048)
    parser.add_argument("--ppl_cache_max", type=int, default=20000)

    # search
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--expansion_k", type=int, default=20)
    parser.add_argument("--filter_p", type=int, default=80)

    # converge / safety (원본 step3b 유지)
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help="(하위호환) >0이면 max_generations로 사용. 0이면 무제한.",
    )
    parser.add_argument(
        "--max_generations", type=int, default=0, help="0이면 수렴 조건까지 진행."
    )
    parser.add_argument("--converge_eps", type=float, default=1e-3)
    parser.add_argument("--converge_rel_eps", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--stable_bits_patience", type=int, default=12)
    parser.add_argument("--time_limit_sec", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
