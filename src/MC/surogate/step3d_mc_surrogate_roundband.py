#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3b — Surrogate-guided Monte-Carlo Beam Search (Round-band budget) + True-PPL eval top-k
===========================================================================================

요구사항 반영 (당신이 요청한 최종 형태):
- **이 스크립트는 step3b_ppl_montecarlo_prebaked_converge_round.py (roundband 포함) 를 뼈대로 유지**
- real PPL evaluator를 "전체 후보 평가"에 쓰지 않고,
  **surrogate가 예측한 score(낮을수록 좋음)** 로 filter를 수행
- 매 generation마다:
  Expand -> surrogate로 all_candidates 스코어링 -> top filter_p 선정 ->
  그 중 **top beam_size(=top10)** 만 True PPL 측정 -> beam 갱신 -> 수렴까지 반복
- global_best_true_ppl은 **세대가 지날수록 실제 PPL이 더 낮아질 때만 갱신(min)**

주의:
- surrogate 출력은 "낮을수록 좋다" 가정 (validate에서 쓰던 것과 동일)
- surrogate가 proxy 입력(use_proxy=True)일 때는 proxy_loss(L(b))를 proxy feature로 넣음
- budget은 기본적으로 exact(Σw·b 정확히)로 맞추되,
  --use_round_band 사용 시 round_quantum 밴드([avg-0.5q, avg+0.5q)) 안으로 프로젝션

Usage:
CUDA_VISIBLE_DEVICES=1 nohup \
python montecarlo/step3d_mc_surrogate_roundband.py \
  --model_id meta-llama/Llama-3.2-3B \
  --gpu_id 0 \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --prebake_root ../artifacts/montecarlo/prebake \
  --init_assign_csv ../artifacts/montecarlo/step3b_refine/bit_assign.csv \
  --avg_bits 2.50 \
  --use_round_band --round_quantum 0.1 \
  --beam_size 10 --expansion_k 20 --filter_p 80 \
  --surrogate_ckpt ../artifacts/surrogate_checkpoint_brp/best.pt \
  --sur_batch 512 \
  --eval_seq_len 2048 \
  --converge_eps 1e-3 --converge_rel_eps 1e-3 \
  --patience 12 --stable_bits_patience 12 \
  --ppl_cache_max 20000 \
  --output_dir ../artifacts/bitmin/step3b_surrogate_roundband \
  > step3b_surrogate_roundband.log 2>&1 &
"""

import os, gc, csv, json, math, random, argparse, re, time, sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# =========================================================
# Repo src/ 탐색 + step2_alpha_estimation / step3b_sa_refine import (원본 step3b_roundband와 동일 계열)
# =========================================================
CURR_DIR = Path(__file__).resolve().parent


def _detect_src_dir(curr_dir: Path) -> Path:
    search_roots = [curr_dir] + list(curr_dir.parents)
    for root in search_roots:
        direct = root / "step2_alpha_estimation.py"
        if direct.exists():
            return root
        nested_src = root / "src" / "step2_alpha_estimation.py"
        if nested_src.exists():
            return root / "src"
    return curr_dir.parent if curr_dir.parents else curr_dir


SRC_DIR = _detect_src_dir(CURR_DIR)
for path in (SRC_DIR, CURR_DIR):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def _import_step2_alpha_estimation():
    module_name = "step2_alpha_estimation"
    module_variants = (
        module_name,
        "src.step2_alpha_estimation",
        "RAQ.src.step2_alpha_estimation",
    )
    for name in module_variants:
        try:
            return importlib.import_module(name)
        except ImportError as e:
            missing = name.split(".")[-1]
            if getattr(e, "name", missing) != missing:
                raise
    module_path = SRC_DIR / f"{module_name}.py"
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module_name] = module
            return module
    raise ImportError(f"'{module_name}' 모듈을 {module_path} 에서 찾을 수 없습니다.")


def _import_step3b_sa_refine():
    module_name = "step3b_sa_refine"
    module_variants = (
        module_name,
        "montecarlo.step3b_sa_refine",
        "src.montecarlo.step3b_sa_refine",
        "RAQ.src.montecarlo.step3b_sa_refine",
    )
    for name in module_variants:
        try:
            return importlib.import_module(name)
        except ImportError as e:
            missing = name.split(".")[-1]
            if getattr(e, "name", missing) != missing:
                raise
    candidate_paths = [
        CURR_DIR / f"{module_name}.py",
        SRC_DIR / "montecarlo" / f"{module_name}.py",
    ]
    for path in candidate_paths:
        if path.exists():
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[module_name] = module
                return module
    raise ImportError(f"'{module_name}' 모듈을 찾을 수 없습니다.")


try:
    _step2_alpha = _import_step2_alpha_estimation()
    _canonical_dataset_name = (
        _step2_alpha._canonical_dataset_name
    )  # noqa: F401 (원본 호환)
except ImportError as e:
    print(f"오류: step2_alpha_estimation.py 로딩 실패 — {e}")
    raise

try:
    _sa_refine = _import_step3b_sa_refine()
    load_sens_table = _sa_refine.load_sens_table
    build_loss_table = _sa_refine.build_loss_table
    waterfill_budget = _sa_refine.waterfill_budget
    round_and_fix_budget = _sa_refine.round_and_fix_budget
    KAPPA = _sa_refine.KAPPA
except ImportError as e:
    print(f"오류: step3b_sa_refine.py 로딩 실패 — {e}")
    raise


# =========================================================
# Helpers (roundband 원본 기반)
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
    return float(math.exp(total_nll / total_tok))


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


def build_c_prime_map(
    sens_csv,
    alpha_csv,
    alpha_bit=3,
    alpha_default=1.0,
    bmin=2,
    bmax=4,
):
    """
    Step3b-SA 유틸을 이용해 레이어별 (C', w, loss_table) 정보를 구축.
    반환:
      - C_prime_map: layer_name → C'
      - W_map      : layer_name → w_j
      - loss_lookup: layer_name → {bit: loss}
      - layer_order: 입력 CSV 순서의 레이어 이름 리스트
      - bits_tuple : 사용된 비트 집합(tuple)
    """
    layers = load_sens_table(sens_csv, sens_col="C_mean_per_batch")
    bits = list(range(int(bmin), int(bmax) + 1))
    if alpha_bit not in bits:
        bits.append(alpha_bit)
    bits_tuple = tuple(sorted(set(bits)))

    loss_table, _ = build_loss_table(
        layers=layers,
        bits=bits_tuple,
        loss_table_csv=None,
        alpha_csv=alpha_csv,
        alpha_default=alpha_default,
    )

    C_prime_map, W_map, loss_lookup = {}, {}, {}
    exp_factor = math.exp(KAPPA * float(alpha_bit))
    for layer, loss_row in zip(layers, loss_table):
        loss_lookup[layer.name] = dict(loss_row)
        if alpha_bit in loss_row:
            Cp = float(loss_row[alpha_bit]) * exp_factor
        else:
            Cp = float(layer.C) * float(alpha_default)
        C_prime_map[layer.name] = max(0.0, Cp)
        W_map[layer.name] = int(layer.w)

    layer_order = [layer.name for layer in layers]
    return C_prime_map, W_map, loss_lookup, layer_order, bits_tuple


def calculate_proxy_loss(bit_assignment, C_prime_map, bmin=2):
    total_loss = 0.0
    for name, cp in C_prime_map.items():
        bit = bit_assignment.get(name, bmin)
        total_loss += float(cp) * (2.0 ** (-2.0 * float(bit)))
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
    # gcd 정렬로 "도달 가능한" 가장 가까운 타깃 합으로 스냅
    sum_w = sum(int(w) for w in W_map.values())
    g = gcd_list([int(w) for w in W_map.values()]) if W_map else 1
    raw = float(avg_bits) * float(sum_w)
    snapped = int(round(raw / g) * g)
    return int(snapped)


def ensure_complete_assignment(
    b_assign: Dict[str, int], layer_names: List[str], bmin: int
) -> Dict[str, int]:
    out = dict(b_assign)
    for n in layer_names:
        if n not in out:
            out[n] = int(bmin)
    return out


def project_to_weighted_budget(
    b_assign: Dict[str, int],
    W_map: Dict[str, int],
    C_prime_map: Dict[str, float],
    B_target: int,
    bmin: int,
    bmax: int,
    max_steps: int = 200000,
) -> Dict[str, int]:
    """
    정수 비트 제약 하에서 Σ w_j b_j = B_target 으로 투영.
    delta>0 : 예산 미달 → +1을 넣을 레이어를 '개선/가중치' 최대 기준으로 선택
    delta<0 : 예산 초과 → -1을 뺄 레이어를 '손실/가중치' 최소 기준으로 선택
    """
    b = dict(b_assign)
    names = [n for n in b.keys() if n in W_map]
    if not names:
        return b

    def marg_gain_up(n):
        b0 = int(b[n])
        if b0 >= bmax:
            return -float("inf")
        cp = float(C_prime_map.get(n, 0.0))
        return cp * ((2.0 ** (-2.0 * b0)) - (2.0 ** (-2.0 * (b0 + 1))))

    def marg_harm_down(n):
        b0 = int(b[n])
        if b0 <= bmin:
            return float("inf")
        cp = float(C_prime_map.get(n, 0.0))
        return cp * ((2.0 ** (-2.0 * (b0 - 1))) - (2.0 ** (-2.0 * b0)))

    g = gcd_list([int(W_map[n]) for n in names])
    if g <= 0:
        g = 1
    if int(B_target) % g != 0:
        B_target = int(round(int(B_target) / g) * g)

    steps = 0
    while steps < max_steps:
        S = weighted_sum_bits(b, W_map)
        delta = int(B_target) - int(S)
        if delta == 0:
            break

        if delta > 0:
            cand = [
                (marg_gain_up(n) / float(W_map[n]), n)
                for n in names
                if int(b[n]) < bmax
            ]
            if not cand:
                break
            cand.sort(key=lambda x: x[0], reverse=True)
            n_best = cand[0][1]
            b[n_best] = int(b[n_best]) + 1
        else:
            cand = [
                (marg_harm_down(n) / float(W_map[n]), n)
                for n in names
                if int(b[n]) > bmin
            ]
            if not cand:
                break
            cand.sort(key=lambda x: x[0])
            n_best = cand[0][1]
            b[n_best] = int(b[n_best]) - 1

        steps += 1

    return b


def project_to_weighted_band(
    b_assign: Dict[str, int],
    W_map: Dict[str, int],
    C_prime_map: Dict[str, float],
    B_lo: int,
    B_hi: int,
    bmin: int,
    bmax: int,
) -> Dict[str, int]:
    """
    Σ w_j b_j 가 [B_lo, B_hi] 밴드에 들어오도록 프로젝션.
    밴드 밖이면 가까운 경계로 project_to_weighted_budget 호출.
    """
    if not W_map:
        return b_assign
    S = weighted_sum_bits(b_assign, W_map)
    if B_lo > B_hi:
        mid = (int(B_lo) + int(B_hi)) // 2
        return project_to_weighted_budget(b_assign, W_map, C_prime_map, mid, bmin, bmax)
    if S < int(B_lo):
        return project_to_weighted_budget(
            b_assign, W_map, C_prime_map, int(B_lo), bmin, bmax
        )
    if S > int(B_hi):
        return project_to_weighted_budget(
            b_assign, W_map, C_prime_map, int(B_hi), bmin, bmax
        )
    return b_assign


def get_initial_seed(
    C_prime_map,
    W_map,
    avg_bits,
    bmin=2,
    bmax=4,
    loss_lookup=None,
    bits_tuple=None,
    layer_order=None,
) -> Dict[str, int]:
    names_order = [
        n
        for n in (layer_order or list(C_prime_map.keys()))
        if n in W_map and n in C_prime_map
    ]
    if not names_order:
        print("[경고] 초기 시드 생성 실패: 유효한 레이어가 없습니다.")
        return {}

    Cp_vals = [float(C_prime_map[n]) for n in names_order]
    w_vals = [int(W_map[n]) for n in names_order]
    if sum(w_vals) == 0:
        print("[경고] 초기 시드 생성 실패: 유효한 가중치 합이 0입니다.")
        return {}

    B_target = target_weighted_sum(
        float(avg_bits), {n: int(W_map[n]) for n in names_order}
    )
    R_cont = waterfill_budget(
        Cp_vals, w_vals, float(B_target), float(bmin), float(bmax)
    )

    b_list = None
    if loss_lookup and bits_tuple:
        try:
            loss_table = [loss_lookup[n] for n in names_order]
            b_list = round_and_fix_budget(
                R_cont, w_vals, loss_table, bits_tuple, float(B_target)
            )
        except Exception as e:
            print(f"[경고] SA 라운딩 실패, 기본 라운딩 사용: {e}")

    if b_list is None:
        b_list = [max(bmin, min(bmax, int(round(r)))) for r in R_cont]

    b_seed = {names_order[i]: int(b_list[i]) for i in range(len(names_order))}
    b_seed = project_to_weighted_budget(
        b_seed,
        {n: int(W_map[n]) for n in names_order},
        C_prime_map,
        int(B_target),
        bmin,
        bmax,
    )
    return b_seed


def generate_random_neighbor(
    b_assign: Dict[str, int], layer_names: List[str], bmin=2, bmax=4
) -> Dict[str, int]:
    new_b = b_assign.copy()
    c_up = [n for n in layer_names if int(new_b.get(n, bmin)) < bmax]
    c_down = [n for n in layer_names if int(new_b.get(n, bmin)) > bmin]
    if not c_up or not c_down:
        return None
    j = random.choice(c_up)
    k = random.choice(c_down)
    if j == k and (len(c_up) > 1 or len(c_down) > 1):
        return generate_random_neighbor(b_assign, layer_names, bmin, bmax)
    if j == k:
        return None
    new_b[j] = int(new_b.get(j, bmin)) + 1
    new_b[k] = int(new_b.get(k, bmin)) - 1
    return new_b


# =========================================================
# PPL Evaluator (prebaked Wq/A/B) — topK만 true ppl 평가에 사용
# =========================================================
class PplEvaluator:
    """
    Pre-baked Wq/A/B 로드 후 W_eff = Wq + A@B를 주입해 PPL 측정.
    동일 조합 반복 평가 방지를 위해 LRU 캐시 사용.
    """

    def __init__(
        self,
        model,
        tokenizer,
        original_state_dict,
        eval_input_ids,
        prebake_root,
        eval_seq_len,
        device: torch.device,
        cache_maxsize: int = 5000,
    ):
        self.model = model
        self.device = device
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
                bit = bit_assignment.get(layer_name, None)
                module = self._get_module(layer_name)
                if bit is None or module is None:
                    continue
                safe_name = _safe_name(layer_name)
                file_path = self.prebake_root / f"bit{int(bit)}" / f"{safe_name}.pt"
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
# Surrogate (validate에서 사용하던 형태)
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
        # "낮을수록 좋다"
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

    return model, list(layer_names), (C_mu, C_sd, W_mu, W_sd), use_proxy


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

    for i in range(0, bits.shape[0], int(batch_size)):
        j = min(bits.shape[0], i + int(batch_size))
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
# 모델 로딩 (device_map 이슈 안전 처리)
# =========================================================
def load_model_tokenizer(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # 안전 로딩: 가능한 한 단일 GPU로
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": device.index} if device.type == "cuda" else None,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        model.to(device)
    model.eval()
    return model, tokenizer


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
    model, tokenizer = load_model_tokenizer(args.model_id, device=device)

    # 모델 원본 가중치 백업 (CPU)
    original_state_dict = {
        k: v.detach().clone().cpu() for k, v in model.state_dict().items()
    }

    print("PPL 평가용 데이터셋 로드 (wikitext-2-raw-v1)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    eval_input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids

    print("프록시(C', W) 로드...")
    C_prime_map_all, W_map_all, loss_lookup_all, _layer_order_all, bits_tuple = (
        build_c_prime_map(
            args.sens_csv,
            args.alpha_csv,
            args.alpha_bit,
            args.alpha_default,
            args.bmin,
            args.bmax,
        )
    )

    # Phase 2: PPL evaluator (true ppl은 topK에서만)
    print("--- 🌱 Phase 2: 평가기(PPL) 초기화 ---")
    ppl_eval = PplEvaluator(
        model=model,
        tokenizer=tokenizer,
        original_state_dict=original_state_dict,
        eval_input_ids=eval_input_ids,
        prebake_root=args.prebake_root,
        eval_seq_len=args.eval_seq_len,
        device=device,
        cache_maxsize=args.ppl_cache_max,
    )

    target_layers_list = list(ppl_eval.target_layers)
    W_map = {k: int(W_map_all[k]) for k in target_layers_list if k in W_map_all}
    C_prime_filtered = {
        k: float(C_prime_map_all[k]) for k in target_layers_list if k in C_prime_map_all
    }
    loss_lookup_filtered = {
        k: dict(loss_lookup_all[k]) for k in target_layers_list if k in loss_lookup_all
    }

    sum_w = sum(W_map.values())
    if sum_w <= 0:
        raise RuntimeError("Σw 가 0입니다. W_map 구성이 비정상입니다.")

    B_target = target_weighted_sum(args.avg_bits, W_map)

    # ---- roundband 계산 ----
    use_band = bool(getattr(args, "use_round_band", False))
    quantum = float(getattr(args, "round_quantum", 0.1))
    if use_band:
        eps = 1e-9
        avg_lo = float(args.avg_bits) - 0.5 * quantum
        avg_hi = float(args.avg_bits) + 0.5 * quantum - eps

        g = gcd_list(list(W_map.values())) or 1
        B_lo = int(math.ceil((avg_lo * sum_w) / g) * g)
        B_hi = int(math.floor((avg_hi * sum_w) / g) * g)
        if B_lo > B_hi:
            B_lo = B_hi = int(B_target)

        print(
            f"[Budget Band] round→{args.avg_bits:.2f} (quantum={quantum}) → "
            f"허용 가중평균 [{B_lo/sum_w:.6f}, {B_hi/sum_w:.6f}) "
            f"(Σw·b ∈ [{B_lo}, {B_hi}]); "
            f"기준 타깃≈{B_target/sum_w:.6f}"
        )
    else:
        B_lo = B_hi = int(B_target)
        print(f"[Budget Exact] 목표 가중평균≈{B_target/sum_w:.6f} (Σw·b={B_target})")

    def project_budget(b: Dict[str, int]) -> Dict[str, int]:
        b = ensure_complete_assignment(b, target_layers_list, args.bmin)
        return project_to_weighted_band(
            b, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax
        )

    proxy_loss_calc = lambda b: calculate_proxy_loss(b, C_prime_filtered, args.bmin)

    # Phase 2.5: Load surrogate
    print("--- 🧠 Phase 2.5: surrogate 로드 ---")
    surrogate, sur_layer_order, (C_mu, C_sd, W_mu, W_sd), sur_use_proxy = (
        load_surrogate(args.surrogate_ckpt, device=device)
    )

    # static features normalization (surrogate layer order 기준)
    C_log = np.zeros((len(sur_layer_order),), dtype=np.float32)
    W_log = np.zeros((len(sur_layer_order),), dtype=np.float32)
    for i, ln in enumerate(sur_layer_order):
        cp = float(C_prime_filtered.get(ln, 0.0))
        w = float(W_map.get(ln, 1.0))
        C_log[i] = math.log(max(cp, 1e-30))
        W_log[i] = math.log(max(w, 1.0))
    C_log_n = (C_log - float(C_mu)) / (float(C_sd) + 1e-12)
    W_log_n = (W_log - float(W_mu)) / (float(W_sd) + 1e-12)
    C_log_t = torch.tensor(C_log_n, dtype=torch.float32, device=device)
    W_log_t = torch.tensor(W_log_n, dtype=torch.float32, device=device)

    # Phase 3: seed 생성
    print("--- 🌱 Phase 3: 시드 생성 ---")
    b_seed = get_initial_seed(
        C_prime_filtered,
        W_map,
        args.avg_bits,
        args.bmin,
        args.bmax,
        loss_lookup=loss_lookup_filtered,
        bits_tuple=bits_tuple,
        layer_order=target_layers_list,
    )
    if args.init_assign_csv and os.path.exists(args.init_assign_csv):
        print(f"--- 덮어쓰기: {args.init_assign_csv} 에서 시드 로드 ---")
        seed_csv = load_seed_from_csv(args.init_assign_csv)
        if seed_csv:
            b_seed.update(seed_csv)
        else:
            print(f"[경고] {args.init_assign_csv}에서 유효한 시드를 찾지 못함.")

    b_seed = project_budget(b_seed)

    # Gen0 initial candidate pool (원본 방식 유지)
    initial_candidates = {}
    b_seed_proj = project_budget(b_seed)
    initial_candidates[tuple(sorted(b_seed_proj.items()))] = proxy_loss_calc(
        b_seed_proj
    )

    for _ in range(int(args.beam_size) * int(args.expansion_k)):
        neighbor = generate_random_neighbor(
            b_seed, target_layers_list, args.bmin, args.bmax
        )
        if neighbor:
            neighbor = project_budget(neighbor)
            initial_candidates[tuple(sorted(neighbor.items()))] = proxy_loss_calc(
                neighbor
            )

    # surrogate로 initial pool 스코어링 -> top filter_p -> 그 중 topK만 true ppl
    init_list = [(dict(bt), float(l)) for bt, l in initial_candidates.items()]
    init_bits = [b for (b, _) in init_list]
    init_proxy = [l for (_, l) in init_list]

    init_scores = surrogate_score_candidates(
        surrogate=surrogate,
        candidates=init_bits,
        proxy_losses=init_proxy,
        layer_order=sur_layer_order,
        C_log_t=C_log_t,
        W_log_t=W_log_t,
        use_proxy=sur_use_proxy,
        bmin=args.bmin,
        device=device,
        batch_size=args.sur_batch,
    )

    idx_sorted = np.argsort(init_scores)  # 낮을수록 좋음
    topP_idx = idx_sorted[: min(int(args.filter_p), len(idx_sorted))]
    topK_idx = topP_idx[: min(int(args.beam_size), len(topP_idx))]

    beam: List[Tuple[float, float, float, Dict[str, int]]] = []
    print(f"--- 초기 True PPL 평가 (surrogate top-{len(topK_idx)} only) ---")
    for i in tqdm(topK_idx.tolist(), desc="Gen0 true_ppl(topK)"):
        b = project_budget(init_bits[int(i)])
        l = float(proxy_loss_calc(b))
        pred = float(init_scores[int(i)])
        ppl = ppl_eval.evaluate(b)
        beam.append((float(ppl), float(pred), float(l), b))
    beam.sort(key=lambda x: x[0])

    # global best init
    global_best_true = float(beam[0][0])
    global_best_bits = dict(beam[0][3])
    global_best_pred = float(beam[0][1])
    global_best_L = float(beam[0][2])

    wavg0 = weighted_sum_bits(beam[0][3], W_map) / sum_w
    print(
        f"[Gen0] best_true={beam[0][0]:.6f} | best_pred={beam[0][1]:.6f} | wavg={wavg0:.6f} | global_best_true={global_best_true:.6f}"
    )

    # Logging
    os.makedirs(args.output_dir, exist_ok=True)
    curve_path = os.path.join(args.output_dir, "surrogate_roundband_curve.csv")
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

    # log init row
    wcurve.writerow(
        [
            -1,
            len(initial_candidates),
            min(int(args.filter_p), len(initial_candidates)),
            len(topK_idx),
            f"{beam[0][0]:.6f}",
            f"{beam[0][1]:.8f}",
            f"{beam[0][2]:.6e}",
            f"{wavg0:.6f}",
            f"{global_best_true:.6f}",
            ppl_eval.cache_hits,
            ppl_eval.cache_misses,
            int(time.time()),
            "init",
        ]
    )
    fcurve.flush()

    # Converge state
    start_ts = time.time()
    gen = 0
    no_improve = 0
    stable_bits = 0
    prev_best_bits = beam[0][3]  # 현재 세대 best bits 추적 (stable_bits용)

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
    print("--- 🎲 Phase 4: Surrogate-guided MC Beam Search (converge, roundband) ---")
    while True:
        # Expand
        all_candidates = set()
        for _, _, _, b_assign in beam:
            all_candidates.add(tuple(sorted(b_assign.items())))
            for _ in range(int(args.expansion_k)):
                neighbor = generate_random_neighbor(
                    b_assign, target_layers_list, args.bmin, args.bmax
                )
                if neighbor:
                    neighbor = project_budget(neighbor)
                    all_candidates.add(tuple(sorted(neighbor.items())))

        cand_bits: List[Dict[str, int]] = []
        cand_proxy: List[float] = []
        for bt in all_candidates:
            b = project_budget(dict(bt))
            cand_bits.append(b)
            cand_proxy.append(float(proxy_loss_calc(b)))

        # Filter by surrogate (top filter_p)
        sur_scores = surrogate_score_candidates(
            surrogate=surrogate,
            candidates=cand_bits,
            proxy_losses=cand_proxy,
            layer_order=sur_layer_order,
            C_log_t=C_log_t,
            W_log_t=W_log_t,
            use_proxy=sur_use_proxy,
            bmin=args.bmin,
            device=device,
            batch_size=args.sur_batch,
        )

        order = np.argsort(sur_scores)  # 낮을수록 좋음
        P = min(int(args.filter_p), len(order))
        finalists_idx = order[:P]

        # Evaluate only topK among finalists (top beam_size)
        K = min(int(args.beam_size), P)
        eval_idx = finalists_idx[:K]

        new_beam_candidates: List[Tuple[float, float, float, Dict[str, int]]] = []
        for i in tqdm(
            eval_idx.tolist(), desc=f"G-{gen} true_ppl(topK={K})", leave=False
        ):
            b = cand_bits[int(i)]
            ppl = ppl_eval.evaluate(b)
            new_beam_candidates.append(
                (float(ppl), float(sur_scores[int(i)]), float(cand_proxy[int(i)]), b)
            )

        # Select beam by true ppl
        new_beam_candidates.sort(key=lambda x: x[0])
        beam = new_beam_candidates[: int(args.beam_size)]

        best_true, best_pred, best_L, best_bits = beam[0]
        S = weighted_sum_bits(best_bits, W_map)
        wavg = S / sum_w

        # improvement 기준: "global best true ppl"을 실제로 낮췄는지
        abs_gain = float(global_best_true) - float(best_true)
        rel_gain = (
            (abs_gain / float(global_best_true))
            if (math.isfinite(global_best_true) and global_best_true > 0)
            else float("inf")
        )
        improved = (abs_gain > float(args.converge_eps)) or (
            rel_gain > float(args.converge_rel_eps)
        )

        if improved:
            global_best_true = float(best_true)
            global_best_bits = dict(best_bits)
            global_best_pred = float(best_pred)
            global_best_L = float(best_L)

        note = "improved" if improved else "no-improve"
        print(
            f"--- G-{gen} 완료 | Best TRUE(gen)={best_true:.6f} | Best PRED={best_pred:.6f} "
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

    # Save final assignment: global best를 저장 (요구사항 핵심)
    print("\n--- 🏆 Phase 5: 탐색 완료 ---")
    print(f"Global Best TRUE PPL: {global_best_true:.6f} | pred={global_best_pred:.6f}")

    S_final = weighted_sum_bits(global_best_bits, W_map)
    wavg_final = S_final / sum_w
    if use_band:
        print(
            f"GlobalBest Σw·b={S_final} / Σw={sum_w} → wavg={wavg_final:.6f} "
            f"(허용대역≈[{B_lo/sum_w:.6f}, {B_hi/sum_w:.6f}), 기준표기 {args.avg_bits:.2f})"
        )
    else:
        print(
            f"GlobalBest Σw·b={S_final} / Σw={sum_w} → wavg={wavg_final:.6f} "
            f"(목표≈{B_target/sum_w:.6f})"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "bit_assign_global_best.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_name", "R_int"])
        for name, bit in sorted(global_best_bits.items()):
            writer.writerow([name, int(bit)])
    print(f"Global best 할당 저장: {out_csv}")

    # ppl cache 저장
    cache_path = os.path.join(args.output_dir, "ppl_cache.json")
    str_cache = {str(k): float(v) for k, v in ppl_eval.ppl_cache.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(str_cache, f, indent=2)
    print(f"PPL 캐시 저장: {cache_path}")

    try:
        fcurve.close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Step 3b - Surrogate-guided Monte-Carlo (Prebaked, converge, roundband)"
    )

    # paths
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--sens_csv", type=str, required=True)
    parser.add_argument("--alpha_csv", type=str, required=True)
    parser.add_argument("--prebake_root", type=str, required=True)
    parser.add_argument("--init_assign_csv", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts/bitmin/step3b_surrogate_roundband",
    )

    # device
    parser.add_argument("--gpu_id", type=int, default=0)

    # budget / bits
    parser.add_argument("--avg_bits", type=float, default=2.5)
    parser.add_argument("--alpha_bit", type=int, default=3)
    parser.add_argument("--alpha_default", type=float, default=1.0)
    parser.add_argument("--bmin", type=int, default=2)
    parser.add_argument("--bmax", type=int, default=4)

    # roundband
    parser.add_argument(
        "--use_round_band",
        action="store_true",
        help="avg_bits를 반올림 밴드로 허용(예: 2.45~2.55→표기상 2.50).",
    )
    parser.add_argument(
        "--round_quantum",
        type=float,
        default=0.1,
        help="반올림 자릿수 폭 (0.1이면 소수 첫째자리 반올림).",
    )

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

    # converge / safety
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help="(하위호환) >0이면 max_generations로 사용. 0이면 무제한.",
    )
    parser.add_argument(
        "--max_generations", type=int, default=0, help="0이면 수렴 조건까지 진행."
    )
    parser.add_argument(
        "--converge_eps",
        type=float,
        default=1e-3,
        help="절대 개선 임계값 (global best 기준).",
    )
    parser.add_argument(
        "--converge_rel_eps",
        type=float,
        default=1e-3,
        help="상대 개선 임계값 (global best 기준).",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--stable_bits_patience", type=int, default=12)
    parser.add_argument("--time_limit_sec", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
