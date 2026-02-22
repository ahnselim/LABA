#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3b — PPL-Direct Optimization using Monte-Carlo Beam Search
(수정: Pre-baked Wq/A/B 로더 사용 + generation별 best PPL CSV 로깅 + 수렴까지 반복)
(추가) 가중평균(파라미터수×비트) **반올림 밴드 허용** 옵션:
      예) --avg_bits 2.50, --use_round_band, --round_quantum 0.1
          → [2.45, 2.55) 범위(표기상 2.50) 안에 들면 허용

이 스크립트는 Step 3의 프록시 손실(L(b)) 대신,
실제 PPL을 목적 함수로 사용하여 몬테카를로 빔 서치를 수행합니다.

(수정) Step 4 (prebake)의 결과를 사용하여 PPL 평가 속도를 가속화합니다. (--prebake_root)
(추가) 빔 서치를 "세대 수 고정"이 아니라 "수렴 조건"으로 중단
(추가) 모든 후보를 weighted budget으로 **정확히** 맞추거나(기본),
      또는 **반올림 밴드** 안으로 프로젝션(--use_round_band)합니다.

CUDA_VISIBLE_DEVICES=0 nohup \
python montecarlo/real_ppl/step3b_ppl_mc_converge_roun_v2.py \
  --model_id meta-llama/Llama-3.2-3B \
  --gpu_id 0 \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --prebake_root ../artifacts/montecarlo/prebake \
  --avg_bits 2.50 \
  --use_round_band --round_quantum 0.1 \
  --beam_size 10 --expansion_k 20 --filter_p 80 \
  --patience 12 --stable_bits_patience 12 \
  --output_dir ../artifacts/montecarlo/step3b_ppl_roundband > llama3_8b_ppl_roundband.log 2>&1 &


      
"""
import os, gc, csv, json, math, random, argparse, re, time, sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict
from functools import reduce
import operator

# Ensure repo-relative modules (src/*, proxy_codes/*) can be imported even when
# this script is executed from arbitrary working directories.
_FILE_PATH = Path(__file__).resolve()
_PARENTS = _FILE_PATH.parents
_SRC_ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _FILE_PATH.parent
_REPO_ROOT = _PARENTS[2] if len(_PARENTS) > 2 else _SRC_ROOT
_WORKSPACE_ROOT = _PARENTS[3] if len(_PARENTS) > 3 else _REPO_ROOT
_PROJECT_ROOT = _WORKSPACE_ROOT.parent if _WORKSPACE_ROOT != _WORKSPACE_ROOT.parent else _WORKSPACE_ROOT
for _path in (_SRC_ROOT, _REPO_ROOT, _WORKSPACE_ROOT, _PROJECT_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# -----------------------------------------------
# Step 2/4에서 가져오기 (경로/네이밍)
# -----------------------------------------------
try:
    from step2_alpha_estimation import (
        _canonical_dataset_name,  # Step 4의 캐시 이름 매칭용 (여기선 직접 사용하진 않음)
    )
except ImportError:
    print("오류: step2_alpha_estimation.py가 같은 디렉토리에 필요합니다.")
    exit(1)

# -----------------------------------------------
# Step 3에서 가져오기 (프록시 모델 및 시드 생성)
# -----------------------------------------------
try:
    from RAQ.proxy_codes.step3_bit_optimization import (
        load_sensitivity_csv,
        load_alpha_csv,
        solve_mu_for_budget,  # 연속 워터필링
        greedy_integer_refine_budget,  # 정수화(가중예산 고려)
    )
except ImportError:
    print("오류: step3_bit_optimization.py가 같은 디렉토리에 필요합니다.")
    exit(1)

# ===============================================
# 이식/헬퍼 함수
# ===============================================


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
    ppl = math.exp(total_nll / total_tok)
    return ppl


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
        W_map[name] = int(w[i])  # 정수 파라미터 수로 강제
    return C_prime_map, W_map


def calculate_proxy_loss(bit_assignment, C_prime_map, bmin=2):
    total_loss = 0.0
    for name, cp in C_prime_map.items():
        bit = bit_assignment.get(name, bmin)
        total_loss += cp * (2.0 ** (-2.0 * bit))
    return total_loss


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
        b0 = b[n]
        if b0 >= bmax:
            return -float("inf")
        cp = float(C_prime_map.get(n, 0.0))
        # 개선량(양수): L(b)-L(b+1)
        return cp * ((2.0 ** (-2.0 * b0)) - (2.0 ** (-2.0 * (b0 + 1))))

    def marg_harm_down(n):
        b0 = b[n]
        if b0 <= bmin:
            return float("inf")
        cp = float(C_prime_map.get(n, 0.0))
        # 손실(양수): L(b-1)-L(b)
        return cp * ((2.0 ** (-2.0 * (b0 - 1))) - (2.0 ** (-2.0 * b0)))

    sum_w = sum(int(W_map[n]) for n in names)
    g = gcd_list([int(W_map[n]) for n in names])
    # 타깃 자체를 gcd 배수로 스냅(안전)
    if B_target % g != 0:
        B_target = int(round(B_target / g) * g)
    steps = 0
    while steps < max_steps:
        S = weighted_sum_bits(b, W_map)
        delta = B_target - S
        if delta == 0:
            break
        if delta > 0:
            # +1할 후보 중 (개선량/가중치) 최대
            cand = [
                (marg_gain_up(n) / float(W_map[n]), n) for n in names if b[n] < bmax
            ]
            if not cand:
                break
            cand.sort(key=lambda x: x[0], reverse=True)
            n_best = cand[0][1]
            b[n_best] += 1
        else:  # delta < 0
            # -1할 후보 중 (손실/가중치) 최소
            cand = [
                (marg_harm_down(n) / float(W_map[n]), n) for n in names if b[n] > bmin
            ]
            if not cand:
                break
            cand.sort(key=lambda x: x[0])
            n_best = cand[0][1]
            b[n_best] -= 1
        steps += 1
    return b


# -------- (추가) 반올림 밴드 프로젝션 --------
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
        # 드문 엣지: 밴드가 비정상적이면 중앙으로 수렴
        mid = (B_lo + B_hi) // 2
        return project_to_weighted_budget(b_assign, W_map, C_prime_map, mid, bmin, bmax)
    if S < B_lo:
        return project_to_weighted_budget(
            b_assign, W_map, C_prime_map, B_lo, bmin, bmax
        )
    if S > B_hi:
        return project_to_weighted_budget(
            b_assign, W_map, C_prime_map, B_hi, bmin, bmax
        )
    return b_assign


# -----------------------------------------------


def get_initial_seed(C_prime_map, W_map, avg_bits, bmin=2, bmax=4) -> Dict[str, int]:
    names = [n for n in C_prime_map.keys() if n in W_map]
    Cp_arr = np.array([C_prime_map[n] for n in names], dtype=np.float64)
    w_arr = np.array([W_map[n] for n in names], dtype=np.float64)
    if w_arr.sum() == 0:
        print("[경고] 초기 시드 생성 실패: 유효한 가중치 맵이 없습니다.")
        return {}
    # 가용한 정수 제약을 고려해 타깃 합을 스냅
    B_target = target_weighted_sum(float(avg_bits), {n: int(W_map[n]) for n in names})
    # 연속 해 → 정수화
    mu, R_cont, R_clamped, S_cont, L_cont, info = solve_mu_for_budget(
        Cp_arr, w_arr, float(B_target), bmin, bmax
    )
    b_init = np.clip(np.floor(R_clamped + 1e-12), bmin, bmax).astype(np.int64)
    b_int, L_int, S_int = greedy_integer_refine_budget(
        Cp_arr, w_arr, b_init, float(B_target), bmin, bmax
    )
    b_seed = {names[i]: int(b_int[i]) for i in range(len(names))}
    # 안전하게 한 번 더 투영(정확 타깃 기준)
    b_seed = project_to_weighted_budget(
        b_seed,
        {n: int(W_map[n]) for n in names},
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
    c_up = [n for n in layer_names if new_b.get(n, bmin) < bmax]
    c_down = [n for n in layer_names if new_b.get(n, bmin) > bmin]
    if not c_up or not c_down:
        return None
    j = random.choice(c_up)
    k = random.choice(c_down)
    if j == k and (len(c_up) > 1 or len(c_down) > 1):
        return generate_random_neighbor(b_assign, layer_names, bmin, bmax)
    if j == k:
        return None
    new_b[j] = new_b.get(j, bmin) + 1
    new_b[k] = new_b.get(k, bmin) - 1
    return new_b


# ===============================================
# PPL 평가 오케스트레이터
# ===============================================
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
        cache_maxsize: int = 5000,
    ):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.original_state_dict = original_state_dict
        self.eval_input_ids = eval_input_ids.to(self.device)
        self.prebake_root = Path(prebake_root)
        self.eval_seq_len = eval_seq_len
        self.cache_maxsize = int(cache_maxsize)
        self.ppl_cache = OrderedDict()

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
            val = self.ppl_cache.pop(b_tuple)  # LRU 갱신
            self.ppl_cache[b_tuple] = val
            return val

        # 모델 패치
        try:
            for layer_name in self.target_layers:
                bit = bit_assignment.get(layer_name)
                module = self._get_module(layer_name)
                if bit is None or module is None:
                    continue
                safe_name = _safe_name(layer_name)
                file_path = self.prebake_root / f"bit{bit}" / f"{safe_name}.pt"
                if not file_path.exists():
                    print(f"[경고] Pre-baked 파일 없음: {file_path}. 원본 유지.")
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

        self.ppl_cache[b_tuple] = ppl
        if len(self.ppl_cache) > self.cache_maxsize:
            self.ppl_cache.popitem(last=False)  # LRU 제거
        gc.collect()
        torch.cuda.empty_cache()
        return ppl

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


# ===============================================
# 메인
# ===============================================
def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Phase 1: 로드
    print("--- 🔬 Phase 1: 기반 데이터 준비 ---")
    # 주: 기존 스크립트와 호환 위해 device_map 인자로 device를 넘기던 형태 유지
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

    print("프록시 모델(C', W) 로드...")
    C_prime_map, W_map_all = build_c_prime_map(
        args.sens_csv, args.alpha_csv, args.alpha_bit, args.alpha_default
    )

    # Phase 2: 평가기/시드
    print("--- 🌱 Phase 2: 시드 생성 및 평가기 초기화 ---")
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

    # ---- 반올림 밴드 계산 (옵션) ----
    use_band = bool(getattr(args, "use_round_band", False))
    quantum = float(getattr(args, "round_quantum", 0.1))
    if use_band:
        eps = 1e-9  # 상한을 반열린구간으로
        avg_lo = args.avg_bits - 0.5 * quantum  # 예: 2.5 - 0.05 = 2.45
        avg_hi = args.avg_bits + 0.5 * quantum - eps  # 예: 2.5 + 0.05 - ε = 2.55-ε

        g = gcd_list(list(W_map.values())) or 1
        B_lo = int(math.ceil((avg_lo * sum_w) / g) * g)
        B_hi = int(math.floor((avg_hi * sum_w) / g) * g)

        # 안전장치: 밴드가 비정상이면 타깃으로 강제
        if B_lo > B_hi:
            B_lo = B_hi = B_target

        print(
            f"[Budget Band] round→{args.avg_bits:.2f} (quantum={quantum}) → "
            f"허용 가중평균 [{B_lo/sum_w:.6f}, {B_hi/sum_w:.6f}) "
            f"(Σw·b ∈ [{B_lo}, {B_hi}]); "
            f"기준 타깃≈{B_target/sum_w:.6f}"
        )
    else:
        B_lo = B_target
        B_hi = B_target
        print(
            f"[Budget Exact] 목표 가중평균 ≈ {B_target/sum_w:.6f} (Σw·b = {B_target})"
        )

    proxy_loss_calc = lambda b: calculate_proxy_loss(b, C_prime_filtered, args.bmin)

    print(
        f"Step 3 프록시로 *기본* 시드 생성 (Avg Bits 목표: {args.avg_bits} → "
        f"Target Σ w·b = {B_target} / Σw={sum_w}, 목표 가중평균 ≈ {B_target/sum_w:.6f})"
    )
    b_seed = get_initial_seed(
        C_prime_filtered, W_map, args.avg_bits, args.bmin, args.bmax
    )

    if args.init_assign_csv and os.path.exists(args.init_assign_csv):
        print(f"--- 덮어쓰기: {args.init_assign_csv} 에서 시드 로드 ---")
        b_seed_from_csv = load_seed_from_csv(args.init_assign_csv)
        if b_seed_from_csv:
            b_seed.update(b_seed_from_csv)
        else:
            print(f"[경고] {args.init_assign_csv}에서 유효한 시드를 찾지 못함.")

    # 시드 완성/프로젝션 → **밴드 허용 시 밴드로, 아니면 정확예산으로**
    b_seed = ensure_complete_assignment(b_seed, target_layers_list, args.bmin)
    b_seed = project_to_weighted_band(
        b_seed, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax
    )

    # 초기 후보 생성
    initial_candidates = {}
    b_seed_proj = project_to_weighted_band(
        b_seed, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax
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
            neighbor = project_to_weighted_band(
                neighbor, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax
            )
            initial_candidates[tuple(sorted(neighbor.items()))] = proxy_loss_calc(
                neighbor
            )

    top_n_L_b = sorted(initial_candidates.items(), key=lambda x: x[1])[: args.beam_size]

    beam = []
    print(f"--- 초기 PPL 평가 ({args.beam_size}개) ---")
    for b_tuple, l_score in tqdm(top_n_L_b, desc="초기 PPL 평가"):
        b_dict = dict(b_tuple)
        ppl = ppl_eval.evaluate(b_dict)
        beam.append((ppl, l_score, b_dict))
    beam.sort(key=lambda x: x[0])
    wavg0 = weighted_sum_bits(beam[0][2], W_map) / sum_w if sum_w > 0 else 0.0
    print(f"초기 빔 PPL: {beam[0][0]:.4f} | 초기 가중평균={wavg0:.6f}")

    # Phase 3: 수렴까지 반복
    print("--- 🎲 Phase 3: PPL 빔 서치(수렴까지) 시작 ---")
    os.makedirs(args.output_dir, exist_ok=True)
    ppl_curve_csv = os.path.join(args.output_dir, "ppl_curve.csv")
    new_file = not os.path.exists(ppl_curve_csv)
    curve_f = open(ppl_curve_csv, "a", newline="", encoding="utf-8")
    curve_w = csv.writer(curve_f)
    if new_file:
        curve_w.writerow(
            [
                "generation",
                "best_ppl",
                "best_L",
                "weighted_avg_bits",
                "timestamp",
                "note",
            ]
        )
        curve_f.flush()

    start_ts = time.time()
    gen = 0
    no_improve = 0
    stable_bits = 0
    prev_best_ppl = float("inf")
    prev_best_bits = None

    # generations >0 이면 max_generations로 사용
    max_g = (
        args.max_generations
        if args.max_generations > 0
        else (args.generations if args.generations > 0 else 0)
    )

    def time_over():
        return (args.time_limit_sec > 0) and (
            time.time() - start_ts >= args.time_limit_sec
        )

    while True:
        # Expand
        all_candidates = set()
        for _, _, b_assign in beam:
            all_candidates.add(tuple(sorted(b_assign.items())))
            for _ in range(args.expansion_k):
                neighbor = generate_random_neighbor(
                    b_assign, target_layers_list, args.bmin, args.bmax
                )
                if neighbor:
                    neighbor = ensure_complete_assignment(
                        neighbor, target_layers_list, args.bmin
                    )
                    neighbor = project_to_weighted_band(
                        neighbor,
                        W_map,
                        C_prime_filtered,
                        B_lo,
                        B_hi,
                        args.bmin,
                        args.bmax,
                    )
                    all_candidates.add(tuple(sorted(neighbor.items())))

        # Filter (proxy)
        candidate_L_scores = [(proxy_loss_calc(dict(bt)), bt) for bt in all_candidates]
        candidate_L_scores.sort(key=lambda x: x[0])
        finalists = candidate_L_scores[: args.filter_p]

        # Evaluate (PPL)
        new_beam_candidates = []
        for l_score, b_tuple in tqdm(finalists, desc=f"G-{gen} PPL 평가", leave=False):
            b_dict = dict(b_tuple)
            # 안전: **밴드/정확예산** 프로젝션을 재적용
            b_dict = project_to_weighted_band(
                b_dict, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax
            )
            ppl = ppl_eval.evaluate(b_dict)
            new_beam_candidates.append((ppl, l_score, b_dict))

        # Select
        new_beam_candidates.sort(key=lambda x: x[0])
        beam = new_beam_candidates[: args.beam_size]

        best_ppl, best_L, best_bits = beam[0]
        S = weighted_sum_bits(best_bits, W_map)
        wavg = S / sum_w if sum_w > 0 else 0.0

        abs_gain = prev_best_ppl - best_ppl
        rel_gain = (
            abs_gain / prev_best_ppl
            if math.isfinite(prev_best_ppl) and prev_best_ppl > 0
            else float("inf")
        )
        improved = (abs_gain > args.converge_eps) or (rel_gain > args.converge_rel_eps)

        note = "improved" if improved else "no-improve"
        tqdm.write(
            f"--- G-{gen} 완료 | Best PPL: {best_ppl:.4f} (L={best_L:.2e}) | w-avg={wavg:.6f} | {note}"
        )
        try:
            curve_w.writerow(
                [
                    gen,
                    f"{best_ppl:.6f}",
                    f"{best_L:.6e}",
                    f"{wavg:.6f}",
                    int(time.time()),
                    note,
                ]
            )
            curve_f.flush()
        except Exception as e:
            tqdm.write(f"[경고] ppl_curve.csv 기록 실패: {e}")

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        if prev_best_bits is not None and best_bits == prev_best_bits:
            stable_bits += 1
        else:
            stable_bits = 0

        prev_best_ppl = min(prev_best_ppl, best_ppl)
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
            tqdm.write(f"✔️ 수렴/중단: {', '.join(stop_reasons)}")
            break

        gen += 1

    # Phase 4: 결과
    print("\n--- 🏆 Phase 4: 탐색 완료 ---")
    best_ppl, best_L, best_b = beam[0]
    print(f"최종 Best PPL: {best_ppl:.4f}")
    S_final = weighted_sum_bits(best_b, W_map)
    wavg_final = S_final / sum_w if sum_w > 0 else 0.0
    if use_band:
        print(
            f"최종 Σ w·b = {S_final} / Σw={sum_w} → 가중평균={wavg_final:.6f} "
            f"(허용대역≈[{B_lo/sum_w:.6f}, {B_hi/sum_w:.6f}), 기준표기 {args.avg_bits:.2f})"
        )
    else:
        print(
            f"최종 Σ w·b = {S_final} / Σw={sum_w} → 가중평균={wavg_final:.6f} "
            f"(목표≈{B_target/sum_w:.6f})"
        )

    try:
        curve_f.close()
    except Exception:
        pass

    os.makedirs(args.output_dir, exist_ok=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Step 3b - PPL-Direct Monte-Carlo Search (Pre-baked, convergence)"
    )
    # 경로
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--sens_csv", type=str, required=True)
    parser.add_argument("--alpha_csv", type=str, required=True)
    parser.add_argument("--prebake_root", type=str, required=True)
    parser.add_argument("--init_assign_csv", type=str, default=None)
    parser.add_argument(
        "--output_dir", type=str, default="./artifacts/bitmin/step3b_ppl_search"
    )
    parser.add_argument("--gpu_id", type=int, default=0)

    # 시드/프록시
    parser.add_argument("--avg_bits", type=float, default=2.5)
    parser.add_argument("--alpha_bit", type=int, default=3)
    parser.add_argument("--alpha_default", type=float, default=1.0)

    # (추가) 반올림 밴드 옵션
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

    # PPL 평가
    parser.add_argument("--eval_seq_len", type=int, default=2048)

    # 빔 서치
    parser.add_argument("--bmin", type=int, default=2)
    parser.add_argument("--bmax", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--expansion_k", type=int, default=20)
    parser.add_argument("--filter_p", type=int, default=50)

    # 수렴/안전
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
        "--converge_eps", type=float, default=1e-3, help="절대 개선 임계값."
    )
    parser.add_argument(
        "--converge_rel_eps", type=float, default=1e-3, help="상대 개선 임계값."
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="개선 없는 세대 허용치."
    )
    parser.add_argument(
        "--stable_bits_patience",
        type=int,
        default=10,
        help="동일 best bit 반복 허용치.",
    )
    parser.add_argument(
        "--time_limit_sec", type=int, default=0, help="0=무제한, >0이면 시간 제한(초)."
    )
    parser.add_argument(
        "--ppl_cache_max", type=int, default=20000, help="LRU 캐시 최대 엔트리 수."
    )

    args = parser.parse_args()
    main(args)
