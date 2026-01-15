#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3b — Surrogate-guided Monte-Carlo Beam Search (Pre-baked, convergence) [Mixer]
============================================================================
요구사항(v2 기반) 유지:
- neighbor 생성(generate_random_neighbor) / budget projection(project_to_weighted_band/budget) 로직 유지
- finalists(filter_p)는 proxy_loss로 1차 필터
- finalists는 surrogate score로 랭킹 후 top-K만 true PPL 측정
- beam 갱신은 "측정된 true PPL" 기준
- surrogate score 캐시(LRU) + true PPL 캐시(LRU) 유지
- init에서도 surrogate 랭킹 후 top-K만 true PPL 측정

주의:
- 이 스크립트는 "Mixer 기반 surrogate" 체크포인트(best.pt)와 config.json을 기대합니다.
  (Transformer로 학습된 ckpt와는 일반적으로 호환되지 않음)

[요청 반영 - 저장 방식 변경]
1) global best 변수 추가
2) ppl_curve.csv에서 num_finalists / num_true_eval 저장 제거
3) ppl_curve.csv 컬럼:
   generation,best_ppl,global_best_ppl,best_L,best_sur,weighted_avg_bits,timestamp,note
4) bit_assign.csv는 "global best가 갱신될 때만" 저장/갱신 (초기 1회 + 갱신 시만)
5) (중요) global best 갱신을 먼저 수행한 뒤, 같은 generation row에 갱신된 global_best_ppl 기록 (lag 제거)

Usage 예시:
CUDA_VISIBLE_DEVICES=1 nohup \
python montecarlo/mixer/step3b_in3_ppl_mc_converge_round_mixer_v3.py \
  --model_id meta-llama/Llama-3.2-3B \
  --gpu_id 0 \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --prebake_root ../artifacts/montecarlo/prebake \
  --avg_bits 2.50 \
  --use_round_band --round_quantum 0.1 \
  --beam_size 10 --expansion_k 20 --filter_p 80 \
  --true_eval_topk 10 \
  --surrogate_ckpt ../artifacts/surrogate_checkpoint_input3_mixer/best.pt \
  --surrogate_config ../artifacts/surrogate_checkpoint_input3_mixer/config.json \
  --patience 12 --stable_bits_patience 12 \
  --output_dir ../artifacts/montecarlo/step3b_surrogate_roundband_mixer \
  > ./log/run_mixer_3b_check.log 2>&1 &
"""

import os, gc, csv, json, math, random, argparse, re, time, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from functools import reduce

# Ensure repo-relative modules can be imported
_FILE_PATH = Path(__file__).resolve()
_PARENTS = _FILE_PATH.parents
_SRC_ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _FILE_PATH.parent
_REPO_ROOT = _PARENTS[2] if len(_PARENTS) > 2 else _SRC_ROOT
_WORKSPACE_ROOT = _PARENTS[3] if len(_PARENTS) > 3 else _REPO_ROOT
_PROJECT_ROOT = (
    _WORKSPACE_ROOT.parent
    if _WORKSPACE_ROOT != _WORKSPACE_ROOT.parent
    else _WORKSPACE_ROOT
)
for _path in (_SRC_ROOT, _REPO_ROOT, _WORKSPACE_ROOT, _PROJECT_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# -----------------------------------------------
# Step 2/4에서 가져오기 (경로/네이밍)
# -----------------------------------------------
try:
    from step2_alpha_estimation import _canonical_dataset_name
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
        solve_mu_for_budget,
        greedy_integer_refine_budget,
    )
except ImportError:
    print("오류: step3_bit_optimization.py가 같은 디렉토리에 필요합니다.")
    exit(1)


# =========================================================
# Helper
# =========================================================
def _safe_name(s) -> str:
    if s is None:
        s = "none"
    s = str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def atomic_save_bit_assign_csv(path: str, bits: Dict[str, int]):
    """global best 갱신 시에만 호출."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer_name", "R_int"])
        for name, bit in sorted(bits.items()):
            w.writerow([name, int(bit)])
    os.replace(tmp, path)


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
        print(f"[경고] 시드 CSV 로드 실패: {e}", flush=True)
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
    sum_w = sum(int(w) for w in W_map.values())
    g = gcd_list([int(w) for w in W_map.values()]) if W_map else 1
    raw = avg_bits * sum_w
    return int(round(raw / g) * g)


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
    b = dict(b_assign)
    names = [n for n in b.keys() if n in W_map]
    if not names:
        return b

    def marg_gain_up(n):
        b0 = b[n]
        if b0 >= bmax:
            return -float("inf")
        cp = float(C_prime_map.get(n, 0.0))
        return cp * ((2.0 ** (-2.0 * b0)) - (2.0 ** (-2.0 * (b0 + 1))))

    def marg_harm_down(n):
        b0 = b[n]
        if b0 <= bmin:
            return float("inf")
        cp = float(C_prime_map.get(n, 0.0))
        return cp * ((2.0 ** (-2.0 * (b0 - 1))) - (2.0 ** (-2.0 * b0)))

    g = gcd_list([int(W_map[n]) for n in names])
    if B_target % g != 0:
        B_target = int(round(B_target / g) * g)

    steps = 0
    while steps < max_steps:
        S = weighted_sum_bits(b, W_map)
        delta = B_target - S
        if delta == 0:
            break

        if delta > 0:
            cand = [
                (marg_gain_up(n) / float(W_map[n]), n) for n in names if b[n] < bmax
            ]
            if not cand:
                break
            cand.sort(key=lambda x: x[0], reverse=True)
            b[cand[0][1]] += 1
        else:
            cand = [
                (marg_harm_down(n) / float(W_map[n]), n) for n in names if b[n] > bmin
            ]
            if not cand:
                break
            cand.sort(key=lambda x: x[0])
            b[cand[0][1]] -= 1

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
    if not W_map:
        return b_assign
    S = weighted_sum_bits(b_assign, W_map)
    if B_lo > B_hi:
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


def get_initial_seed(C_prime_map, W_map, avg_bits, bmin=2, bmax=4) -> Dict[str, int]:
    names = [n for n in C_prime_map.keys() if n in W_map]
    Cp_arr = np.array([C_prime_map[n] for n in names], dtype=np.float64)
    w_arr = np.array([W_map[n] for n in names], dtype=np.float64)
    if w_arr.sum() == 0:
        print("[경고] 초기 시드 생성 실패: 유효한 가중치 맵이 없습니다.", flush=True)
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
) -> Optional[Dict[str, int]]:
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


# =========================================================
# Mixer Surrogate (input_3 호환: [logC, logW, logit(alpha(bit))])
#  - train_brp_pairwise_surrogate_mixer_input3.py 와 "동일한" 모듈 구조/이름
# =========================================================

# ---- Alpha parsing helper (trainer와 동일) ----
from typing import Any, Union, Tuple

AlphaSpec = Union[float, int, Dict[str, float], Dict[int, float], List[float], Tuple[float, float, float]]

def parse_alpha_per_bit(am: Optional[AlphaSpec]) -> Tuple[float, float, float]:
    """
    alpha_map[layer] 형태가 무엇이든 (a2,a3,a4)로 통일.
      - dict: {"2":a2,"3":a3,"4":a4} 또는 {2:a2,3:a3,4:a4}
      - float/int: a2=a3=a4=value
      - list/tuple(len=3): [a2,a3,a4]
      - None: 1,1,1
    """
    if am is None:
        return 1.0, 1.0, 1.0
    if isinstance(am, (float, int)):
        v = float(am)
        return v, v, v
    if isinstance(am, (list, tuple)) and len(am) == 3:
        return float(am[0]), float(am[1]), float(am[2])
    if isinstance(am, dict):
        a2 = float(am.get("2", am.get(2, 1.0)))
        a3 = float(am.get("3", am.get(3, 1.0)))
        a4 = float(am.get("4", am.get(4, 1.0)))
        return a2, a3, a4
    return 1.0, 1.0, 1.0


class MixerBlock(nn.Module):
    def __init__(self, L: int, d_model: int, token_mlp_dim: int, channel_mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mlp = nn.Sequential(
            nn.Linear(L, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, L),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, channel_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)          # (B,d,L)
        y = self.token_mlp(y).transpose(1, 2)      # (B,L,d)
        x = x + y
        y = self.channel_mlp(self.norm2(x))
        x = x + y
        return x


class BitSequenceEncoderInput3(nn.Module):
    """
    input_3:
      per-layer numeric feature = [logC, logW, logit(alpha_sel(bit))]
    """
    def __init__(
        self,
        L: int,
        d_model: int = 128,
        bit_emb_dim: int = 32,
        nlayers: int = 2,
        ff_dim: int = 256,
        token_mlp_dim: int = 128,
        dropout: float = 0.1,
        use_proxy: bool = True,
    ):
        super().__init__()
        self.L = L
        self.use_proxy = use_proxy

        # bits {2,3,4} -> idx {0,1,2}
        self.bit_emb = nn.Embedding(3, bit_emb_dim)

        # numeric features: [logC, logW, logit(alpha_sel)]
        self.num_proj = nn.Linear(3, bit_emb_dim)

        self.in_proj = nn.Linear(bit_emb_dim * 2, d_model)
        self.pos_emb = nn.Embedding(L, d_model)
        self.dropout = nn.Dropout(dropout)

        self.mixer = nn.Sequential(
            *[
                MixerBlock(L=L, d_model=d_model, token_mlp_dim=token_mlp_dim, channel_mlp_dim=ff_dim, dropout=dropout)
                for _ in range(nlayers)
            ]
        )

        head_in = d_model + (1 if use_proxy else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        bits: torch.Tensor,                 # (B,L) in {2,3,4}
        C_log: torch.Tensor,                # (L,) normalized
        W_log: torch.Tensor,                # (L,) normalized
        alpha_logit_table: torch.Tensor,    # (L,3) normalized
        proxy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = bits.shape
        assert L == self.L

        bit_idx = torch.clamp(bits - 2, 0, 2)                    # (B,L)
        e_bit = self.bit_emb(bit_idx)                            # (B,L,bit_emb_dim)

        # alpha gather: (B,L,1)
        a_tbl = alpha_logit_table.view(1, L, 3).expand(B, L, 3)
        a_sel = torch.gather(a_tbl, 2, bit_idx.unsqueeze(-1))     # (B,L,1)

        C = C_log.view(1, L, 1).expand(B, L, 1)
        W = W_log.view(1, L, 1).expand(B, L, 1)
        x_num = torch.cat([C, W, a_sel], dim=-1)                  # (B,L,3)
        e_num = self.num_proj(x_num)                              # (B,L,bit_emb_dim)

        x = self.in_proj(torch.cat([e_bit, e_num], dim=-1))       # (B,L,d_model)
        pos = torch.arange(L, device=bits.device).view(1, L)
        x = x + self.pos_emb(pos)

        x = self.mixer(self.dropout(x))
        x_pool = x.mean(dim=1)

        if self.use_proxy:
            if proxy is None:
                proxy = torch.zeros((B, 1), device=bits.device, dtype=x_pool.dtype)
            x_pool = torch.cat([x_pool, proxy], dim=-1)

        return self.head(x_pool)                                  # (B,1) lower is better


class BRPPairwiseSurrogate(nn.Module):
    def __init__(self, encoder: nn.Module, tau_pair: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.tau_pair = float(tau_pair)

    def score_single(
        self,
        bits: torch.Tensor,
        C_log: torch.Tensor,
        W_log: torch.Tensor,
        alpha_logit_table: torch.Tensor,
        proxy: Optional[torch.Tensor] = None,
    ):
        return self.encoder(bits, C_log=C_log, W_log=W_log, alpha_logit_table=alpha_logit_table, proxy=proxy)


class SurrogateScorer:
    def __init__(self, ckpt_path: str, config_path: str, device: torch.device, bmin: int = 2):
        self.device = device
        self.bmin = int(bmin)

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model_type = str(cfg.get("model_type", "")).lower()
        if ("mixer" not in model_type) or ("input3" not in model_type):
            raise ValueError(
                f"[Surrogate] input3 Mixer ckpt 전용입니다. "
                f"config.model_type={cfg.get('model_type')} 를 확인하세요."
            )

        self.layer_names: List[str] = cfg["layer_names"]
        self.L = int(cfg["L"])
        h = cfg["hparams"]
        self.use_proxy = not bool(h.get("no_proxy", False))

        static_json_path = cfg.get("json_path", None)
        if static_json_path is None:
            raise ValueError("[Surrogate] config.json에 json_path가 없습니다.")
        if not os.path.exists(static_json_path):
            base = os.path.dirname(os.path.abspath(config_path))
            cand = os.path.join(base, static_json_path)
            if os.path.exists(cand):
                static_json_path = cand
            else:
                raise FileNotFoundError(f"[Surrogate] static_info.json not found: {static_json_path}")

        with open(static_json_path, "r", encoding="utf-8") as f:
            static_info = json.load(f)

        # input3 static: C_map, W_map, alpha_map
        C_map: Dict[str, float] = static_info.get("C_map", None)
        if C_map is None:
            # 혹시 예전 포맷이면 fallback
            C_map = static_info.get("C_prime_map", {})
            print("[Surrogate][Warn] static_info에 C_map이 없어 C_prime_map을 C_map으로 사용합니다.", flush=True)

        W_map: Dict[str, int] = static_info.get("W_map", None)
        if W_map is None:
            raise ValueError("[Surrogate] static_info.json missing W_map")

        alpha_map: Dict[str, Any] = static_info.get("alpha_map", None)
        if alpha_map is None:
            alpha_map = {}
            print("[Surrogate][Warn] static_info에 alpha_map이 없어 alpha=1.0으로 처리합니다.", flush=True)

        # ----- build raw arrays -----
        C_log = np.zeros((self.L,), dtype=np.float32)
        W_log = np.zeros((self.L,), dtype=np.float32)
        alpha_tbl = np.zeros((self.L, 3), dtype=np.float32)  # bit2/3/4

        for i, ln in enumerate(self.layer_names):
            c = float(C_map.get(ln, 0.0))
            w = float(W_map.get(ln, 1.0))
            C_log[i] = math.log(max(c, 1e-30))
            W_log[i] = math.log(max(w, 1.0))

            a2, a3, a4 = parse_alpha_per_bit(alpha_map.get(ln, None))
            alpha_tbl[i, 0] = a2
            alpha_tbl[i, 1] = a3
            alpha_tbl[i, 2] = a4

        # ----- normalize exactly like trainer config -----
        norm = cfg.get("norm", {})

        C_mu = float(norm.get("C_log_mu", float(C_log.mean())))
        C_sd = float(norm.get("C_log_sd", float(C_log.std() + 1e-6)))
        W_mu = float(norm.get("W_log_mu", float(W_log.mean())))
        W_sd = float(norm.get("W_log_sd", float(W_log.std() + 1e-6)))

        C_log_n = (C_log - C_mu) / (C_sd + 1e-12)
        W_log_n = (W_log - W_mu) / (W_sd + 1e-12)

        # alpha: logit then (alpha_logit - mu)/sd (trainer와 동일)
        eps = 1e-6
        a = np.clip(alpha_tbl, eps, 1.0 - eps).astype(np.float32)
        alpha_logit = np.log(a / (1.0 - a)).astype(np.float32)

        A_mu = float(norm.get("alpha_logit_mu", float(alpha_logit.reshape(-1).mean())))
        A_sd = float(norm.get("alpha_logit_sd", float(alpha_logit.reshape(-1).std() + 1e-6)))
        alpha_logit_n = (alpha_logit - A_mu) / (A_sd + 1e-12)

        self.C_log_t = torch.tensor(C_log_n, dtype=torch.float32, device=device)
        self.W_log_t = torch.tensor(W_log_n, dtype=torch.float32, device=device)
        self.alpha_logit_table_t = torch.tensor(alpha_logit_n, dtype=torch.float32, device=device)  # (L,3)

        encoder = BitSequenceEncoderInput3(
            L=self.L,
            d_model=int(h.get("d_model", 128)),
            bit_emb_dim=int(h.get("bit_emb_dim", 32)),
            nlayers=int(h.get("nlayers", 2)),
            ff_dim=int(h.get("ff_dim", 256)),
            token_mlp_dim=int(h.get("token_mlp_dim", 128)),
            dropout=float(h.get("dropout", 0.1)),
            use_proxy=self.use_proxy,
        )
        self.model = BRPPairwiseSurrogate(encoder=encoder, tau_pair=float(h.get("tau_pair", 1.0))).to(device)
        self.model.eval()

        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=True)

        self.cache = OrderedDict()
        self.cache_max = int(h.get("score_cache_max", 200000))
        print(
            f"[Surrogate] loaded (Mixer input3): L={self.L}, use_proxy={self.use_proxy}, cache_max={self.cache_max}",
            flush=True,
        )

    def _bits_to_vec(self, b_assign: Dict[str, int]) -> List[int]:
        return [int(b_assign.get(ln, self.bmin)) for ln in self.layer_names]

    @torch.no_grad()
    def score_batch(
        self,
        assigns: List[Dict[str, int]],
        proxy_vals: Optional[List[float]] = None,
        batch: int = 1024,
    ) -> np.ndarray:
        n = len(assigns)
        if n == 0:
            return np.zeros((0,), dtype=np.float64)

        scores = np.empty((n,), dtype=np.float64)
        todo_idx, todo_bits, todo_proxy = [], [], []

        for i, b in enumerate(assigns):
            bitvec = self._bits_to_vec(b)
            key = tuple(bitvec)
            if key in self.cache:
                v = self.cache.pop(key)
                self.cache[key] = v
                scores[i] = float(v)
            else:
                todo_idx.append(i)
                todo_bits.append(bitvec)
                todo_proxy.append(0.0 if proxy_vals is None else float(proxy_vals[i]))

        if todo_idx:
            bits_np = np.asarray(todo_bits, dtype=np.int64)
            proxy_np = np.asarray(todo_proxy, dtype=np.float32).reshape(-1, 1)

            for s in range(0, len(todo_idx), batch):
                e = min(len(todo_idx), s + batch)
                b_t = torch.tensor(bits_np[s:e], dtype=torch.long, device=self.device)
                p_t = torch.tensor(proxy_np[s:e], dtype=torch.float32, device=self.device) if self.use_proxy else None

                out = self.model.score_single(
                    b_t,
                    C_log=self.C_log_t,
                    W_log=self.W_log_t,
                    alpha_logit_table=self.alpha_logit_table_t,
                    proxy=p_t,
                ).squeeze(-1)
                out_np = out.detach().cpu().numpy().astype(np.float64)

                for k_local, idx in enumerate(todo_idx[s:e]):
                    v = float(out_np[k_local])
                    scores[idx] = v
                    key = tuple(todo_bits[s + k_local])
                    self.cache[key] = v
                    if len(self.cache) > self.cache_max:
                        self.cache.popitem(last=False)

        return scores



# =========================================================
# PPL Evaluator (그대로 유지)
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
        self.eval_seq_len = eval_seq_len
        self.cache_maxsize = int(cache_maxsize)
        self.ppl_cache = OrderedDict()

        self.target_layers = []
        bit2_dir = self.prebake_root / "bit2"
        if not bit2_dir.exists():
            raise FileNotFoundError(f"Pre-bake 디렉토리를 찾을 수 없습니다: {bit2_dir}")
        print(f"[PplEvaluator] {bit2_dir} 스캔하여 대상 레이어 찾는 중...", flush=True)
        for f in bit2_dir.glob("*.pt"):
            try:
                payload = torch.load(f, map_location="cpu")
                module_name = payload.get("module")
                if module_name and f"{module_name}.weight" in original_state_dict:
                    self.target_layers.append(module_name)
                del payload
            except Exception as e:
                print(f"[경고] {f} 로드 실패: {e}", flush=True)
        self.target_layers = sorted(list(set(self.target_layers)))
        if not self.target_layers:
            raise ValueError(
                f"Pre-bake 디렉토리({bit2_dir})에서 유효한 레이어를 찾지 못했습니다."
            )
        print(
            f"[PplEvaluator] 탐색 대상 레이어 {len(self.target_layers)}개 확인.",
            flush=True,
        )

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
            val = self.ppl_cache.pop(b_tuple)
            self.ppl_cache[b_tuple] = val
            return val

        try:
            for layer_name in self.target_layers:
                bit = bit_assignment.get(layer_name)
                module = self._get_module(layer_name)
                if bit is None or module is None:
                    continue
                safe_name = _safe_name(layer_name)
                file_path = self.prebake_root / f"bit{bit}" / f"{safe_name}.pt"
                if not file_path.exists():
                    print(
                        f"[경고] Pre-baked 파일 없음: {file_path}. 원본 유지.",
                        flush=True,
                    )
                    continue
                payload = torch.load(file_path, map_location=self.device)
                Wq, A, B = payload["Wq"], payload["A"], payload["B"]
                compute_dtype = Wq.dtype
                W_eff = Wq + (A.to(compute_dtype) @ B.to(compute_dtype))
                module.weight.data.copy_(W_eff.to(module.weight.dtype))
                del payload, Wq, A, B, W_eff
        except Exception as e:
            print(f"!! PPL 평가 중 오류: {e}", flush=True)
            self.restore_original_weights()
            gc.collect()
            torch.cuda.empty_cache()
            return float("inf")

        ppl = run_live_ppl_eval(self.model, self.eval_input_ids, self.eval_seq_len)
        self.restore_original_weights()

        self.ppl_cache[b_tuple] = ppl
        if len(self.ppl_cache) > self.cache_maxsize:
            self.ppl_cache.popitem(last=False)
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


# =========================================================
# Main
# =========================================================
def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    print("--- 🔬 Phase 1: 기반 데이터 준비 ---", flush=True)

    # device_map 안전 처리
    if torch.cuda.is_available():
        device_map = {"": int(args.gpu_id)}
        torch_dtype = torch.float16
    else:
        device_map = None
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    original_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    print("PPL 평가용 데이터셋 로드 (wikitext-2-raw-v1)...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    eval_input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids

    print("프록시 모델(C', W) 로드...", flush=True)
    C_prime_map, W_map_all = build_c_prime_map(
        args.sens_csv, args.alpha_csv, args.alpha_bit, args.alpha_default
    )

    print("--- 🌱 Phase 2: 시드 생성 및 평가기/서로게이트 초기화 ---", flush=True)
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

    surrogate_device = (
        torch.device(args.surrogate_device)
        if args.surrogate_device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    surrogate = SurrogateScorer(
        ckpt_path=args.surrogate_ckpt,
        config_path=args.surrogate_config,
        device=surrogate_device,
        bmin=args.bmin,
    )

    # ---- 반올림 밴드 계산 (옵션) ----
    use_band = bool(getattr(args, "use_round_band", False))
    quantum = float(getattr(args, "round_quantum", 0.1))
    if use_band:
        eps = 1e-9
        avg_lo = args.avg_bits - 0.5 * quantum
        avg_hi = args.avg_bits + 0.5 * quantum - eps

        g = gcd_list(list(W_map.values())) or 1
        B_lo = int(math.ceil((avg_lo * sum_w) / g) * g)
        B_hi = int(math.floor((avg_hi * sum_w) / g) * g)
        if B_lo > B_hi:
            B_lo = B_hi = B_target

        print(
            f"[Budget Band] round→{args.avg_bits:.2f} (quantum={quantum}) → "
            f"허용 가중평균 [{B_lo/sum_w:.6f}, {B_hi/sum_w:.6f}) "
            f"(Σw·b ∈ [{B_lo}, {B_hi}]); 기준 타깃≈{B_target/sum_w:.6f}",
            flush=True,
        )
    else:
        B_lo = B_target
        B_hi = B_target
        print(
            f"[Budget Exact] 목표 가중평균 ≈ {B_target/sum_w:.6f} (Σw·b = {B_target})",
            flush=True,
        )

    proxy_loss_calc = lambda b: calculate_proxy_loss(b, C_prime_filtered, args.bmin)

    print(
        f"Step 3 프록시로 *기본* 시드 생성 (Avg Bits 목표: {args.avg_bits} → "
        f"Target Σ w·b = {B_target} / Σw={sum_w}, 목표 가중평균 ≈ {B_target/sum_w:.6f})",
        flush=True,
    )
    b_seed = get_initial_seed(
        C_prime_filtered, W_map, args.avg_bits, args.bmin, args.bmax
    )

    if args.init_assign_csv and os.path.exists(args.init_assign_csv):
        print(f"--- 덮어쓰기: {args.init_assign_csv} 에서 시드 로드 ---", flush=True)
        b_seed_from_csv = load_seed_from_csv(args.init_assign_csv)
        if b_seed_from_csv:
            b_seed.update(b_seed_from_csv)
        else:
            print(
                f"[경고] {args.init_assign_csv}에서 유효한 시드를 찾지 못함.",
                flush=True,
            )

    b_seed = ensure_complete_assignment(b_seed, target_layers_list, args.bmin)
    b_seed = project_to_weighted_band(
        b_seed, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax
    )

    # -------------------------------------------------------
    # 초기 후보 생성 -> surrogate topK만 true PPL
    # -------------------------------------------------------
    initial_candidates: Dict[Tuple[Tuple[str, int], ...], float] = {}
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

    init_items = list(initial_candidates.items())
    init_assigns = [dict(bt) for (bt, _) in init_items]
    init_proxys = [float(Lv) for (_, Lv) in init_items]
    init_scores = surrogate.score_batch(
        init_assigns, proxy_vals=init_proxys, batch=args.surrogate_batch
    )

    K0 = max(int(args.true_eval_topk), int(args.beam_size))
    idx_sorted = np.argsort(init_scores)[: min(K0, len(init_items))].tolist()

    beam = []
    print(
        f"--- 초기 true PPL 평가 (surrogate top-{min(K0,len(init_items))}) ---",
        flush=True,
    )
    for idx in tqdm(idx_sorted, desc="초기 PPL 평가"):
        bt, l_score = init_items[idx]
        b_dict = dict(bt)
        ppl = ppl_eval.evaluate(b_dict)
        beam.append((ppl, float(l_score), b_dict, float(init_scores[idx])))

    if not beam:
        raise RuntimeError(
            "초기 beam이 비었습니다. (후보 생성/프로젝션/레이어 매칭을 확인)"
        )

    beam.sort(key=lambda x: x[0])
    beam = beam[: args.beam_size]

    wavg0 = weighted_sum_bits(beam[0][2], W_map) / sum_w if sum_w > 0 else 0.0
    print(
        f"초기 빔 Best PPL: {beam[0][0]:.4f} | w-avg={wavg0:.6f} | sur={beam[0][3]:.6f}",
        flush=True,
    )

    # -------------------------------------------------------
    # Global best tracker (NEW)
    # -------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    global_bit_csv = os.path.join(args.output_dir, "bit_assign.csv")

    global_best_ppl, global_best_L, global_best_bits, global_best_sur = beam[0]
    global_best_gen = 0
    global_best_ts = int(time.time())

    atomic_save_bit_assign_csv(global_bit_csv, global_best_bits)
    print(
        f"[GlobalBest] init saved: {global_bit_csv} (ppl={global_best_ppl:.6f})",
        flush=True,
    )

    # -------------------------------------------------------
    # Phase 3: 수렴까지 반복
    # -------------------------------------------------------
    print("--- 🎲 Phase 3: Surrogate-guided 빔 서치(수렴까지) 시작 ---", flush=True)
    ppl_curve_csv = os.path.join(args.output_dir, "ppl_curve.csv")
    new_file = not os.path.exists(ppl_curve_csv)
    curve_f = open(ppl_curve_csv, "a", newline="", encoding="utf-8")
    curve_w = csv.writer(curve_f)
    if new_file:
        curve_w.writerow(
            [
                "generation",
                "best_ppl",
                "global_best_ppl",
                "best_L",
                "best_sur",
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
    prev_best_bits = None

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
        for _, _, b_assign, _ in beam:
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

        # Surrogate rank
        fin_assigns = [dict(bt) for (L, bt) in finalists]
        fin_proxys = [float(L) for (L, bt) in finalists]
        fin_scores = surrogate.score_batch(
            fin_assigns, proxy_vals=fin_proxys, batch=args.surrogate_batch
        )

        K_true = max(int(args.true_eval_topk), int(args.beam_size))
        top_idx = np.argsort(fin_scores)[: min(K_true, len(finalists))].tolist()

        # True eval only surrogate top-K
        new_beam_candidates = []
        for idx in tqdm(
            top_idx, desc=f"G-{gen} true PPL (sur-top{len(top_idx)})", leave=False
        ):
            l_score, b_tuple = finalists[idx]
            b_dict = dict(b_tuple)
            b_dict = project_to_weighted_band(
                b_dict, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax
            )
            ppl = ppl_eval.evaluate(b_dict)
            new_beam_candidates.append(
                (ppl, float(l_score), b_dict, float(fin_scores[idx]))
            )

        new_beam_candidates.sort(key=lambda x: x[0])
        if not new_beam_candidates:
            print(
                "[경고] 이번 generation에서 true PPL 평가 후보가 비었습니다. 종료합니다.",
                flush=True,
            )
            break

        beam = new_beam_candidates[: args.beam_size]
        best_ppl, best_L, best_bits, best_sur = beam[0]

        S = weighted_sum_bits(best_bits, W_map)
        wavg = S / sum_w if sum_w > 0 else 0.0

        # -------------------------------
        # Global-best update (IMPORTANT: update first, then log/write CSV)
        # -------------------------------
        if math.isfinite(global_best_ppl) and global_best_ppl > 0:
            abs_gain_g = global_best_ppl - best_ppl
            rel_gain_g = abs_gain_g / global_best_ppl
        else:
            abs_gain_g = float("inf")
            rel_gain_g = float("inf")

        global_improved = (abs_gain_g > args.converge_eps) or (
            rel_gain_g > args.converge_rel_eps
        )

        if global_improved:
            global_best_ppl, global_best_L, global_best_bits, global_best_sur = (
                best_ppl,
                best_L,
                best_bits,
                best_sur,
            )
            global_best_gen = gen
            global_best_ts = int(time.time())

            atomic_save_bit_assign_csv(global_bit_csv, global_best_bits)
            tqdm.write(
                f"[GlobalBest] UPDATED @G-{gen} ppl={global_best_ppl:.6f} saved={global_bit_csv}"
            )

            no_improve = 0
            note = "global-improved"
        else:
            no_improve += 1
            note = "no-improve"

        tqdm.write(
            f"--- G-{gen} 완료 | Best PPL: {best_ppl:.4f} | sur={best_sur:.6f} | L={best_L:.2e} | "
            f"w-avg={wavg:.6f} | {note} | GlobalBest: {global_best_ppl:.4f}"
        )

        # CSV: global_best_ppl은 '갱신 후' 값으로 기록 (lag 제거)
        try:
            curve_w.writerow(
                [
                    gen,
                    f"{best_ppl:.6f}",
                    f"{global_best_ppl:.6f}",
                    f"{best_L:.6e}",
                    f"{best_sur:.6f}",
                    f"{wavg:.6f}",
                    int(time.time()),
                    note,
                ]
            )
            curve_f.flush()
        except Exception as e:
            tqdm.write(f"[경고] ppl_curve.csv 기록 실패: {e}")

        # stable-bits tracking (세대 best 기준)
        if prev_best_bits is not None and best_bits == prev_best_bits:
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
            tqdm.write(f"✔️ 수렴/중단: {', '.join(stop_reasons)}")
            break

        gen += 1

    # -------------------------------------------------------
    # Phase 4: 결과 (global best 기준)
    # -------------------------------------------------------
    print("\n--- 🏆 Phase 4: 탐색 완료 ---", flush=True)
    print(
        f"최종 Global Best PPL: {global_best_ppl:.4f} | best_sur={global_best_sur:.6f} | found@G-{global_best_gen}",
        flush=True,
    )
    print(f"최종 할당 파일(글로벌 베스트): {global_bit_csv}", flush=True)

    best_b = global_best_bits
    S_final = weighted_sum_bits(best_b, W_map)
    wavg_final = S_final / sum_w if sum_w > 0 else 0.0

    if use_band:
        print(
            f"최종 Σ w·b = {S_final} / Σw={sum_w} → 가중평균={wavg_final:.6f} "
            f"(허용대역≈[{B_lo/sum_w:.6f}, {B_hi/sum_w:.6f}), 기준표기 {args.avg_bits:.2f})",
            flush=True,
        )
    else:
        print(
            f"최종 Σ w·b = {S_final} / Σw={sum_w} → 가중평균={wavg_final:.6f} "
            f"(목표≈{B_target/sum_w:.6f})",
            flush=True,
        )

    try:
        curve_f.close()
    except Exception:
        pass

    cache_path = os.path.join(args.output_dir, "ppl_cache.json")
    str_cache = {str(k): v for k, v in ppl_eval.ppl_cache.items()}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(str_cache, f, indent=2)
    print(f"True PPL 캐시 저장: {cache_path}", flush=True)

    sur_cache_path = os.path.join(args.output_dir, "surrogate_cache.json")
    with open(sur_cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in surrogate.cache.items()}, f, indent=2)
    print(f"Surrogate 캐시 저장: {sur_cache_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Step 3b - Surrogate-guided MC Search (Mixer v2 base)"
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
        default="./artifacts/bitmin/step3b_surrogate_search_mixer",
    )
    parser.add_argument("--gpu_id", type=int, default=0)

    # seed/proxy
    parser.add_argument("--avg_bits", type=float, default=2.5)
    parser.add_argument("--alpha_bit", type=int, default=3)
    parser.add_argument("--alpha_default", type=float, default=1.0)

    # round band
    parser.add_argument(
        "--use_round_band", action="store_true", help="avg_bits 반올림 밴드 허용"
    )
    parser.add_argument(
        "--round_quantum", type=float, default=0.1, help="반올림 자릿수 폭"
    )

    # ppl eval
    parser.add_argument("--eval_seq_len", type=int, default=2048)
    parser.add_argument(
        "--ppl_cache_max",
        type=int,
        default=20000,
        help="true PPL LRU 캐시 최대 엔트리 수",
    )

    # beam search
    parser.add_argument("--bmin", type=int, default=2)
    parser.add_argument("--bmax", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--expansion_k", type=int, default=20)
    parser.add_argument("--filter_p", type=int, default=80)

    # surrogate (Mixer)
    parser.add_argument(
        "--surrogate_ckpt",
        type=str,
        required=True,
        help="train_brp_pairwise_surrogate_mixer best.pt",
    )
    parser.add_argument(
        "--surrogate_config",
        type=str,
        required=True,
        help="train output_dir/config.json",
    )
    parser.add_argument(
        "--true_eval_topk", type=int, default=10, help="surrogate top-K만 true PPL 측정"
    )
    parser.add_argument(
        "--surrogate_batch", type=int, default=1024, help="surrogate batch inference"
    )
    parser.add_argument(
        "--surrogate_device",
        type=str,
        default="",
        help="예: cuda:0 / cpu (빈 문자열이면 자동)",
    )

    # convergence/safety
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help="(하위호환) >0이면 max_generations로 사용",
    )
    parser.add_argument(
        "--max_generations", type=int, default=0, help="0이면 수렴 조건까지 진행"
    )
    parser.add_argument(
        "--converge_eps", type=float, default=1e-3, help="절대 개선 임계값"
    )
    parser.add_argument(
        "--converge_rel_eps", type=float, default=1e-3, help="상대 개선 임계값"
    )
    parser.add_argument(
        "--patience", type=int, default=12, help="개선 없는 세대 허용치"
    )
    parser.add_argument(
        "--stable_bits_patience", type=int, default=12, help="동일 best bit 반복 허용치"
    )
    parser.add_argument(
        "--time_limit_sec", type=int, default=0, help="0=무제한, >0이면 시간 제한(초)"
    )

    # misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
