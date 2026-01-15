"""
Utilities shared by Monte-Carlo mixer scripts.

The functions collected here originally lived in multiple mixer variants.
Having them in one place keeps the implementations in sync and makes it
easy for future scripts to import common helpers.
"""

import csv
import math
import os
import random
import re
from functools import reduce
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from RAQ.proxy_codes.step3_bit_optimization import (
    greedy_integer_refine_budget,
    load_alpha_csv,
    load_sensitivity_csv,
    solve_mu_for_budget,
)

__all__ = [
    "_safe_name",
    "atomic_save_bit_assign_csv",
    "run_live_ppl_eval",
    "load_seed_from_csv",
    "build_c_prime_map",
    "calculate_proxy_loss",
    "weighted_sum_bits",
    "gcd_list",
    "target_weighted_sum",
    "ensure_complete_assignment",
    "project_to_weighted_budget",
    "project_to_weighted_band",
    "get_initial_seed",
    "generate_random_neighbor",
]


def _safe_name(s) -> str:
    if s is None:
        s = "none"
    s = str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def atomic_save_bit_assign_csv(path: str, bits: Dict[str, int]):
    """Persist bit assignments via a temp file to avoid partial writes."""
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

