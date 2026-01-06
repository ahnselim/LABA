#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3c — Data Collection for Surrogate Model
(Step 3b 기반 수정: 몬테카를로 탐색 과정에서 측정되는 모든 PPL 데이터를 로깅)

목적:
  - Surrogate Neural Net 학습을 위한 (Bit_Config -> PPL) 데이터셋 구축
  - 매 Generation마다 filter_p 개수만큼의 후보를 평가하고, 그 결과를 모두 저장

출력:
  1) static_info.json: 레이어별 C, Alpha, W 정보 (학습 시 Input Feature 구성용)
  2) training_samples.csv: [generation, bit_assign_json, proxy_loss, real_ppl]

사용법:
  CUDA_VISIBLE_DEVICES=1 nohup \
  python neural_net/step3c_ppl_data_collection.py \
    --model_id meta-llama/Llama-3.2-3B \
    --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv --prebake_root ../artifacts/montecarlo/prebake \
    --generations 50 \
    --filter_p 100 \
    --output_dir ../artifacts/surrogate_data > collect_ppl_dataset.log 2>&1 &
"""
import os, gc, csv, json, math, random, argparse, re, time
from pathlib import Path
from typing import Dict, List
from collections import OrderedDict
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# -------------------------------------------------------------------------
# (기존 유틸리티 임포트)
try:
    from RAQ.src.step3_bit_optimization import (
        load_sensitivity_csv,
        load_alpha_csv,
        solve_mu_for_budget,
        greedy_integer_refine_budget,
    )
except ImportError:
    print("오류: step3_bit_optimization.py가 필요합니다.")
    exit(1)
# -------------------------------------------------------------------------


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s) if s else "none")


@torch.no_grad()
def run_live_ppl_eval(model, eval_input_ids, seq_len=2048) -> float:
    model.eval()
    total_nll, total_tok = 0.0, 0
    for i in range(0, eval_input_ids.size(1), seq_len):
        begin, end = i, min(i + seq_len, eval_input_ids.size(1))
        if end - begin <= 1:
            continue
        x = eval_input_ids[:, begin:end]
        out = model(x)
        logits = out.logits
        loss = F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            x[..., 1:].contiguous().view(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tok += end - begin - 1
    return math.exp(total_nll / total_tok) if total_tok > 0 else 0.0


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


def weighted_sum_bits(b_assign, W_map):
    return sum(int(W_map[n]) * int(b) for n, b in b_assign.items() if n in W_map)


def gcd_list(int_list):
    return (
        reduce(math.gcd, int_list)
        if len(int_list) > 1
        else (int_list[0] if int_list else 1)
    )


def target_weighted_sum(avg_bits, W_map):
    sum_w = sum(int(w) for w in W_map.values())
    g = gcd_list([int(w) for w in W_map.values()]) if W_map else 1
    raw = avg_bits * sum_w
    return int(round(raw / g) * g)


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
        print(f"[경고] CSV 로드 실패: {e}", flush=True)
    return seed_map


def project_to_weighted_budget(
    b_assign, W_map, C_prime_map, B_target, bmin, bmax, max_steps=None
):
    """
    Step 3의 greedy_integer_refine_budget 로직을 100% 동일하게 이식.
    - 예산 초과 시: 목표 이하가 될 때까지 가장 효율 낮은 비트 제거
    - 예산 미달 시: 목표 이상이 될 때까지 가장 효율 높은 비트 추가
    - 진동 없음, 무한루프 없음.
    """
    b = dict(b_assign)
    names = [n for n in b.keys() if n in W_map]
    if not names:
        return b

    # 현재 상태 계산
    S_cur = sum(int(W_map[n]) * int(b[n]) for n in names)

    # 1. 예산 초과 시 (줄여야 함) -> 줄여서 B_target 이하가 되는 순간 멈춤
    if S_cur > B_target:
        while S_cur > B_target + 1e-9:  # 부동소수점 오차 고려
            # 줄일 수 있는 후보 탐색 (b > bmin)
            candidates = [n for n in names if b[n] > bmin]
            if not candidates:
                break

            # 손실(증가량)/비용(절감량) 이 가장 작은 레이어 찾기
            # minimize: (L(b-1) - L(b)) / w
            best_node = None
            min_score = float("inf")

            for n in candidates:
                w_val = float(W_map[n])
                cp_val = float(C_prime_map.get(n, 0.0))
                # 비트를 줄일 때 늘어나는 손실 (Delta Harm)
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

    # 2. 예산 미달 시 (늘려야 함) -> 늘려서 B_target 이상이 되는 순간 멈춤
    elif S_cur < B_target - 1e-9:
        while S_cur < B_target - 1e-9:
            # 늘릴 수 있는 후보 탐색 (b < bmax)
            candidates = [n for n in names if b[n] < bmax]
            if not candidates:
                break

            # 이득(감소량)/비용(증가량) 이 가장 큰 레이어 찾기
            # maximize: (L(b) - L(b+1)) / w
            best_node = None
            max_score = -float("inf")

            for n in candidates:
                w_val = float(W_map[n])
                cp_val = float(C_prime_map.get(n, 0.0))
                # 비트를 늘릴 때 줄어드는 손실 (Delta Gain)
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


def get_initial_seed(C_prime_map, W_map, avg_bits, bmin=2, bmax=4):
    names = [n for n in C_prime_map.keys() if n in W_map]
    Cp_arr = np.array([C_prime_map[n] for n in names], dtype=np.float64)
    w_arr = np.array([W_map[n] for n in names], dtype=np.float64)
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


def generate_random_neighbor(b_assign, layer_names, bmin=2, bmax=4):
    new_b = b_assign.copy()
    c_up = [n for n in layer_names if new_b.get(n, bmin) < bmax]
    c_down = [n for n in layer_names if new_b.get(n, bmin) > bmin]
    if not c_up or not c_down:
        return None
    j, k = random.choice(c_up), random.choice(c_down)
    if j == k:
        return None
    new_b[j] += 1
    new_b[k] -= 1
    return new_b


# ===============================================
class PplEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        original_state_dict,
        eval_input_ids,
        prebake_root,
        eval_seq_len,
    ):
        self.model = model
        self.device = model.device
        self.original_state_dict = original_state_dict
        self.eval_input_ids = eval_input_ids.to(self.device)
        self.prebake_root = Path(prebake_root)
        self.eval_seq_len = eval_seq_len
        self.target_layers = []

        bit2_dir = self.prebake_root / "bit2"
        files = list(bit2_dir.glob("*.pt"))
        print(
            f"[PplEvaluator] Pre-bake 폴더 스캔 중... ({len(files)} files)", flush=True
        )

        for f in tqdm(files, desc="Loading Pre-bake Info"):
            try:
                payload = torch.load(f, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(f, map_location="cpu")
            mod_name = payload.get("module")
            if mod_name and f"{mod_name}.weight" in original_state_dict:
                self.target_layers.append(mod_name)
            del payload

        self.target_layers = sorted(list(set(self.target_layers)))
        print(
            f"[PplEvaluator] 유효 대상 레이어 {len(self.target_layers)}개 식별 완료.",
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
        try:
            for layer_name in self.target_layers:
                bit = bit_assignment.get(layer_name)
                module = self._get_module(layer_name)
                if bit is None or module is None:
                    continue

                fpath = self.prebake_root / f"bit{bit}" / f"{_safe_name(layer_name)}.pt"
                if not fpath.exists():
                    continue

                try:
                    payload = torch.load(
                        fpath, map_location=self.device, weights_only=False
                    )
                except TypeError:
                    payload = torch.load(fpath, map_location=self.device)

                Wq, A, B = payload["Wq"], payload["A"], payload["B"]
                W_eff = Wq + (A.to(Wq.dtype) @ B.to(Wq.dtype))
                module.weight.data.copy_(W_eff.to(module.weight.dtype))
                del payload, Wq, A, B, W_eff
        except Exception as e:
            print(f"Eval Error: {e}", flush=True)
            self.restore_original_weights()
            return float("inf")

        ppl = run_live_ppl_eval(self.model, self.eval_input_ids, self.eval_seq_len)
        self.restore_original_weights()
        gc.collect()
        torch.cuda.empty_cache()
        return ppl

    @torch.no_grad()
    def restore_original_weights(self):
        for layer_name in self.target_layers:
            module = self._get_module(layer_name)
            if module:
                module.weight.data.copy_(
                    self.original_state_dict[f"{layer_name}.weight"].to(
                        module.weight.device
                    )
                )


# ===============================================
def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("--- Phase 1: 모델 및 데이터 준비 ---", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    orig_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_ids = tokenizer(
        "\n\n".join(ds["text"]), return_tensors="pt", add_special_tokens=False
    ).input_ids

    print("--- Phase 2: PPL Evaluator 초기화 ---", flush=True)
    C_prime_map, W_map_all = build_c_prime_map(
        args.sens_csv, args.alpha_csv, args.alpha_bit, 1.0
    )

    ppl_eval = PplEvaluator(
        model, tokenizer, orig_sd, eval_ids, args.prebake_root, args.eval_seq_len
    )
    targets = ppl_eval.target_layers

    W_map = {k: int(W_map_all[k]) for k in targets if k in W_map_all}
    C_prime = {k: float(C_prime_map[k]) for k in targets if k in C_prime_map}

    static_info = {
        "layer_names": targets,
        "W_map": W_map,
        "C_prime_map": C_prime,
        "model_id": args.model_id,
        "avg_bits_target": args.avg_bits,
    }
    with open(
        os.path.join(args.output_dir, "static_info.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(static_info, f, indent=2)

    # 3. 예산 및 시드
    sum_w = sum(W_map.values())
    B_target = target_weighted_sum(args.avg_bits, W_map)
    print(f"Target Budget: {B_target} (Avg: {args.avg_bits})", flush=True)

    print("--- [Math] 초기 시드 설정 중... ---", flush=True)
    b_seed = get_initial_seed(C_prime, W_map, args.avg_bits, args.bmin, args.bmax)

    # [추가] CSV 로드 기능
    if args.init_assign_csv and os.path.exists(args.init_assign_csv):
        print(f"--- [Info] CSV에서 시드 로드: {args.init_assign_csv} ---", flush=True)
        csv_seed = load_seed_from_csv(args.init_assign_csv)
        if csv_seed:
            b_seed.update(csv_seed)

    # 누락된 키 채우기
    for t in targets:
        if t not in b_seed:
            b_seed[t] = args.bmin

    print("--- [Math] 시드 예산 투영 (Fast Mode) ---", flush=True)
    b_seed = project_to_weighted_budget(
        b_seed, W_map, C_prime, B_target, args.bmin, args.bmax
    )

    # 4. 데이터 수집 루프
    samples_csv = os.path.join(args.output_dir, "training_samples.csv")
    file_exists = os.path.exists(samples_csv)
    csv_f = open(samples_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    if not file_exists:
        writer.writerow(
            [
                "generation",
                "sample_idx",
                "proxy_loss",
                "measured_ppl",
                "bit_assignment_json",
            ]
        )
        csv_f.flush()

    print("--- [Beam] 초기 이웃 후보군 생성 중... ---", flush=True)
    initial_cands = {}
    seed_proj = project_to_weighted_budget(
        b_seed, W_map, C_prime, B_target, args.bmin, args.bmax
    )
    initial_cands[tuple(sorted(seed_proj.items()))] = calculate_proxy_loss(
        seed_proj, C_prime
    )

    # 이제 빠를 겁니다
    for i in range(args.beam_size * 5):
        nbr = generate_random_neighbor(b_seed, targets, args.bmin, args.bmax)
        if nbr:
            nbr = project_to_weighted_budget(
                nbr, W_map, C_prime, B_target, args.bmin, args.bmax
            )
            initial_cands[tuple(sorted(nbr.items()))] = calculate_proxy_loss(
                nbr, C_prime
            )

    print(
        f"--- [Beam] 초기 생성 완료 ({len(initial_cands)}개). 평가 시작... ---",
        flush=True,
    )
    top_cands = sorted(initial_cands.items(), key=lambda x: x[1])[: args.filter_p]

    beam = []

    for idx, (b_tuple, l_score) in enumerate(tqdm(top_cands, desc="[Gen 0] Measuring")):
        b_dict = dict(b_tuple)
        ppl = ppl_eval.evaluate(b_dict)
        writer.writerow([0, idx, l_score, ppl, json.dumps(b_dict)])
        csv_f.flush()
        os.fsync(csv_f.fileno())
        beam.append((ppl, l_score, b_dict))

    beam.sort(key=lambda x: x[0])
    beam = beam[: args.beam_size]

    for gen in range(1, args.generations + 1):
        candidates = set()
        for _, _, b_parent in beam:
            candidates.add(tuple(sorted(b_parent.items())))
            for _ in range(args.expansion_k):
                nbr = generate_random_neighbor(b_parent, targets, args.bmin, args.bmax)
                if nbr:
                    nbr = project_to_weighted_budget(
                        nbr, W_map, C_prime, B_target, args.bmin, args.bmax
                    )
                    candidates.add(tuple(sorted(nbr.items())))

        cand_list = [(calculate_proxy_loss(dict(bt), C_prime), bt) for bt in candidates]
        cand_list.sort(key=lambda x: x[0])
        finalists = cand_list[: args.filter_p]

        next_beam = []
        pbar = tqdm(
            finalists, desc=f"[Gen {gen}/{args.generations}] Collection", leave=False
        )
        for idx, (l_score, b_tuple) in enumerate(pbar):
            b_dict = dict(b_tuple)
            ppl = ppl_eval.evaluate(b_dict)
            writer.writerow([gen, idx, l_score, ppl, json.dumps(b_dict)])
            csv_f.flush()
            os.fsync(csv_f.fileno())
            next_beam.append((ppl, l_score, b_dict))
            pbar.set_postfix({"curr_ppl": f"{ppl:.2f}"})

        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[: args.beam_size]
        print(f"Gen {gen} Best: {beam[0][0]:.4f}", flush=True)

    csv_f.close()
    print("완료.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--sens_csv", type=str, required=True)
    parser.add_argument("--alpha_csv", type=str, required=True)
    parser.add_argument("--prebake_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./surrogate_data")
    parser.add_argument(
        "--init_assign_csv",
        type=str,
        default=None,
        help="Step3b의 결과 csv가 있다면 로드",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--avg_bits", type=float, default=2.5)
    parser.add_argument("--alpha_bit", type=int, default=3)
    parser.add_argument("--eval_seq_len", type=int, default=2048)
    parser.add_argument("--bmin", type=int, default=2)
    parser.add_argument("--bmax", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--expansion_k", type=int, default=20)
    parser.add_argument("--filter_p", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)

    args = parser.parse_args()
    main(args)
