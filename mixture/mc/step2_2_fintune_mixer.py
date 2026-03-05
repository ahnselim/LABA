#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3b — v10_2: Target-bit MC-Beam + Online Micro-Finetune (Mixer input3)
==========================================================================
목표:
- v10_1에서 만든 "범용 surrogate ckpt(2.5_to_3.0)"를 warm-start로 로드
- warmup은 스킵(=비싼 full-eval warmup 없음)
- target avg_bits(예: 2.6)에서 MC-beam 수행
  * 후보 80개(filter_p)는 proxy_loss로 1차 필터
  * surrogate로 80개 모두 스코어링
  * true PPL은 topK + probe만 측정
  * beam update는 true PPL 기준
- 매 gen 끝마다 (측정된 true 라벨) -> online_training_samples.csv에 누적
- warmup_core(범용 데이터 anchor) + current(target gens)로 micro-finetune 수행
- fine-tune된 surrogate로 다음 gen 진행

Usage 예시:
CUDA_VISIBLE_DEVICES=1 nohup \
python MC_NN/step3b_v10_2_target_finetune_mixer.py \
  --model_id meta-llama/Llama-3.2-3B \
  --gpu_id 0 \
  --sens_csv ../artifacts/bitmin/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/bitmin/step2_dyn/alpha_2bit_qmode_rankvar.csv \
  --prebake_root ../artifacts/bitmin/prebake \
  --target_avg_bits 2.5 \
  --use_round_band --round_quantum 0.1 \
  --beam_size 10 --expansion_k 20 --filter_p 80 \
  --true_eval_topk 10 --probe_n 8 \
  --base_training_csv ../artifacts/bitmin/step3b_surrogate_range_2p5_3p0_v10_1/training_samples.csv \
  --warmup_ckpt_json ../artifacts/bitmin/step3b_surrogate_range_2p5_3p0_v10_1/warmup_ckpt.json \
  --base_surrogate_ckpt ../artifacts/bitmin/step3b_surrogate_range_2p5_3p0_v10_1/surrogate_ckpt/best.pt \
  --base_surrogate_config ../artifacts/bitmin/step3b_surrogate_range_2p5_3p0_v10_1/surrogate_ckpt/config.json \
  --surrogate_static_info ../artifacts/data/surrogate_data/static_info_v3.json \
  --output_dir ../artifacts/bitmin/step3b_target_2p5_v10_2 \
  --online_epochs 3 \
  --online_pairs_per_gen 1200 \
  --online_replay_keep_gens 12 \
  > ./log/run_v10_2_target_2p5_1234.log 2>&1 &
"""

import os, gc, csv, json, math, random, argparse, time, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
from copy import deepcopy

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------- sys.path injection (v9 스타일 유지) --------
_FILE_PATH = Path(__file__).resolve()
_PARENTS = _FILE_PATH.parents
_SRC_ROOT = _PARENTS[1] if len(_PARENTS) > 1 else _FILE_PATH.parent
_REPO_ROOT = _PARENTS[2] if len(_PARENTS) > 2 else _SRC_ROOT
_WORKSPACE_ROOT = _PARENTS[3] if len(_PARENTS) > 3 else _REPO_ROOT
_PROJECT_ROOT = (_WORKSPACE_ROOT.parent if _WORKSPACE_ROOT != _WORKSPACE_ROOT.parent else _WORKSPACE_ROOT)
for _path in (_SRC_ROOT, _REPO_ROOT, _WORKSPACE_ROOT, _PROJECT_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Standalone trainer (v9 동일 로직을 그대로 사용)
from module.train_brp_pairwise_surrogate_mixer_input3 import (
    BitSequenceEncoderInput3,
    BRPPairIterableDataset,
    BRPPairwiseSurrogate,
    collate_fn,
    evaluate_ranking_metrics,
    safe_makedirs,
    load_generation_pool,
    parse_alpha_per_bit,
    set_all_seeds,
    zscore,
)

from module.surrogate import SurrogateScorer

from module.montecarlo import (
    _beam_from_serializable,
    _beam_to_serializable,
    append_training_samples,
    bits_to_json,
    compute_budget_band_for_avg_bits,
    build_c_prime_map,
    calculate_proxy_loss,
    check_convergence,
    ensure_complete_assignment,
    generate_random_neighbor,
    get_initial_seed,
    load_seed_from_csv,
    project_to_weighted_band,
    weighted_sum_bits,
)
from module.evaluator import PplEvaluator

# ---------------------------
# Util
# ---------------------------
TRAINING_SAMPLE_COLUMNS = ["generation", "proxy_loss", "measured_ppl", "bit_assignment_json", "avg_bits_target"]
MICROFT_METRIC_COLUMNS = ["generation", "epoch", "train_loss", "val_overlap@10"]

def read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    if (not csv_path) or (not os.path.exists(csv_path)):
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv_rows(csv_path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAINING_SAMPLE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def atomic_save_bit_assign_csv(path: str, bits: Dict[str, int]):
    """global best 갱신 시에만 호출."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer_name", "R_int"])
        for name, bit in sorted(bits.items()):
            w.writerow([name, int(bit)])
    os.replace(tmp, path)

def append_microft_metrics(csv_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MICROFT_METRIC_COLUMNS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def save_json_atomic(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------
# Micro-Finetune Trainer (standalone 로직 그대로 사용)
# ---------------------------
class MicroFinetuneTrainer:
    """
    - v9 standalone trainer와 동일한 데이터 구성(gen2cands, gen 내부 pair sampling)을 사용
    - 단, epochs/pairs_per_gen 등을 v10_2 online용으로 작게 가져간다
    - warm_start_ckpt(이전 best)를 받아서 warm-start fine-tune 후 best.pt 업데이트
    """
    def __init__(self, static_info_path: str, device: torch.device):
        self.static_info_path = static_info_path
        self.device = device

    def run(
        self,
        csv_path: str,
        output_dir: str,
        warm_start_ckpt: str,
        args,
        gen_idx: int,
        global_metrics_csv: Optional[str] = None,
    ) -> Tuple[str, str, float]:
        safe_makedirs(output_dir)
        set_all_seeds(args.seed)

        layer_names, gen2cands, static_info = load_generation_pool(
            csv_path=csv_path,
            static_info_path=self.static_info_path,
            bmin=args.bmin,
        )
        L = len(layer_names)
        gens_all = sorted(gen2cands.keys())
        if len(gens_all) < 2:
            raise RuntimeError(f"[MicroFT] Not enough generations to split: {gens_all}")

        rng = random.Random(int(args.seed))
        gens_shuf = gens_all[:]
        rng.shuffle(gens_shuf)
        n_val = max(1, int(round(len(gens_shuf) * args.val_ratio)))
        val_gens = sorted(gens_shuf[:n_val])
        train_gens = sorted(gens_shuf[n_val:])

        # static features
        C_map: Dict[str, float] = static_info.get("C_map", None)
        if C_map is None:
            C_map = static_info.get("C_prime_map", {})
        W_map: Dict[str, int] = static_info.get("W_map", None)
        if W_map is None:
            raise ValueError("[MicroFT] static_info.json missing W_map")
        alpha_map: Dict[str, Any] = static_info.get("alpha_map", None) or {}

        C_log = np.zeros((L,), dtype=np.float32)
        W_log = np.zeros((L,), dtype=np.float32)
        num_bits = int(args.bmax) - int(args.bmin) + 1
        if num_bits <= 0:
            raise ValueError(f"[MicroFT] Invalid bit range: bmin={args.bmin}, bmax={args.bmax}")
        alpha_table = np.zeros((L, num_bits), dtype=np.float32)
        for i, ln in enumerate(layer_names):
            c = float(C_map.get(ln, 0.0))
            w = float(W_map.get(ln, 1.0))
            C_log[i] = math.log(max(c, 1e-30))
            W_log[i] = math.log(max(w, 1.0))
            avals = parse_alpha_per_bit(
                alpha_map.get(ln, None),
                bmin=args.bmin,
                bmax=args.bmax,
            )
            for j, av in enumerate(avals):
                alpha_table[i, j] = av

        C_log_n, C_mu, C_sd = zscore(C_log)
        W_log_n, W_mu, W_sd = zscore(W_log)
        eps = 1e-6
        a = np.clip(alpha_table, eps, 1.0 - eps).astype(np.float32)
        alpha_logit = np.log(a / (1.0 - a)).astype(np.float32)
        alpha_flat_n, A_mu, A_sd = zscore(alpha_logit.reshape(-1))
        alpha_logit_n = alpha_flat_n.reshape(L, num_bits).astype(np.float32)

        C_log_t = torch.tensor(C_log_n, dtype=torch.float32, device=self.device)
        W_log_t = torch.tensor(W_log_n, dtype=torch.float32, device=self.device)
        alpha_logit_table_t = torch.tensor(alpha_logit_n, dtype=torch.float32, device=self.device)

        # dataset/loader
        train_ds = BRPPairIterableDataset(
            gen2cands=gen2cands,
            gens=train_gens,
            layer_names=layer_names,
            pairs_per_gen=int(args.pairs_per_gen),
            top_frac=float(args.top_frac),
            hard_frac=float(args.hard_frac),
            hard_window=int(args.hard_window),
            tau_soft=float(args.tau_soft),
            seed=int(args.seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

        encoder = BitSequenceEncoderInput3(
            L=L,
            d_model=int(args.d_model),
            bit_emb_dim=int(args.bit_emb_dim),
            nlayers=int(args.nlayers),
            ff_dim=int(args.ff_dim),
            token_mlp_dim=int(args.token_mlp_dim),
            dropout=float(args.dropout),
            use_proxy=True,
            bmin=args.bmin,
            bmax=args.bmax,
        )
        model = BRPPairwiseSurrogate(encoder=encoder, tau_pair=float(args.tau_pair)).to(self.device)

        # warm-start
        if warm_start_ckpt and os.path.exists(warm_start_ckpt):
            ckpt = torch.load(warm_start_ckpt, map_location=self.device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=True)
            print(f"[MicroFT] warm-start from {warm_start_ckpt}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

        config_out = {
            "model_type": "BRP_pairwise_surrogate_mixer_input3",
            "csv_path": csv_path,
            "json_path": self.static_info_path,
            "layer_names": layer_names,
            "L": L,
            "norm": {
                "C_log_mu": C_mu, "C_log_sd": C_sd,
                "W_log_mu": W_mu, "W_log_sd": W_sd,
                "alpha_logit_mu": A_mu, "alpha_logit_sd": A_sd,
            },
            "hparams": {
                "d_model": int(args.d_model),
                "bit_emb_dim": int(args.bit_emb_dim),
                "nlayers": int(args.nlayers),
                "ff_dim": int(args.ff_dim),
                "token_mlp_dim": int(args.token_mlp_dim),
                "dropout": float(args.dropout),
                "no_proxy": False,
                "tau_pair": float(args.tau_pair),
                "bmin": int(args.bmin),
                "bmax": int(args.bmax),
                "score_cache_max": int(args.score_cache_max),
            },
            "split": {"train_gens": train_gens, "val_gens": val_gens},
            "split_meta": {"seed": int(args.seed), "val_ratio": float(args.val_ratio)},
        }

        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_out, f, indent=2)

        best_metric = -1.0
        best_path = os.path.join(output_dir, "best.pt")
        # per-gen(폴더) microft metrics
        per_gen_metrics_csv = os.path.join(output_dir, "microft_metrics.csv")

        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            run_loss, run_n = 0.0, 0
            for batch in train_loader:
                bits_A = batch["bits_A"].to(self.device, non_blocking=True)
                bits_B = batch["bits_B"].to(self.device, non_blocking=True)
                q = batch["q"].to(self.device, non_blocking=True)
                proxy_A = batch["proxy_A"].to(self.device, non_blocking=True)
                proxy_B = batch["proxy_B"].to(self.device, non_blocking=True)

                p, _, _ = model.forward_pair(
                    bits_A, bits_B,
                    C_log=C_log_t, W_log=W_log_t, alpha_logit_table=alpha_logit_table_t,
                    proxy_A=proxy_A, proxy_B=proxy_B,
                )
                logp = torch.log(p)
                loss = F.kl_div(logp, q, reduction="batchmean")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                opt.step()

                run_loss += float(loss.item()) * bits_A.size(0)
                run_n += bits_A.size(0)

            model.eval()
            val_metrics = evaluate_ranking_metrics(
                model=model,
                gen2cands=gen2cands,
                gens=val_gens,
                C_log_t=C_log_t, W_log_t=W_log_t, alpha_logit_table_t=alpha_logit_table_t,
                device=self.device,
                topk=int(args.eval_topk),
                batch_eval=int(args.eval_batch),
            )
            score = float(val_metrics["topk_overlap"])
            avg_loss = run_loss / max(run_n, 1)
            print(f"[MicroFT][Epoch {epoch}] loss={avg_loss:.6f} val_overlap@{int(args.eval_topk)}={score:.4f}", flush=True)

            # (요청) gen | train_loss | overlap@10 저장 (epoch 단위)
            row = {
                "generation": int(gen_idx),
                "epoch": int(epoch),
                "train_loss": float(avg_loss),
                "val_overlap@10": float(score),
            }
            append_microft_metrics(per_gen_metrics_csv, [row])
            if global_metrics_csv:
                append_microft_metrics(global_metrics_csv, [row])
            if score > best_metric:
                best_metric = score
                torch.save({"model_state": model.state_dict(), "best_metric": best_metric, "epoch": epoch, "config": config_out}, best_path)

        torch.cuda.empty_cache()
        return best_path, config_path, best_metric


# ---------------------------
# Core: one generation (surrogate-guided)
# ---------------------------
def run_generation_surrogate_guided(
    gen_idx: int,
    beam: List[Tuple[float, float, Dict[str, int], float]],
    args,
    target_layers_list: List[str],
    ppl_eval: PplEvaluator,
    W_map: Dict[str, int],
    C_prime_filtered: Dict[str, float],
    B_lo: int,
    B_hi: int,
    proxy_loss_calc,
    surrogate: SurrogateScorer,
    desc: str,
    rng: random.Random,
):
    # 후보 생성
    all_candidates = set()
    for _, _, b_assign, _ in beam:
        b0 = ensure_complete_assignment(dict(b_assign), target_layers_list, args.bmin)
        b0 = project_to_weighted_band(b0, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
        all_candidates.add(tuple(sorted(b0.items())))
        for _ in range(int(args.expansion_k)):
            nb = generate_random_neighbor(b0, target_layers_list, args.bmin, args.bmax)
            if nb:
                nb = ensure_complete_assignment(nb, target_layers_list, args.bmin)
                nb = project_to_weighted_band(nb, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
                all_candidates.add(tuple(sorted(nb.items())))

    # proxy filter
    candidate_L_scores = [(proxy_loss_calc(dict(bt)), bt) for bt in all_candidates]
    candidate_L_scores.sort(key=lambda x: x[0])
    finalists = candidate_L_scores[: int(args.filter_p)]

    fin_assigns = [dict(bt) for (L, bt) in finalists]
    fin_proxys = [float(L) for (L, _) in finalists]

    # surrogate score
    sur_scores = surrogate.score_batch(fin_assigns, proxy_vals=fin_proxys, batch=int(args.surrogate_batch))

    # eval set: topK + probe
    K = int(args.true_eval_topk)
    probe_n = int(args.probe_n)
    order = np.argsort(sur_scores)  # 낮을수록 좋다고 가정(모델이 그렇게 학습되었다는 전제)
    top_idx = order[: min(K, len(finalists))].tolist()

    remaining = [i for i in range(len(finalists)) if i not in set(top_idx)]
    probe_idx = []
    if probe_n > 0 and remaining:
        # probe는 랜덤으로(혹은 surrogate uncertainty를 넣고 싶으면 v11에서)
        rng.shuffle(remaining)
        probe_idx = remaining[: min(probe_n, len(remaining))]

    eval_indices = top_idx + probe_idx
    eval_indices = sorted(list(set(eval_indices)))

    evaluated = []
    for idx in tqdm(eval_indices, desc=f"G-{gen_idx} true PPL ({desc})", leave=False):
        proxy_val, b_tuple = finalists[idx]
        b_dict = dict(b_tuple)
        b_dict = project_to_weighted_band(b_dict, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
        ppl = ppl_eval.evaluate(b_dict)
        evaluated.append({
            "ppl": float(ppl),
            "proxy_loss": float(proxy_val),
            "bits": b_dict,
            "candidate_idx": int(idx),
            "surrogate_score": float(sur_scores[idx] if len(sur_scores) > idx else float("nan")),
        })

    if not evaluated:
        raise RuntimeError("No candidates evaluated for current generation.")

    # (요청) overlap@10: (topK+probe=18개) 내부에서 surrogate-top10 vs true-top10
    # 기준은 candidate_idx (finalists 내 인덱스)
    eval_sorted_by_sur = sorted(evaluated, key=lambda x: x["surrogate_score"])
    eval_sorted_by_true = sorted(evaluated, key=lambda x: x["ppl"])
    sur_top10 = [x["candidate_idx"] for x in eval_sorted_by_sur[: min(10, len(eval_sorted_by_sur))]]
    true_top10 = [x["candidate_idx"] for x in eval_sorted_by_true[: min(10, len(eval_sorted_by_true))]]
    overlap10_cnt = len(set(sur_top10).intersection(set(true_top10)))
    overlap10_rate = float(overlap10_cnt) / 10.0

    measured = sorted(evaluated, key=lambda x: x["ppl"])
    new_beam = [(m["ppl"], m["proxy_loss"], m["bits"], m["surrogate_score"]) for m in measured[: int(args.beam_size)]]
    return new_beam, evaluated, overlap10_cnt, overlap10_rate


# ---------------------------
# Build micro-finetune CSV: warmup_core + recent target gens
# ---------------------------
def build_replay_csv(
    base_csv: str,
    online_csv: str,
    warmup_core_gens: List[int],
    keep_recent_target_gens: int,
    out_csv: str,
):
    base_rows = read_csv_rows(base_csv)
    online_rows = read_csv_rows(online_csv)

    # base: warmup_core만 유지(범용 anchor)
    core_set = set(int(x) for x in warmup_core_gens)
    base_keep = []
    for r in base_rows:
        try:
            g = int(float(r["generation"]))
        except Exception:
            continue
        if g in core_set:
            base_keep.append(r)

    # online: 최근 keep_recent_target_gens만 유지
    # (generation 값은 v10_2에서 자체 카운터로 증가)
    online_rows_parsed = []
    for r in online_rows:
        try:
            g = int(float(r["generation"]))
        except Exception:
            continue
        online_rows_parsed.append((g, r))
    online_rows_parsed.sort(key=lambda x: x[0])

    if keep_recent_target_gens > 0:
        gens_sorted = sorted(list(set(g for g, _ in online_rows_parsed)))
        keep_gens = set(gens_sorted[-keep_recent_target_gens:]) if gens_sorted else set()
        online_keep = [r for g, r in online_rows_parsed if g in keep_gens]
    else:
        online_keep = [r for _, r in online_rows_parsed]

    # concat + normalize types
    merged = []
    def norm_row(r: Dict[str, str]) -> Dict[str, Any]:
        return {
            "generation": int(float(r["generation"])),
            "proxy_loss": float(r["proxy_loss"]),
            "measured_ppl": float(r["measured_ppl"]),
            "bit_assignment_json": r["bit_assignment_json"],
            "avg_bits_target": float(r.get("avg_bits_target", 0.0)),
        }
    for r in base_keep:
        merged.append(norm_row(r))
    for r in online_keep:
        merged.append(norm_row(r))

    write_csv_rows(out_csv, merged)
    return len(base_keep), len(online_keep), len(merged)


# ---------------------------
# Main
# ---------------------------
def main(args):
    # seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # devices
    model_device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_map = {"": int(args.gpu_id)}
        torch_dtype = torch.float16
    else:
        device_map = None
        torch_dtype = torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    # outputs
    online_csv = os.path.join(args.output_dir, "online_training_samples.csv")
    replay_csv = os.path.join(args.output_dir, "replay_for_microft.csv")
    run_ckpt_json = os.path.join(args.output_dir, "run_ckpt.json")
    ppl_curve_csv = os.path.join(args.output_dir, "ppl_curve.csv")
    surrogate_out_dir = os.path.join(args.output_dir, "surrogate_ckpt_target")
    microft_global_csv = os.path.join(args.output_dir, "microft_metrics.csv")
    global_bit_csv = os.path.join(args.output_dir, "bit_assign.csv")

    # load model/tokenizer
    print("--- Phase 0: load model/tokenizer/dataset ---", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    original_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    print("PPL eval dataset: wikitext-2-raw-v1...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    eval_input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids

    # proxy maps
    print("--- Phase 1: load proxy maps ---", flush=True)
    C_prime_map, W_map_all = build_c_prime_map(args.sens_csv, args.alpha_csv, args.alpha_bit, args.alpha_default)

    # evaluator / target layers
    print("--- Phase 2: init evaluator ---", flush=True)
    ppl_eval = PplEvaluator(
        model=model,
        tokenizer=tokenizer,
        original_state_dict=original_state_dict,
        eval_input_ids=eval_input_ids,
        prebake_root=args.prebake_root,
        eval_seq_len=args.eval_seq_len,
        cache_maxsize=args.ppl_cache_max,
    )
    target_layers_list = list(ppl_eval.target_layers)
    W_map = {k: int(W_map_all[k]) for k in target_layers_list if k in W_map_all}
    C_prime_filtered = {k: float(C_prime_map[k]) for k in target_layers_list if k in C_prime_map}
    sum_w = int(sum(W_map.values()))
    proxy_loss_calc = lambda b: calculate_proxy_loss(b, C_prime_filtered, args.bmin)

    # load warmup ckpt (for core gens)
    if not args.warmup_ckpt_json or (not os.path.exists(args.warmup_ckpt_json)):
        raise FileNotFoundError("--warmup_ckpt_json (v10_1 warmup_ckpt.json) 필요")
    warmup_ckpt = load_json(args.warmup_ckpt_json)
    warmup_core_gens = [int(x) for x in warmup_ckpt.get("warmup_core_gens", [])]
    if not warmup_core_gens:
        print("[Warn] warmup_core_gens가 비었습니다. base warmup anchor가 약해질 수 있음.", flush=True)

    # load base surrogate (범용 ckpt)
    if not (args.base_surrogate_ckpt and os.path.exists(args.base_surrogate_ckpt)):
        raise FileNotFoundError("--base_surrogate_ckpt 필요")
    if not (args.base_surrogate_config and os.path.exists(args.base_surrogate_config)):
        raise FileNotFoundError("--base_surrogate_config 필요")

    surrogate_device = torch.device(args.surrogate_device) if args.surrogate_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = SurrogateScorer(args.base_surrogate_ckpt, args.base_surrogate_config, device=surrogate_device, bmin=args.bmin)
    surrogate.clear_score_cache()

    # target budget band
    target_avg = float(args.target_avg_bits)
    B_lo, B_hi, _, B_target = compute_budget_band_for_avg_bits(target_avg, W_map, float(args.round_quantum), bool(args.use_round_band))
    print(
        f"[Target] avg_bits={target_avg:.3f} band={'on' if args.use_round_band else 'off'} q={args.round_quantum} "
        f"Σw·b in [{B_lo},{B_hi}] target≈{B_target}/{sum_w}={B_target/sum_w:.6f}",
        flush=True,
    )

    # init beam from warmup beam? (선택)
    if args.init_from_warmup_beam:
        print("[Init] warmup_ckpt의 beam을 가져와 target band로 projection 후 초기화", flush=True)
        beam = _beam_from_serializable(warmup_ckpt["beam"])
        # projection to target band
        proj_beam = []
        for ppl, L, bits, sur in beam:
            b = ensure_complete_assignment(bits, target_layers_list, args.bmin)
            b = project_to_weighted_band(b, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
            proj_beam.append((ppl, float(proxy_loss_calc(b)), b, sur))
        proj_beam.sort(key=lambda x: x[0])
        beam = proj_beam[: int(args.beam_size)]
        if not beam:
            raise RuntimeError("warmup beam init 실패(beam empty)")
    else:
        b_seed: Dict[str, int] = {}
        init_assign_csv = (args.init_assign_csv or "").strip()
        if init_assign_csv:
            if not os.path.exists(init_assign_csv):
                print(
                    f"[Warn] init_assign_csv not found, fallback to convex seed: {init_assign_csv}",
                    flush=True,
                )
            else:
                print(f"[Init] init_assign_csv에서 seed 로드: {init_assign_csv}", flush=True)
                b_seed = load_seed_from_csv(init_assign_csv)
                if not b_seed:
                    print(
                        f"[Warn] init_assign_csv has no valid rows, fallback to convex seed: {init_assign_csv}",
                        flush=True,
                    )

        if not b_seed:
            print("[Init] target_avg_bits 기준 convex seed로 초기 beam 구성", flush=True)
            b_seed = get_initial_seed(C_prime_filtered, W_map, target_avg, args.bmin, args.bmax)

        b_seed = ensure_complete_assignment(b_seed, target_layers_list, args.bmin)
        b_seed = project_to_weighted_band(b_seed, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)

        # init 후보 생성: proxy topK만 true eval
        initial_candidates: Dict[Tuple[Tuple[str, int], ...], float] = {}
        initial_candidates[tuple(sorted(b_seed.items()))] = proxy_loss_calc(b_seed)
        for _ in range(int(args.beam_size) * int(args.expansion_k)):
            nb = generate_random_neighbor(b_seed, target_layers_list, args.bmin, args.bmax)
            if nb:
                nb = ensure_complete_assignment(nb, target_layers_list, args.bmin)
                nb = project_to_weighted_band(nb, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
                initial_candidates[tuple(sorted(nb.items()))] = proxy_loss_calc(nb)
        init_items = sorted(initial_candidates.items(), key=lambda x: x[1])
        K0 = max(int(args.true_eval_topk), int(args.beam_size))
        beam = []
        for bt, l_score in tqdm(init_items[: min(K0, len(init_items))], desc="init true PPL"):
            b_dict = dict(bt)
            ppl = ppl_eval.evaluate(b_dict)
            beam.append((ppl, float(l_score), b_dict, float(l_score)))
        beam.sort(key=lambda x: x[0])
        beam = beam[: int(args.beam_size)]
        if not beam:
            raise RuntimeError("초기 beam이 비었습니다.")

    # resume?
    gen_counter = 0
    global_best_ppl = float("inf")
    global_best_bits = None
    best_ckpt_path = args.base_surrogate_ckpt
    best_config_path = args.base_surrogate_config

    if args.resume and os.path.exists(run_ckpt_json):
        ck = load_json(run_ckpt_json)
        gen_counter = int(ck["gen_counter"])
        beam = _beam_from_serializable(ck["beam"])
        global_best_ppl = float(ck.get("global_best_ppl", global_best_ppl))
        best_ckpt_path = ck.get("best_surrogate_ckpt", best_ckpt_path)
        best_config_path = ck.get("best_surrogate_config", best_config_path)
        if best_ckpt_path and best_config_path and os.path.exists(best_ckpt_path) and os.path.exists(best_config_path):
            surrogate = SurrogateScorer(best_ckpt_path, best_config_path, device=surrogate_device, bmin=args.bmin)
            surrogate.clear_score_cache()
        print(f"[Resume] gen={gen_counter} best_ppl={global_best_ppl}", flush=True)

    if beam:
        global_best_bits = deepcopy(beam[0][2])
        if not math.isfinite(global_best_ppl) or global_best_ppl == float("inf"):
            global_best_ppl = float(beam[0][0])
            if (not args.resume) or (not os.path.exists(global_bit_csv)):
                atomic_save_bit_assign_csv(global_bit_csv, global_best_bits)

    # ppl_curve init
    if not os.path.exists(ppl_curve_csv):
        with open(ppl_curve_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "generation",
                "best_ppl",
                "global_best_ppl",
                "best_L",
                "best_sur",
                "weighted_avg_bits",
                "overlap10_cnt",
                "overlap10_rate",
            ])

    # micro finetune trainer
    if not args.surrogate_static_info or (not os.path.exists(args.surrogate_static_info)):
        raise FileNotFoundError("--surrogate_static_info 필요")
    ft_device = torch.device(args.surrogate_trainer_device) if args.surrogate_trainer_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    microft = MicroFinetuneTrainer(static_info_path=args.surrogate_static_info, device=ft_device)

    # main loop
    print("--- Phase 3: target MC + online micro-finetune ---", flush=True)
    rng = random.Random(int(args.seed) + 999)

    no_improve = 0
    stable_bits = 0
    prev_best_bits = None
    start_ts = time.time()
    max_g = int(args.max_generations) if int(args.max_generations) > 0 else (int(args.generations) if int(args.generations) > 0 else 0)

    while True:
        # 1) run one gen
        new_beam, evaluated, overlap10_cnt, overlap10_rate = run_generation_surrogate_guided(
            gen_idx=gen_counter,
            beam=beam,
            args=args,
            target_layers_list=target_layers_list,
            ppl_eval=ppl_eval,
            W_map=W_map,
            C_prime_filtered=C_prime_filtered,
            B_lo=B_lo,
            B_hi=B_hi,
            proxy_loss_calc=proxy_loss_calc,
            surrogate=surrogate,
            desc="target-surrogate",
            rng=rng,
        )
        beam = new_beam

        # 2) update best
        best_ppl_this_gen = float(beam[0][0])
        best_bits_this_gen = deepcopy(beam[0][2])
        best_L_this_gen = float(beam[0][1])
        best_sur_this_gen = float(beam[0][3])
        S = weighted_sum_bits(best_bits_this_gen, W_map)
        wavg = S / sum_w if sum_w > 0 else 0.0
        global_improved, no_improve, stable_bits, prev_best_bits, stop_reasons = check_convergence(
            best_ppl=best_ppl_this_gen,
            best_bits=best_bits_this_gen,
            global_best_ppl=global_best_ppl,
            no_improve=no_improve,
            stable_bits=stable_bits,
            prev_best_bits=prev_best_bits,
            converge_eps=float(args.converge_eps),
            converge_rel_eps=float(args.converge_rel_eps),
            patience=int(args.patience),
            stable_bits_patience=int(args.stable_bits_patience),
            time_limit_sec=int(args.time_limit_sec),
            start_ts=float(start_ts),
            gen_idx=int(gen_counter),
            max_g=int(max_g),
        )

        if global_improved:
            global_best_ppl = best_ppl_this_gen
            global_best_bits = deepcopy(best_bits_this_gen)
            atomic_save_bit_assign_csv(global_bit_csv, global_best_bits)

        # 3) log curve
        with open(ppl_curve_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                gen_counter,
                f"{best_ppl_this_gen:.6f}",
                f"{global_best_ppl:.6f}",
                f"{best_L_this_gen:.6e}",
                f"{best_sur_this_gen:.6f}",
                f"{wavg:.6f}",
                int(overlap10_cnt),
                float(overlap10_rate),
            ])

        # 4) append online labels
        rows = [{
            "generation": gen_counter,
            "proxy_loss": e["proxy_loss"],
            "measured_ppl": e["ppl"],
            "bit_assignment_json": bits_to_json(e["bits"]),
            "avg_bits_target": float(target_avg),
        } for e in evaluated]
        append_training_samples(online_csv, rows)

        # 5) build replay csv (warmup_core + recent target gens)
        base_keep, online_keep, merged = build_replay_csv(
            base_csv=args.base_training_csv,
            online_csv=online_csv,
            warmup_core_gens=warmup_core_gens,
            keep_recent_target_gens=int(args.online_replay_keep_gens),
            out_csv=replay_csv,
        )
        print(f"[Replay] base_core={base_keep} online_keep={online_keep} merged={merged}", flush=True)

        # 6) micro-finetune (warm-start from current best_ckpt_path)
        #    - epoch 적게, pairs_per_gen 적게 -> 빠르게 적응
        os.makedirs(surrogate_out_dir, exist_ok=True)

        ft_args = argparse.Namespace(
            seed=int(args.seed) + gen_counter,  # gen마다 변형
            bmin=int(args.bmin),
            bmax=int(args.bmax),

            epochs=int(args.online_epochs),
            val_ratio=float(args.online_val_ratio),
            pairs_per_gen=int(args.online_pairs_per_gen),
            top_frac=float(args.online_top_frac),
            hard_frac=float(args.online_hard_frac),
            hard_window=int(args.online_hard_window),
            tau_soft=float(args.online_tau_soft),

            batch_size=int(args.online_batch_size),
            num_workers=int(args.online_num_workers),

            d_model=int(args.online_d_model),
            bit_emb_dim=int(args.online_bit_emb_dim),
            nlayers=int(args.online_nlayers),
            ff_dim=int(args.online_ff_dim),
            token_mlp_dim=int(args.online_token_mlp_dim),
            dropout=float(args.online_dropout),
            tau_pair=float(args.online_tau_pair),

            lr=float(args.online_lr),
            weight_decay=float(args.online_weight_decay),
            grad_clip=float(args.online_grad_clip),

            eval_topk=int(args.online_eval_topk),
            eval_batch=int(args.online_eval_batch),
            score_cache_max=int(args.surrogate_score_cache_max),
        )

        # fine-tune 결과를 generation별 폴더에 저장 (선호하면 latest로 덮어쓰기 가능)
        gen_ckpt_dir = os.path.join(surrogate_out_dir, f"gen_{gen_counter:04d}")
        os.makedirs(gen_ckpt_dir, exist_ok=True)

        ft_best_ckpt, ft_best_cfg, ft_best_metric = microft.run(
            csv_path=replay_csv,
            output_dir=gen_ckpt_dir,
            warm_start_ckpt=best_ckpt_path,
            args=ft_args,
            gen_idx=int(gen_counter),
            global_metrics_csv=microft_global_csv,
        )

        # 7) update surrogate to latest fine-tuned ckpt
        best_ckpt_path = ft_best_ckpt
        best_config_path = ft_best_cfg
        surrogate = SurrogateScorer(best_ckpt_path, best_config_path, device=surrogate_device, bmin=args.bmin)
        surrogate.clear_score_cache()  # ★ 요청 취지 반영: finetune 직후 캐시 강제 클리어

        # 8) save run ckpt
        ck_payload = {
            "gen_counter": gen_counter + 1,
            "beam": _beam_to_serializable(beam),
            "global_best_ppl": float(global_best_ppl),
            "best_surrogate_ckpt": best_ckpt_path,
            "best_surrogate_config": best_config_path,
            "meta": {
                "target_avg_bits": float(target_avg),
                "use_round_band": bool(args.use_round_band),
                "round_quantum": float(args.round_quantum),
                "true_eval_topk": int(args.true_eval_topk),
                "probe_n": int(args.probe_n),
                "online_epochs": int(args.online_epochs),
                "online_pairs_per_gen": int(args.online_pairs_per_gen),
                "online_replay_keep_gens": int(args.online_replay_keep_gens),
                "overlap10_cnt": int(overlap10_cnt),
                "overlap10_rate": float(overlap10_rate),
            },
        }
        save_json_atomic(run_ckpt_json, ck_payload)

        print(
            f"[G-{gen_counter}] best_ppl={best_ppl_this_gen:.4f} global_best={global_best_ppl:.4f} "
            f"| overlap10={int(overlap10_cnt)}/10 ({overlap10_rate:.3f}) "
            f"| microft_overlap@{int(args.online_eval_topk)}={ft_best_metric:.4f}",
            flush=True
        )

        if stop_reasons:
            print(f"✔️ 수렴/중단: {', '.join(stop_reasons)}", flush=True)
            break

        gen_counter += 1

    print(f"[Done] gens={gen_counter} global_best_ppl={global_best_ppl:.4f}", flush=True)
    return {
        "output_dir": args.output_dir,
        "bit_assign_csv": global_bit_csv,
        "ppl_curve_csv": ppl_curve_csv,
        "run_ckpt_json": run_ckpt_json,
        "online_csv": online_csv,
        "replay_csv": replay_csv,
        "microft_metrics_csv": microft_global_csv,
        "surrogate_ckpt_target_dir": surrogate_out_dir,
        "global_best_ppl": float(global_best_ppl),
        "generations_done": int(gen_counter),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Step3b v10_2 target-bit finetune (Mixer input3)")

    # basic
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    # proxy files
    p.add_argument("--sens_csv", type=str, required=True)
    p.add_argument("--alpha_csv", type=str, required=True)
    p.add_argument("--alpha_bit", type=int, default=3)
    p.add_argument("--alpha_default", type=float, default=1.0)

    # prebake
    p.add_argument("--prebake_root", type=str, required=True)

    # bit range + target
    p.add_argument("--bmin", type=int, default=1)
    p.add_argument("--bmax", type=int, default=4)
    p.add_argument("--target_avg_bits", type=float, required=True)

    # round band
    p.add_argument("--use_round_band", action="store_true")
    p.add_argument("--round_quantum", type=float, default=0.1)

    # ppl eval
    p.add_argument("--eval_seq_len", type=int, default=2048)
    p.add_argument("--ppl_cache_max", type=int, default=20000)

    # MC-beam params
    p.add_argument("--beam_size", type=int, default=10)
    p.add_argument("--expansion_k", type=int, default=20)
    p.add_argument("--filter_p", type=int, default=80)
    p.add_argument("--true_eval_topk", type=int, default=10)
    p.add_argument("--probe_n", type=int, default=8)
    p.add_argument("--surrogate_batch", type=int, default=1024)

    # v10_1 artifacts (범용)
    p.add_argument("--base_training_csv", type=str, required=True, help="v10_1 training_samples.csv")
    p.add_argument("--warmup_ckpt_json", type=str, required=True, help="v10_1 warmup_ckpt.json")
    p.add_argument("--base_surrogate_ckpt", type=str, required=True, help="v10_1 surrogate best.pt")
    p.add_argument("--base_surrogate_config", type=str, required=True, help="v10_1 surrogate config.json")
    p.add_argument("--surrogate_static_info", type=str, required=True, help="static_info_v3.json (Input3)")

    # init
    p.add_argument("--init_assign_csv", type=str, default="", help="Optional seed CSV (layer_name,R_int)")
    p.add_argument("--init_from_warmup_beam", action="store_true", help="warmup_ckpt의 beam으로 초기화")
    p.add_argument("--resume", action="store_true")

    # devices
    p.add_argument("--surrogate_device", type=str, default="", help="scorer device")
    p.add_argument("--surrogate_trainer_device", type=str, default="", help="trainer device")

    # loop
    # convergence/safety
    p.add_argument("--generations", type=int, default=0, help="(하위호환) >0이면 max_generations로 사용")
    p.add_argument("--max_generations", type=int, default=0, help="0이면 수렴 조건까지 진행")
    p.add_argument("--converge_eps", type=float, default=1e-3, help="절대 개선 임계값")
    p.add_argument("--converge_rel_eps", type=float, default=1e-3, help="상대 개선 임계값")
    p.add_argument("--patience", type=int, default=12, help="개선 없는 세대 허용치")
    p.add_argument("--stable_bits_patience", type=int, default=12, help="동일 best bit 반복 허용치")
    p.add_argument("--time_limit_sec", type=int, default=0, help="0=무제한, >0이면 시간 제한(초)")

    # online micro-finetune (빠르게)
    p.add_argument("--online_epochs", type=int, default=3)
    p.add_argument("--online_val_ratio", type=float, default=0.2)
    p.add_argument("--online_pairs_per_gen", type=int, default=1200)
    p.add_argument("--online_replay_keep_gens", type=int, default=12)

    p.add_argument("--online_top_frac", type=float, default=0.3)
    p.add_argument("--online_hard_frac", type=float, default=0.3)
    p.add_argument("--online_hard_window", type=int, default=8)
    p.add_argument("--online_tau_soft", type=float, default=1.0)

    p.add_argument("--online_batch_size", type=int, default=256)
    p.add_argument("--online_num_workers", type=int, default=2)

    p.add_argument("--online_d_model", type=int, default=128)
    p.add_argument("--online_bit_emb_dim", type=int, default=32)
    p.add_argument("--online_nlayers", type=int, default=2)
    p.add_argument("--online_ff_dim", type=int, default=256)
    p.add_argument("--online_token_mlp_dim", type=int, default=128)
    p.add_argument("--online_dropout", type=float, default=0.1)
    p.add_argument("--online_tau_pair", type=float, default=1.0)

    p.add_argument("--online_lr", type=float, default=3e-4)
    p.add_argument("--online_weight_decay", type=float, default=0.01)
    p.add_argument("--online_grad_clip", type=float, default=1.0)

    p.add_argument("--online_eval_topk", type=int, default=10)
    p.add_argument("--online_eval_batch", type=int, default=512)

    # scorer cache max (config에도 저장)
    p.add_argument("--surrogate_score_cache_max", type=int, default=200000)

    return p


def run(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = build_arg_parser().parse_args()
    return main(args)


if __name__ == "__main__":
    run()
