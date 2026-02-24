#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3b — v10_1: Warmup Range Dataset Builder + Offline Surrogate Train (Mixer input3)
====================================================================================

목표:
- Warmup 단계에서 avg_bits를 고정(2.5)하지 않고, 지정한 범위 [lo, hi]에서 gen마다 target avg_bits를 선택
- 각 gen은 해당 target avg_bits에 대한 roundband(quantum)로 후보를 projection하여 true PPL을 측정
- warmup 결과(beam/global_state/warmup_core/meta)를 warmup_ckpt.json에 저장
- warmup 데이터(training_samples.csv)로 offline surrogate를 학습해 "범용(2.5_to_3.0 등)" ckpt(best.pt/config.json)를 만든다.
- v10_2에서: 이 범용 ckpt를 warm-start로 target avg_bits에 맞춰 micro-finetune + MC 탐색을 수행

Usage 예시:
CUDA_VISIBLE_DEVICES=1 nohup \
python MC_NN/step3b_v10_1_warmup_range_train_mixer.py \
  --model_id meta-llama/Llama-3.2-3B \
  --gpu_id 0 \
  --sens_csv ../artifacts/bitmin/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/bitmin/step2_dyn/alpha_layerwise_rankvar.csv \
  --prebake_root ../artifacts/bitmin/prebake \
  --warmup_generations 10 \
  --warmup_bits_lo 2.5 --warmup_bits_hi 3.0 \
  --warmup_bits_sampling uniform \
  --use_round_band --round_quantum 0.1 \
  --beam_size 10 --expansion_k 20 --filter_p 80 \
  --true_eval_topk 10 \
  --surrogate_static_info ../artifacts/bitmin/surrogate_data/static_info_v3.json \
  --output_dir ../artifacts/bitmin/step3b_surrogate_range_2p5_3p0_v10_1 \
  > ./log/run_v10_1_range_warmup_1234.log 2>&1 &
"""

import os, gc, csv, json, math, random, argparse, time, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
from copy import deepcopy

# Ensure repo-relative modules can be imported before referencing them
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

from module.train_brp_pairwise_surrogate_mixer_input3 import (
    BitSequenceEncoderInput3,
    BRPPairIterableDataset,
    BRPPairwiseSurrogate,
    collate_fn,
    evaluate_ranking_metrics,
    safe_makedirs,
    compute_ndcg_at_k,
    load_generation_pool,
    parse_alpha_per_bit,
    set_all_seeds,
    topk_overlap,
    zscore,
)

from module.surrogate import SurrogateScorer

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader

from module.montecarlo import (
    _beam_from_serializable,
    _beam_to_serializable,
    append_training_samples,
    bits_to_json,
    compute_budget_band_for_avg_bits,
    atomic_save_bit_assign_csv,
    build_c_prime_map,
    calculate_proxy_loss,
    ensure_complete_assignment,
    generate_random_neighbor,
    get_initial_seed,
    load_seed_from_csv,
    project_to_weighted_band,
    weighted_sum_bits,
)
from module.evaluator import PplEvaluator

# -----------------------------------------------
# Step 2/4에서 가져오기 (경로/네이밍)
# -----------------------------------------------
try:
    from cvx.step1_2_alpha_estimation import _canonical_dataset_name  # noqa: F401
except ImportError:
    try:
        from ..cvx.step1_2_alpha_estimation import _canonical_dataset_name  # noqa: F401
    except ImportError:
        print("오류: mixture/cvx/step1_2_alpha_estimation.py import 실패")
        sys.exit(1)


# =========================================================
# Offline Surrogate Trainer Wrapper (v9와 동일)
# =========================================================
class OfflineSurrogateTrainer:
    def __init__(self, csv_path: str, static_info_path: str, output_dir: str, trainer_args: argparse.Namespace, device=None):
        self.csv_path = csv_path
        self.static_info_path = static_info_path
        self.output_dir = output_dir
        self.args = trainer_args
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        safe_makedirs(self.output_dir)

    def train(self, warm_start_ckpt: Optional[str] = None) -> Tuple[str, str, float]:
        set_all_seeds(self.args.sur_train_seed)

        layer_names, gen2cands, static_info = load_generation_pool(
            csv_path=self.csv_path,
            static_info_path=self.static_info_path,
            bmin=self.args.bmin,
        )
        L = len(layer_names)
        gens_all = sorted(gen2cands.keys())
        if len(gens_all) < 2:
            raise RuntimeError(f"[Trainer] Not enough generations to split: {gens_all}")

        rng = random.Random(int(self.args.sur_train_seed))
        gens_shuf = gens_all[:]
        rng.shuffle(gens_shuf)
        n_val = max(1, int(round(len(gens_shuf) * self.args.sur_train_val_ratio)))
        val_gens = sorted(gens_shuf[:n_val])
        train_gens = sorted(gens_shuf[n_val:])

        print(f"[Trainer] split: total={len(gens_all)} train={len(train_gens)} val={len(val_gens)}", flush=True)

        C_map: Dict[str, float] = static_info.get("C_map", None)
        W_map: Dict[str, int] = static_info.get("W_map", None)
        alpha_map: Dict[str, Any] = static_info.get("alpha_map", None)

        if C_map is None:
            C_map = static_info.get("C_prime_map", {})
            print("[Trainer] static_info missing C_map. Falling back to C_prime_map.", flush=True)
        if W_map is None:
            raise ValueError("[Trainer] static_info.json missing W_map")
        if alpha_map is None:
            alpha_map = {}
            print("[Trainer] static_info missing alpha_map. Using alpha=1.0.", flush=True)

        C_log = np.zeros((L,), dtype=np.float32)
        W_log = np.zeros((L,), dtype=np.float32)
        num_bits = int(self.args.bmax) - int(self.args.bmin) + 1
        if num_bits <= 0:
            raise ValueError(f"[Trainer] Invalid bit range: bmin={self.args.bmin}, bmax={self.args.bmax}")
        alpha_table = np.zeros((L, num_bits), dtype=np.float32)
        for i, ln in enumerate(layer_names):
            c = float(C_map.get(ln, 0.0))
            w = float(W_map.get(ln, 1.0))
            C_log[i] = math.log(max(c, 1e-30))
            W_log[i] = math.log(max(w, 1.0))
            avals = parse_alpha_per_bit(
                alpha_map.get(ln, None),
                bmin=self.args.bmin,
                bmax=self.args.bmax,
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

        train_ds = BRPPairIterableDataset(
            gen2cands=gen2cands,
            gens=train_gens,
            layer_names=layer_names,
            pairs_per_gen=int(self.args.sur_train_pairs_per_gen),
            top_frac=float(self.args.sur_train_top_frac),
            hard_frac=float(self.args.sur_train_hard_frac),
            hard_window=int(self.args.sur_train_hard_window),
            tau_soft=float(self.args.sur_train_tau_soft),
            seed=int(self.args.sur_train_seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(self.args.sur_train_batch_size),
            num_workers=int(self.args.sur_train_num_workers),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

        encoder = BitSequenceEncoderInput3(
            L=L,
            d_model=int(self.args.sur_train_d_model),
            bit_emb_dim=int(self.args.sur_train_bit_emb_dim),
            nlayers=int(self.args.sur_train_nlayers),
            ff_dim=int(self.args.sur_train_ff_dim),
            token_mlp_dim=int(self.args.sur_train_token_mlp_dim),
            dropout=float(self.args.sur_train_dropout),
            use_proxy=(not bool(self.args.sur_train_no_proxy)),
            bmin=self.args.bmin,
            bmax=self.args.bmax,
        )
        model = BRPPairwiseSurrogate(encoder=encoder, tau_pair=float(self.args.sur_train_tau_pair)).to(self.device)

        if warm_start_ckpt and os.path.exists(warm_start_ckpt):
            ckpt = torch.load(warm_start_ckpt, map_location=self.device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=True)
            print(f"[Trainer] warm-started from {warm_start_ckpt}", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=float(self.args.sur_train_lr), weight_decay=float(self.args.sur_train_weight_decay))

        config_out = {
            "model_type": "BRP_pairwise_surrogate_mixer_input3",
            "csv_path": self.csv_path,
            "json_path": self.static_info_path,
            "layer_names": layer_names,
            "L": L,
            "norm": {
                "C_log_mu": C_mu, "C_log_sd": C_sd,
                "W_log_mu": W_mu, "W_log_sd": W_sd,
                "alpha_logit_mu": A_mu, "alpha_logit_sd": A_sd,
            },
            "hparams": {
                "d_model": int(self.args.sur_train_d_model),
                "bit_emb_dim": int(self.args.sur_train_bit_emb_dim),
                "nlayers": int(self.args.sur_train_nlayers),
                "ff_dim": int(self.args.sur_train_ff_dim),
                "token_mlp_dim": int(self.args.sur_train_token_mlp_dim),
                "dropout": float(self.args.sur_train_dropout),
                "no_proxy": bool(self.args.sur_train_no_proxy),
                "tau_pair": float(self.args.sur_train_tau_pair),
                "bmin": int(self.args.bmin),
                "bmax": int(self.args.bmax),
                "score_cache_max": int(self.args.surrogate_score_cache_max),
            },
            "static_info": {k: static_info.get(k) for k in ["model_id", "avg_bits_target"]},
            "split": {"train_gens": train_gens, "val_gens": val_gens},
            "split_meta": {"seed": int(self.args.sur_train_seed), "val_ratio": float(self.args.sur_train_val_ratio)},
        }
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_out, f, indent=2)

        best_metric = -1.0
        best_path = os.path.join(self.output_dir, "best.pt")
        print("[Trainer] start", flush=True)
        global_step = 0

        for epoch in range(1, int(self.args.sur_train_epochs) + 1):
            model.train()
            t0 = time.time()
            run_loss = 0.0
            run_n = 0

            for batch in train_loader:
                bits_A = batch["bits_A"].to(self.device, non_blocking=True)
                bits_B = batch["bits_B"].to(self.device, non_blocking=True)
                q = batch["q"].to(self.device, non_blocking=True)
                proxy_A = batch["proxy_A"].to(self.device, non_blocking=True) if (not self.args.sur_train_no_proxy) else None
                proxy_B = batch["proxy_B"].to(self.device, non_blocking=True) if (not self.args.sur_train_no_proxy) else None

                p, _, _ = model.forward_pair(
                    bits_A, bits_B,
                    C_log=C_log_t, W_log=W_log_t, alpha_logit_table=alpha_logit_table_t,
                    proxy_A=proxy_A, proxy_B=proxy_B,
                )
                logp = torch.log(p)
                loss = F.kl_div(logp, q, reduction="batchmean")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(self.args.sur_train_grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.args.sur_train_grad_clip))
                opt.step()

                run_loss += float(loss.item()) * bits_A.size(0)
                run_n += bits_A.size(0)
                global_step += 1

                if int(self.args.sur_train_log_interval) > 0 and global_step % int(self.args.sur_train_log_interval) == 0:
                    avg_loss = run_loss / max(run_n, 1)
                    print(f"[Trainer] epoch={epoch} step={global_step} loss={avg_loss:.6f}", flush=True)

            avg_train_loss = run_loss / max(run_n, 1)
            dt = time.time() - t0

            model.eval()
            val_metrics = evaluate_ranking_metrics(
                model=model,
                gen2cands=gen2cands,
                gens=val_gens,
                C_log_t=C_log_t, W_log_t=W_log_t, alpha_logit_table_t=alpha_logit_table_t,
                device=self.device,
                topk=int(self.args.sur_train_topk),
                batch_eval=int(self.args.sur_train_eval_batch),
            )
            score = float(val_metrics["topk_overlap"])
            print(
                f"[Trainer][Epoch {epoch}] train_loss={avg_train_loss:.6f} "
                f"val_top{int(self.args.sur_train_topk)}_overlap={score:.4f} "
                f"val_ndcg@{int(self.args.sur_train_topk)}={float(val_metrics['ndcg@k']):.4f} "
                f"time={dt:.1f}s",
                flush=True,
            )
            if score > best_metric:
                best_metric = score
                torch.save({"model_state": model.state_dict(), "best_metric": best_metric, "epoch": epoch, "config": config_out}, best_path)
                print(f"[Trainer] best updated -> {best_path} (overlap={best_metric:.4f})", flush=True)

        torch.cuda.empty_cache()
        return best_path, config_path, best_metric


# =========================================================
# Dataset helpers
# =========================================================
def save_warmup_ckpt(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_warmup_ckpt(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_warmup_avg_bits(gen_idx: int, args, rng: random.Random) -> float:
    lo = float(args.warmup_bits_lo)
    hi = float(args.warmup_bits_hi)
    if hi < lo:
        lo, hi = hi, lo

    mode = str(args.warmup_bits_sampling).lower()
    if mode == "uniform":
        return float(rng.uniform(lo, hi))
    if mode == "grid":
        step = float(args.warmup_bits_grid_step)
        step = max(step, 1e-6)
        n = int(round((hi - lo) / step)) + 1
        grid = [lo + i * step for i in range(max(n, 1))]
        grid = [min(x, hi) for x in grid]
        return float(grid[gen_idx % len(grid)])
    if mode == "cycle":
        # cycle: lo, lo+q, lo+2q, ... <= hi 반복 (q = round_quantum)
        step = float(args.round_quantum)
        step = max(step, 1e-6)
        n = int(round((hi - lo) / step)) + 1
        grid = [lo + i * step for i in range(max(n, 1))]
        grid = [min(x, hi) for x in grid]
        return float(grid[gen_idx % len(grid)])
    raise ValueError(f"Unknown --warmup_bits_sampling={args.warmup_bits_sampling}")


# =========================================================
# run_generation (v10_1: beam seed도 현재 밴드에 맞춰 projection)
# =========================================================
def run_generation(
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
    surrogate: Optional[SurrogateScorer],
    measurement_mode: str,
    desc: str,
    rng: Optional[random.Random] = None,
):
    if rng is None:
        rng = random.Random(1234 + gen_idx)

    all_candidates = set()
    for _, _, b_assign, _ in beam:
        # ★ v10_1: beam 자체도 현재 밴드로 맞춘 버전을 후보에 넣음 (proxy ranking 안정)
        b0 = ensure_complete_assignment(dict(b_assign), target_layers_list, args.bmin)
        b0 = project_to_weighted_band(b0, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
        all_candidates.add(tuple(sorted(b0.items())))

        for _ in range(args.expansion_k):
            neighbor = generate_random_neighbor(b0, target_layers_list, args.bmin, args.bmax)
            if neighbor:
                neighbor = ensure_complete_assignment(neighbor, target_layers_list, args.bmin)
                neighbor = project_to_weighted_band(neighbor, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
                all_candidates.add(tuple(sorted(neighbor.items())))

    candidate_L_scores = [(proxy_loss_calc(dict(bt)), bt) for bt in all_candidates]
    candidate_L_scores.sort(key=lambda x: x[0])
    finalists = candidate_L_scores[: args.filter_p]

    fin_assigns = [dict(bt) for (_, bt) in finalists]
    fin_proxys = [float(L) for (L, _) in finalists]

    if surrogate is not None:
        surrogate_scores = surrogate.score_batch(fin_assigns, proxy_vals=fin_proxys, batch=args.surrogate_batch)
    else:
        surrogate_scores = np.array(fin_proxys, dtype=np.float64)

    if measurement_mode == "full":
        eval_indices = list(range(len(finalists)))
    elif measurement_mode == "sur_topk":
        K_true = max(int(args.true_eval_topk), int(args.beam_size))
        order = np.argsort(surrogate_scores)
        eval_indices = order[: min(K_true, len(finalists))].tolist()
    else:
        raise ValueError(f"Unknown measurement_mode={measurement_mode}")

    evaluated = []
    for idx in tqdm(eval_indices, desc=f"G-{gen_idx} true PPL ({desc})", leave=False):
        proxy_val, b_tuple = finalists[idx]
        b_dict = dict(b_tuple)
        b_dict = project_to_weighted_band(b_dict, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
        ppl = ppl_eval.evaluate(b_dict)
        evaluated.append(
            {
                "ppl": float(ppl),
                "proxy_loss": float(proxy_val),
                "bits": b_dict,
                "candidate_idx": int(idx),
                "surrogate_score": float(surrogate_scores[idx] if len(surrogate_scores) > idx else float("nan")),
            }
        )

    if not evaluated:
        raise RuntimeError("No candidates evaluated for current generation.")

    measured = sorted(evaluated, key=lambda x: x["ppl"])
    new_beam = [(m["ppl"], m["proxy_loss"], m["bits"], m["surrogate_score"]) for m in measured[: args.beam_size]]
    return new_beam, evaluated


# =========================================================
# Main (v10_1)
# =========================================================
def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        device_map = {"": int(args.gpu_id)}
        torch_dtype = torch.float16
    else:
        device_map = None
        torch_dtype = torch.float32

    print("--- Phase 1: 모델/데이터 준비 ---", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    original_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    print("PPL 평가용 데이터셋 로드 (wikitext-2-raw-v1)...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    eval_input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids

    print("프록시 모델(C', W) 로드...", flush=True)
    C_prime_map, W_map_all = build_c_prime_map(args.sens_csv, args.alpha_csv, args.alpha_bit, args.alpha_default)

    print("--- Phase 2: 평가기 초기화 / 대상 레이어 스캔 ---", flush=True)
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

    # output paths
    os.makedirs(args.output_dir, exist_ok=True)
    training_csv = args.training_samples_csv if args.training_samples_csv else os.path.join(args.output_dir, "training_samples.csv")
    surrogate_ckpt_dir = args.surrogate_ckpt_dir if args.surrogate_ckpt_dir else os.path.join(args.output_dir, "surrogate_ckpt")
    os.makedirs(os.path.dirname(training_csv) or ".", exist_ok=True)
    os.makedirs(surrogate_ckpt_dir, exist_ok=True)

    warmup_ckpt_path = args.warmup_ckpt_path.strip() if args.warmup_ckpt_path else os.path.join(args.output_dir, "warmup_ckpt.json")

    # -------------------------------------------------------
    # 초기 beam seed는 "init_avg_bits" 기준으로 한번만 생성
    # -------------------------------------------------------
    init_avg_bits = float(args.init_avg_bits)
    use_band = bool(args.use_round_band)
    quantum = float(args.round_quantum)

    B_lo0, B_hi0, _, B_target0 = compute_budget_band_for_avg_bits(init_avg_bits, W_map, quantum, use_band)
    print(
        f"[Init] init_avg_bits={init_avg_bits:.3f} "
        f"(band={'on' if use_band else 'off'}, quantum={quantum}) "
        f"→ Σw·b in [{B_lo0}, {B_hi0}], target≈{B_target0}/{sum_w}={B_target0/sum_w:.6f}",
        flush=True,
    )

    b_seed = get_initial_seed(C_prime_filtered, W_map, init_avg_bits, args.bmin, args.bmax)
    if args.init_assign_csv and os.path.exists(args.init_assign_csv):
        print(f"--- 덮어쓰기: {args.init_assign_csv} 에서 시드 로드 ---", flush=True)
        b_seed_from_csv = load_seed_from_csv(args.init_assign_csv)
        if b_seed_from_csv:
            b_seed.update(b_seed_from_csv)

    b_seed = ensure_complete_assignment(b_seed, target_layers_list, args.bmin)
    b_seed = project_to_weighted_band(b_seed, W_map, C_prime_filtered, B_lo0, B_hi0, args.bmin, args.bmax)

    # init 후보 생성 (proxy topK만 PPL)
    initial_candidates: Dict[Tuple[Tuple[str, int], ...], float] = {}
    initial_candidates[tuple(sorted(b_seed.items()))] = proxy_loss_calc(b_seed)

    for _ in range(args.beam_size * args.expansion_k):
        nb = generate_random_neighbor(b_seed, target_layers_list, args.bmin, args.bmax)
        if nb:
            nb = ensure_complete_assignment(nb, target_layers_list, args.bmin)
            nb = project_to_weighted_band(nb, W_map, C_prime_filtered, B_lo0, B_hi0, args.bmin, args.bmax)
            initial_candidates[tuple(sorted(nb.items()))] = proxy_loss_calc(nb)

    init_items = sorted(initial_candidates.items(), key=lambda x: x[1])
    K0 = max(int(args.true_eval_topk), int(args.beam_size))

    beam = []
    print(f"--- 초기 true PPL 평가 (proxy top-{min(K0,len(init_items))}) ---", flush=True)
    for bt, l_score in tqdm(init_items[: min(K0, len(init_items))], desc="초기 PPL 평가"):
        b_dict = dict(bt)
        ppl = ppl_eval.evaluate(b_dict)
        beam.append((ppl, float(l_score), b_dict, float(l_score)))
    beam.sort(key=lambda x: x[0])
    beam = beam[: args.beam_size]

    if not beam:
        raise RuntimeError("초기 beam이 비었습니다. (후보 생성/프로젝션/레이어 매칭 확인)")

    print(f"[Init] beam best ppl={beam[0][0]:.4f} | wavg={weighted_sum_bits(beam[0][2], W_map)/sum_w:.6f}", flush=True)

    # -------------------------------------------------------
    # Warmup reuse / or run warmup
    # -------------------------------------------------------
    gen_counter = 0
    warmup_core_gens: List[int] = []

    if args.reuse_warmup and os.path.exists(warmup_ckpt_path):
        ck = load_warmup_ckpt(warmup_ckpt_path)
        gen_counter = int(ck["gen_counter"])
        beam = _beam_from_serializable(ck["beam"])
        warmup_core_gens = [int(x) for x in ck.get("warmup_core_gens", [])]
        print(f"[Warmup-Resume] loaded: {warmup_ckpt_path}", flush=True)
        print(f"[Warmup-Resume] gen_counter={gen_counter} | beam_size={len(beam)}", flush=True)
    else:
        print(
            f"--- Stage 1 (v10_1): Warmup Range "
            f"gens={int(args.warmup_generations)} "
            f"avg_bits in [{float(args.warmup_bits_lo):.3f}, {float(args.warmup_bits_hi):.3f}] "
            f"sampling={args.warmup_bits_sampling} ---",
            flush=True,
        )
        rng = random.Random(int(args.seed) + 12345)

        warmup_targets: List[float] = []
        for _ in range(int(args.warmup_generations)):
            avg_bits_t = sample_warmup_avg_bits(gen_counter, args, rng)
            warmup_targets.append(float(avg_bits_t))

            B_lo, B_hi, _, B_target = compute_budget_band_for_avg_bits(avg_bits_t, W_map, quantum, use_band)
            tqdm.write(
                f"[Warmup] G-{gen_counter} avg_bits_target={avg_bits_t:.4f} "
                f"(band={'on' if use_band else 'off'}) "
                f"Σw·b in [{B_lo},{B_hi}] target≈{B_target}/{sum_w}={B_target/sum_w:.6f}"
            )

            new_beam, evaluated = run_generation(
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
                surrogate=None,
                measurement_mode="full",
                desc="warmup-full-range",
                rng=random.Random(int(args.seed) + 1000 + gen_counter),
            )
            beam = new_beam

            rows = [{
                "generation": gen_counter,
                "proxy_loss": e["proxy_loss"],
                "measured_ppl": e["ppl"],
                "bit_assignment_json": bits_to_json(e["bits"]),
                "avg_bits_target": float(avg_bits_t),
            } for e in evaluated]
            append_training_samples(training_csv, rows)

            # warmup_core: 초반 N개 gen만 유지 (v9 유지). 필요하면 v10_2에서 budget-bin core로 확장 가능.
            if len(warmup_core_gens) < int(args.warmup_core_keep_gens):
                warmup_core_gens.append(gen_counter)

            gen_counter += 1

        ck_payload = {
            "gen_counter": gen_counter,
            "beam": _beam_to_serializable(beam),
            "warmup_core_gens": list(warmup_core_gens),
            "warmup_targets": warmup_targets,
            "meta": {
                "seed": int(args.seed),
                "beam_size": int(args.beam_size),
                "expansion_k": int(args.expansion_k),
                "filter_p": int(args.filter_p),
                "bmin": int(args.bmin),
                "bmax": int(args.bmax),
                "use_round_band": bool(args.use_round_band),
                "round_quantum": float(args.round_quantum),
                "init_avg_bits": float(args.init_avg_bits),
                "warmup_generations": int(args.warmup_generations),
                "warmup_bits_lo": float(args.warmup_bits_lo),
                "warmup_bits_hi": float(args.warmup_bits_hi),
                "warmup_bits_sampling": str(args.warmup_bits_sampling),
                "warmup_bits_grid_step": float(args.warmup_bits_grid_step),
            },
        }
        save_warmup_ckpt(warmup_ckpt_path, ck_payload)
        print(f"[Warmup-CKPT] saved: {warmup_ckpt_path}", flush=True)

    # -------------------------------------------------------
    # Stage 2: Offline surrogate training (범용 ckpt 생성)
    # -------------------------------------------------------
    print("--- Stage 2 (v10_1): Offline surrogate training on warmup dataset ---", flush=True)
    if not args.surrogate_static_info:
        raise ValueError("--surrogate_static_info 경로가 필요합니다.")

    trainer_args = argparse.Namespace(
        bmin=args.bmin,
        bmax=args.bmax,
        sur_train_seed=args.sur_train_seed,
        sur_train_val_ratio=args.sur_train_val_ratio,
        sur_train_pairs_per_gen=args.sur_train_pairs_per_gen,
        sur_train_top_frac=args.sur_train_top_frac,
        sur_train_hard_frac=args.sur_train_hard_frac,
        sur_train_hard_window=args.sur_train_hard_window,
        sur_train_tau_soft=args.sur_train_tau_soft,
        sur_train_batch_size=args.sur_train_batch_size,
        sur_train_num_workers=args.sur_train_num_workers,
        sur_train_d_model=args.sur_train_d_model,
        sur_train_bit_emb_dim=args.sur_train_bit_emb_dim,
        sur_train_nlayers=args.sur_train_nlayers,
        sur_train_ff_dim=args.sur_train_ff_dim,
        sur_train_token_mlp_dim=args.sur_train_token_mlp_dim,
        sur_train_dropout=args.sur_train_dropout,
        sur_train_no_proxy=args.sur_train_no_proxy,
        sur_train_tau_pair=args.sur_train_tau_pair,
        sur_train_lr=args.sur_train_lr,
        sur_train_weight_decay=args.sur_train_weight_decay,
        sur_train_grad_clip=args.sur_train_grad_clip,
        sur_train_log_interval=args.sur_train_log_interval,
        sur_train_topk=args.sur_train_topk,
        sur_train_eval_batch=args.sur_train_eval_batch,
        sur_train_epochs=args.sur_train_epochs,
        surrogate_score_cache_max=args.surrogate_score_cache_max,
    )

    trainer_device = torch.device(args.surrogate_trainer_device) if args.surrogate_trainer_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = OfflineSurrogateTrainer(
        csv_path=training_csv,
        static_info_path=args.surrogate_static_info,
        output_dir=surrogate_ckpt_dir,
        trainer_args=trainer_args,
        device=trainer_device,
    )
    best_ckpt, best_config, best_metric = trainer.train()
    print(f"[v10_1] Offline training complete: best_ckpt={best_ckpt} best_overlap@{int(args.sur_train_topk)}={best_metric:.4f}", flush=True)

    # scorer 로드(검증용)
    surrogate_device = torch.device(args.surrogate_device) if args.surrogate_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = SurrogateScorer(ckpt_path=best_ckpt, config_path=best_config, device=surrogate_device, bmin=args.bmin)
    surrogate.clear_score_cache()

    # 결과 요약 저장
    summary_path = os.path.join(args.output_dir, "v10_1_summary.json")
    summary = {
        "warmup_ckpt_path": warmup_ckpt_path,
        "training_csv": training_csv,
        "surrogate_best_ckpt": best_ckpt,
        "surrogate_config": best_config,
        "best_metric_overlap": best_metric,
        "meta": {
            "warmup_bits_lo": float(args.warmup_bits_lo),
            "warmup_bits_hi": float(args.warmup_bits_hi),
            "warmup_bits_sampling": str(args.warmup_bits_sampling),
            "round_quantum": float(args.round_quantum),
            "use_round_band": bool(args.use_round_band),
            "warmup_generations": int(args.warmup_generations),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[v10_1] Summary saved: {summary_path}", flush=True)

    # 캐시 저장(선택)
    cache_path = os.path.join(args.output_dir, "ppl_cache.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in ppl_eval.ppl_cache.items()}, f, indent=2)
    print(f"[v10_1] True PPL cache saved: {cache_path}", flush=True)
    return {
        "summary_path": summary_path,
        "warmup_ckpt_path": warmup_ckpt_path,
        "training_csv": training_csv,
        "surrogate_best_ckpt": best_ckpt,
        "surrogate_config": best_config,
        "ppl_cache_path": cache_path,
        "output_dir": args.output_dir,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Step 3b - v10_1 WarmupRange + OfflineTrain (Mixer input3)")

    # paths
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--sens_csv", type=str, required=True)
    parser.add_argument("--alpha_csv", type=str, required=True)
    parser.add_argument("--prebake_root", type=str, required=True)
    parser.add_argument("--init_assign_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)

    # seed/proxy
    parser.add_argument("--alpha_bit", type=int, default=3)
    parser.add_argument("--alpha_default", type=float, default=1.0)

    # init avg_bits (seed/초기 beam용)
    parser.add_argument("--init_avg_bits", type=float, default=2.75, help="초기 seed/beam 생성에 사용하는 avg_bits")

    # warmup range (요청사항 핵심)
    parser.add_argument("--warmup_generations", type=int, default=10)
    parser.add_argument("--warmup_bits_lo", type=float, default=2.5)
    parser.add_argument("--warmup_bits_hi", type=float, default=3.0)
    parser.add_argument("--warmup_bits_sampling", type=str, default="uniform", choices=["uniform", "grid", "cycle"])
    parser.add_argument("--warmup_bits_grid_step", type=float, default=0.1, help="sampling=grid일 때 step")
    parser.add_argument("--warmup_core_keep_gens", type=int, default=3)

    # round band
    parser.add_argument("--use_round_band", action="store_true")
    parser.add_argument("--round_quantum", type=float, default=0.1)

    # ppl eval
    parser.add_argument("--eval_seq_len", type=int, default=2048)
    parser.add_argument("--ppl_cache_max", type=int, default=20000)

    # beam search
    parser.add_argument("--bmin", type=int, default=1)
    parser.add_argument("--bmax", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--expansion_k", type=int, default=20)
    parser.add_argument("--filter_p", type=int, default=80)
    parser.add_argument("--true_eval_topk", type=int, default=10)  # init 때만 사용

    # surrogate / training assets
    parser.add_argument("--training_samples_csv", type=str, default="", help="warmup 라벨 CSV 경로")
    parser.add_argument("--surrogate_ckpt_dir", type=str, default="", help="surrogate ckpt 출력 디렉토리")
    parser.add_argument("--surrogate_static_info", type=str, required=True)
    parser.add_argument("--surrogate_score_cache_max", type=int, default=200000)
    parser.add_argument("--surrogate_batch", type=int, default=1024)
    parser.add_argument("--surrogate_device", type=str, default="", help="scorer 디바이스")
    parser.add_argument("--surrogate_trainer_device", type=str, default="", help="trainer 디바이스")

    # warmup reuse
    parser.add_argument("--reuse_warmup", action="store_true")
    parser.add_argument("--warmup_ckpt_path", type=str, default="")

    # surrogate training hparams
    parser.add_argument("--sur_train_seed", type=int, default=42)
    parser.add_argument("--sur_train_val_ratio", type=float, default=0.2)
    parser.add_argument("--sur_train_pairs_per_gen", type=int, default=2000)
    parser.add_argument("--sur_train_top_frac", type=float, default=0.3)
    parser.add_argument("--sur_train_hard_frac", type=float, default=0.3)
    parser.add_argument("--sur_train_hard_window", type=int, default=8)
    parser.add_argument("--sur_train_tau_soft", type=float, default=1.0)
    parser.add_argument("--sur_train_tau_pair", type=float, default=1.0)
    parser.add_argument("--sur_train_d_model", type=int, default=128)
    parser.add_argument("--sur_train_bit_emb_dim", type=int, default=32)
    parser.add_argument("--sur_train_nlayers", type=int, default=2)
    parser.add_argument("--sur_train_ff_dim", type=int, default=256)
    parser.add_argument("--sur_train_token_mlp_dim", type=int, default=128)
    parser.add_argument("--sur_train_dropout", type=float, default=0.1)
    parser.add_argument("--sur_train_no_proxy", action="store_true")
    parser.add_argument("--sur_train_epochs", type=int, default=50)
    parser.add_argument("--sur_train_batch_size", type=int, default=256)
    parser.add_argument("--sur_train_lr", type=float, default=3e-4)
    parser.add_argument("--sur_train_weight_decay", type=float, default=0.01)
    parser.add_argument("--sur_train_grad_clip", type=float, default=1.0)
    parser.add_argument("--sur_train_num_workers", type=int, default=2)
    parser.add_argument("--sur_train_log_interval", type=int, default=200)
    parser.add_argument("--sur_train_topk", type=int, default=10)
    parser.add_argument("--sur_train_eval_batch", type=int, default=512)

    # misc
    parser.add_argument("--seed", type=int, default=42)

    return parser


def run(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = build_arg_parser().parse_args()
    return main(args)


if __name__ == "__main__":
    run()
