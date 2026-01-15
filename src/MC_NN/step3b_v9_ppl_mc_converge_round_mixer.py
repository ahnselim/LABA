#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3b — Surrogate-guided Monte-Carlo Beam Search (Pre-baked, convergence) [Mixer]  [v9 patch]
=================================================================================
[v9] “매 generation마다” micro finetune (warmup_core + current)  ← quality_bad 트리거 제거

✅ v2 기반 요구사항 유지
- neighbor 생성(generate_random_neighbor) / budget projection(project_to_weighted_band/budget) 로직 유지
- finalists(filter_p)는 proxy_loss로 1차 필터
- finalists는 surrogate score로 랭킹 후 top-K(+probe)만 true PPL 측정
- beam 갱신은 "측정된 true PPL" 기준
- surrogate score 캐시(LRU) + true PPL 캐시(LRU) 유지
- init에서도 surrogate 랭킹 후 top-K만 true PPL 측정

✅ v6의 global best/로깅/저장 규칙 유지
1) global best 변수 추가
2) ppl_curve.csv에서 num_finalists / num_true_eval 저장 제거
3) ppl_curve.csv 컬럼:
   generation,best_ppl,global_best_ppl,best_L,best_sur,weighted_avg_bits,timestamp,note
4) bit_assign.csv는 "global best가 갱신될 때만" 저장/갱신 (초기 1회 + 갱신 시만)
5) global best 갱신을 먼저 수행한 뒤, 같은 generation row에 갱신된 global_best_ppl 기록 (lag 제거)

✅ v9 patch:
- Gate 제거 유지, Stage4에서만 운용
- 매 generation:
  * topK + probe(고정)만 실제 PPL 측정
  * 측정된 샘플을 즉시 training_samples.csv에 append
  * replay = warmup_core + current(gen) 로만 구성
  * 1 epoch micro finetune (pair_budget 기반) → 즉시 surrogate 갱신
  * quality_bad(EMA/overlap/ndcg/cooldown/probe-adaptive) 로직 제거

Usage 예시:
CUDA_VISIBLE_DEVICES=1 nohup \
python MC_NN/step3b_v9_ppl_mc_converge_round_mixer.py \
  --model_id meta-llama/Llama-3.2-3B \
  --gpu_id 0 \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --prebake_root ../artifacts/montecarlo/prebake \
  --avg_bits 2.50 \
  --use_round_band --round_quantum 0.1 \
  --beam_size 10 --expansion_k 20 --filter_p 80 \
  --true_eval_topk 10 --s4_probe_n 8 \
  --surrogate_static_info ../artifacts/data/surrogate_data/static_info_v3.json \
  --output_dir ../artifacts/montecarlo/step3b_surrogate_roundband_mixer_v9 \
  --reuse_warmup \
  --warmup_ckpt_path ../artifacts/montecarlo/step3b_surrogate_roundband_mixer_v9/warmup_ckpt.json \
  > ./log/run_mixer_3b_v7.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=1 nohup \
python MC_NN/step3b_v9_ppl_mc_converge_round_mixer.py \
  --model_id meta-llama/Llama-3.2-3B \
  --gpu_id 0 \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --prebake_root ../artifacts/montecarlo/prebake \
  --avg_bits 2.50 \
  --use_round_band --round_quantum 0.1 \
  --beam_size 10 --expansion_k 20 --filter_p 80 \
  --true_eval_topk 10 \
  --s4_probe_n 8 \
  --s4_probe_pool 30 \
  --s4_pair_budget 6000 \
  --reuse_warmup \
  --warmup_ckpt_path ../artifacts/montecarlo/step3b_surrogate_roundband_mixer_v9/warmup_ckpt.json \
  --surrogate_static_info ../artifacts/data/surrogate_data/static_info_v3.json \
  --output_dir ../artifacts/montecarlo/step3b_surrogate_roundband_mixer_v9 \
  > ./log/run_mixer_3b_v9_scratch.log 2>&1 &

"""

import os, gc, csv, json, math, random, argparse, time, sys
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
from copy import deepcopy

from dataclasses import dataclass

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

from neural_net.train_brp_pairwise_surrogate_mixer_input3 import (
    BitSequenceEncoderInput3,
    BRPPairIterableDataset,
    BRPPairwiseSurrogate,
    collate_fn,
    evaluate_ranking_metrics,
    safe_makedirs,
    compute_ndcg_at_k,
    load_generation_pool,
    parse_alpha_per_bit,
    parse_bit_assignment,
    set_all_seeds,
    topk_overlap,
    zscore,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from module.montecarlo import (
    _safe_name,
    atomic_save_bit_assign_csv,
    build_c_prime_map,
    calculate_proxy_loss,
    ensure_complete_assignment,
    generate_random_neighbor,
    get_initial_seed,
    gcd_list,
    load_seed_from_csv,
    project_to_weighted_band,
    project_to_weighted_budget,
    run_live_ppl_eval,
    target_weighted_sum,
    weighted_sum_bits,
)

# -----------------------------------------------
# Step 2/4에서 가져오기 (경로/네이밍)
# -----------------------------------------------
try:
    from step2_alpha_estimation import _canonical_dataset_name
except ImportError:
    print("오류: step2_alpha_estimation.py가 같은 디렉토리에 필요합니다.")
    exit(1)

# =========================================================
# Helper functions for Monte-Carlo search logic are imported from
# module.montecarlo to avoid duplication across mixer variants.
# =========================================================


# =========================================================
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

        C_map: Dict[str, float] = static_info.get("C_map", None)
        if C_map is None:
            C_map = static_info.get("C_prime_map", {})
            print("[Surrogate][Warn] static_info에 C_map이 없어 C_prime_map을 C_map으로 사용합니다.", flush=True)

        W_map: Dict[str, int] = static_info.get("W_map", None)
        if W_map is None:
            raise ValueError("[Surrogate] static_info.json missing W_map")

        alpha_map: Dict[str, Any] = static_info.get("alpha_map", None)
        if alpha_map is None:
            alpha_map = {}
            print("[Surrogate][Warn] static_info에 alpha_map이 없어 alpha=1.0으로 처리합니다.", flush=True)

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

        norm = cfg.get("norm", {})

        C_mu = float(norm.get("C_log_mu", float(C_log.mean())))
        C_sd = float(norm.get("C_log_sd", float(C_log.std() + 1e-6)))
        W_mu = float(norm.get("W_log_mu", float(W_log.mean())))
        W_sd = float(norm.get("W_log_sd", float(W_log.std() + 1e-6)))

        C_log_n = (C_log - C_mu) / (C_sd + 1e-12)
        W_log_n = (W_log - W_mu) / (W_sd + 1e-12)

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

    def clear_score_cache(self):
        self.cache.clear()


# =========================================================
# Offline Surrogate Trainer Wrapper (v7: replay_gens + pair_budget)
# =========================================================
class OfflineSurrogateTrainer:
    def __init__(
        self,
        csv_path: str,
        static_info_path: str,
        output_dir: str,
        trainer_args: argparse.Namespace,
        device: Optional[torch.device] = None,
    ):
        self.csv_path = csv_path
        self.static_info_path = static_info_path
        self.output_dir = output_dir
        self.args = trainer_args
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        safe_makedirs(self.output_dir)

    def train(
        self,
        warm_start_ckpt: Optional[str] = None,
        replay_gens: Optional[List[int]] = None,
        pair_budget: int = 0,
        max_epochs: Optional[int] = None,
        split_seed: Optional[int] = None,
    ) -> Tuple[str, str, float]:
        """
        - replay_gens가 주어지면: 그 gens만 학습/검증 풀로 사용
        - pair_budget>0이면: (대략) update당 pair 총량을 고정하려고 pairs_per_gen을 동적으로 조절
          => epochs=1 기본(또는 max_epochs로 상한)
        """
        set_all_seeds(self.args.sur_train_seed)

        layer_names, gen2cands, static_info = load_generation_pool(
            csv_path=self.csv_path,
            static_info_path=self.static_info_path,
            bmin=self.args.bmin,
        )

        if replay_gens is not None:
            replay_set = set(int(g) for g in replay_gens)
            gen2cands = {g: gen2cands[g] for g in gen2cands.keys() if int(g) in replay_set}

        L = len(layer_names)
        gens_all = sorted(gen2cands.keys())
        if len(gens_all) < 2:
            raise RuntimeError(f"[Trainer] Not enough generations to split: {gens_all}")

        _split_seed = int(self.args.sur_train_seed if split_seed is None else split_seed)
        rng = random.Random(_split_seed)
        gens_shuf = gens_all[:]
        rng.shuffle(gens_shuf)
        n_val = max(1, int(round(len(gens_shuf) * self.args.sur_train_val_ratio)))
        val_gens = sorted(gens_shuf[:n_val])
        train_gens = sorted(gens_shuf[n_val:])

        print(
            f"[Trainer] split: total={len(gens_all)} train={len(train_gens)} val={len(val_gens)} "
            f"(replay={'on' if replay_gens is not None else 'off'})",
            flush=True,
        )

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
        alpha_table = np.zeros((L, 3), dtype=np.float32)
        for i, ln in enumerate(layer_names):
            c = float(C_map.get(ln, 0.0))
            w = float(W_map.get(ln, 1.0))
            C_log[i] = math.log(max(c, 1e-30))
            W_log[i] = math.log(max(w, 1.0))
            a2, a3, a4 = parse_alpha_per_bit(alpha_map.get(ln, None))
            alpha_table[i, 0] = a2
            alpha_table[i, 1] = a3
            alpha_table[i, 2] = a4

        C_log_n, C_mu, C_sd = zscore(C_log)
        W_log_n, W_mu, W_sd = zscore(W_log)
        eps = 1e-6
        a = np.clip(alpha_table, eps, 1.0 - eps).astype(np.float32)
        alpha_logit = np.log(a / (1.0 - a)).astype(np.float32)
        alpha_flat_n, A_mu, A_sd = zscore(alpha_logit.reshape(-1))
        alpha_logit_n = alpha_flat_n.reshape(L, 3).astype(np.float32)

        C_log_t = torch.tensor(C_log_n, dtype=torch.float32, device=self.device)
        W_log_t = torch.tensor(W_log_n, dtype=torch.float32, device=self.device)
        alpha_logit_table_t = torch.tensor(alpha_logit_n, dtype=torch.float32, device=self.device)

        # --------- dynamic pair budget -> pairs_per_gen ----------
        pairs_per_gen = int(self.args.sur_train_pairs_per_gen)
        epochs = int(self.args.sur_train_epochs)

        if pair_budget and pair_budget > 0:
            # update당 총 pair_budget을 목표로: train_gens에 균등 분배
            denom = max(1, len(train_gens))
            pairs_per_gen = max(64, int(math.ceil(float(pair_budget) / float(denom))))
            epochs = 1
            if max_epochs is not None:
                epochs = min(epochs, int(max_epochs))
            print(
                f"[Trainer] pair_budget={pair_budget} -> pairs_per_gen≈{pairs_per_gen} (epochs={epochs})",
                flush=True,
            )

        train_ds = BRPPairIterableDataset(
            gen2cands=gen2cands,
            gens=train_gens,
            layer_names=layer_names,
            pairs_per_gen=pairs_per_gen,
            top_frac=self.args.sur_train_top_frac,
            hard_frac=self.args.sur_train_hard_frac,
            hard_window=self.args.sur_train_hard_window,
            tau_soft=self.args.sur_train_tau_soft,
            seed=self.args.sur_train_seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.sur_train_batch_size,
            num_workers=self.args.sur_train_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )

        encoder = BitSequenceEncoderInput3(
            L=L,
            d_model=self.args.sur_train_d_model,
            bit_emb_dim=self.args.sur_train_bit_emb_dim,
            nlayers=self.args.sur_train_nlayers,
            ff_dim=self.args.sur_train_ff_dim,
            token_mlp_dim=self.args.sur_train_token_mlp_dim,
            dropout=self.args.sur_train_dropout,
            use_proxy=(not self.args.sur_train_no_proxy),
        )
        model = BRPPairwiseSurrogate(encoder=encoder, tau_pair=self.args.sur_train_tau_pair).to(self.device)

        if warm_start_ckpt and os.path.exists(warm_start_ckpt):
            ckpt = torch.load(warm_start_ckpt, map_location=self.device)
            state = ckpt.get("model_state", ckpt)
            model.load_state_dict(state, strict=True)
            print(f"[Trainer] warm-started from {warm_start_ckpt}", flush=True)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=self.args.sur_train_lr,
            weight_decay=self.args.sur_train_weight_decay,
        )

        config_out = {
            "model_type": "BRP_pairwise_surrogate_mixer_input3",
            "csv_path": self.csv_path,
            "json_path": self.static_info_path,
            "layer_names": layer_names,
            "L": L,
            "norm": {
                "C_log_mu": C_mu,
                "C_log_sd": C_sd,
                "W_log_mu": W_mu,
                "W_log_sd": W_sd,
                "alpha_logit_mu": A_mu,
                "alpha_logit_sd": A_sd,
            },
            "hparams": {
                "d_model": self.args.sur_train_d_model,
                "bit_emb_dim": self.args.sur_train_bit_emb_dim,
                "nlayers": self.args.sur_train_nlayers,
                "ff_dim": self.args.sur_train_ff_dim,
                "token_mlp_dim": self.args.sur_train_token_mlp_dim,
                "dropout": self.args.sur_train_dropout,
                "no_proxy": self.args.sur_train_no_proxy,
                "tau_pair": self.args.sur_train_tau_pair,
                "score_cache_max": self.args.surrogate_score_cache_max,
            },
            "static_info": {k: static_info.get(k) for k in ["model_id", "avg_bits_target"]},
            "split": {"train_gens": train_gens, "val_gens": val_gens},
            "split_meta": {"seed": _split_seed, "val_ratio": self.args.sur_train_val_ratio},
            "replay_meta": {
                "replay": (replay_gens is not None),
                "pair_budget": pair_budget,
                "pairs_per_gen": pairs_per_gen,
                "epochs": epochs,
            },
        }
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_out, f, indent=2)

        best_metric = -1.0
        best_path = os.path.join(self.output_dir, "best.pt")
        print("[Trainer] start", flush=True)
        global_step = 0

        for epoch in range(1, epochs + 1):
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
                    bits_A,
                    bits_B,
                    C_log=C_log_t,
                    W_log=W_log_t,
                    alpha_logit_table=alpha_logit_table_t,
                    proxy_A=proxy_A,
                    proxy_B=proxy_B,
                )
                logp = torch.log(p)
                loss = F.kl_div(logp, q, reduction="batchmean")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.args.sur_train_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.sur_train_grad_clip)
                opt.step()

                run_loss += float(loss.item()) * bits_A.size(0)
                run_n += bits_A.size(0)
                global_step += 1

                if self.args.sur_train_log_interval > 0 and global_step % self.args.sur_train_log_interval == 0:
                    avg_loss = run_loss / max(run_n, 1)
                    print(f"[Trainer] epoch={epoch} step={global_step} loss={avg_loss:.6f}", flush=True)

            avg_train_loss = run_loss / max(run_n, 1)
            dt = time.time() - t0

            model.eval()
            val_metrics = evaluate_ranking_metrics(
                model=model,
                gen2cands=gen2cands,
                gens=val_gens,
                C_log_t=C_log_t,
                W_log_t=W_log_t,
                alpha_logit_table_t=alpha_logit_table_t,
                device=self.device,
                topk=self.args.sur_train_topk,
                batch_eval=self.args.sur_train_eval_batch,
            )
            score = val_metrics["topk_overlap"]
            print(
                f"[Trainer][Epoch {epoch}] train_loss={avg_train_loss:.6f} "
                f"val_top{self.args.sur_train_topk}_overlap={val_metrics['topk_overlap']:.4f} "
                f"val_ndcg@{self.args.sur_train_topk}={val_metrics['ndcg@k']:.4f} "
                f"time={dt:.1f}s",
                flush=True,
            )
            if score > best_metric:
                best_metric = score
                torch.save(
                    {"model_state": model.state_dict(), "best_metric": best_metric, "epoch": epoch, "config": config_out},
                    best_path,
                )
                print(f"[Trainer] best updated -> {best_path} (top{self.args.sur_train_topk}_overlap={best_metric:.4f})", flush=True)

        torch.cuda.empty_cache()
        return best_path, config_path, best_metric


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
            raise ValueError(f"Pre-bake 디렉토리({bit2_dir})에서 유효한 레이어를 찾지 못했습니다.")
        print(f"[PplEvaluator] 탐색 대상 레이어 {len(self.target_layers)}개 확인.", flush=True)

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
                    print(f"[경고] Pre-baked 파일 없음: {file_path}. 원본 유지.", flush=True)
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
            module.weight.data.copy_(orig_w.to(device=module.weight.device, dtype=module.weight.dtype))


# =========================================================
# Dataset & Generation Helpers
# =========================================================
TRAINING_SAMPLE_COLUMNS = [
    "generation",
    "proxy_loss",
    "measured_ppl",
    "bit_assignment_json",
]


def bits_to_json(bit_dict: Dict[str, int]) -> str:
    return json.dumps({k: int(v) for k, v in sorted(bit_dict.items())})


def _beam_to_serializable(beam: List[Tuple[float, float, Dict[str,int], float]]) -> List[Dict[str, Any]]:
    out = []
    for (ppl, L, bits, sur) in beam:
        out.append({
            "ppl": float(ppl),
            "L": float(L),
            "sur": float(sur),
            "bits": {k: int(v) for k, v in sorted(bits.items())},
        })
    return out


def _beam_from_serializable(rows: List[Dict[str, Any]]) -> List[Tuple[float, float, Dict[str,int], float]]:
    beam = []
    for r in rows:
        bits = {k: int(v) for k, v in r["bits"].items()}
        beam.append((float(r["ppl"]), float(r["L"]), bits, float(r.get("sur", r["L"])) ))
    beam.sort(key=lambda x: x[0])
    return beam


def save_warmup_ckpt(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def load_warmup_ckpt(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_training_samples(csv_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    dirpath = os.path.dirname(csv_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_SAMPLE_COLUMNS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pick_probe_indices(
    total: int,
    topk_idx: List[int],
    probe_n: int,
    probe_pool: int,
    rng: random.Random,
) -> List[int]:
    """
    probe 후보: topk 다음 구간([topk, topk+probe_pool))에서 우선 샘플링,
    부족하면 나머지에서 랜덤.
    """
    if probe_n <= 0:
        return []
    topk_set = set(topk_idx)
    cand = [i for i in range(min(total, max(topk_idx) + 1 if topk_idx else 0), min(total, (len(topk_idx) + probe_pool))) if i not in topk_set]
    # 위 cand가 비거나 topk_idx가 비어있을 수 있으니 좀 더 안전하게
    if not cand:
        cand = [i for i in range(total) if i not in topk_set]
    if not cand:
        return []
    if len(cand) <= probe_n:
        return cand
    return rng.sample(cand, probe_n)


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
) -> Tuple[
    List[Tuple[float, float, Dict[str, int], float]],
    List[Dict[str, Any]],
    np.ndarray,
    List[Tuple[float, Tuple[Tuple[str, int], ...]]],
    List[Dict[str, Any]],
]:
    if rng is None:
        rng = random.Random(1234 + gen_idx)

    all_candidates = set()
    for _, _, b_assign, _ in beam:
        all_candidates.add(tuple(sorted(b_assign.items())))
        for _ in range(args.expansion_k):
            neighbor = generate_random_neighbor(b_assign, target_layers_list, args.bmin, args.bmax)
            if neighbor:
                neighbor = ensure_complete_assignment(neighbor, target_layers_list, args.bmin)
                neighbor = project_to_weighted_band(
                    neighbor, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax
                )
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

    # --------- choose eval indices ----------
    if measurement_mode == "full":
        eval_indices = list(range(len(finalists)))
    elif measurement_mode in {"sur_topk", "sur_topk_probe"}:
        K_true = max(int(args.true_eval_topk), int(args.beam_size))
        order = np.argsort(surrogate_scores)  # (주의: lower-is-better 가정)
        top_idx = order[: min(K_true, len(finalists))].tolist()
        eval_indices = top_idx

        probe_idx: List[int] = []
        if measurement_mode == "sur_topk":
            probe_pool = order[
                min(K_true, len(finalists)) : min(K_true + int(args.s4_probe_pool), len(finalists))
            ].tolist()
            if int(args.s4_probe_n) > 0 and len(probe_pool) > 0:
                if len(probe_pool) <= int(args.s4_probe_n):
                    probe_idx = probe_pool
                else:
                    probe_idx = rng.sample(probe_pool, int(args.s4_probe_n))
        else:  # sur_topk_probe
            probe_idx = _pick_probe_indices(
                total=len(finalists),
                topk_idx=top_idx,
                probe_n=int(args.s4_probe_n),
                probe_pool=int(args.s4_probe_pool),
                rng=rng,
            )

        eval_indices = sorted(list(dict.fromkeys(eval_indices + probe_idx)))
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
    return new_beam, measured, surrogate_scores, finalists, evaluated


def update_and_log_generation(
    gen_idx: int,
    beam: List[Tuple[float, float, Dict[str, int], float]],
    global_state: Dict[str, Any],
    args,
    W_map: Dict[str, int],
    sum_w: int,
    curve_writer,
    curve_file,
    global_bit_csv: str,
    note_prefix: str = "",
):
    best_ppl, best_L, best_bits, best_sur = beam[0]
    S = weighted_sum_bits(best_bits, W_map)
    wavg = S / sum_w if sum_w > 0 else 0.0

    gb_ppl = global_state["ppl"]
    if math.isfinite(gb_ppl) and gb_ppl > 0:
        abs_gain = gb_ppl - best_ppl
        rel_gain = abs_gain / gb_ppl
    else:
        abs_gain = float("inf")
        rel_gain = float("inf")
    improved = (abs_gain > args.converge_eps) or (rel_gain > args.converge_rel_eps)
    note = (f"{note_prefix}-global-improved" if improved else f"{note_prefix}-no-improve").strip("-")

    if improved:
        global_state.update(
            {"ppl": best_ppl, "L": best_L, "bits": best_bits, "sur": best_sur, "gen": gen_idx, "ts": int(time.time())}
        )
        atomic_save_bit_assign_csv(global_bit_csv, best_bits)
        tqdm.write(f"[GlobalBest] UPDATED @G-{gen_idx} ppl={best_ppl:.6f} saved={global_bit_csv}")

    tqdm.write(
        f"--- G-{gen_idx} 완료 | Best PPL: {best_ppl:.4f} | sur={best_sur:.6f} | "
        f"L={best_L:.2e} | w-avg={wavg:.6f} | {note} | GlobalBest: {global_state['ppl']:.4f}"
    )

    try:
        curve_writer.writerow(
            [gen_idx, f"{best_ppl:.6f}", f"{global_state['ppl']:.6f}", f"{best_L:.6e}", f"{best_sur:.6f}",
             f"{wavg:.6f}", int(time.time()), note]
        )
        curve_file.flush()
    except Exception as e:
        tqdm.write(f"[경고] ppl_curve.csv 기록 실패: {e}")

    return improved


def compute_quality_signal_from_evaluated(
    evaluated: List[Dict[str, Any]],
    kq: int,
) -> Tuple[float, float]:
    """
    evaluated subset에서 품질 신호 계산:
      - overlap@kq: surrogate_score 기반 topk vs true(ppl) topk overlap
      - ndcg@kq: surrogate_score 랭킹이 true(ppl) 랭킹을 얼마나 잘 보존하는지(작을수록 나쁨)
    주의: subset 기반 신호라 "드리프트/붕괴 감지용"으로만 씀.
    """
    if not evaluated:
        return 0.0, 0.0

    # NOTE: trainer(train_brp_pairwise_surrogate_mixer_input3.py) 정의와 1:1로 맞춤:
    #   - true_vals (PPL): 낮을수록 좋음
    #   - pred_scores (surrogate score): 낮을수록 좋음
    ppl = np.array([e["ppl"] for e in evaluated], dtype=np.float64)
    sur = np.array([e.get("surrogate_score", float("nan")) for e in evaluated], dtype=np.float64)

    # 방어적 처리: NaN/Inf score는 "최악"으로 간주 (큰 값)
    bad = ~np.isfinite(sur)
    if np.any(bad):
        sur = sur.copy()
        sur[bad] = np.nanmax(sur[np.isfinite(sur)]) + 1e6 if np.any(np.isfinite(sur)) else 1e12

    k = max(1, min(int(kq), len(ppl)))
    ov = topk_overlap(ppl, sur, k=k)
    nd = compute_ndcg_at_k(ppl, sur, k=k)
    return float(ov), float(nd)


def _ema_update(prev: Optional[float], x: float, beta: float) -> float:
    """EMA helper with prev=None fallback."""
    b = float(beta)
    if prev is None or (not math.isfinite(prev)):
        return float(x)
    return b * float(prev) + (1.0 - b) * float(x)


# =========================================================
# Main
# =========================================================
def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print("--- Phase 1: 기반 데이터 준비 ---", flush=True)

    if torch.cuda.is_available():
        device_map = {"": int(args.gpu_id)}
        torch_dtype = torch.float16
    else:
        device_map = None
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch_dtype, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    original_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    print("PPL 평가용 데이터셋 로드 (wikitext-2-raw-v1)...", flush=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    eval_input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids

    print("프록시 모델(C', W) 로드...", flush=True)
    C_prime_map, W_map_all = build_c_prime_map(args.sens_csv, args.alpha_csv, args.alpha_bit, args.alpha_default)

    print("--- Phase 2: 시드 생성 및 평가기/서로게이트 초기화 ---", flush=True)
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
    C_prime_filtered = {k: float(C_prime_map[k]) for k in target_layers_list if k in C_prime_map}

    sum_w = sum(W_map.values())
    B_target = target_weighted_sum(args.avg_bits, W_map)

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
        print(f"[Budget Exact] 목표 가중평균 ≈ {B_target/sum_w:.6f} (Σw·b = {B_target})", flush=True)

    proxy_loss_calc = lambda b: calculate_proxy_loss(b, C_prime_filtered, args.bmin)

    print(
        f"Step 3 프록시로 *기본* 시드 생성 (Avg Bits 목표: {args.avg_bits} → "
        f"Target Σ w·b = {B_target} / Σw={sum_w}, 목표 가중평균 ≈ {B_target/sum_w:.6f})",
        flush=True,
    )
    b_seed = get_initial_seed(C_prime_filtered, W_map, args.avg_bits, args.bmin, args.bmax)

    if args.init_assign_csv and os.path.exists(args.init_assign_csv):
        print(f"--- 덮어쓰기: {args.init_assign_csv} 에서 시드 로드 ---", flush=True)
        b_seed_from_csv = load_seed_from_csv(args.init_assign_csv)
        if b_seed_from_csv:
            b_seed.update(b_seed_from_csv)
        else:
            print(f"[경고] {args.init_assign_csv}에서 유효한 시드를 찾지 못함.", flush=True)

    b_seed = ensure_complete_assignment(b_seed, target_layers_list, args.bmin)
    b_seed = project_to_weighted_band(b_seed, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)

    # -------------------------------------------------------
    # 초기 후보 생성 -> surrogate 없음 -> proxy 순으로 true PPL
    # -------------------------------------------------------
    initial_candidates: Dict[Tuple[Tuple[str, int], ...], float] = {}
    b_seed_proj = project_to_weighted_band(b_seed, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
    initial_candidates[tuple(sorted(b_seed_proj.items()))] = proxy_loss_calc(b_seed_proj)

    for _ in range(args.beam_size * args.expansion_k):
        neighbor = generate_random_neighbor(b_seed, target_layers_list, args.bmin, args.bmax)
        if neighbor:
            neighbor = ensure_complete_assignment(neighbor, target_layers_list, args.bmin)
            neighbor = project_to_weighted_band(neighbor, W_map, C_prime_filtered, B_lo, B_hi, args.bmin, args.bmax)
            initial_candidates[tuple(sorted(neighbor.items()))] = proxy_loss_calc(neighbor)

    init_items = sorted(initial_candidates.items(), key=lambda x: x[1])
    K0 = max(int(args.true_eval_topk), int(args.beam_size))

    beam = []
    print(f"--- 초기 true PPL 평가 (proxy top-{min(K0,len(init_items))}) ---", flush=True)
    for bt, l_score in tqdm(init_items[: min(K0, len(init_items))], desc="초기 PPL 평가"):
        b_dict = dict(bt)
        ppl = ppl_eval.evaluate(b_dict)
        beam.append((ppl, float(l_score), b_dict, float(l_score)))

    if not beam:
        raise RuntimeError("초기 beam이 비었습니다. (후보 생성/프로젝션/레이어 매칭을 확인)")

    beam.sort(key=lambda x: x[0])
    beam = beam[: args.beam_size]

    wavg0 = weighted_sum_bits(beam[0][2], W_map) / sum_w if sum_w > 0 else 0.0
    print(f"초기 빔 Best PPL: {beam[0][0]:.4f} | w-avg={wavg0:.6f} | sur={beam[0][3]:.6f}", flush=True)

    # -------------------------------------------------------
    # Global best tracker
    # -------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    global_bit_csv = os.path.join(args.output_dir, "bit_assign.csv")

    global_best_ppl, global_best_L, global_best_bits, global_best_sur = beam[0]
    atomic_save_bit_assign_csv(global_bit_csv, global_best_bits)
    print(f"[GlobalBest] init saved: {global_bit_csv} (ppl={global_best_ppl:.6f})", flush=True)

    ppl_curve_csv = os.path.join(args.output_dir, "ppl_curve.csv")
    new_file = not os.path.exists(ppl_curve_csv)
    curve_f = open(ppl_curve_csv, "a", newline="", encoding="utf-8")
    curve_w = csv.writer(curve_f)
    if new_file:
        curve_w.writerow(
            ["generation", "best_ppl", "global_best_ppl", "best_L", "best_sur", "weighted_avg_bits", "timestamp", "note"]
        )
        curve_f.flush()

    start_ts = time.time()
    global_state = {
        "ppl": global_best_ppl,
        "L": global_best_L,
        "bits": global_best_bits,
        "sur": global_best_sur,
        "gen": 0,
        "ts": int(time.time()),
    }

    def time_over():
        return (args.time_limit_sec > 0) and (time.time() - start_ts >= args.time_limit_sec)

    gen_counter = 0
    training_csv = args.training_samples_csv if args.training_samples_csv else os.path.join(args.output_dir, "training_samples.csv")
    training_dir = os.path.dirname(training_csv)
    if training_dir:
        os.makedirs(training_dir, exist_ok=True)

    surrogate_ckpt_dir = args.surrogate_ckpt_dir if args.surrogate_ckpt_dir else os.path.join(args.output_dir, "surrogate_ckpt")
    os.makedirs(surrogate_ckpt_dir, exist_ok=True)

    surrogate_device = torch.device(args.surrogate_device) if args.surrogate_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer_device = torch.device(args.surrogate_trainer_device) if args.surrogate_trainer_device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------
    # v7: Warmup cache / resume
    # -------------------------------------------------------
    warmup_ckpt_path = args.warmup_ckpt_path.strip() if args.warmup_ckpt_path else ""
    if warmup_ckpt_path == "":
        warmup_ckpt_path = os.path.join(args.output_dir, "warmup_ckpt.json")

    # -------------------------------------------------------
    # v9 Replay set: warmup_core + current only
    # -------------------------------------------------------
    warmup_core_gens: List[int] = []  # 고정 앵커 gen들만 유지

    # -------------------------------------------------------
    # Option: reuse warmup results (skip Stage1)
    # -------------------------------------------------------
    if args.reuse_warmup and os.path.exists(warmup_ckpt_path):
        ck = load_warmup_ckpt(warmup_ckpt_path)
        # 안전장치: 주요 하이퍼파라미터가 바뀌었는데 재사용하는 실수 방지용(강제 중단은 안 함, 경고만)
        meta = ck.get("meta", {})
        key_fields = ["seed","beam_size","expansion_k","filter_p","avg_bits","bmin","bmax","use_round_band","round_quantum","warmup_generations"]
        diffs = []
        for k in key_fields:
            if k in meta and str(meta[k]) != str(getattr(args, k, None)):
                diffs.append((k, meta[k], getattr(args, k, None)))
        if diffs:
            print("[Warmup-Resume][Warn] warmup_ckpt meta와 현재 args가 다릅니다. 동일 설정 재사용을 권장합니다:", flush=True)
            for (k,a,b) in diffs[:12]:
                print(f"  - {k}: ckpt={a} vs now={b}", flush=True)

        gen_counter = int(ck["gen_counter"])
        beam = _beam_from_serializable(ck["beam"])
        global_state.update(ck["global_state"])

        warmup_core_gens = [int(x) for x in ck.get("warmup_core_gens", [])]
        print(f"[Warmup-Resume] loaded warmup_ckpt: {warmup_ckpt_path}", flush=True)
        print(f"[Warmup-Resume] gen_counter={gen_counter} | beam_size={len(beam)} | global_best_ppl={global_state['ppl']:.6f}", flush=True)
    else:
        # -------------------------------------------------------
        # Stage 1: Warmup (전수 측정)
        # -------------------------------------------------------
        print(f"--- Stage 1: Warmup (gen {args.warmup_generations}) ---", flush=True)
        for _ in range(args.warmup_generations):
            new_beam, _, _, _, evaluated = run_generation(
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
                desc="warmup-full",
                rng=random.Random(args.seed + 1000 + gen_counter),
            )
            beam = new_beam

            rows = [{
                "generation": gen_counter,
                "proxy_loss": e["proxy_loss"],
                "measured_ppl": e["ppl"],
                "bit_assignment_json": bits_to_json(e["bits"]),
            } for e in evaluated]
            append_training_samples(training_csv, rows)

            update_and_log_generation(
                gen_counter, beam, global_state, args, W_map, sum_w, curve_w, curve_f, global_bit_csv, note_prefix="warmup"
            )

            if len(warmup_core_gens) < int(args.warmup_core_keep_gens):
                warmup_core_gens.append(gen_counter)

            gen_counter += 1

        # warmup 끝나면 체크포인트 저장
        ck_payload = {
            "gen_counter": gen_counter,
            "beam": _beam_to_serializable(beam),
            "global_state": deepcopy(global_state),
            "warmup_core_gens": list(warmup_core_gens),
            "meta": {
                "seed": args.seed,
                "beam_size": args.beam_size,
                "expansion_k": args.expansion_k,
                "filter_p": args.filter_p,
                "avg_bits": args.avg_bits,
                "bmin": args.bmin,
                "bmax": args.bmax,
                "use_round_band": bool(args.use_round_band),
                "round_quantum": float(args.round_quantum),
                "warmup_generations": int(args.warmup_generations),
            },
        }
        save_warmup_ckpt(warmup_ckpt_path, ck_payload)
        print(f"[Warmup-CKPT] saved: {warmup_ckpt_path}", flush=True)

    # -------------------------------------------------------
    # Stage 2: Offline training (scratch)
    # -------------------------------------------------------
    print("--- Stage 2: Offline surrogate training ---", flush=True)
    if not args.surrogate_static_info:
        raise ValueError("--surrogate_static_info 경로가 필요합니다.")

    trainer_args = argparse.Namespace(
        bmin=args.bmin,
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

    trainer = OfflineSurrogateTrainer(
        csv_path=training_csv,
        static_info_path=args.surrogate_static_info,
        output_dir=surrogate_ckpt_dir,
        trainer_args=trainer_args,
        device=trainer_device,
    )
    best_ckpt, best_config, best_metric = trainer.train()
    print(f"[Stage2] offline training complete (best overlap@{args.sur_train_topk}={best_metric:.4f})", flush=True)

    surrogate = SurrogateScorer(ckpt_path=best_ckpt, config_path=best_config, device=surrogate_device, bmin=args.bmin)
    surrogate.clear_score_cache()

    # -------------------------------------------------------
    # Stage 3 (v9): Gate 제거 → Stage 4로 바로 진입
    # -------------------------------------------------------
    print("--- Stage 3 (v9): Gate 제거 → Stage 4 (S4)로 바로 진입 ---", flush=True)

    # -------------------------------------------------------
    # Stage 4 (v9): Surrogate-guided top-K(+probe) true eval + EVERY-GEN micro finetune
    # -------------------------------------------------------
    no_improve = 0
    stable_bits = 0
    prev_best_bits = None
    stage4_gen = 0
    max_g = args.max_generations if args.max_generations > 0 else (args.generations if args.generations > 0 else 0)

    # v9: probe는 고정 사용 (args.s4_probe_n)
    # v9: quality_bad/EMA/cooldown/적응형 probe 제거

    while True:
        # S4: topK + probe (고정 probe_n)
        new_beam, _, _, _, evaluated = run_generation(
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
            measurement_mode="sur_topk_probe",
            desc="s4-topk+probe",
            rng=random.Random(args.seed + 3000 + gen_counter),
        )
        beam = new_beam

        # measured 라벨은 항상 저장(나중에 replay 대상)
        rows = [{
            "generation": gen_counter,
            "proxy_loss": e["proxy_loss"],
            "measured_ppl": e["ppl"],
            "bit_assignment_json": bits_to_json(e["bits"]),
        } for e in evaluated]
        append_training_samples(training_csv, rows)

        # 로그/글로벌 베스트 업데이트
        improved = update_and_log_generation(
            gen_counter, beam, global_state, args, W_map, sum_w, curve_w, curve_f, global_bit_csv, note_prefix="s4"
        )

        best_bits = beam[0][2]
        if prev_best_bits is not None and best_bits == prev_best_bits:
            stable_bits += 1
        else:
            stable_bits = 0
        prev_best_bits = best_bits

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        # ---------------------------------------------------
        # v9: EVERY-GEN micro finetune
        #   replay = warmup_core + current(gen_counter)
        #   - warm_start
        #   - pair_budget로 update 비용을 고정(근사)
        #   - 1 epoch만 수행
        # ---------------------------------------------------
        replay_gens = sorted(set(warmup_core_gens) | {int(gen_counter)})
        tqdm.write(f"[S4-Update][v9] @G-{gen_counter} micro-finetune (replay_gens={len(replay_gens)} = warmup_core({len(warmup_core_gens)}) + current)")

        best_ckpt, best_config, _ = trainer.train(
            warm_start_ckpt=best_ckpt,
            replay_gens=replay_gens,
            pair_budget=int(args.s4_pair_budget),
            max_epochs=1,
            split_seed=int(args.sur_train_seed) + 30000 + int(gen_counter),
        )
        surrogate = SurrogateScorer(
            ckpt_path=best_ckpt,
            config_path=best_config,
            device=surrogate_device,
            bmin=args.bmin,
        )
        surrogate.clear_score_cache()  # 매 gen 업데이트이므로 score cache는 항상 폐기

        gen_counter += 1
        stage4_gen += 1

        stop_reasons = []
        if args.patience > 0 and no_improve >= args.patience:
            stop_reasons.append(f"no-improve≥{args.patience}")
        if args.stable_bits_patience > 0 and stable_bits >= args.stable_bits_patience:
            stop_reasons.append(f"stable-bits≥{args.stable_bits_patience}")
        if time_over():
            stop_reasons.append("time-limit")
        if max_g > 0 and stage4_gen >= max_g:
            stop_reasons.append("max-generations")

        if stop_reasons:
            tqdm.write(f"수렴/중단: {', '.join(stop_reasons)}")
            break

    # -------------------------------------------------------
    # Stage 5: 최종 결과 요약
    # -------------------------------------------------------
    print("\n--- Stage 5: 탐색 완료 ---", flush=True)
    print(
        f"최종 Global Best PPL: {global_state['ppl']:.4f} | best_sur={global_state['sur']:.6f} | found@G-{global_state['gen']}",
        flush=True,
    )
    print(f"최종 할당 파일(글로벌 베스트): {global_bit_csv}", flush=True)

    best_b = global_state["bits"]
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
    parser = argparse.ArgumentParser("Step 3b - Surrogate-guided MC Search (Mixer v2 base) [v9]")

    # paths
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--sens_csv", type=str, required=True)
    parser.add_argument("--alpha_csv", type=str, required=True)
    parser.add_argument("--prebake_root", type=str, required=True)
    parser.add_argument("--init_assign_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./artifacts/bitmin/step3b_surrogate_search_mixer_v7")
    parser.add_argument("--gpu_id", type=int, default=0)

    # seed/proxy
    parser.add_argument("--avg_bits", type=float, default=2.5)
    parser.add_argument("--alpha_bit", type=int, default=3)
    parser.add_argument("--alpha_default", type=float, default=1.0)

    # round band
    parser.add_argument("--use_round_band", action="store_true", help="avg_bits 반올림 밴드 허용")
    parser.add_argument("--round_quantum", type=float, default=0.1, help="반올림 자릿수 폭")

    # ppl eval
    parser.add_argument("--eval_seq_len", type=int, default=2048)
    parser.add_argument("--ppl_cache_max", type=int, default=20000, help="true PPL LRU 캐시 최대 엔트리 수")

    # beam search
    parser.add_argument("--bmin", type=int, default=2)
    parser.add_argument("--bmax", type=int, default=4)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--expansion_k", type=int, default=20)
    parser.add_argument("--filter_p", type=int, default=80)

    # surrogate / training assets
    parser.add_argument("--training_samples_csv", type=str, default="", help="warmup/gate/s4 true-PPL 기록 CSV 경로")
    parser.add_argument("--surrogate_ckpt_dir", type=str, default="", help="surrogate checkpoint 출력 디렉토리")
    parser.add_argument("--surrogate_static_info", type=str, required=True, help="static_info.json 경로 (trainer가 요구)")
    parser.add_argument("--surrogate_score_cache_max", type=int, default=200000, help="surrogate score cache 크기(config에 저장)")
    parser.add_argument("--true_eval_topk", type=int, default=10, help="surrogate top-K만 true PPL 측정 (S4)")
    parser.add_argument("--surrogate_batch", type=int, default=1024, help="surrogate batch inference")
    parser.add_argument("--surrogate_device", type=str, default="", help="예: cuda:0 / cpu (빈 문자열이면 자동)")
    parser.add_argument("--surrogate_trainer_device", type=str, default="", help="surrogate 학습 디바이스 지정 (빈 문자열이면 자동)")

    # warmup / gate
    parser.add_argument("--warmup_generations", type=int, default=10)
    parser.add_argument("--warmup_core_keep_gens", type=int, default=3, help="warmup 중 core로 유지할 gen 개수")
    # warmup reuse
    parser.add_argument("--reuse_warmup", action="store_true",
                        help="output_dir/warmup_ckpt.json이 있으면 Stage1 warmup을 스킵하고 재사용")
    parser.add_argument("--warmup_ckpt_path", type=str, default="",
                        help="warmup 체크포인트 경로(기본: output_dir/warmup_ckpt.json)")
    parser.add_argument("--gate_round_generations", type=int, default=2)
    parser.add_argument("--gate_eval_topk", type=int, default=10)
    parser.add_argument("--gate_avg_threshold", type=float, default=0.8)
    parser.add_argument("--gate_min_threshold", type=float, default=0.7)
    parser.add_argument("--gate_pair_budget", type=int, default=20000, help="gate fail 시 finetune update당 pair budget(근사)")

    # v9: replay는 warmup_core + current만 사용 (recent/hard 미사용)

    # S4: probe (고정)
    parser.add_argument("--s4_probe_n", type=int, default=8, help="(하위호환) S4 기본 probe 개수 (v9에서는 고정 probe_n으로 사용)")
    parser.add_argument("--s4_probe_pool", type=int, default=30, help="probe를 topK 다음 구간에서 우선 뽑는 폭")
    parser.add_argument("--s4_pair_budget", type=int, default=6000, help="v9: 매 gen micro-finetune update당 pair budget(근사)")

    # surrogate training hparams (trainer parity)
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

    # convergence/safety
    parser.add_argument("--generations", type=int, default=0, help="(하위호환) >0이면 max_generations로 사용")
    parser.add_argument("--max_generations", type=int, default=0, help="0이면 수렴 조건까지 진행")
    parser.add_argument("--converge_eps", type=float, default=1e-3, help="절대 개선 임계값")
    parser.add_argument("--converge_rel_eps", type=float, default=1e-3, help="상대 개선 임계값")
    parser.add_argument("--patience", type=int, default=12, help="개선 없는 세대 허용치")
    parser.add_argument("--stable_bits_patience", type=int, default=12, help="동일 best bit 반복 허용치")
    parser.add_argument("--time_limit_sec", type=int, default=0, help="0=무제한, >0이면 시간 제한(초)")

    # misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
