#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRP-style Pairwise Surrogate Trainer (Mixer encoder) — input_3 version
=====================================================================

input_3 변경점:
- static_info.json에서 C_prime_map 대신:
    C_map, W_map, alpha_map(layer x bit)을 사용
- per-layer numeric feature를 3개로 구성:
    [log C_j, log W_j, logit(alpha_j(b_j))]
  여기서 alpha_j(b_j)는 bits로 gather해서 선택.

학습/샘플링/손실(KL), generation split, scoring 로직은 동일.

Usage:
CUDA_VISIBLE_DEVICES=1 nohup \
python neural_net/train_brp_pairwise_surrogate_mixer_input3.py \
  --csv_path ../artifacts/surrogate_data/training_samples.csv \
  --json_path ../artifacts/surrogate_data/static_info_v3.json \
  --output_dir ../artifacts/surrogate_checkpoint_input3_mixer \
  --nlayers 3 --d_model 128 --bit_emb_dim 32 --token_mlp_dim 128 --ff_dim 512 \
  --dropout 0.1 --epochs 50 --batch_size 256 --pairs_per_gen 2000 --topk 10 \
  --debug_alpha_once \
  > ./log/train_input3_mixer.log 2>&1 &

"""
import os
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader


# -----------------------------
# Utils
# -----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def parse_bit_assignment(bit_json: str, layer_names: List[str], bmin: int = 2) -> List[int]:
    """
    Accepts:
      - dict: {"layer.name": 2/3/4, ...}
      - list: [2,3,2,...] aligned with layer_names
    Returns:
      - list of ints length L
    """
    obj = json.loads(bit_json) if isinstance(bit_json, str) else bit_json

    if isinstance(obj, list):
        if len(obj) != len(layer_names):
            raise ValueError(f"bit list length mismatch: got {len(obj)} vs L={len(layer_names)}")
        return [int(x) for x in obj]

    if isinstance(obj, dict):
        out = []
        for ln in layer_names:
            out.append(int(obj.get(ln, bmin)))
        return out

    raise ValueError(f"Unsupported bit_assignment_json type: {type(obj)}")


def compute_ndcg_at_k(true_vals: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    """
    true_vals: lower is better (PPL)
    pred_scores: lower is better (model score)
    NDCG uses relevance; here we define relevance as -true_vals (monotone).
    """
    n = true_vals.shape[0]
    k = min(k, n)
    pred_rank = np.argsort(pred_scores)[:k]
    ideal_rank = np.argsort(true_vals)[:k]

    def dcg(idxs):
        rel = -true_vals[idxs]
        denom = np.log2(np.arange(2, len(idxs) + 2))
        return float(np.sum((2.0**rel - 1.0) / denom))

    dcg_val = dcg(pred_rank)
    idcg_val = dcg(ideal_rank)
    return dcg_val / (idcg_val + 1e-12)


def topk_overlap(true_vals: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    n = true_vals.shape[0]
    k = min(k, n)
    tset = set(np.argsort(true_vals)[:k].tolist())
    pset = set(np.argsort(pred_scores)[:k].tolist())
    return float(len(tset & pset)) / float(k)


def zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(x.mean())
    sd = float(x.std() + 1e-6)
    return (x - mu) / sd, mu, sd


# -----------------------------
# Data container
# -----------------------------
@dataclass
class Cand:
    bitvec: List[int]
    ppl: float
    proxy: float
    gen: int


def load_generation_pool(
    csv_path: str,
    static_info_path: str,
    bmin: int = 2,
) -> Tuple[List[str], Dict[int, List[Cand]], Dict[str, Any]]:
    with open(static_info_path, "r", encoding="utf-8") as f:
        static_info = json.load(f)

    layer_names: List[str] = static_info["layer_names"]
    df = pd.read_csv(csv_path)

    required = {"generation", "proxy_loss", "measured_ppl", "bit_assignment_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    gen2cands: Dict[int, List[Cand]] = {}
    for row in df.itertuples(index=False):
        gen = int(getattr(row, "generation"))
        proxy = float(getattr(row, "proxy_loss"))
        ppl = float(getattr(row, "measured_ppl"))
        bjson = getattr(row, "bit_assignment_json")
        bitvec = parse_bit_assignment(bjson, layer_names, bmin=bmin)
        gen2cands.setdefault(gen, []).append(Cand(bitvec=bitvec, ppl=ppl, proxy=proxy, gen=gen))

    for g in gen2cands:
        gen2cands[g].sort(key=lambda c: (c.ppl, c.proxy))

    return layer_names, gen2cands, static_info


# -----------------------------
# Pair Sampler (BRP-style)
# -----------------------------
class BRPPairIterableDataset(IterableDataset):
    """
    Yields sampled pairs (A,B) from within the same generation.
    """

    def __init__(
        self,
        gen2cands: Dict[int, List[Cand]],
        gens: List[int],
        layer_names: List[str],
        pairs_per_gen: int,
        top_frac: float,
        hard_frac: float,
        hard_window: int,
        tau_soft: float,
        seed: int,
    ):
        super().__init__()
        self.gen2cands = gen2cands
        self.gens = gens
        self.layer_names = layer_names
        self.pairs_per_gen = pairs_per_gen
        self.top_frac = top_frac
        self.hard_frac = hard_frac
        self.hard_window = hard_window
        self.tau_soft = tau_soft
        self.seed = seed
        self.L = len(layer_names)

    def _make_pair(self, rng: random.Random, cands: List[Cand]) -> Tuple[Cand, Cand]:
        n = len(cands)
        m_top = max(2, int(self.top_frac * n))

        if rng.random() < self.hard_frac:
            i = rng.randrange(0, n)
            j_lo = max(0, i - self.hard_window)
            j_hi = min(n - 1, i + self.hard_window)
            j = rng.randrange(j_lo, j_hi + 1)
            if j == i:
                j = (j + 1) % n
            return cands[i], cands[j]
        else:
            i = rng.randrange(0, m_top)
            j = rng.randrange(m_top, n) if m_top < n else rng.randrange(0, n)
            return cands[i], cands[j]

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        base_seed = self.seed + (worker.id if worker is not None else 0)
        rng = random.Random(base_seed)

        gens = self.gens[:]
        rng.shuffle(gens)

        for g in gens:
            cands = self.gen2cands.get(g, [])
            if len(cands) < 2:
                continue

            for _ in range(self.pairs_per_gen):
                A, B = self._make_pair(rng, cands)

                q = np.exp(np.array([-A.ppl / self.tau_soft, -B.ppl / self.tau_soft], dtype=np.float64))
                q = q / (q.sum() + 1e-12)

                yield {
                    "bits_A": np.asarray(A.bitvec, dtype=np.int64),
                    "bits_B": np.asarray(B.bitvec, dtype=np.int64),
                    "proxy_A": float(A.proxy),
                    "proxy_B": float(B.proxy),
                    "q": q.astype(np.float32),
                }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    bits_A = torch.tensor(np.stack([b["bits_A"] for b in batch], axis=0), dtype=torch.long)
    bits_B = torch.tensor(np.stack([b["bits_B"] for b in batch], axis=0), dtype=torch.long)
    q = torch.tensor(np.stack([b["q"] for b in batch], axis=0), dtype=torch.float32)  # (B,2)
    proxy_A = torch.tensor([b["proxy_A"] for b in batch], dtype=torch.float32).unsqueeze(-1)
    proxy_B = torch.tensor([b["proxy_B"] for b in batch], dtype=torch.float32).unsqueeze(-1)
    return {"bits_A": bits_A, "bits_B": bits_B, "q": q, "proxy_A": proxy_A, "proxy_B": proxy_B}


# -----------------------------
# Model (Mixer)
# -----------------------------
class MixerBlock(nn.Module):
    def __init__(self, L: int, d_model: int, token_mlp_dim: int = 128, channel_mlp_dim: int = 256, dropout: float = 0.1):
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
        y = self.norm1(x)
        y = y.transpose(1, 2)      # (B,d,L)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)      # (B,L,d)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mlp(y)
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
        debug_alpha_once: bool = False,  # PATCH
    ):
        super().__init__()
        self.L = L
        self.use_proxy = use_proxy

        # PATCH
        self.debug_alpha_once = bool(debug_alpha_once)
        self._dbg_printed = False

        # bits {2,3,4} -> idx {0,1,2}
        self.bit_emb = nn.Embedding(3, bit_emb_dim)

        # numeric features: [logC, logW, logit(alpha_sel)]
        self.num_proj = nn.Linear(3, bit_emb_dim)

        self.in_proj = nn.Linear(bit_emb_dim * 2, d_model)
        self.pos_emb = nn.Embedding(L, d_model)

        blocks = []
        for _ in range(nlayers):
            blocks.append(MixerBlock(L=L, d_model=d_model, token_mlp_dim=token_mlp_dim, channel_mlp_dim=ff_dim, dropout=dropout))
        self.mixer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)

        head_in = d_model + (1 if use_proxy else 0)
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(
        self,
        bits: torch.Tensor,                 # (B,L) in {2,3,4}
        C_log: torch.Tensor,                # (L,)
        W_log: torch.Tensor,                # (L,)
        alpha_logit_table: torch.Tensor,    # (L,3) normalized
        proxy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = bits.shape
        assert L == self.L

        bit_idx = torch.clamp(bits - 2, 0, 2)                  # (B,L)
        e_bit = self.bit_emb(bit_idx)                          # (B,L,bit_emb_dim)

        # alpha gather: (B,L,1)
        a_tbl = alpha_logit_table.view(1, L, 3).expand(B, L, 3)
        a_sel = torch.gather(a_tbl, 2, bit_idx.unsqueeze(-1))   # (B,L,1)

        # -------------------------
        # PATCH-DBG: one-time alpha gather sanity check
        # -------------------------
        if self.debug_alpha_once and (not self._dbg_printed):
            self._dbg_printed = True
            try:
                def _alpha_sel_for_bit(bb: int) -> float:
                    bi = torch.full((1, L), bb, device=bits.device, dtype=bits.dtype)  # (1,L)
                    idx = torch.clamp(bi - 2, 0, 2)                                   # (1,L)
                    a = torch.gather(
                        alpha_logit_table.view(1, L, 3),
                        2,
                        idx.unsqueeze(-1)
                    )  # (1,L,1)
                    return float(a[0, 0, 0].detach().item())

                a2 = _alpha_sel_for_bit(2)
                a3 = _alpha_sel_for_bit(3)
                a4 = _alpha_sel_for_bit(4)

                print("[DBG] alpha_gather sanity (layer0, logit space):", flush=True)
                print(f"  bit=2 -> {a2:.6f}", flush=True)
                print(f"  bit=3 -> {a3:.6f}", flush=True)
                print(f"  bit=4 -> {a4:.6f}", flush=True)
            except Exception as e:
                print(f"[DBG] alpha_gather sanity failed: {e}", flush=True)

        C = C_log.view(1, L, 1).expand(B, L, 1)
        W = W_log.view(1, L, 1).expand(B, L, 1)
        x_num = torch.cat([C, W, a_sel], dim=-1)                # (B,L,3)
        e_num = self.num_proj(x_num)                            # (B,L,bit_emb_dim)

        x = torch.cat([e_bit, e_num], dim=-1)                   # (B,L,2*bit_emb_dim)
        x = self.in_proj(x)                                     # (B,L,d_model)

        pos = torch.arange(L, device=bits.device).view(1, L)
        x = x + self.pos_emb(pos)

        x = self.mixer(self.dropout(x))                         # (B,L,d_model)
        x_pool = x.mean(dim=1)                                  # (B,d_model)

        if self.use_proxy:
            if proxy is None:
                proxy = torch.zeros((B, 1), device=bits.device, dtype=x_pool.dtype)
            h = torch.cat([x_pool, proxy], dim=-1)
        else:
            h = x_pool

        return self.head(h)                                     # (B,1) lower is better


class BRPPairwiseSurrogate(nn.Module):
    """
    Pairwise probability derived from scalar scores:
      p(A better than B) = sigmoid((sB - sA) / tau_pair)
    """

    def __init__(self, encoder: nn.Module, tau_pair: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.tau_pair = float(tau_pair)

    def forward_pair(
        self,
        bits_A: torch.Tensor,
        bits_B: torch.Tensor,
        C_log: torch.Tensor,
        W_log: torch.Tensor,
        alpha_logit_table: torch.Tensor,
        proxy_A: Optional[torch.Tensor] = None,
        proxy_B: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sA = self.encoder(bits_A, C_log=C_log, W_log=W_log, alpha_logit_table=alpha_logit_table, proxy=proxy_A)  # (B,1)
        sB = self.encoder(bits_B, C_log=C_log, W_log=W_log, alpha_logit_table=alpha_logit_table, proxy=proxy_B)  # (B,1)

        delta = (sB - sA) / max(self.tau_pair, 1e-6)
        pA = torch.sigmoid(delta).clamp(1e-6, 1 - 1e-6)
        pB = (1.0 - pA).clamp(1e-6, 1 - 1e-6)
        p = torch.cat([pA, pB], dim=-1)  # (B,2)
        return p, sA, sB

    def score_single(
        self,
        bits: torch.Tensor,
        C_log: torch.Tensor,
        W_log: torch.Tensor,
        alpha_logit_table: torch.Tensor,
        proxy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(bits, C_log=C_log, W_log=W_log, alpha_logit_table=alpha_logit_table, proxy=proxy)  # (B,1)


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate_ranking_metrics(
    model: BRPPairwiseSurrogate,
    gen2cands: Dict[int, List[Cand]],
    gens: List[int],
    C_log_t: torch.Tensor,
    W_log_t: torch.Tensor,
    alpha_logit_table_t: torch.Tensor,
    device: torch.device,
    topk: int,
    batch_eval: int = 512,
) -> Dict[str, float]:
    overlaps = []
    ndcgs = []

    for g in gens:
        cands = gen2cands.get(g, [])
        if len(cands) < 2:
            continue

        true_ppl = np.array([c.ppl for c in cands], dtype=np.float64)
        proxy = np.array([c.proxy for c in cands], dtype=np.float32).reshape(-1, 1)
        bits = np.array([c.bitvec for c in cands], dtype=np.int64)

        scores = np.zeros((len(cands),), dtype=np.float64)
        for i in range(0, len(cands), batch_eval):
            j = min(len(cands), i + batch_eval)
            b = torch.tensor(bits[i:j], dtype=torch.long, device=device)
            p = torch.tensor(proxy[i:j], dtype=torch.float32, device=device)
            s = model.score_single(b, C_log=C_log_t, W_log=W_log_t, alpha_logit_table=alpha_logit_table_t, proxy=p).squeeze(-1)
            scores[i:j] = s.detach().cpu().numpy().astype(np.float64)

        overlaps.append(topk_overlap(true_ppl, scores, topk))
        ndcgs.append(compute_ndcg_at_k(true_ppl, scores, topk))

    if not overlaps:
        return {"topk_overlap": 0.0, "ndcg@k": 0.0}
    return {"topk_overlap": float(np.mean(overlaps)), "ndcg@k": float(np.mean(ndcgs))}


# -----------------------------
# Alpha parsing helper
# -----------------------------
AlphaSpec = Union[float, int, Dict[str, float], Dict[int, float], List[float], Tuple[float, float, float]]

def parse_alpha_per_bit(am: Optional[AlphaSpec]) -> Tuple[float, float, float]:
    """
    alpha_map[layer] 가 아래 중 무엇이든 받아서 (a2,a3,a4)로 통일.
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


# -----------------------------
# Main
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # split
    parser.add_argument("--val_ratio", type=float, default=0.2, help="generation 단위 split 비율")
    parser.add_argument("--seed", type=int, default=42)

    # bits range (for parsing fallback only)
    parser.add_argument("--bmin", type=int, default=2)
    parser.add_argument("--bmax", type=int, default=4)

    # pair sampling
    parser.add_argument("--pairs_per_gen", type=int, default=2000)
    parser.add_argument("--top_frac", type=float, default=0.3)
    parser.add_argument("--hard_frac", type=float, default=0.3)
    parser.add_argument("--hard_window", type=int, default=8)

    # BRP temperatures
    parser.add_argument("--tau_soft", type=float, default=1.0)
    parser.add_argument("--tau_pair", type=float, default=1.0)

    # model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--bit_emb_dim", type=int, default=32)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--token_mlp_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_proxy", action="store_true")

    # PATCH: debug flag
    parser.add_argument("--debug_alpha_once", action="store_true", help="print alpha sanity check once")

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_interval", type=int, default=200)

    # eval
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--eval_batch", type=int, default=512)

    # misc
    parser.add_argument(
        "--score_cache_max",
        type=int,
        default=200000,
        help="surrogate score cache 크기(config.hparams에 기록)",
    )

    return parser


def train_surrogate_mixer(
    args: argparse.Namespace,
    replay_gens: Optional[List[int]] = None,
    pair_budget: int = 0,
    max_epochs: Optional[int] = None,
    warm_start_ckpt: Optional[str] = None,
    split_seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[str, str, float]:
    safe_makedirs(args.output_dir)
    set_all_seeds(int(args.seed))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not getattr(args, "_suppress_env_log", False):
        print(f"[Env] device={device}", flush=True)

    layer_names, gen2cands, static_info = load_generation_pool(
        csv_path=args.csv_path,
        static_info_path=args.json_path,
        bmin=args.bmin,
    )

    if replay_gens is not None:
        replay_set = set(int(g) for g in replay_gens)
        gen2cands = {g: gen2cands[g] for g in gen2cands if int(g) in replay_set}
        if not gen2cands:
            raise RuntimeError("[Train] replay_gens filtering removed all generations.")

    L = len(layer_names)
    gens_all = sorted(gen2cands.keys())
    if len(gens_all) < 2:
        raise RuntimeError(f"Not enough generations in CSV: {gens_all}")

    split_seed_eff = int(args.seed if split_seed is None else split_seed)
    rng = random.Random(split_seed_eff)
    gens_shuf = gens_all[:]
    rng.shuffle(gens_shuf)
    n_val = max(1, int(round(len(gens_shuf) * args.val_ratio)))
    val_gens = sorted(gens_shuf[:n_val])
    train_gens = sorted(gens_shuf[n_val:])

    print(
        f"[Split] total_gens={len(gens_all)} train={len(train_gens)} val={len(val_gens)} "
        f"(replay={'on' if replay_gens is not None else 'off'})",
        flush=True,
    )

    C_map: Dict[str, float] = static_info.get("C_map", None)
    W_map: Dict[str, int] = static_info.get("W_map", None)
    alpha_map: Dict[str, Any] = static_info.get("alpha_map", None)

    if C_map is None:
        C_map = static_info.get("C_prime_map", {})
        print("[Warn] static_info.json has no C_map. Falling back to C_prime_map as C_map.", flush=True)

    if W_map is None:
        raise ValueError("static_info.json missing W_map")

    if alpha_map is None:
        alpha_map = {}
        print("[Warn] static_info.json has no alpha_map. Using alpha=1.0 for all bits.", flush=True)

    missing_alpha = sum(1 for ln in layer_names if ln not in alpha_map)
    if missing_alpha > 0:
        miss_list = [ln for ln in layer_names if ln not in alpha_map][:10]
        print(f"[Check] missing_alpha={missing_alpha}/{len(layer_names)} (show up to 10):", flush=True)
        for ln in miss_list:
            print(f"  - {ln}", flush=True)
    else:
        print(f"[Check] missing_alpha=0/{len(layer_names)} (all covered)", flush=True)

    C_log = np.zeros((L,), dtype=np.float32)
    W_log = np.zeros((L,), dtype=np.float32)
    alpha_table = np.zeros((L, 3), dtype=np.float32)

    for i, ln in enumerate(layer_names):
        c = float(C_map.get(ln, 0.0))
        w = float(W_map.get(ln, 1.0))
        C_log[i] = math.log(max(c, 1e-30))
        W_log[i] = math.log(max(w, 1.0))

        am = alpha_map.get(ln, None)
        a2, a3, a4 = parse_alpha_per_bit(am)
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

    C_log_t = torch.tensor(C_log_n, dtype=torch.float32, device=device)
    W_log_t = torch.tensor(W_log_n, dtype=torch.float32, device=device)
    alpha_logit_table_t = torch.tensor(alpha_logit_n, dtype=torch.float32, device=device)

    pairs_per_gen = int(args.pairs_per_gen)
    epochs = int(args.epochs)

    if pair_budget and pair_budget > 0:
        denom = max(1, len(train_gens))
        pairs_per_gen = max(64, int(math.ceil(float(pair_budget) / float(denom))))
        epochs = 1
        if max_epochs is not None:
            epochs = max(1, min(epochs, int(max_epochs)))
        print(
            f"[Train] pair_budget={pair_budget} -> pairs_per_gen≈{pairs_per_gen} (epochs={epochs})",
            flush=True,
        )

    train_ds = BRPPairIterableDataset(
        gen2cands=gen2cands,
        gens=train_gens,
        layer_names=layer_names,
        pairs_per_gen=pairs_per_gen,
        top_frac=args.top_frac,
        hard_frac=args.hard_frac,
        hard_window=args.hard_window,
        tau_soft=args.tau_soft,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    encoder = BitSequenceEncoderInput3(
        L=L,
        d_model=args.d_model,
        bit_emb_dim=args.bit_emb_dim,
        nlayers=args.nlayers,
        ff_dim=args.ff_dim,
        token_mlp_dim=args.token_mlp_dim,
        dropout=args.dropout,
        use_proxy=(not args.no_proxy),
        debug_alpha_once=getattr(args, "debug_alpha_once", False),
    )
    model = BRPPairwiseSurrogate(encoder=encoder, tau_pair=args.tau_pair).to(device)

    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        ckpt = torch.load(warm_start_ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"[Train] warm-started from {warm_start_ckpt}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")
    config_path = os.path.join(args.output_dir, "config.json")

    hparams = dict(vars(args))
    hparams["score_cache_max"] = int(getattr(args, "score_cache_max", hparams.get("score_cache_max", 200000)))

    config_out = {
        "model_type": "BRP_pairwise_surrogate_mixer_input3",
        "csv_path": args.csv_path,
        "json_path": args.json_path,
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
        "hparams": hparams,
        "static_info": {k: static_info.get(k) for k in ["model_id", "avg_bits_target"]},
        "split": {"train_gens": train_gens, "val_gens": val_gens},
        "split_meta": {"seed": split_seed_eff, "val_ratio": args.val_ratio},
        "replay_meta": {
            "replay": (replay_gens is not None),
            "pair_budget": pair_budget,
            "pairs_per_gen": pairs_per_gen,
            "epochs": epochs,
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    print("[Train] start", flush=True)
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_n = 0

        for batch in train_loader:
            bits_A = batch["bits_A"].to(device, non_blocking=True)
            bits_B = batch["bits_B"].to(device, non_blocking=True)
            q = batch["q"].to(device, non_blocking=True)

            proxy_A = batch["proxy_A"].to(device, non_blocking=True) if (not args.no_proxy) else None
            proxy_B = batch["proxy_B"].to(device, non_blocking=True) if (not args.no_proxy) else None

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
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            run_loss += float(loss.item()) * bits_A.size(0)
            run_n += bits_A.size(0)
            global_step += 1

            if args.log_interval > 0 and global_step % args.log_interval == 0:
                avg_loss = run_loss / max(run_n, 1)
                print(f"[Train] epoch={epoch} step={global_step} loss={avg_loss:.6f}", flush=True)

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
            device=device,
            topk=args.topk,
            batch_eval=args.eval_batch,
        )

        score = val_metrics["topk_overlap"]
        print(
            f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} "
            f"val_top{args.topk}_overlap={val_metrics['topk_overlap']:.4f} "
            f"val_ndcg@{args.topk}={val_metrics['ndcg@k']:.4f} "
            f"time={dt:.1f}s",
            flush=True,
        )

        if score > best_metric:
            best_metric = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_metric": best_metric,
                    "epoch": epoch,
                    "config": config_out,
                },
                best_path,
            )
            print(f"[Save] best -> {best_path} (top{args.topk}_overlap={best_metric:.4f})", flush=True)

    return best_path, config_path, best_metric


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    _, _, best_metric = train_surrogate_mixer(args)
    print(f"[Done] best_metric(top{args.topk}_overlap)={best_metric:.4f}", flush=True)


if __name__ == "__main__":
    main()
