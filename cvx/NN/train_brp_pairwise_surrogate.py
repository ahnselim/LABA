#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRP-style Pairwise Surrogate Trainer (for PPL ranking)
=====================================================

Input:
  - static_info.json: {layer_names, W_map, C_prime_map, model_id, avg_bits_target}
  - training_samples.csv: columns
      generation, sample_idx, proxy_loss, measured_ppl, bit_assignment_json

Output:
  - output_dir/best.pt
  - output_dir/config.json
  - output_dir/train_log.txt (stdout redirect recommended)

Key idea (BRP-NAS):
  - Train a binary relation predictor via KL( soft_target || predicted_pair_prob )
  - soft_target is derived from measured_ppl(A), measured_ppl(B)

Usage:
  python neural_net/train_brp_pairwise_surrogate.py \
    --csv_path ../artifacts/surrogate_data/training_samples.csv \
    --json_path ../artifacts/surrogate_data/static_info.json \
    --output_dir ../artifacts/surrogate_checkpoint_brp \
    --epochs 50 --batch_size 256 --pairs_per_gen 2000 --topk 10

Tips:
  - Split is by generation (prevents leakage).
"""

import os
import json
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

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


def parse_bit_assignment(
    bit_json: str, layer_names: List[str], bmin: int = 2
) -> List[int]:
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
            raise ValueError(
                f"bit list length mismatch: got {len(obj)} vs L={len(layer_names)}"
            )
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
    NDCG uses relevance; we define relevance as -true_vals.
    """
    n = true_vals.shape[0]
    k = min(k, n)
    # order by predicted
    pred_rank = np.argsort(pred_scores)[:k]
    # ideal order by true
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
    csv_path: str, static_info_path: str, bmin: int = 2
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
        gen2cands.setdefault(gen, []).append(
            Cand(bitvec=bitvec, ppl=ppl, proxy=proxy, gen=gen)
        )

    # ensure deterministic ordering inside each gen (optional)
    for g in gen2cands:
        gen2cands[g].sort(key=lambda c: (c.ppl, c.proxy))

    return layer_names, gen2cands, static_info


# -----------------------------
# Pair Sampler (BRP-style)
# -----------------------------
class BRPPairIterableDataset(IterableDataset):
    """
    Yields sampled pairs (A,B) from within the same generation.

    Sampling strategy:
      - top-biased: pick A from top m, B from rest (most of the time)
      - hard pairs: pick pairs with small rank gap (some fraction)
    """

    def __init__(
        self,
        gen2cands: Dict[int, List[Cand]],
        gens: List[int],
        layer_names: List[str],
        C_log: np.ndarray,
        W_log: np.ndarray,
        pairs_per_gen: int,
        top_frac: float,
        hard_frac: float,
        hard_window: int,
        tau_soft: float,
        bmin: int,
        bmax: int,
        seed: int,
    ):
        super().__init__()
        self.gen2cands = gen2cands
        self.gens = gens
        self.layer_names = layer_names
        self.C_log = C_log
        self.W_log = W_log
        self.pairs_per_gen = pairs_per_gen
        self.top_frac = top_frac
        self.hard_frac = hard_frac
        self.hard_window = hard_window
        self.tau_soft = tau_soft
        self.bmin = bmin
        self.bmax = bmax
        self.seed = seed

        self.L = len(layer_names)

    def _make_pair(self, rng: random.Random, cands: List[Cand]) -> Tuple[Cand, Cand]:
        n = len(cands)
        # sorted by ppl already (ascending)
        m_top = max(2, int(self.top_frac * n))

        if rng.random() < self.hard_frac:
            # hard pair: close ranks
            i = rng.randrange(0, n)
            j_lo = max(0, i - self.hard_window)
            j_hi = min(n - 1, i + self.hard_window)
            j = rng.randrange(j_lo, j_hi + 1)
            if j == i:
                j = (j + 1) % n
            return cands[i], cands[j]
        else:
            # top-biased
            i = rng.randrange(0, m_top)
            j = rng.randrange(m_top, n) if m_top < n else rng.randrange(0, n)
            return cands[i], cands[j]

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        base_seed = self.seed + (worker.id if worker is not None else 0)
        rng = random.Random(base_seed)

        # shuffle generation order each epoch
        gens = self.gens[:]
        rng.shuffle(gens)

        for g in gens:
            cands = self.gen2cands.get(g, [])
            if len(cands) < 2:
                continue

            for _ in range(self.pairs_per_gen):
                A, B = self._make_pair(rng, cands)

                # soft target from ppl(A), ppl(B)
                # q = softmax([-pplA/tau, -pplB/tau])
                q = np.exp(
                    np.array(
                        [-A.ppl / self.tau_soft, -B.ppl / self.tau_soft],
                        dtype=np.float64,
                    )
                )
                q = q / (q.sum() + 1e-12)

                yield {
                    "bits_A": np.asarray(A.bitvec, dtype=np.int64),
                    "bits_B": np.asarray(B.bitvec, dtype=np.int64),
                    "ppl_A": float(A.ppl),
                    "ppl_B": float(B.ppl),
                    "proxy_A": float(A.proxy),
                    "proxy_B": float(B.proxy),
                    "q": q.astype(np.float32),
                    "gen": int(g),
                }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    bits_A = torch.tensor(
        np.stack([b["bits_A"] for b in batch], axis=0), dtype=torch.long
    )
    bits_B = torch.tensor(
        np.stack([b["bits_B"] for b in batch], axis=0), dtype=torch.long
    )
    q = torch.tensor(
        np.stack([b["q"] for b in batch], axis=0), dtype=torch.float32
    )  # (B,2)
    proxy_A = torch.tensor(
        [b["proxy_A"] for b in batch], dtype=torch.float32
    ).unsqueeze(-1)
    proxy_B = torch.tensor(
        [b["proxy_B"] for b in batch], dtype=torch.float32
    ).unsqueeze(-1)
    return {
        "bits_A": bits_A,
        "bits_B": bits_B,
        "q": q,
        "proxy_A": proxy_A,
        "proxy_B": proxy_B,
    }


# -----------------------------
# Model
# -----------------------------
class BitSequenceEncoder(nn.Module):
    """
    Encodes bit vector of length L with per-layer static features (log C', log W).
    Produces pooled embedding and scalar score s(b) (lower is better).
    """

    def __init__(
        self,
        L: int,
        d_model: int = 128,
        bit_emb_dim: int = 32,
        nhead: int = 4,
        nlayers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        use_proxy: bool = True,
    ):
        super().__init__()
        self.L = L
        self.use_proxy = use_proxy

        # bits are in {2,3,4} => map to {0,1,2}
        self.bit_emb = nn.Embedding(3, bit_emb_dim)

        # numeric features per layer: [logC', logW]
        self.num_proj = nn.Linear(2, bit_emb_dim)

        # combine bit_emb + num_proj -> d_model
        self.in_proj = nn.Linear(bit_emb_dim * 2, d_model)

        self.pos_emb = nn.Embedding(L, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.dropout = nn.Dropout(dropout)

        # pooling -> hidden -> score
        head_in = d_model + (1 if use_proxy else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        bits: torch.Tensor,
        C_log: torch.Tensor,
        W_log: torch.Tensor,
        proxy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        bits: (B,L) values in {2,3,4}
        C_log, W_log: (L,) normalized floats (broadcasted)
        proxy: (B,1) optional
        returns: score s(b) shape (B,1)
        """
        B, L = bits.shape
        assert L == self.L

        bit_idx = torch.clamp(bits - 2, 0, 2)  # {2,3,4} -> {0,1,2}
        e_bit = self.bit_emb(bit_idx)  # (B,L,bit_emb_dim)

        # build numeric per-layer tensor
        # (L,) -> (B,L,1)
        C = C_log.view(1, L, 1).expand(B, L, 1)
        W = W_log.view(1, L, 1).expand(B, L, 1)
        x_num = torch.cat([C, W], dim=-1)  # (B,L,2)
        e_num = self.num_proj(x_num)  # (B,L,bit_emb_dim)

        x = torch.cat([e_bit, e_num], dim=-1)  # (B,L,2*bit_emb_dim)
        x = self.in_proj(x)  # (B,L,d_model)

        pos = torch.arange(L, device=bits.device).view(1, L)
        x = x + self.pos_emb(pos)

        x = self.encoder(self.dropout(x))  # (B,L,d_model)
        x_pool = x.mean(dim=1)  # (B,d_model)

        if self.use_proxy:
            if proxy is None:
                proxy = torch.zeros((B, 1), device=bits.device, dtype=x_pool.dtype)
            h = torch.cat([x_pool, proxy], dim=-1)
        else:
            h = x_pool

        s = self.head(h)  # (B,1) lower is better
        return s


class BRPPairwiseSurrogate(nn.Module):
    """
    Pairwise probability derived from scalar scores:
      p(A better than B) = sigmoid((sB - sA) / tau_pair)
    """

    def __init__(self, encoder: BitSequenceEncoder, tau_pair: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.tau_pair = float(tau_pair)

    def forward_pair(
        self,
        bits_A: torch.Tensor,
        bits_B: torch.Tensor,
        C_log: torch.Tensor,
        W_log: torch.Tensor,
        proxy_A: Optional[torch.Tensor] = None,
        proxy_B: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sA = self.encoder(bits_A, C_log=C_log, W_log=W_log, proxy=proxy_A)  # (B,1)
        sB = self.encoder(bits_B, C_log=C_log, W_log=W_log, proxy=proxy_B)  # (B,1)

        # A better if sA < sB  -> (sB - sA) positive
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
        proxy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(bits, C_log=C_log, W_log=W_log, proxy=proxy)  # (B,1)


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate_ranking_metrics(
    model: BRPPairwiseSurrogate,
    gen2cands: Dict[int, List[Cand]],
    gens: List[int],
    layer_names: List[str],
    C_log_t: torch.Tensor,
    W_log_t: torch.Tensor,
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

        # batch inference
        for i in range(0, len(cands), batch_eval):
            j = min(len(cands), i + batch_eval)
            b = torch.tensor(bits[i:j], dtype=torch.long, device=device)
            p = torch.tensor(proxy[i:j], dtype=torch.float32, device=device)
            s = model.score_single(b, C_log=C_log_t, W_log=W_log_t, proxy=p).squeeze(-1)
            scores[i:j] = s.detach().cpu().numpy().astype(np.float64)

        overlaps.append(topk_overlap(true_ppl, scores, topk))
        ndcgs.append(compute_ndcg_at_k(true_ppl, scores, topk))

    if not overlaps:
        return {"topk_overlap": 0.0, "ndcg@k": 0.0}

    return {
        "topk_overlap": float(np.mean(overlaps)),
        "ndcg@k": float(np.mean(ndcgs)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # split
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="generation 단위 split 비율"
    )
    parser.add_argument("--seed", type=int, default=42)

    # bits range (for safety in parsing)
    parser.add_argument("--bmin", type=int, default=2)
    parser.add_argument("--bmax", type=int, default=4)

    # pair sampling
    parser.add_argument("--pairs_per_gen", type=int, default=2000)
    parser.add_argument(
        "--top_frac", type=float, default=0.3, help="top-biased sampling에서 top 비율"
    )
    parser.add_argument("--hard_frac", type=float, default=0.3, help="hard pair 비율")
    parser.add_argument(
        "--hard_window", type=int, default=8, help="hard pair: rank window"
    )

    # BRP temperatures
    parser.add_argument(
        "--tau_soft",
        type=float,
        default=1.0,
        help="soft target temperature (PPL softmax)",
    )
    parser.add_argument(
        "--tau_pair", type=float, default=1.0, help="pairwise sigmoid temperature"
    )

    # model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--bit_emb_dim", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--no_proxy", action="store_true", help="proxy_loss를 입력에서 제거"
    )

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

    args = parser.parse_args()
    safe_makedirs(args.output_dir)
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] device={device}")

    # -------------------------
    # Load data
    # -------------------------
    layer_names, gen2cands, static_info = load_generation_pool(
        csv_path=args.csv_path,
        static_info_path=args.json_path,
        bmin=args.bmin,
    )
    L = len(layer_names)
    gens_all = sorted(gen2cands.keys())

    if len(gens_all) < 2:
        raise RuntimeError(f"Not enough generations in CSV: {gens_all}")

    rng = random.Random(args.seed)
    gens_shuf = gens_all[:]
    rng.shuffle(gens_shuf)

    n_val = max(1, int(round(len(gens_shuf) * args.val_ratio)))
    val_gens = sorted(gens_shuf[:n_val])
    train_gens = sorted(gens_shuf[n_val:])

    print(
        f"[Split] total_gens={len(gens_all)} train={len(train_gens)} val={len(val_gens)}"
    )
    print(f"[Split] val_gens(sample)={val_gens[:min(10,len(val_gens))]}")

    # -------------------------
    # Static per-layer features (normalized)
    # -------------------------
    C_prime_map: Dict[str, float] = static_info["C_prime_map"]
    W_map: Dict[str, int] = static_info["W_map"]

    C_log = np.zeros((L,), dtype=np.float32)
    W_log = np.zeros((L,), dtype=np.float32)
    for i, ln in enumerate(layer_names):
        cp = float(C_prime_map.get(ln, 0.0))
        w = float(W_map.get(ln, 1.0))
        C_log[i] = math.log(max(cp, 1e-30))
        W_log[i] = math.log(max(w, 1.0))

    # normalize (z-score) across layers
    def z(x):
        mu = float(x.mean())
        sd = float(x.std() + 1e-6)
        return (x - mu) / sd, mu, sd

    C_log_n, C_mu, C_sd = z(C_log)
    W_log_n, W_mu, W_sd = z(W_log)

    C_log_t = torch.tensor(C_log_n, dtype=torch.float32, device=device)
    W_log_t = torch.tensor(W_log_n, dtype=torch.float32, device=device)

    # -------------------------
    # Dataset / Loader
    # -------------------------
    train_ds = BRPPairIterableDataset(
        gen2cands=gen2cands,
        gens=train_gens,
        layer_names=layer_names,
        C_log=C_log_n,
        W_log=W_log_n,
        pairs_per_gen=args.pairs_per_gen,
        top_frac=args.top_frac,
        hard_frac=args.hard_frac,
        hard_window=args.hard_window,
        tau_soft=args.tau_soft,
        bmin=args.bmin,
        bmax=args.bmax,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    # -------------------------
    # Model
    # -------------------------
    encoder = BitSequenceEncoder(
        L=L,
        d_model=args.d_model,
        bit_emb_dim=args.bit_emb_dim,
        nhead=args.nhead,
        nlayers=args.nlayers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_proxy=(not args.no_proxy),
    )
    model = BRPPairwiseSurrogate(encoder=encoder, tau_pair=args.tau_pair).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # -------------------------
    # Training loop
    # -------------------------
    best_metric = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")

    # save config early
    config_out = {
        "model_type": "BRP_pairwise_surrogate",
        "csv_path": args.csv_path,
        "json_path": args.json_path,
        "layer_names": layer_names,
        "L": L,
        "norm": {
            "C_log_mu": C_mu,
            "C_log_sd": C_sd,
            "W_log_mu": W_mu,
            "W_log_sd": W_sd,
        },
        "hparams": vars(args),
        "static_info": {k: static_info.get(k) for k in ["model_id", "avg_bits_target"]},
        "split": {"train_gens": train_gens, "val_gens": val_gens},
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    print("[Train] start")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        run_n = 0

        for batch in train_loader:
            bits_A = batch["bits_A"].to(device, non_blocking=True)
            bits_B = batch["bits_B"].to(device, non_blocking=True)
            q = batch["q"].to(device, non_blocking=True)  # (B,2)
            proxy_A = (
                batch["proxy_A"].to(device, non_blocking=True)
                if (not args.no_proxy)
                else None
            )
            proxy_B = (
                batch["proxy_B"].to(device, non_blocking=True)
                if (not args.no_proxy)
                else None
            )

            p, sA, sB = model.forward_pair(
                bits_A,
                bits_B,
                C_log=C_log_t,
                W_log=W_log_t,
                proxy_A=proxy_A,
                proxy_B=proxy_B,
            )
            logp = torch.log(p)

            # KL(q || p)  (batchmean)
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
                print(
                    f"[Train] epoch={epoch} step={global_step} loss={avg_loss:.6f}",
                    flush=True,
                )

        avg_train_loss = run_loss / max(run_n, 1)
        dt = time.time() - t0

        # -------------------------
        # Validation (ranking metrics by generation)
        # -------------------------
        model.eval()
        val_metrics = evaluate_ranking_metrics(
            model=model,
            gen2cands=gen2cands,
            gens=val_gens,
            layer_names=layer_names,
            C_log_t=C_log_t,
            W_log_t=W_log_t,
            device=device,
            topk=args.topk,
            batch_eval=args.eval_batch,
        )

        score = val_metrics["topk_overlap"]  # primary
        print(
            f"[Epoch {epoch}] train_loss={avg_train_loss:.6f} "
            f"val_top{args.topk}_overlap={val_metrics['topk_overlap']:.4f} "
            f"val_ndcg@{args.topk}={val_metrics['ndcg@k']:.4f} "
            f"time={dt:.1f}s",
            flush=True,
        )

        # save best
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
            print(
                f"[Save] best -> {best_path} (top{args.topk}_overlap={best_metric:.4f})",
                flush=True,
            )

    print(f"[Done] best_metric(top{args.topk}_overlap)={best_metric:.4f}", flush=True)


if __name__ == "__main__":
    main()
