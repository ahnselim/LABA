"""
Surrogate scorer utilities for mixer input3 models.
"""

import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from NN.train_brp_pairwise_surrogate_mixer_input3 import (
    BitSequenceEncoderInput3,
    BRPPairwiseSurrogate,
    parse_alpha_per_bit,
)


class SurrogateScorer:
    def __init__(self, ckpt_path: str, config_path: str, device: torch.device, bmin: int = 1):
        self.device = device

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model_type = str(cfg.get("model_type", "")).lower()
        if ("mixer" not in model_type) or ("input3" not in model_type):
            raise ValueError(f"[Surrogate] input3 Mixer ckpt 전용. config.model_type={cfg.get('model_type')}")

        self.layer_names: List[str] = cfg["layer_names"]
        self.L = int(cfg["L"])
        h = cfg["hparams"]
        self.use_proxy = not bool(h.get("no_proxy", False))
        self.model_bmin = int(h.get("bmin", 1))
        self.model_bmax = int(h.get("bmax", 4))
        self.num_bits = self.model_bmax - self.model_bmin + 1
        self.bmin = int(bmin)
        if self.bmin < self.model_bmin or self.bmin > self.model_bmax:
            self.bmin = self.model_bmin

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
            print("[Surrogate][Warn] static_info에 C_map이 없어 C_prime_map 사용", flush=True)
        W_map: Dict[str, int] = static_info.get("W_map", None)
        if W_map is None:
            raise ValueError("[Surrogate] static_info.json missing W_map")
        alpha_map: Dict[str, Any] = static_info.get("alpha_map", None)
        if alpha_map is None:
            alpha_map = {}

        C_log = np.zeros((self.L,), dtype=np.float32)
        W_log = np.zeros((self.L,), dtype=np.float32)
        alpha_tbl = np.zeros((self.L, self.num_bits), dtype=np.float32)

        for i, ln in enumerate(self.layer_names):
            c = float(C_map.get(ln, 0.0))
            w = float(W_map.get(ln, 1.0))
            C_log[i] = math.log(max(c, 1e-30))
            W_log[i] = math.log(max(w, 1.0))
            avals = parse_alpha_per_bit(
                alpha_map.get(ln, None),
                bmin=self.model_bmin,
                bmax=self.model_bmax,
            )
            for j, av in enumerate(avals):
                alpha_tbl[i, j] = av

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
        self.alpha_logit_table_t = torch.tensor(alpha_logit_n, dtype=torch.float32, device=device)

        encoder = BitSequenceEncoderInput3(
            L=self.L,
            d_model=int(h.get("d_model", 128)),
            bit_emb_dim=int(h.get("bit_emb_dim", 32)),
            nlayers=int(h.get("nlayers", 2)),
            ff_dim=int(h.get("ff_dim", 256)),
            token_mlp_dim=int(h.get("token_mlp_dim", 128)),
            dropout=float(h.get("dropout", 0.1)),
            use_proxy=self.use_proxy,
            bmin=self.model_bmin,
            bmax=self.model_bmax,
        )
        self.model = BRPPairwiseSurrogate(encoder=encoder, tau_pair=float(h.get("tau_pair", 1.0))).to(device)
        self.model.eval()

        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=True)

        self.cache = OrderedDict()
        self.cache_max = int(h.get("score_cache_max", 200000))
        print(f"[Surrogate] loaded: L={self.L}, use_proxy={self.use_proxy}, cache_max={self.cache_max}", flush=True)

    def _bits_to_vec(self, b_assign: Dict[str, int]) -> List[int]:
        return [int(b_assign.get(ln, self.bmin)) for ln in self.layer_names]

    @torch.no_grad()
    def score_batch(self, assigns: List[Dict[str, int]], proxy_vals: Optional[List[float]] = None, batch: int = 1024) -> np.ndarray:
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
