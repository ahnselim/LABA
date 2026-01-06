#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collect training data: (bit assignment) -> (true PPL)
- Uses prebaked Wq/A/B per module for bits 2/3/4.
- Evaluates PPL on wikitext2 test, same style as your prebaked MC evaluator.

Outputs:
  output_dir/
    ppl_dataset.csv
    bit_assign/
      arch_000001.csv
      arch_000002.csv
      ...

Usage:
CUDA_VISIBLE_DEVICES=0 nohup python collect_ppl_dataset_prebaked.py \
  --model_id meta-llama/Llama-3.2-3B \
  --prebake_root ../artifacts/montecarlo/prebake \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --avg_bits 2.50 \
  --use_round_band --round_quantum 0.1 \
  --num_samples 20000 \
  --eval_seq_len 2048 \
  --gpu_id 0 \
  --output_dir ../artifacts/ppl_dataset_20000
"""

import os, re, gc, csv, math, json, time, random, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# -------------------------
# 0) helpers
# -------------------------
TARGET_7_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
)


def is_llama_7_module(module_name: str) -> bool:
    return any(
        module_name.endswith("." + s) or module_name.endswith(s)
        for s in TARGET_7_SUFFIXES
    )


def _safe_name(s) -> str:
    if s is None:
        s = "none"
    s = str(s)
    if s.strip() == "" or s.strip().lower() == "none":
        s = "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def gcd_list(xs: List[int]) -> int:
    g = 0
    for x in xs:
        g = x if g == 0 else gcd(g, x)
    return max(g, 1)


def weighted_sum_bits(b_assign: Dict[str, int], W_map: Dict[str, int]) -> int:
    s = 0
    for n, b in b_assign.items():
        w = W_map.get(n)
        if w is not None:
            s += int(w) * int(b)
    return int(s)


def target_weighted_sum(avg_bits: float, W_map: Dict[str, int]) -> int:
    sum_w = sum(int(w) for w in W_map.values())
    g = gcd_list([int(w) for w in W_map.values()]) if W_map else 1
    raw = avg_bits * sum_w
    snapped = int(round(raw / g) * g)
    return snapped


def ensure_complete_assignment(
    b: Dict[str, int], names: List[str], bmin: int
) -> Dict[str, int]:
    out = dict(b)
    for n in names:
        if n not in out:
            out[n] = bmin
    return out


# -------------------------
# 1) load Step1/Step2 CSV (minimal robust)
# -------------------------
def read_csv_dicts(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        return [
            {(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rdr
        ]


def pick_col(header: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in header:
            return c
    return None


def load_sens_map(sens_csv: str, c_col: str = "C_mean_per_batch") -> Dict[str, float]:
    rows = read_csv_dicts(sens_csv)
    if not rows:
        raise ValueError(f"Empty sens_csv: {sens_csv}")
    header = rows[0]

    name_col = pick_col(header, ["layer_name", "module", "name"])
    if not name_col:
        raise ValueError("sens_csv must have layer_name/module/name")

    if c_col not in header:
        c_col = pick_col(
            header, ["C_mean_per_batch", "C_j", "C", "sens", "sensitivity"]
        )
        if not c_col:
            raise ValueError("Cannot find C column in sens_csv")

    out = {}
    for r in rows:
        nm = r.get(name_col, "")
        if nm:
            try:
                out[nm] = float(r[c_col])
            except Exception:
                pass
    return out


def load_alpha_map(alpha_csv: str) -> Dict[str, Dict[int, float]]:
    rows = read_csv_dicts(alpha_csv)
    if not rows:
        raise ValueError(f"Empty alpha_csv: {alpha_csv}")

    out: Dict[str, Dict[int, float]] = {}
    for r in rows:
        nm = r.get("layer_name") or r.get("module") or r.get("name") or ""
        if not nm:
            continue
        if "bit" not in r or "alpha" not in r:
            continue
        try:
            b = int(float(r["bit"]))
            a = float(r["alpha"])
        except Exception:
            continue
        out.setdefault(nm, {})[b] = a
    return out


# -------------------------
# 2) PPL evaluation (prebaked Wq/A/B patch & restore)
# -------------------------
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
        return float("inf")
    return math.exp(total_nll / total_tok)


class PplEvaluator:
    def __init__(
        self,
        model,
        original_state_dict,
        eval_input_ids,
        prebake_root: str,
        eval_seq_len: int = 2048,
        cache_maxsize: int = 20000,
    ):
        self.model = model
        self.device = model.device
        self.original_state_dict = original_state_dict
        self.eval_input_ids = eval_input_ids.to(self.device)
        self.prebake_root = Path(prebake_root)
        self.eval_seq_len = int(eval_seq_len)

        self.cache_maxsize = int(cache_maxsize)
        self.ppl_cache = OrderedDict()

        bit2_dir = self.prebake_root / "bit2"
        if not bit2_dir.exists():
            raise FileNotFoundError(f"Missing prebake dir: {bit2_dir}")

        # scan modules from prebake files
        self.target_layers: List[str] = []
        for f in sorted(bit2_dir.glob("*.pt")):
            payload = torch.load(f, map_location="cpu")
            mn = payload.get("module")
            del payload
            if not mn:
                continue
            if not is_llama_7_module(mn):
                continue
            if f"{mn}.weight" in self.original_state_dict:
                self.target_layers.append(mn)
        self.target_layers = sorted(list(set(self.target_layers)))
        if not self.target_layers:
            raise ValueError("No valid llama-7 modules found from prebake/bit2 scan.")

    def _get_module(self, name: str):
        mod = self.model
        for p in name.split("."):
            mod = getattr(mod, p)
        return mod

    @torch.no_grad()
    def restore_original(self):
        for mn in self.target_layers:
            mod = self._get_module(mn)
            wname = f"{mn}.weight"
            mod.weight.data.copy_(
                self.original_state_dict[wname].to(
                    device=mod.weight.device, dtype=mod.weight.dtype
                )
            )

    @torch.no_grad()
    def evaluate(self, bit_assignment: Dict[str, int]) -> float:
        # cache by tuple (sorted) restricted to target layers
        key = tuple((mn, int(bit_assignment.get(mn, 4))) for mn in self.target_layers)
        if key in self.ppl_cache:
            v = self.ppl_cache.pop(key)
            self.ppl_cache[key] = v
            return v

        try:
            for mn in self.target_layers:
                bit = int(bit_assignment.get(mn, 4))
                bit = 2 if bit < 2 else (4 if bit > 4 else bit)

                mod = self._get_module(mn)
                safe = _safe_name(mn)
                fpath = self.prebake_root / f"bit{bit}" / f"{safe}.pt"
                if not fpath.exists():
                    continue

                payload = torch.load(fpath, map_location=self.device)
                Wq, A, B = payload["Wq"], payload["A"], payload["B"]
                compute_dtype = Wq.dtype
                W_eff = Wq + (A.to(compute_dtype) @ B.to(compute_dtype))
                mod.weight.data.copy_(W_eff.to(mod.weight.dtype))

                del payload, Wq, A, B, W_eff

            ppl = run_live_ppl_eval(self.model, self.eval_input_ids, self.eval_seq_len)

        except Exception:
            ppl = float("inf")
        finally:
            self.restore_original()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.ppl_cache[key] = ppl
        if len(self.ppl_cache) > self.cache_maxsize:
            self.ppl_cache.popitem(last=False)
        return ppl


# -------------------------
# 3) Budget projection (exact or rounding band)
#    proxy: score_j * 2^{-2b}
# -------------------------
def project_to_weighted_budget(
    b_assign: Dict[str, int],
    W_map: Dict[str, int],
    score_map: Dict[str, float],
    B_target: int,
    bmin: int,
    bmax: int,
    max_steps: int = 300000,
) -> Dict[str, int]:
    b = dict(b_assign)
    names = [n for n in b.keys() if n in W_map]
    if not names:
        return b

    def proxy_term(n: str, bit: int) -> float:
        s = float(score_map.get(n, 0.0))
        return s * (2.0 ** (-2.0 * float(bit)))

    def marg_gain_up(n: str) -> float:
        bit = b[n]
        if bit >= bmax:
            return -float("inf")
        # positive means proxy decreases
        return proxy_term(n, bit) - proxy_term(n, bit + 1)

    def marg_harm_down(n: str) -> float:
        bit = b[n]
        if bit <= bmin:
            return float("inf")
        # positive harm
        return proxy_term(n, bit - 1) - proxy_term(n, bit)

    # snap target to gcd
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
            # need increase bits: pick best (gain / w)
            cand = [
                (marg_gain_up(n) / float(W_map[n]), n) for n in names if b[n] < bmax
            ]
            if not cand:
                break
            cand.sort(key=lambda x: x[0], reverse=True)
            n_best = cand[0][1]
            b[n_best] += 1
        else:
            # need decrease bits: pick smallest (harm / w)
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


def project_to_weighted_band(
    b_assign: Dict[str, int],
    W_map: Dict[str, int],
    score_map: Dict[str, float],
    B_lo: int,
    B_hi: int,
    bmin: int,
    bmax: int,
) -> Dict[str, int]:
    S = weighted_sum_bits(b_assign, W_map)
    if S < B_lo:
        return project_to_weighted_budget(b_assign, W_map, score_map, B_lo, bmin, bmax)
    if S > B_hi:
        return project_to_weighted_budget(b_assign, W_map, score_map, B_hi, bmin, bmax)
    return b_assign


def calc_proxy(b_assign: Dict[str, int], score_map: Dict[str, float]) -> float:
    s = 0.0
    for n, bit in b_assign.items():
        sc = float(score_map.get(n, 0.0))
        s += sc * (2.0 ** (-2.0 * float(bit)))
    return float(s)


# -------------------------
# 4) main: sample architectures -> eval ppl -> save csv
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--prebake_root", required=True)
    ap.add_argument("--sens_csv", required=True)
    ap.add_argument("--alpha_csv", required=True)

    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--avg_bits", type=float, default=2.5)
    ap.add_argument("--bmin", type=int, default=2)
    ap.add_argument("--bmax", type=int, default=4)

    ap.add_argument("--use_round_band", action="store_true")
    ap.add_argument("--round_quantum", type=float, default=0.1)

    ap.add_argument("--alpha_bit", type=int, default=3)
    ap.add_argument("--alpha_default", type=float, default=1.0)

    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--max_tries_per_sample", type=int, default=50)

    ap.add_argument("--eval_seq_len", type=int, default=2048)
    ap.add_argument("--ppl_cache_max", type=int, default=20000)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", required=True)

    args = ap.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bit_assign").mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map=device,  # keep your style
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    # backup original weights (CPU)
    original_state = {
        k: v.detach().clone().cpu() for k, v in model.state_dict().items()
    }

    # load eval tokens (wikitext2 test)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    eval_input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids

    # evaluator (scans prebake and keeps only llama 7 modules)
    ppl_eval = PplEvaluator(
        model=model,
        original_state_dict=original_state,
        eval_input_ids=eval_input_ids,
        prebake_root=args.prebake_root,
        eval_seq_len=args.eval_seq_len,
        cache_maxsize=args.ppl_cache_max,
    )
    target_layers = ppl_eval.target_layers

    # W_map from original weights numel
    W_map = {mn: int(original_state[f"{mn}.weight"].numel()) for mn in target_layers}

    # score_map = C_j * alpha_j(alpha_bit)  (if missing alpha, use default)
    sens_map = load_sens_map(args.sens_csv, c_col="C_mean_per_batch")
    alpha_map = load_alpha_map(args.alpha_csv)
    score_map: Dict[str, float] = {}
    for mn in target_layers:
        C = float(sens_map.get(mn, 0.0))
        a = float(alpha_map.get(mn, {}).get(int(args.alpha_bit), args.alpha_default))
        score_map[mn] = max(0.0, C * a)

    sum_w = sum(W_map.values())
    B_target = target_weighted_sum(args.avg_bits, W_map)

    # rounding band -> [B_lo, B_hi]
    if args.use_round_band:
        eps = 1e-9
        avg_lo = args.avg_bits - 0.5 * args.round_quantum
        avg_hi = args.avg_bits + 0.5 * args.round_quantum - eps
        g = gcd_list(list(W_map.values()))
        B_lo = int(math.ceil((avg_lo * sum_w) / g) * g)
        B_hi = int(math.floor((avg_hi * sum_w) / g) * g)
        if B_lo > B_hi:
            B_lo = B_hi = B_target
    else:
        B_lo = B_hi = B_target

    # output csv append
    dataset_csv = out_dir / "ppl_dataset.csv"
    new_file = not dataset_csv.exists()
    f = open(dataset_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(
            ["sample_id", "ppl", "wavg_bits", "sum_wb", "proxy", "bit_assign_csv"]
        )
        f.flush()

    seen = set()

    def make_random_assignment() -> Dict[str, int]:
        # random bits, then project to band
        b = {mn: random.choice([args.bmin, 3, args.bmax]) for mn in target_layers}
        b = ensure_complete_assignment(b, target_layers, args.bmin)
        b = project_to_weighted_band(
            b, W_map, score_map, B_lo, B_hi, args.bmin, args.bmax
        )
        # if still out of band (rare), force to nearest boundary
        S = weighted_sum_bits(b, W_map)
        if S < B_lo:
            b = project_to_weighted_budget(
                b, W_map, score_map, B_lo, args.bmin, args.bmax
            )
        elif S > B_hi:
            b = project_to_weighted_budget(
                b, W_map, score_map, B_hi, args.bmin, args.bmax
            )
        return b

    print(f"[info] target modules: N={len(target_layers)} (llama 7 modules only)")
    print(
        f"[budget] sum_w={sum_w}, B_target={B_target}, band=[{B_lo},{B_hi}] -> avg~[{B_lo/sum_w:.6f},{(B_hi+1)/sum_w:.6f})"
    )

    pbar = tqdm(range(1, args.num_samples + 1), desc="Collect PPL samples")
    for sid in pbar:
        b = None
        for _ in range(args.max_tries_per_sample):
            cand = make_random_assignment()
            key = tuple((mn, int(cand[mn])) for mn in target_layers)
            if key in seen:
                continue
            seen.add(key)
            b = cand
            break
        if b is None:
            print("[warn] failed to find a new unique assignment; stopping early.")
            break

        # eval
        ppl = ppl_eval.evaluate(b)
        S = weighted_sum_bits(b, W_map)
        wavg = S / sum_w if sum_w > 0 else 0.0
        proxy = calc_proxy(b, score_map)

        # save bit_assign csv
        bit_csv = out_dir / "bit_assign" / f"arch_{sid:06d}.csv"
        with open(bit_csv, "w", newline="", encoding="utf-8") as bf:
            bw = csv.writer(bf)
            bw.writerow(["layer_name", "R_int"])
            for mn in target_layers:
                bw.writerow([mn, int(b[mn])])

        # append dataset row
        w.writerow(
            [sid, f"{ppl:.6f}", f"{wavg:.6f}", int(S), f"{proxy:.6e}", str(bit_csv)]
        )
        f.flush()

        pbar.set_postfix({"ppl": f"{ppl:.3f}", "wavg": f"{wavg:.4f}"})

    f.close()
    print(f"[done] saved: {dataset_csv}")


if __name__ == "__main__":
    main()
