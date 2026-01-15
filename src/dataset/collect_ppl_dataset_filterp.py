#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_ppl_dataset_filterp_localmix_prebaked.py

정책(고정안):
- 후보 생성 분포:
  • 80%: local 이웃 (Step1/2/3 proxy-best b*에서 시작, swap 기반 -1/+1 1~3회)
  • 20%: global 랜덤 + projection
- local mutation은 swap(+1,-1)만 사용 (예산 유지 목적)
  • 단, weight numel이 달라서 exact budget이 깨질 수 있으므로
    마지막에 project_to_weighted_budget(...)로 Σ w_j b_j = B_target를 정확히 맞춤.
- 레이어 선택:
  • +1: proxy가 많이 줄어드는 레이어 우선
  • -1: proxy 증가가 작은 레이어 우선

[변경사항 - step3b와 동일]
- B_target 계산을 "gcd 스냅"으로 변경:
    raw_target = avg_bits * sum_w
    g = gcd({w_j})
    B_target = round(raw_target / g) * g
  → 즉 target이 항상 g의 배수(도달 가능한 예산)로 스냅됨.

출력:
  output_dir/
    ppl_dataset.csv
    bit_assign/
      gen000001_rank001.csv
      ...

사용 예:
CUDA_VISIBLE_DEVICES=1 nohup python neural_net/collect_ppl_dataset_filterp.py \
  --model_id meta-llama/Llama-3.2-3B\
  --prebake_root ../artifacts/montecarlo/prebake \
  --sens_csv ../artifacts/montecarlo/step1/layerwise_sensitivity.csv \
  --alpha_csv ../artifacts/montecarlo/step2/alpha_layerwise_rank64.csv \
  --init_assign_csv ../artifacts/montecarlo/step3_sa/bit_assign.csv \
  --avg_bits 2.50 \
  --generations 400 \
  --candidates_per_gen 256 \
  --filter_p 32 \
  --local_ratio 0.8 \
  --local_swaps_min 1 --local_swaps_max 3 \
  --eval_seq_len 2048 \
  --gpu_id 0 \
  --output_dir ../artifacts/ppl_dataset_filterp_2.5 > collect_ppl_dataset.log 2>&1 &
"""

import os, re, gc, csv, math, random, argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


def gcd_list(nums) -> int:
    g = 0
    for n in nums:
        g = math.gcd(g, int(n))
    return max(int(g), 1)


def snap_to_gcd(target_sum_raw: float, g: int) -> int:
    # nearest multiple of g
    return int(round(float(target_sum_raw) / float(g)) * int(g))


def weighted_sum_bits(b_assign: Dict[str, int], W_map: Dict[str, int]) -> int:
    s = 0
    for n, b in b_assign.items():
        w = W_map.get(n)
        if w is not None:
            s += int(w) * int(b)
    return int(s)


def ensure_complete_assignment(
    b: Dict[str, int], names: List[str], bmin: int
) -> Dict[str, int]:
    out = dict(b)
    for n in names:
        if n not in out:
            out[n] = bmin
    return out


# -------------------------
# 1) load Step1/Step2 CSV
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
        c_col2 = pick_col(
            header, ["C_mean_per_batch", "C_j", "C", "sens", "sensitivity"]
        )
        if not c_col2:
            raise ValueError("Cannot find C column in sens_csv")
        c_col = c_col2

    out: Dict[str, float] = {}
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
# 1.5) load init_assign_csv (proxy-best b*)
# -------------------------
def load_init_assign_csv(path: str) -> Dict[str, int]:
    """
    expects header like: layer_name,R_int  (or module,bit)
    """
    out: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None:
            raise ValueError(f"Empty CSV: {path}")
        fields = [c.strip() for c in rdr.fieldnames]
        name_col = None
        bit_col = None
        for c in ["layer_name", "module", "name"]:
            if c in fields:
                name_col = c
                break
        for c in ["R_int", "bit", "bits"]:
            if c in fields:
                bit_col = c
                break
        if name_col is None or bit_col is None:
            raise ValueError(
                f"init_assign_csv must have layer_name/module and R_int/bit columns: {path}"
            )

        for r in rdr:
            nm = (r.get(name_col) or "").strip()
            if not nm:
                continue
            try:
                out[nm] = int(float(r[bit_col]))
            except Exception:
                continue
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
# 3) proxy + projection
#    proxy: score_j * 2^{-2b}
# -------------------------
def calc_proxy(b_assign: Dict[str, int], score_map: Dict[str, float]) -> float:
    s = 0.0
    for n, bit in b_assign.items():
        sc = float(score_map.get(n, 0.0))
        s += sc * (2.0 ** (-2.0 * float(bit)))
    return float(s)


def project_to_weighted_budget(
    b_assign: Dict[str, int],
    W_map: Dict[str, int],
    score_map: Dict[str, float],
    B_target: int,
    bmin: int,
    bmax: int,
    max_steps: int = 300000,
    gcd_step: Optional[int] = None,  # <-- 추가(안전)
) -> Dict[str, int]:
    """
    Greedy projection so that sum_j W_j * b_j == B_target.
    Uses proxy delta per weight as selection criterion.
    """
    b = dict(b_assign)
    names = [n for n in b.keys() if n in W_map]
    if not names:
        return b

    if gcd_step is not None and int(gcd_step) > 1:
        if int(B_target) % int(gcd_step) != 0:
            B_target = snap_to_gcd(float(B_target), int(gcd_step))

    def proxy_term(n: str, bit: int) -> float:
        s = float(score_map.get(n, 0.0))
        return s * (2.0 ** (-2.0 * float(bit)))

    def marg_gain_up(n: str) -> float:
        bit = b[n]
        if bit >= bmax:
            return -float("inf")
        return proxy_term(n, bit) - proxy_term(n, bit + 1)

    def marg_harm_down(n: str) -> float:
        bit = b[n]
        if bit <= bmin:
            return float("inf")
        return proxy_term(n, bit - 1) - proxy_term(n, bit)

    steps = 0
    while steps < max_steps:
        S = weighted_sum_bits(b, W_map)
        delta = int(B_target) - int(S)
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
            cand.sort(key=lambda x: x[0])  # smallest harm first
            b[cand[0][1]] -= 1
        steps += 1
    return b


# -------------------------
# 3.5) local swap mutation around b*
# -------------------------
def _proxy_term(score_map: Dict[str, float], n: str, bit: int) -> float:
    s = float(score_map.get(n, 0.0))
    return s * (2.0 ** (-2.0 * float(bit)))


def local_swap_mutate(
    b_star: Dict[str, int],
    target_layers: List[str],
    W_map: Dict[str, int],
    score_map: Dict[str, float],
    bmin: int,
    bmax: int,
    n_swaps: int,
    tau_up: float = 1.0,
    tau_down: float = 1.0,
) -> Dict[str, int]:
    """
    - 시작점: b_star
    - swap을 n_swaps번 적용:
      • +1 후보 i: benefit 큰(=proxy 많이 감소) 레이어 우선
      • -1 후보 j: harm 작은(=proxy 증가 작은) 레이어 우선
    - 마지막에 projection으로 budget 정확히 맞출 예정(이 함수 밖에서)
    """
    b = dict(b_star)

    for _ in range(int(n_swaps)):
        up_cands = []
        down_cands = []

        for n in target_layers:
            bit = int(b.get(n, bmin))

            if bit < bmax:
                benefit = _proxy_term(score_map, n, bit) - _proxy_term(
                    score_map, n, bit + 1
                )
                benefit = float(benefit) / float(W_map[n])  # weight-normalize
                if benefit > 0:
                    up_cands.append((benefit, n))

            if bit > bmin:
                harm = _proxy_term(score_map, n, bit - 1) - _proxy_term(
                    score_map, n, bit
                )
                harm = float(harm) / float(W_map[n])
                if math.isfinite(harm):
                    down_cands.append((harm, n))

        if not up_cands or not down_cands:
            break

        # sample i with prob ~ exp(benefit/tau_up)
        up_cands.sort(key=lambda x: x[0], reverse=True)
        up_vals = [x[0] for x in up_cands]
        up_max = max(up_vals)
        up_w = [math.exp((v - up_max) / max(1e-8, tau_up)) for v in up_vals]
        i = random.choices([x[1] for x in up_cands], weights=up_w, k=1)[0]

        # sample j with prob ~ exp(-harm/tau_down) (smaller harm => larger prob)
        down_cands.sort(key=lambda x: x[0])  # small harm first
        dn_vals = [x[0] for x in down_cands]
        dn_min = min(dn_vals)
        dn_w = [math.exp(-(v - dn_min) / max(1e-8, tau_down)) for v in dn_vals]
        j = random.choices([x[1] for x in down_cands], weights=dn_w, k=1)[0]

        if i == j:
            continue

        # apply swap
        if b[i] < bmax:
            b[i] += 1
        if b[j] > bmin:
            b[j] -= 1

    return b


# -------------------------
# 4) main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", required=True)
    ap.add_argument("--prebake_root", required=True)
    ap.add_argument("--sens_csv", required=True)
    ap.add_argument("--alpha_csv", required=True)

    ap.add_argument(
        "--init_assign_csv",
        required=True,
        help="Step1/2/3 proxy-best bit assignment CSV (b*)",
    )

    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--avg_bits", type=float, default=2.5)
    ap.add_argument("--bmin", type=int, default=2)
    ap.add_argument("--bmax", type=int, default=4)

    ap.add_argument("--alpha_bit", type=int, default=3)
    ap.add_argument("--alpha_default", type=float, default=1.0)

    ap.add_argument("--generations", type=int, default=200)
    ap.add_argument("--candidates_per_gen", type=int, default=256)
    ap.add_argument("--filter_p", type=int, default=32)
    ap.add_argument("--max_tries_per_candidate", type=int, default=50)

    # local/global mixture
    ap.add_argument("--local_ratio", type=float, default=0.8)
    ap.add_argument("--local_swaps_min", type=int, default=1)
    ap.add_argument("--local_swaps_max", type=int, default=3)
    ap.add_argument("--local_tau_up", type=float, default=1.0)
    ap.add_argument("--local_tau_down", type=float, default=1.0)

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
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

    # backup original weights (CPU)
    original_state = {
        k: v.detach().clone().cpu() for k, v in model.state_dict().items()
    }

    # load eval tokens
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    eval_input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).input_ids

    # evaluator
    ppl_eval = PplEvaluator(
        model=model,
        original_state_dict=original_state,
        eval_input_ids=eval_input_ids,
        prebake_root=args.prebake_root,
        eval_seq_len=args.eval_seq_len,
        cache_maxsize=args.ppl_cache_max,
    )
    target_layers = ppl_eval.target_layers

    # W_map
    W_map = {mn: int(original_state[f"{mn}.weight"].numel()) for mn in target_layers}
    sum_w = int(sum(W_map.values()))
    g = gcd_list(W_map.values())
    raw_target = float(args.avg_bits) * float(sum_w)
    B_target = snap_to_gcd(raw_target, g)

    # score_map = C_j * alpha_j(alpha_bit)
    sens_map = load_sens_map(args.sens_csv, c_col="C_mean_per_batch")
    alpha_map = load_alpha_map(args.alpha_csv)
    score_map: Dict[str, float] = {}
    for mn in target_layers:
        C = float(sens_map.get(mn, 0.0))
        a = float(alpha_map.get(mn, {}).get(int(args.alpha_bit), args.alpha_default))
        score_map[mn] = max(0.0, C * a)

    print(f"[info] target modules: N={len(target_layers)}")
    print(
        f"[budget] sum_w={sum_w}, gcd={g}, raw_target={raw_target:.6f}, "
        f"B_target={B_target} => wavg_bits={((B_target/sum_w) if sum_w>0 else 0.0):.6f}"
    )

    # load b*
    b_star_raw = load_init_assign_csv(args.init_assign_csv)
    b_star = {mn: int(b_star_raw.get(mn, 4)) for mn in target_layers}
    for mn in target_layers:
        b_star[mn] = max(args.bmin, min(args.bmax, int(b_star[mn])))
    b_star = ensure_complete_assignment(b_star, target_layers, args.bmin)
    b_star = project_to_weighted_budget(
        b_star, W_map, score_map, B_target, args.bmin, args.bmax, gcd_step=g
    )

    print(f"[init] loaded b*: {args.init_assign_csv}")
    print(
        f"[init] b* proxy={calc_proxy(b_star, score_map):.6e}, "
        f"wavg_bits={(weighted_sum_bits(b_star, W_map)/sum_w if sum_w>0 else 0.0):.6f}"
    )

    # output csv
    dataset_csv = out_dir / "ppl_dataset.csv"
    new_file = not dataset_csv.exists()
    f = open(dataset_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new_file:
        w.writerow(
            [
                "generation",
                "proxy_rank_in_gen",
                "proxy",
                "ppl",
                "wavg_bits",
                "sum_wb",
                "bit_assign_csv",
                "source",  # local/global
                "n_swaps",  # local swap count (global=0)
            ]
        )
        f.flush()

    seen = set()

    def make_global_random_assignment() -> Dict[str, int]:
        b = {mn: random.choice([args.bmin, 3, args.bmax]) for mn in target_layers}
        b = ensure_complete_assignment(b, target_layers, args.bmin)
        b = project_to_weighted_budget(
            b, W_map, score_map, B_target, args.bmin, args.bmax, gcd_step=g
        )
        return b

    def make_local_assignment() -> Tuple[Dict[str, int], int]:
        n_swaps = random.randint(int(args.local_swaps_min), int(args.local_swaps_max))
        b = local_swap_mutate(
            b_star=b_star,
            target_layers=target_layers,
            W_map=W_map,
            score_map=score_map,
            bmin=args.bmin,
            bmax=args.bmax,
            n_swaps=n_swaps,
            tau_up=args.local_tau_up,
            tau_down=args.local_tau_down,
        )
        # exact budget
        b = project_to_weighted_budget(
            b, W_map, score_map, B_target, args.bmin, args.bmax, gcd_step=g
        )
        return b, n_swaps

    gen_bar = tqdm(range(1, args.generations + 1), desc="Generations")
    for gidx in gen_bar:
        cands: List[Tuple[float, Dict[str, int], str, int]] = (
            []
        )  # (proxy, b, source, n_swaps)

        for _ in range(args.candidates_per_gen):
            b = None
            src = "global"
            n_swaps = 0

            for _t in range(args.max_tries_per_candidate):
                if random.random() < float(args.local_ratio):
                    cand, n_swaps = make_local_assignment()
                    src = "local"
                else:
                    cand = make_global_random_assignment()
                    src = "global"
                    n_swaps = 0

                key = tuple((mn, int(cand[mn])) for mn in target_layers)
                if key in seen:
                    continue
                seen.add(key)
                b = cand
                break

            if b is None:
                # fallback allow dup
                if random.random() < float(args.local_ratio):
                    b, n_swaps = make_local_assignment()
                    src = "local"
                else:
                    b = make_global_random_assignment()
                    src = "global"
                    n_swaps = 0

            px = calc_proxy(b, score_map)
            cands.append((px, b, src, n_swaps))

        # proxy-top filter_p
        cands.sort(key=lambda x: x[0])
        top = cands[: max(1, min(args.filter_p, len(cands)))]

        # evaluate ppl only for top filter_p
        for r, (px, b, src, n_swaps) in enumerate(top, start=1):
            ppl = ppl_eval.evaluate(b)
            S = weighted_sum_bits(b, W_map)
            wavg = (S / sum_w) if sum_w > 0 else 0.0

            bit_csv = out_dir / "bit_assign" / f"gen{gidx:06d}_rank{r:03d}.csv"
            with open(bit_csv, "w", newline="", encoding="utf-8") as bf:
                bw = csv.writer(bf)
                bw.writerow(["layer_name", "R_int"])
                for mn in target_layers:
                    bw.writerow([mn, int(b.get(mn, 4))])

            w.writerow(
                [
                    gidx,
                    r,
                    f"{px:.6e}",
                    f"{ppl:.6f}",
                    f"{wavg:.6f}",
                    int(S),
                    str(bit_csv),
                    src,
                    int(n_swaps),
                ]
            )
            f.flush()

        gen_bar.set_postfix({"saved": len(top), "best_proxy": f"{top[0][0]:.3e}"})

    f.close()
    print(f"[done] saved: {dataset_csv}")


if __name__ == "__main__":
    main()
