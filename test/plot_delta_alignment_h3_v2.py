#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_delta_alignment_h3_v2.py

✅ 목적 (H3 검증용, corr 대신 "분포 정렬"로 안정적 검증)
- code usage 분포 p(q) 와 Δ-shift 분포 s(q) 가 얼마나 "가까운지"를 측정
- uv vs no_uv 비교 (diff = uv - no_uv, 음수면 uv가 더 정렬됨)

핵심 지표:
  1) JS divergence: JS(p, s)  (대칭, 안정적, 추천)
  2) KL(p||s), KL(s||p)
  3) (옵션) 1D Wasserstein/EMD (q가 "순서형"일 때 의미 있음: 0<1<2<3)

추가:
  - global: 전체 레이어 텐서에서 p(q), s(q)
  - per-group: (O,G)별로 p_g(q), s_g(q) 를 만들고 평균/중앙값으로 집계
    (uv 효과가 group-wise로 나타날 수도 있어서 강추)

입력:
  --step1_dir/qcodes.pt
  --delta_uv  (uv run delta_best.pt)
  --delta_no  (no_uv run delta_best.pt)

출력(out_dir):
  - 01_heat_js_global.png
  - 02_heat_js_groupmean.png
  - 03_box_js_by_module.png
  - 04_box_js_groupmean_by_module.png
  - summary_h3_alignment.csv

사용 예시:
python test/plot_delta_alignment_h3_v2.py \
  --step1_dir ./output/2bit_asym_nonuniform_base \
  --delta_uv  ./output/4bit_output_bo/4bit_bo_block8_v2_20260228_163743/stage2_full/delta_best.pt \
  --delta_no  ./output/4bit_output_bo/4bit_bo_block8_v3_20260228_165131/stage2_full/delta_best.pt \
  --out_dir   ./analysis_h3_v2_4bit \
  --device cuda

Tip:
- 2bit(Q=4)에서는 corr보다 JS/KL이 훨씬 안정적으로 H3를 보여줍니다.
"""

import os, re, csv, argparse, math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
MODULE_ORDER = {
    "q_proj": 0, "k_proj": 1, "v_proj": 2, "o_proj": 3, "out_proj": 4,
    "gate_proj": 5, "up_proj": 6, "down_proj": 7, "fc1": 8, "fc2": 9,
}

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def block_idx(name: str) -> int:
    m = re.search(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1

def module_suffix(name: str) -> str:
    parts = name.split(".")
    return parts[-2] if len(parts) >= 2 else ""

def sort_modules(mods: List[str]) -> List[str]:
    return sorted(mods, key=lambda m: MODULE_ORDER.get(m, 10**6))

def normalize_dist(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s <= eps:
        # fallback: uniform
        return np.ones_like(x, dtype=np.float64) / max(x.size, 1)
    return x / s

def normalize_dist_t(x: torch.Tensor, eps=1e-12, dim=-1):
    x = x.to(torch.float32)
    x = torch.clamp(x, min=0.0)
    s = x.sum(dim=dim, keepdim=True)
    n = max(int(x.size(dim)), 1)
    uniform = torch.full_like(x, 1.0 / float(n))
    return torch.where(s > eps, x / s.clamp_min(eps), uniform)

def kl_div(p, q, eps=1e-12):
    p = normalize_dist(p, eps)
    q = normalize_dist(q, eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

def kl_div_t(p: torch.Tensor, q: torch.Tensor, eps=1e-12, dim=-1):
    p = normalize_dist_t(p, eps=eps, dim=dim).clamp_min(eps)
    q = normalize_dist_t(q, eps=eps, dim=dim).clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=dim)

def js_div(p, q, eps=1e-12):
    p = normalize_dist(p, eps)
    q = normalize_dist(q, eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m, eps) + 0.5 * kl_div(q, m, eps)

def js_div_t(p: torch.Tensor, q: torch.Tensor, eps=1e-12, dim=-1):
    p = normalize_dist_t(p, eps=eps, dim=dim)
    q = normalize_dist_t(q, eps=eps, dim=dim)
    m = 0.5 * (p + q)
    return 0.5 * kl_div_t(p, m, eps=eps, dim=dim) + 0.5 * kl_div_t(q, m, eps=eps, dim=dim)

def wasserstein_1d(p, q, eps=1e-12):
    """
    1D Wasserstein distance on discrete ordered support {0,...,Q-1}.
    For discrete distributions, W1 = sum_k |CDF_p(k) - CDF_q(k)|
    """
    p = normalize_dist(p, eps)
    q = normalize_dist(q, eps)
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))

def wasserstein_1d_t(p: torch.Tensor, q: torch.Tensor, eps=1e-12, dim=-1):
    p = normalize_dist_t(p, eps=eps, dim=dim)
    q = normalize_dist_t(q, eps=eps, dim=dim)
    cdf_p = torch.cumsum(p, dim=dim)
    cdf_q = torch.cumsum(q, dim=dim)
    return torch.sum(torch.abs(cdf_p - cdf_q), dim=dim)

def shift_rms_over_og(delta_ogq: torch.Tensor):
    # delta_ogq: [O,G,Q] -> shift[q] = sqrt(mean_{O,G} delta^2)
    return torch.sqrt(torch.mean(delta_ogq**2, dim=(0,1)))

def shift_absmean_over_og(delta_ogq: torch.Tensor):
    # alt: mean abs
    return torch.mean(delta_ogq.abs(), dim=(0,1))

def group_histograms_from_codes(codes_cs: torch.Tensor, q_levels: int) -> torch.Tensor:
    """
    codes_cs: [C,S] int64, values in [0, Q-1]
    returns counts [C,Q]
    """
    c = codes_cs.shape[0]
    offset = (torch.arange(c, device=codes_cs.device, dtype=torch.long) * q_levels).unsqueeze(1)
    flat = (codes_cs + offset).reshape(-1)
    hist = torch.bincount(flat, minlength=c * q_levels).reshape(c, q_levels)
    return hist.to(torch.float32)

def make_heatmap(M, xlabels, ylabels, title, out_path, cbar_label):
    plt.figure(figsize=(12, 6))
    im = plt.imshow(M, aspect="auto")
    plt.colorbar(im, label=cbar_label)
    plt.xticks(range(len(xlabels)), xlabels, rotation=30, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def boxplot_by_module(values_by_mod, order, title, ylabel, out_path):
    data = [values_by_mod.get(m, []) for m in order]
    plt.figure(figsize=(12, 4))
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def heat_block_module(records, field, blocks, mods):
    bi = {b:i for i,b in enumerate(blocks)}
    mi = {m:i for i,m in enumerate(mods)}
    bucket = [[[] for _ in mods] for __ in blocks]

    for r in records:
        b, m, v = r["block"], r["module"], r.get(field, np.nan)
        if b in bi and m in mi and v is not None and (not np.isnan(v)):
            bucket[bi[b]][mi[m]].append(float(v))

    M = np.full((len(blocks), len(mods)), np.nan, dtype=np.float64)
    for i in range(len(blocks)):
        for j in range(len(mods)):
            if bucket[i][j]:
                M[i, j] = float(np.mean(bucket[i][j]))
    return M


# ----------------------------
# Main
# ----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1_dir", required=True)
    ap.add_argument("--delta_uv", required=True)
    ap.add_argument("--delta_no", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--module_regex", default=None, help='e.g. "q_proj|k_proj|down_proj"')
    ap.add_argument("--shift_mode", default="rms", choices=["rms","absmean"],
                    help="how to compute shift(q) from Δ[O,G,Q]")
    ap.add_argument("--use_wasserstein", action="store_true",
                    help="also compute 1D Wasserstein distance on ordered q")
    ap.add_argument("--group_chunk", type=int, default=65536,
                    help="chunk size over (O*G) for per-group metrics (reduce peak memory)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    dev = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    step1_dir = Path(args.step1_dir)
    qcodes_path = step1_dir / "qcodes.pt"
    if not qcodes_path.exists():
        raise FileNotFoundError(f"missing: {qcodes_path}")

    print("📦 loading qcodes:", qcodes_path)
    qcodes: Dict[str, torch.Tensor] = torch.load(qcodes_path, map_location="cpu")

    print("📦 loading deltas...")
    delta_uv: Dict[str, torch.Tensor] = torch.load(args.delta_uv, map_location="cpu")
    delta_no: Dict[str, torch.Tensor] = torch.load(args.delta_no, map_location="cpu")

    mod_re = re.compile(args.module_regex) if args.module_regex else None

    # common keys
    keys = []
    for k in delta_uv.keys():
        if k in delta_no and k in qcodes:
            m = module_suffix(k)
            if m and (m in MODULE_ORDER):
                if (mod_re is None) or mod_re.search(m):
                    keys.append(k)
    keys = sorted(keys)
    if not keys:
        raise RuntimeError("No common keys found. Check module_regex / paths.")
    print("✅ common keys:", len(keys))

    records = []

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    chunk_size = max(1, int(args.group_chunk))

    for li, k in enumerate(keys, start=1):
        if li == 1 or li % max(1, len(keys) // 10) == 0:
            print(f"[*] processing {li}/{len(keys)}: {k}")

        b = block_idx(k)
        m = module_suffix(k)

        Duv = delta_uv[k].to(device=dev, dtype=torch.float32, non_blocking=True)   # [O,G,Q]
        Dno = delta_no[k].to(device=dev, dtype=torch.float32, non_blocking=True)
        codes = qcodes[k].to(device=dev, dtype=torch.long, non_blocking=True)       # [O,G,S]
        O, G, S = codes.shape
        Q = int(Duv.shape[-1])

        # ---------- global usage p(q)
        hist = torch.bincount(codes.reshape(-1), minlength=Q).to(torch.float32)
        p_global_t = normalize_dist_t(hist, dim=0)

        # ---------- global shift s(q) (from Δ)
        if args.shift_mode == "rms":
            sh_uv_t = shift_rms_over_og(Duv)
            sh_no_t = shift_rms_over_og(Dno)
        else:
            sh_uv_t = shift_absmean_over_og(Duv)
            sh_no_t = shift_absmean_over_og(Dno)

        # divergence (smaller = better aligned)
        js_uv = float(js_div_t(p_global_t, sh_uv_t, dim=0).item())
        js_no = float(js_div_t(p_global_t, sh_no_t, dim=0).item())
        js_diff = js_uv - js_no   # <0 => uv improved alignment

        klp_uv = float(kl_div_t(p_global_t, sh_uv_t, dim=0).item())  # KL(p||s)
        klp_no = float(kl_div_t(p_global_t, sh_no_t, dim=0).item())
        klp_diff = klp_uv - klp_no

        kls_uv = float(kl_div_t(sh_uv_t, p_global_t, dim=0).item())  # KL(s||p)
        kls_no = float(kl_div_t(sh_no_t, p_global_t, dim=0).item())
        kls_diff = kls_uv - kls_no

        w_uv = w_no = w_diff = np.nan
        if args.use_wasserstein:
            w_uv = float(wasserstein_1d_t(p_global_t, sh_uv_t, dim=0).item())
            w_no = float(wasserstein_1d_t(p_global_t, sh_no_t, dim=0).item())
            w_diff = w_uv - w_no

        # ---------- per-group conditional alignment
        codes_og = codes.reshape(O*G, S)  # [OG,S]
        Duv_ogq = Duv.reshape(O*G, Q)     # [OG,Q]
        Dno_ogq = Dno.reshape(O*G, Q)

        total_groups = int(codes_og.shape[0])
        js_g_uv_sum = 0.0
        js_g_no_sum = 0.0
        klp_g_uv_sum = 0.0
        klp_g_no_sum = 0.0
        w_g_uv_sum = 0.0
        w_g_no_sum = 0.0

        for st in range(0, total_groups, chunk_size):
            ed = min(st + chunk_size, total_groups)
            cg = codes_og[st:ed]       # [C,S]
            duv = Duv_ogq[st:ed]       # [C,Q]
            dno = Dno_ogq[st:ed]       # [C,Q]

            pg = normalize_dist_t(group_histograms_from_codes(cg, q_levels=Q), dim=1)
            suv = normalize_dist_t(duv.abs(), dim=1)
            sno = normalize_dist_t(dno.abs(), dim=1)

            js_g_uv_sum += float(js_div_t(pg, suv, dim=1).sum().item())
            js_g_no_sum += float(js_div_t(pg, sno, dim=1).sum().item())
            klp_g_uv_sum += float(kl_div_t(pg, suv, dim=1).sum().item())
            klp_g_no_sum += float(kl_div_t(pg, sno, dim=1).sum().item())
            if args.use_wasserstein:
                w_g_uv_sum += float(wasserstein_1d_t(pg, suv, dim=1).sum().item())
                w_g_no_sum += float(wasserstein_1d_t(pg, sno, dim=1).sum().item())

            del cg, duv, dno, pg, suv, sno

        js_g_uv = js_g_uv_sum / max(total_groups, 1)
        js_g_no = js_g_no_sum / max(total_groups, 1)
        js_g_diff = js_g_uv - js_g_no

        klp_g_uv = klp_g_uv_sum / max(total_groups, 1)
        klp_g_no = klp_g_no_sum / max(total_groups, 1)
        klp_g_diff = klp_g_uv - klp_g_no

        w_g_uv_m = w_g_no_m = w_g_diff = np.nan
        if args.use_wasserstein:
            w_g_uv_m = w_g_uv_sum / max(total_groups, 1)
            w_g_no_m = w_g_no_sum / max(total_groups, 1)
            w_g_diff = w_g_uv_m - w_g_no_m

        records.append({
            "key": k, "block": b, "module": m, "Q": Q,
            "js_uv": js_uv, "js_no": js_no, "js_diff": js_diff,
            "klp_uv": klp_uv, "klp_no": klp_no, "klp_diff": klp_diff,
            "kls_uv": kls_uv, "kls_no": kls_no, "kls_diff": kls_diff,
            "w_uv": w_uv, "w_no": w_no, "w_diff": w_diff,
            "js_g_uv": js_g_uv, "js_g_no": js_g_no, "js_g_diff": js_g_diff,
            "klp_g_uv": klp_g_uv, "klp_g_no": klp_g_no, "klp_g_diff": klp_g_diff,
            "w_g_uv": w_g_uv_m, "w_g_no": w_g_no_m, "w_g_diff": w_g_diff,
        })

        del Duv, Dno, codes, codes_og, Duv_ogq, Dno_ogq, hist, p_global_t, sh_uv_t, sh_no_t

    blocks = sorted(set([r["block"] for r in records if r["block"] >= 0]))
    mods = sort_modules(sorted(set([r["module"] for r in records])))

    # heatmaps: JS diff (uv - no_uv). negative => improved.
    M_js = heat_block_module(records, "js_diff", blocks, mods)
    M_js_g = heat_block_module(records, "js_g_diff", blocks, mods)

    make_heatmap(
        M_js, mods, blocks,
        "H3 alignment diff (JS(p(q), s(q))) GLOBAL : uv - no_uv (negative => better aligned)",
        os.path.join(args.out_dir, "01_heat_js_global.png"),
        cbar_label="JS diff"
    )

    make_heatmap(
        M_js_g, mods, blocks,
        "H3 alignment diff (JS(p_g(q), s_g(q))) GROUP-MEAN : uv - no_uv (negative => better aligned)",
        os.path.join(args.out_dir, "02_heat_js_groupmean.png"),
        cbar_label="JS diff"
    )

    # boxplots by module
    by_mod = {}
    for r in records:
        by_mod.setdefault(r["module"], []).append(r)

    js_by_mod = {mm: [rr["js_diff"] for rr in lst if not np.isnan(rr["js_diff"])] for mm, lst in by_mod.items()}
    js_g_by_mod = {mm: [rr["js_g_diff"] for rr in lst if not np.isnan(rr["js_g_diff"])] for mm, lst in by_mod.items()}

    order = sort_modules(list(js_by_mod.keys()))

    boxplot_by_module(
        js_by_mod, order,
        "H3 alignment diff by module (GLOBAL JS) : uv - no_uv (negative => better aligned)",
        ylabel="JS diff",
        out_path=os.path.join(args.out_dir, "03_box_js_by_module.png")
    )

    boxplot_by_module(
        js_g_by_mod, order,
        "H3 alignment diff by module (GROUP-MEAN JS) : uv - no_uv (negative => better aligned)",
        ylabel="JS diff",
        out_path=os.path.join(args.out_dir, "04_box_js_groupmean_by_module.png")
    )

    # save CSV
    csv_path = os.path.join(args.out_dir, "summary_h3_alignment.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "key","block","module","Q",
            "js_uv","js_no","js_diff",
            "klp_uv","klp_no","klp_diff",
            "kls_uv","kls_no","kls_diff",
            "w_uv","w_no","w_diff",
            "js_g_uv","js_g_no","js_g_diff",
            "klp_g_uv","klp_g_no","klp_g_diff",
            "w_g_uv","w_g_no","w_g_diff",
        ])
        for r in records:
            w.writerow([
                r["key"], r["block"], r["module"], r["Q"],
                r["js_uv"], r["js_no"], r["js_diff"],
                r["klp_uv"], r["klp_no"], r["klp_diff"],
                r["kls_uv"], r["kls_no"], r["kls_diff"],
                r["w_uv"], r["w_no"], r["w_diff"],
                r["js_g_uv"], r["js_g_no"], r["js_g_diff"],
                r["klp_g_uv"], r["klp_g_no"], r["klp_g_diff"],
                r["w_g_uv"], r["w_g_no"], r["w_g_diff"],
            ])

    # quick console summary
    diffs = np.array([r["js_diff"] for r in records if not np.isnan(r["js_diff"])], dtype=np.float64)
    diffs_g = np.array([r["js_g_diff"] for r in records if not np.isnan(r["js_g_diff"])], dtype=np.float64)

    print("✅ saved to:", args.out_dir)
    print("H3 summary (GLOBAL JS diff = uv - no_uv):")
    print("  mean:", float(diffs.mean()), "median:", float(np.median(diffs)),
          "improved fraction(diff<0):", float(np.mean(diffs < 0.0)))
    print("H3 summary (GROUP-MEAN JS diff = uv - no_uv):")
    print("  mean:", float(diffs_g.mean()), "median:", float(np.median(diffs_g)),
          "improved fraction(diff<0):", float(np.mean(diffs_g < 0.0)))
    print("Start here:")
    print("  - 01_heat_js_global.png / 02_heat_js_groupmean.png")
    print("  - 03_box_js_by_module.png / 04_box_js_groupmean_by_module.png")
    print("  - summary_h3_alignment.csv")

if __name__ == "__main__":
    main()
