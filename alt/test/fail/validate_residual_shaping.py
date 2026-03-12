#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validate residual shaping for low-rank recoverability.

Two input modes:
  (A) load W directly from pretrained model + Wq
  (B) direct W tensor + Wq
  (C) load W directly from pretrained model + (codebook, qcodes, meta) -> reconstruct Wq
  (D) direct W tensor + (codebook, qcodes, meta) -> reconstruct Wq

Projector modes:
  - weighted_residual : top-r projector of R = (W - Wq) D
  - unweighted_residual: top-r projector of E = (W - Wq)
  - alpha_metric      : top-r projector of E diag(d^alpha)
  - random            : random rank-r projector in input space
  - external_cov      : projector from an external covariance/data matrix file

Goal:
  Compare plain weighted SVD vs shaped weighted SVD on residual E = W - Wq.

CUDA_VISIBLE_DEVICES=2 nohup python validate_residual_shaping.py \
    --model_id meta-llama/Llama-3.1-8B \
    --w_key model.layers.0.self_attn.q_proj.weight \
    --codebook_path ./output/llama3_8b/step1_quant/codebook.pt \
    --qcodes_path ./output/llama3_8b/step1_quant/qcodes.pt \
    --meta_path ./output/llama3_8b/step1_quant/meta.pt \
    --quant_key model.layers.0.self_attn.q_proj.weight \
    --d_path ./output/llama3_8b/calib_sqrtdiag.pt \
    --d_key model.layers.0.self_attn.q_proj.weight \
    --projector_source alpha_metric \
    --alphas 0.0 0.25 0.5 0.75 1.0 \
    --rank 64 \
    --etas 1.0 0.8 0.6 0.4 0.2 0.1 \
    --output_dir ./validate_shaping_layer0_qproj/alpha_sweep > validate.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# -----------------------------
# utils
# -----------------------------
def ensure_fp32_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu().contiguous()


def fro_sq(x: torch.Tensor) -> float:
    return float((x * x).sum().item())


def topk_capture_ratio_from_svals(s: torch.Tensor, rank: int) -> float:
    num = float((s[:rank] ** 2).sum().item())
    den = float((s ** 2).sum().item()) + 1e-12
    return num / den


def truncated_svd_best_rank(X: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    r = min(rank, S.numel())
    Xr = (U[:, :r] * S[:r]) @ Vh[:r, :]
    return Xr, S


def weighted_residual(E: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return E * d.view(1, -1)


def recover_unweighted_from_weighted(Xw: torch.Tensor, d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    dinv = 1.0 / d.clamp_min(eps)
    return Xw * dinv.view(1, -1)


def compute_top_r_projector(X: torch.Tensor, rank: int) -> torch.Tensor:
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    r = min(rank, Vh.shape[0])
    Vr = Vh[:r, :].T
    return Vr @ Vr.T


def compute_random_projector(dim: int, rank: int, seed: int) -> torch.Tensor:
    r = min(int(rank), int(dim))
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    basis = torch.randn((dim, r), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(basis, mode='reduced')
    return q @ q.T


def compute_projector_from_external_matrix(X: torch.Tensor, rank: int, target_dim: int) -> torch.Tensor:
    if X.ndim != 2:
        raise ValueError(f"external projector matrix must be 2D, got {X.ndim}D")

    # Accept either a square feature covariance [I, I] or a data matrix whose feature axis is I.
    if X.shape == (target_dim, target_dim):
        Xsym = 0.5 * (X + X.T)
        _, evecs = torch.linalg.eigh(Xsym)
        r = min(rank, target_dim)
        Vr = evecs[:, -r:]
        return Vr @ Vr.T

    if X.shape[1] == target_dim:
        return compute_top_r_projector(X, rank=rank)
    if X.shape[0] == target_dim:
        return compute_top_r_projector(X.T, rank=rank)

    raise ValueError(
        f"external projector matrix shape {tuple(X.shape)} is incompatible with in_features={target_dim}"
    )


def compute_alpha_metric_projector(E: torch.Tensor, d: torch.Tensor, rank: int, alpha: float) -> torch.Tensor:
    d_alpha = d.clamp_min(1e-12).pow(float(alpha))
    R_alpha = weighted_residual(E, d_alpha)
    return compute_top_r_projector(R_alpha, rank=rank)


def apply_shaping_matrix(R: torch.Tensor, P: torch.Tensor, eta: float) -> torch.Tensor:
    RP = R @ P
    return eta * R + (1.0 - eta) * RP


def apply_inverse_shaping(X: torch.Tensor, P: torch.Tensor, eta: float, eps: float = 1e-12) -> torch.Tensor:
    eta = max(float(eta), eps)
    XP = X @ P
    return (1.0 / eta) * X + (1.0 - 1.0 / eta) * XP


def evaluate_one_eta(
    E: torch.Tensor,
    d: torch.Tensor,
    rank: int,
    eta: float,
    P: torch.Tensor,
    alpha: Optional[float] = None,
) -> Dict[str, float]:
    R = weighted_residual(E, d)

    Rr_plain, S_plain = truncated_svd_best_rank(R, rank=rank)
    Ehat_plain = recover_unweighted_from_weighted(Rr_plain, d)

    RG = apply_shaping_matrix(R, P, eta=eta)
    RG_r, S_shaped = truncated_svd_best_rank(RG, rank=rank)
    Rr_shaped = apply_inverse_shaping(RG_r, P, eta=eta)
    Ehat_shaped = recover_unweighted_from_weighted(Rr_shaped, d)

    plain_weighted_err = fro_sq(weighted_residual(E - Ehat_plain, d))
    shaped_weighted_err = fro_sq(weighted_residual(E - Ehat_shaped, d))

    plain_unweighted_err = fro_sq(E - Ehat_plain)
    shaped_unweighted_err = fro_sq(E - Ehat_shaped)

    plain_capture = topk_capture_ratio_from_svals(S_plain, rank)
    shaped_capture = topk_capture_ratio_from_svals(S_shaped, rank)

    plain_tail = float((S_plain[rank:] ** 2).sum().item()) if rank < S_plain.numel() else 0.0
    shaped_tail = float((S_shaped[rank:] ** 2).sum().item()) if rank < S_shaped.numel() else 0.0

    return {
        'alpha': (None if alpha is None else float(alpha)),
        'eta': float(eta),
        'plain_capture_ratio': plain_capture,
        'shaped_capture_ratio': shaped_capture,
        'capture_gain': shaped_capture - plain_capture,
        'plain_tail_energy': plain_tail,
        'shaped_tail_energy': shaped_tail,
        'tail_reduction': plain_tail - shaped_tail,
        'plain_weighted_recon_err': plain_weighted_err,
        'shaped_weighted_recon_err': shaped_weighted_err,
        'weighted_err_delta': shaped_weighted_err - plain_weighted_err,
        'plain_unweighted_recon_err': plain_unweighted_err,
        'shaped_unweighted_recon_err': shaped_unweighted_err,
        'unweighted_err_delta': shaped_unweighted_err - plain_unweighted_err,
    }


# -----------------------------
# loading helpers
# -----------------------------
def load_tensor_file(path: str, key: str | None = None) -> torch.Tensor:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, torch.Tensor):
        return ensure_fp32_cpu(obj)

    if isinstance(obj, dict):
        if key is not None:
            if key not in obj:
                raise KeyError(f"Key '{key}' not found in {path}. Available: {list(obj.keys())[:20]}")
            return ensure_fp32_cpu(obj[key])

        for cand in ['W', 'Wq', 'E', 'weight', 'tensor', 'value']:
            if cand in obj and isinstance(obj[cand], torch.Tensor):
                return ensure_fp32_cpu(obj[cand])

        raise ValueError(f"Could not infer tensor key from dict in {path}. Please specify --*_key.")
    raise TypeError(f"Unsupported file format in {path}: {type(obj)}")


def load_d_vector(path: str, key: str | None = None) -> torch.Tensor:
    obj = torch.load(path, map_location='cpu')

    if isinstance(obj, torch.Tensor):
        return ensure_fp32_cpu(obj).flatten()

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported d file format in {path}: {type(obj)}")

    calib_map = obj.get('cov_ops', obj)

    if key is not None:
        entry = calib_map.get(key)
        if entry is None and key.endswith('.weight'):
            entry = calib_map.get(key[:-7])
        if entry is None:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {list(calib_map.keys())[:20]}")
    else:
        if len(calib_map) == 1:
            entry = next(iter(calib_map.values()))
        else:
            raise ValueError(f"Please specify --d_key for {path}")

    if isinstance(entry, torch.Tensor):
        return ensure_fp32_cpu(entry).flatten()

    if isinstance(entry, dict):
        if 's' in entry:
            return ensure_fp32_cpu(entry['s']).flatten()
        if 'sqrt' in entry:
            return ensure_fp32_cpu(entry['sqrt']).flatten()
        if 'var' in entry:
            return torch.sqrt(ensure_fp32_cpu(entry['var']).clamp_min(0.0)).flatten()

    raise ValueError(f"Could not infer d vector from {path} with key={key}")


def load_projector_matrix(path: str, key: str | None = None) -> torch.Tensor:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, torch.Tensor):
        return ensure_fp32_cpu(obj)

    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported projector file format in {path}: {type(obj)}")

    matrix_map = obj.get('cov_ops', obj)
    entry = None
    if key is not None:
        entry = matrix_map.get(key)
        if entry is None and key.endswith('.weight'):
            entry = matrix_map.get(key[:-7])
        if entry is None:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {list(matrix_map.keys())[:20]}")
    elif len(matrix_map) == 1:
        entry = next(iter(matrix_map.values()))
    else:
        for cand in ['cov', 'gram', 'matrix', 'tensor', 'value']:
            if cand in matrix_map and isinstance(matrix_map[cand], torch.Tensor):
                entry = matrix_map[cand]
                break
        if entry is None:
            raise ValueError(f"Please specify --projector_key for {path}")

    if isinstance(entry, torch.Tensor):
        return ensure_fp32_cpu(entry)

    if isinstance(entry, dict):
        for cand in ['cov', 'gram', 'matrix', 'tensor', 'value']:
            val = entry.get(cand)
            if isinstance(val, torch.Tensor):
                return ensure_fp32_cpu(val)

    raise ValueError(f"Could not infer projector matrix from {path} with key={key}")


def _resolve_load_dtype(dtype_name: str) -> Optional[torch.dtype]:
    name = str(dtype_name).strip().lower()
    if name in {'fp16', 'float16', 'half'}:
        return torch.float16
    if name in {'bf16', 'bfloat16'}:
        return torch.bfloat16
    if name in {'fp32', 'float32'}:
        return torch.float32
    if name in {'auto', ''}:
        if not torch.cuda.is_available():
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    raise ValueError(f"Unsupported --model_dtype: {dtype_name}")


def _materialize_weight_from_module(module: nn.Module, full_name: str) -> torch.Tensor:
    """
    Read a module weight safely even when `device_map="auto"` leaves it as a meta tensor
    and relies on accelerate CPU offload hooks.
    """
    w = getattr(module, 'weight', None)
    if w is None:
        raise RuntimeError(f"Module has no weight for {full_name}")

    hook = getattr(module, '_hf_hook', None)
    used_offload_hook = False

    try:
        if getattr(w, 'is_meta', False):
            if hook is None or not hasattr(hook, 'pre_forward'):
                raise RuntimeError(
                    f"Weight is meta for {full_name}, but no accelerate hook is attached. "
                    'Try running with --model_device_map cpu/none.'
                )
            hook.pre_forward(module)
            used_offload_hook = True
            w = getattr(module, 'weight', None)
            if w is None or getattr(w, 'is_meta', False):
                raise RuntimeError(
                    f"Failed to materialize offloaded weight for {full_name} "
                    '(still meta after pre_forward).'
                )

        return ensure_fp32_cpu(w)
    finally:
        if used_offload_hook and hook is not None and hasattr(hook, 'post_forward'):
            try:
                hook.post_forward(module, None)
            except Exception:
                pass


def load_weight_from_model(
    model_id: str,
    full_name: str,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    model_dtype: str = 'auto',
    model_device_map: Optional[str] = 'auto',
) -> torch.Tensor:
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError(
            'transformers is required to load W directly from --model_id'
        ) from e

    resolved_device_map = model_device_map
    if resolved_device_map is not None and str(resolved_device_map).strip().lower() in {'', 'none', 'null'}:
        resolved_device_map = None

    load_dtype = _resolve_load_dtype(model_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=(load_dtype if load_dtype in (torch.float16, torch.bfloat16) else None),
        trust_remote_code=trust_remote_code,
        device_map=resolved_device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    try:
        if full_name.endswith('.weight'):
            module_name = full_name[:-7]
            try:
                module = model.get_submodule(module_name)
            except AttributeError as e:
                raise KeyError(f"Could not resolve module '{module_name}' for weight '{full_name}'") from e
            return _materialize_weight_from_module(module, full_name)

        named_params = dict(model.named_parameters())
        if full_name not in named_params:
            sample = list(named_params.keys())[:20]
            raise KeyError(f"Parameter '{full_name}' not found in model. Available examples: {sample}")
        return ensure_fp32_cpu(named_params[full_name])
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_projector(
    projector_source: str,
    E: torch.Tensor,
    d: torch.Tensor,
    rank: int,
    projector_path: Optional[str] = None,
    projector_key: Optional[str] = None,
    projector_seed: int = 0,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    source = str(projector_source).strip().lower()
    R = weighted_residual(E, d)
    target_dim = int(R.shape[1])

    if source == 'weighted_residual':
        return compute_top_r_projector(R, rank=rank)
    if source == 'unweighted_residual':
        return compute_top_r_projector(E, rank=rank)
    if source == 'alpha_metric':
        if alpha is None:
            raise ValueError('--alpha or --alphas is required when --projector_source alpha_metric')
        return compute_alpha_metric_projector(E, d, rank=rank, alpha=float(alpha))
    if source == 'random':
        return compute_random_projector(dim=target_dim, rank=rank, seed=projector_seed)
    if source == 'external_cov':
        if projector_path is None:
            raise ValueError('--projector_path is required when --projector_source external_cov')
        X = load_projector_matrix(projector_path, projector_key)
        return compute_projector_from_external_matrix(X, rank=rank, target_dim=target_dim)

    raise ValueError(f'Unsupported --projector_source: {projector_source}')


# -----------------------------
# quantized weight reconstruction
# -----------------------------
def reconstruct_wq_from_quant_files(
    codebook_path: str,
    qcodes_path: str,
    meta_path: str,
    quant_key: str,
) -> torch.Tensor:
    codebook_obj = torch.load(codebook_path, map_location='cpu')
    qcodes_obj = torch.load(qcodes_path, map_location='cpu')
    meta_obj = torch.load(meta_path, map_location='cpu')

    if not isinstance(codebook_obj, dict):
        raise TypeError(f'codebook file must be dict: {codebook_path}')
    if not isinstance(qcodes_obj, dict):
        raise TypeError(f'qcodes file must be dict: {qcodes_path}')
    if not isinstance(meta_obj, dict):
        raise TypeError(f'meta file must be dict: {meta_path}')

    if quant_key not in codebook_obj:
        raise KeyError(f"quant_key '{quant_key}' not found in codebook: {list(codebook_obj.keys())[:20]}")
    if quant_key not in qcodes_obj:
        raise KeyError(f"quant_key '{quant_key}' not found in qcodes: {list(qcodes_obj.keys())[:20]}")
    if quant_key not in meta_obj:
        raise KeyError(f"quant_key '{quant_key}' not found in meta: {list(meta_obj.keys())[:20]}")

    codebook = ensure_fp32_cpu(codebook_obj[quant_key])
    qcodes = qcodes_obj[quant_key].detach().cpu()
    meta = meta_obj[quant_key]

    if codebook.ndim != 3:
        raise ValueError(f'codebook must be [O,G,L], got {tuple(codebook.shape)}')
    if qcodes.ndim != 3:
        raise ValueError(f'qcodes must be [O,G,S], got {tuple(qcodes.shape)}')
    if not isinstance(meta, dict):
        raise TypeError(f'meta[{quant_key}] must be dict')

    O, G, _ = codebook.shape
    O2, G2, S = qcodes.shape
    if (O, G) != (O2, G2):
        raise ValueError(
            f'Shape mismatch: codebook {tuple(codebook.shape)} vs qcodes {tuple(qcodes.shape)}'
        )

    orig_shape = meta.get('orig_shape', None)
    if orig_shape is None:
        raise KeyError(f"meta[{quant_key}] missing 'orig_shape'")
    if isinstance(orig_shape, torch.Size):
        orig_shape = tuple(orig_shape)
    orig_O, orig_I = int(orig_shape[0]), int(orig_shape[1])

    qcodes_long = qcodes.to(torch.long)
    xq = torch.gather(codebook, dim=2, index=qcodes_long)
    wq = xq.reshape(O, G * S)[:, :orig_I]

    if wq.shape != (orig_O, orig_I):
        raise ValueError(
            f'Reconstructed Wq shape mismatch: got {tuple(wq.shape)}, expected {(orig_O, orig_I)}'
        )
    return ensure_fp32_cpu(wq)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('--w_path', type=str, default=None, help='Path to full W tensor .pt')
    ap.add_argument('--model_id', type=str, default=None, help='HF model id or local pretrained model path')
    ap.add_argument('--d_path', type=str, required=True, help='Path to right-weight vector d .pt')
    ap.add_argument('--output_dir', type=str, required=True)

    ap.add_argument('--w_key', type=str, default=None)
    ap.add_argument('--d_key', type=str, default=None)
    ap.add_argument('--revision', type=str, default=None)
    ap.add_argument('--trust_remote_code', action='store_true')
    ap.add_argument('--model_dtype', type=str, default='auto', choices=['auto', 'fp16', 'bf16', 'fp32'])
    ap.add_argument('--model_device_map', type=str, default='auto')

    ap.add_argument('--wq_path', type=str, default=None, help='Path to direct Wq tensor .pt')
    ap.add_argument('--wq_key', type=str, default=None)

    ap.add_argument('--codebook_path', type=str, default=None)
    ap.add_argument('--qcodes_path', type=str, default=None)
    ap.add_argument('--meta_path', type=str, default=None)
    ap.add_argument('--quant_key', type=str, default=None, help='Layer key in codebook/qcodes/meta')

    ap.add_argument(
        '--projector_source',
        type=str,
        default='weighted_residual',
        choices=['weighted_residual', 'unweighted_residual', 'alpha_metric', 'random', 'external_cov'],
        help='How to build the shaping projector',
    )
    ap.add_argument('--alpha', type=float, default=None, help='Single alpha for --projector_source alpha_metric')
    ap.add_argument(
        '--alphas',
        type=float,
        nargs='+',
        default=None,
        help='Alpha sweep values for --projector_source alpha_metric',
    )
    ap.add_argument('--projector_path', type=str, default=None, help='Path to external projector/covariance file')
    ap.add_argument('--projector_key', type=str, default=None, help='Key inside external projector/covariance file')
    ap.add_argument('--projector_seed', type=int, default=0, help='Random seed for --projector_source random')

    ap.add_argument('--rank', type=int, default=64)
    ap.add_argument('--etas', type=float, nargs='+', default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1])

    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.w_path is not None:
        W = load_tensor_file(args.w_path, args.w_key)
        w_source = 'direct_w'
    else:
        if args.model_id is None:
            raise ValueError('Provide either --w_path or --model_id.')
        if not args.w_key:
            raise ValueError('--w_key is required when loading W from --model_id.')
        W = load_weight_from_model(
            model_id=args.model_id,
            full_name=args.w_key,
            revision=args.revision,
            trust_remote_code=bool(args.trust_remote_code),
            model_dtype=args.model_dtype,
            model_device_map=args.model_device_map,
        )
        w_source = 'loaded_from_model'

    d = load_d_vector(args.d_path, args.d_key)

    if args.wq_path is not None:
        Wq = load_tensor_file(args.wq_path, args.wq_key)
        wq_source = 'direct_wq'
    else:
        need = [args.codebook_path, args.qcodes_path, args.meta_path, args.quant_key]
        if any(v is None for v in need):
            raise ValueError(
                'Provide either (--wq_path [--wq_key]) or '
                '(--codebook_path --qcodes_path --meta_path --quant_key).'
            )
        Wq = reconstruct_wq_from_quant_files(
            codebook_path=args.codebook_path,
            qcodes_path=args.qcodes_path,
            meta_path=args.meta_path,
            quant_key=args.quant_key,
        )
        wq_source = 'reconstructed_from_codebook_qcodes_meta'

    if W.shape != Wq.shape:
        raise ValueError(f'W shape {tuple(W.shape)} != Wq shape {tuple(Wq.shape)}')
    if W.ndim != 2:
        raise ValueError(f'W must be 2D, got {W.ndim}D')
    if d.numel() != W.shape[1]:
        raise ValueError(f'd length {d.numel()} must equal in_features {W.shape[1]}')

    E = W - Wq
    projector_key = args.projector_key if args.projector_key is not None else args.w_key

    if args.projector_source == 'alpha_metric':
        if args.alphas is not None:
            alpha_values = [float(v) for v in args.alphas]
        elif args.alpha is not None:
            alpha_values = [float(args.alpha)]
        else:
            alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        alpha_values = [None]

    results: List[Dict[str, float]] = []
    for alpha_value in alpha_values:
        P = build_projector(
            projector_source=args.projector_source,
            E=E,
            d=d,
            rank=int(args.rank),
            projector_path=args.projector_path,
            projector_key=projector_key,
            projector_seed=int(args.projector_seed),
            alpha=alpha_value,
        )
        for eta in args.etas:
            results.append(
                evaluate_one_eta(
                    E=E,
                    d=d,
                    rank=args.rank,
                    eta=float(eta),
                    P=P,
                    alpha=alpha_value,
                )
            )

    with open(outdir / 'results.json', 'w') as f:
        json.dump(
            {
                'w_source': w_source,
                'wq_source': wq_source,
                'projector_source': args.projector_source,
                'projector_path': args.projector_path,
                'projector_key': projector_key,
                'projector_seed': int(args.projector_seed),
                'alpha': (None if args.alpha is None else float(args.alpha)),
                'alphas': alpha_values,
                'w_shape': list(W.shape),
                'rank': int(args.rank),
                'etas': list(map(float, args.etas)),
                'results': results,
            },
            f,
            indent=2,
        )

    import csv

    keys = list(results[0].keys()) if results else []
    with open(outdir / 'results.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(results)

    print('=' * 100)
    print(
        f"[Done] rank={args.rank} | w_source={w_source} | wq_source={wq_source} "
        f"| projector_source={args.projector_source}"
    )
    print(
        f"{'alpha':>8} | {'eta':>8} | {'cap_plain':>10} | {'cap_shaped':>11} | "
        f"{'gain':>10} | {'werr_delta':>12} | {'uerr_delta':>12}"
    )
    print('-' * 100)
    for r in results:
        alpha_val = '-' if r['alpha'] is None else f"{r['alpha']:.3f}"
        print(
            f"{alpha_val:>8} | "
            f"{r['eta']:8.3f} | "
            f"{r['plain_capture_ratio']:10.6f} | "
            f"{r['shaped_capture_ratio']:11.6f} | "
            f"{r['capture_gain']:10.6f} | "
            f"{r['weighted_err_delta']:12.4f} | "
            f"{r['unweighted_err_delta']:12.4f}"
        )
    print('=' * 100)
    print(f"saved: {outdir / 'results.csv'}")
    print(f"saved: {outdir / 'results.json'}")


if __name__ == '__main__':
    main()
