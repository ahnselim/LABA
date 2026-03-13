#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPL Evaluator shared by v10_1 / v10_2.
"""

import gc
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

try:
    from .montecarlo import _safe_name, run_live_ppl_eval
except ImportError:
    from module.montecarlo import _safe_name, run_live_ppl_eval


class PplEvaluator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        original_state_dict: Dict[str, torch.Tensor],
        eval_input_ids: torch.Tensor,
        prebake_root: str,
        eval_seq_len: int,
        cache_maxsize: int = 5000,
    ):
        self.model = model
        self.device = model.device
        self.tokenizer = tokenizer
        self.original_state_dict = original_state_dict
        self.eval_input_ids = eval_input_ids.to(self.device)
        self.prebake_root = Path(prebake_root)
        self.eval_seq_len = int(eval_seq_len)
        self.cache_maxsize = int(cache_maxsize)
        self.ppl_cache = OrderedDict()

        self.target_layers = []
        bit_dirs = []
        for b in (1, 2, 3, 4):
            cand = self.prebake_root / f"bit{b}"
            if cand.exists():
                bit_dirs.append(cand)
        if not bit_dirs:
            raise FileNotFoundError(
                f"Pre-bake 디렉토리를 찾을 수 없습니다: {self.prebake_root}/bit[1-4]"
            )
        bit_dir_list = ", ".join(d.name for d in bit_dirs)
        print(f"[PplEvaluator] {bit_dir_list} 스캔하여 대상 레이어 찾는 중...", flush=True)
        for bit_dir in bit_dirs:
            for f in bit_dir.glob("*.pt"):
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
            searched = ", ".join(str(d) for d in bit_dirs)
            raise ValueError(f"Pre-bake 디렉토리({searched})에서 유효한 레이어를 찾지 못했습니다.")
        print(f"[PplEvaluator] 탐색 대상 레이어 {len(self.target_layers)}개 확인.", flush=True)

    def _get_module(self, name: str):
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
            module.weight.data.copy_(
                orig_w.to(device=module.weight.device, dtype=module.weight.dtype)
            )
