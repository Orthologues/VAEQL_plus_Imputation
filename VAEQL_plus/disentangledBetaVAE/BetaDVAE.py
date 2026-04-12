"""Modular adapter for the draft DisentangledBetaVAE tuning utilities.

This module provides stable imports under:
    VAEQL_plus.disentangledBetaVAE.BetaDVAE

while reusing the implementation in:
    codex_CV_beta_C_fine_tuning_draft/DisentangledBetaVAE.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_draft_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    draft_path = repo_root / "codex_CV_beta_C_fine_tuning_draft" / "DisentangledBetaVAE.py"
    spec = importlib.util.spec_from_file_location("_draft_disentangled_beta_vae", draft_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load draft module from {draft_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_draft = _load_draft_module()

iterative_halving_search = _draft.iterative_halving_search
train_and_save_best_model = _draft.train_and_save_best_model

__all__ = [
    "iterative_halving_search",
    "train_and_save_best_model",
]
