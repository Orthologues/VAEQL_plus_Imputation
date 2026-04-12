#!/usr/bin/env python3
"""
Preliminary beta/C fine-tuning entrypoint for disentangled beta-VAE.

Pipeline:
1) Load raw tabular data and feature-type dictionary.
2) Preprocess via FeaturePreprocessor (with configurable pre-imputation, default MICE).
3) Build a complete reference matrix + synthetic-missing matrix for CV.
4) Run halving-grid beta/C search with MAE metric.
5) Train and save best model checkpoint.

Notes:
- This is intentionally a preliminary scaffold for step1 tuning.
- CV tuning is evaluated on synthetically amputed entries (mask == 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# Ensure local imports work when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from VAEQL_plus.conf.feat_types import FeaturesTypeDict
from VAEQL_plus.util.feat_preprocessor import FeaturePreprocessor
from VAEQL_plus.disentangledBetaVAE import iterative_halving_search, train_and_save_best_model


class IdentityScaler:
    """Minimal scaler shim for draft CV utilities expecting inverse_transform."""

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x


def _fill_preexisting_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill pre-existing NaNs only for building a complete reference matrix."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.isna().any():
            fill = float(s.mean()) if s.notna().any() else 0.0
            out[col] = s.fillna(fill)
    return out


def build_cv_matrices(
    input_csv: Path,
    feature_dict_json: Path,
    missing_mechanism: str,
    missing_rate: float,
    pre_imputation_method: str,
    pre_imputation_max_iter: int,
    mice_num_imputations: int,
    random_forest_n_estimators: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build:
      - data_full       : complete reference matrix (float32)
      - data_missing_nan: same matrix with synthetic missing entries as NaN
      - mask            : 0 observed / 1 pre-existing NaN / 2 newly amputed
    """
    raw_df = pd.read_csv(input_csv)
    feat_dict = FeaturesTypeDict.create(str(feature_dict_json))

    preprocessor = FeaturePreprocessor(
        feat_dict=feat_dict,
        missing_mechanism=missing_mechanism,
        missing_rate=missing_rate,
        input_df=raw_df,
        use_spark=False,
        pre_imputation_method=pre_imputation_method,
        pre_imputation_max_iter=pre_imputation_max_iter,
        mice_num_imputations=mice_num_imputations,
        random_forest_n_estimators=random_forest_n_estimators,
    )

    # Preliminary step1 intentionally uses internal helpers to expose both
    # complete/reference and synthetically missing views for CV tuning.
    pre_df, _, _, _, ord_groups, cat_groups = preprocessor._build_preprocessed_dataframe()
    amputed_df, mask = preprocessor._apply_pyampute(pre_df, ord_groups, cat_groups)

    complete_df = _fill_preexisting_nans(pre_df)
    synthetic_missing_df = complete_df.copy()
    synthetic_missing_df.values[mask == 2] = np.nan

    return (
        complete_df.to_numpy(dtype=np.float32, copy=True),
        synthetic_missing_df.to_numpy(dtype=np.float32, copy=True),
        mask.astype(np.int8, copy=False),
    )


def load_and_patch_cv_config(
    cv_config_path: Path,
    results_path: Path,
    model_outdir: Path,
    k_folds_override: int | None,
) -> Dict[str, Any]:
    with open(cv_config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    # Force MAE as requested for this step.
    config["halving_metric"] = "mae"
    config["results_path"] = str(results_path)
    config["model_outdir"] = str(model_outdir)
    if k_folds_override is not None:
        config["k_folds"] = int(k_folds_override)

    required = [
        "beta_range",
        "C_range",
        "latent_size",
        "hidden_size_1",
        "hidden_size_2",
        "learning_rate",
        "batch_size",
        "recycles",
        "m",
        "k_folds",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"cv config missing required keys: {missing}")

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1 beta/C halving CV tuner (preliminary)")
    parser.add_argument("--input_csv", type=Path, required=True, help="Raw tabular dataset CSV path")
    parser.add_argument("--feature_dict_json", type=Path, required=True, help="Feature-type JSON for FeaturesTypeDict")
    parser.add_argument("--cv_config", type=Path, default=REPO_ROOT / "VAEQL_plus" / "disentangledBetaVAE" / "cv_trainer_params.json")
    parser.add_argument("--missing_mechanism", type=str, default="MAR", choices=["MAR", "MNAR", "MCAR"])
    parser.add_argument("--missing_rate", type=float, default=0.1)

    parser.add_argument("--pre_imputation_method", type=str, default="MICE")
    parser.add_argument("--pre_imputation_max_iter", type=int, default=5)
    parser.add_argument("--mice_num_imputations", type=int, default=5)
    parser.add_argument("--random_forest_n_estimators", type=int, default=100)

    parser.add_argument("--k_folds", type=int, default=None)
    parser.add_argument("--results_path", type=Path, default=REPO_ROOT / "VAEQL_plus" / "step1_beta_C_tuning" / "beta_C_halving_mae.csv")
    parser.add_argument("--model_outdir", type=Path, default=REPO_ROOT / "VAEQL_plus" / "step1_beta_C_tuning" / "trained_models")
    parser.add_argument("--summary_json", type=Path, default=REPO_ROOT / "VAEQL_plus" / "step1_beta_C_tuning" / "best_beta_C_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_outdir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    data_full, data_missing_nan, mask = build_cv_matrices(
        input_csv=args.input_csv,
        feature_dict_json=args.feature_dict_json,
        missing_mechanism=args.missing_mechanism,
        missing_rate=float(args.missing_rate),
        pre_imputation_method=args.pre_imputation_method,
        pre_imputation_max_iter=int(args.pre_imputation_max_iter),
        mice_num_imputations=int(args.mice_num_imputations),
        random_forest_n_estimators=int(args.random_forest_n_estimators),
    )

    config = load_and_patch_cv_config(
        cv_config_path=args.cv_config,
        results_path=args.results_path,
        model_outdir=args.model_outdir,
        k_folds_override=args.k_folds,
    )

    scaler = IdentityScaler()
    best_candidate = iterative_halving_search(
        config=config,
        data_full=data_full,
        data_missing_nan=data_missing_nan,
        scaler=scaler,
    )

    checkpoint_path = train_and_save_best_model(
        beta_val=float(best_candidate["beta"]),
        Cval=float(best_candidate["C"]),
        config=config,
        data_full=data_full,
        data_missing_nan=data_missing_nan,
        scaler=scaler,
    )

    summary = {
        "best_beta": float(best_candidate["beta"]),
        "best_C": float(best_candidate["C"]),
        "best_score": float(best_candidate["score"]),
        "halving_metric": "mae",
        "checkpoint_path": str(checkpoint_path),
        "results_path": str(args.results_path),
        "mask_counts": {
            "observed_0": int((mask == 0).sum()),
            "preexisting_nan_1": int((mask == 1).sum()),
            "newly_amputed_2": int((mask == 2).sum()),
        },
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
