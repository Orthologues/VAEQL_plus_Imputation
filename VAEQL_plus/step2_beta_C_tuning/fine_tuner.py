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
- CV tuning is evaluated on beta-C validation entries (validation mask == 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure local imports work when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from VAEQL_plus.conf.feat_types import FeaturesTypeDict
from VAEQL_plus.conf.config import DisentangledBetaVaeTuningConfig
from VAEQL_plus.step1_preprocessing.feat_preprocessor import FeaturePreprocessor
from VAEQL_plus.beta_DVAE.lightning_mod import BetaGausMixedDVAETrainer
from VAEQL_plus.beta_DVAE.torch_nn import BetaGausMixedDVAE


def _available_devices() -> list[str]:
    """Prefer multi-GPU. If only one or none, fall back to CPU for multiprocessing."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def _run_candidates_parallel(
    candidates: list[tuple[float, float]],
    config: Dict[str, Any],
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    data_X_mask: np.ndarray,
) -> list[dict]:
    """Execute candidate CV jobs in parallel using GPUs when available, otherwise CPU."""
    devices = _available_devices()
    jobs = []
    for idx, (b, c) in enumerate(candidates):
        jobs.append(
            {
                "beta": b,
                "C": c,
                "config": config,
                "data_full": data_full,
                "data_missing_nan": data_missing_nan,
                "data_X_mask": data_X_mask,
                "device": devices[idx % len(devices)],
            }
        )

    if len(jobs) == 1:
        return [BetaGausMixedDVAETrainer.run_candidate_cv(jobs[0])]

    workers = min(len(jobs), cpu_count()) if devices == ["cpu"] else min(len(jobs), len(devices))
    with Pool(processes=workers) as pool:
        return list(pool.imap_unordered(BetaGausMixedDVAETrainer.run_candidate_cv, jobs))


def iterative_halving_search(
    config: Dict[str, Any],
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    data_X_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Iteratively shrink beta/C ranges by evaluating four corner combos
    until ranges are smaller than granularity * original_span.
    """
    beta_min, beta_max = config["beta_range"]
    C_min, C_max = config["C_range"]
    granularity = config.get("granularity", 0.01)

    beta_span0 = beta_max - beta_min
    C_span0 = C_max - C_min
    best_global = None
    iteration = 0

    while True:
        iteration += 1
        combos = [
            (beta_min, C_min),
            (beta_min, C_max),
            (beta_max, C_min),
            (beta_max, C_max),
        ]
        print(f"[Iter {iteration}] evaluating beta in [{beta_min}, {beta_max}] C in [{C_min}, {C_max}]")
        results = _run_candidates_parallel(combos, config, data_full, data_missing_nan, data_X_mask)
        results = sorted(results, key=lambda r: r["score"])
        best_round = results[0]
        if best_global is None or best_round["score"] < best_global["score"]:
            best_global = best_round

        beta_m = (beta_min + beta_max) / 2.0
        C_m = (C_min + C_max) / 2.0

        if best_round["beta"] == beta_min and best_round["C"] == C_min:
            beta_max, C_max = beta_m, C_m
        elif best_round["beta"] == beta_min and best_round["C"] == C_max:
            beta_max, C_min = beta_m, C_m
        elif best_round["beta"] == beta_max and best_round["C"] == C_min:
            beta_min, C_max = beta_m, C_m
        else:
            beta_min, C_min = beta_m, C_m

        if (beta_max - beta_min) <= granularity * beta_span0 and (C_max - C_min) <= granularity * C_span0:
            break

    print(f"[Search done] best beta={best_global['beta']}, C={best_global['C']}, score={best_global['score']}")
    return best_global


def train_and_save_best_model(
    beta_val: float,
    capacity_C: float,
    config: Dict[str, Any],
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    data_X_mask: np.ndarray,
):
    """Train one fold with selected hyperparameters and persist checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_chunk, final_epochs = BetaGausMixedDVAETrainer._epoch_settings_from_config(config)
    metric = config.get("halving_metric", "mae")
    tolerance = config.get("convergence_tolerance", 1e-4)
    patience = config.get("convergence_patience", 2)
    min_epochs = config.get("min_epochs_before_convergence", epoch_chunk)

    split_output = BetaGausMixedDVAETrainer.split_training_and_validation(
        config, 0, data_missing_nan, data_full, data_X_mask
    )
    training_input_np = split_output.training_input
    validation_input = split_output.validation_input
    validation_mask = split_output.validation_mask
    pre_val_amputation_mask = split_output.pre_val_amputation_mask

    train_tensor = torch.tensor(training_input_np, dtype=torch.float32)
    if "features_type_dict" not in config:
        raise KeyError("Missing required config key: `features_type_dict`")
    feat_type_dict: FeaturesTypeDict = dict(config["features_type_dict"])
    model = BetaGausMixedDVAE(
        input_dim=train_tensor.shape[1],
        latent_dim=config["vae_cont_lat_dim"],
        hidden_dim1=config["hidden_size_1"],
        hidden_dim2=config["hidden_size_2"],
        n_gmm_components=int(config["n_gmm_components"]),
        num_feat_loss_metric=config["num_feat_loss_metric"],
    ).to(device)
    optimizer = BetaGausMixedDVAETrainer.build_optimizer(model, lr=config["learning_rate"], use_adam=config.get("use_adam_optimizer", False))

    best_metric, completed_epochs = BetaGausMixedDVAETrainer.train_fold_with_eval(
        model=model,
        optimizer=optimizer,
        train_tensor=train_tensor,
        beta=beta_val,
        capacity_C=capacity_C,
        device=device,
        feat_type_dict=feat_type_dict,
        batch_size=config["batch_size"],
        epoch_chunk=epoch_chunk,
        final_epochs=final_epochs,
        metric=metric,
        validation_input=validation_input,
        validation_mask=validation_mask, # expected to contain values {0, 1, 2, 3, 4} with 3 as $\beta$-C validation entries
        imputation_mask=pre_val_amputation_mask, # expected to contain values {0, 1, 2} with 2 as imputation entries for final evaluation
        recycles=config["recycles"],
        num_imputations=config["m"],
        fold_idx=0,
        results_path=config["results_path"],
        tolerance=tolerance,
        patience=patience,
        min_epochs=min_epochs,
    )

    outdir = config.get("model_outdir", "./trained_models")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(Path(outdir) / f"beta{beta_val}_C{capacity_C}_fold0.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "beta": beta_val,
            "C": capacity_C,
            "fold": 0,
            "epochs": completed_epochs,
            "best_metric": best_metric,
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build:
      - data_full       : complete reference matrix (float32)
      - data_missing_nan: pre-imputed numeric matrix used as model input
      - mask            : 0 observed / 1 pre-existing NaN / 2 newly amputed
      - model_feat_dict : feature-type metadata aligned to preprocessed columns
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
    model_feat_dict: Dict[str, Any] = dict(feat_dict)
    model_feat_dict["all_feats"] = set(pre_df.columns)

    return (
        complete_df.to_numpy(dtype=np.float32, copy=True),
        synthetic_missing_df.to_numpy(dtype=np.float32, copy=True),
        mask.astype(np.int8, copy=False),
        model_feat_dict,
    )


def load_and_patch_cv_config(
    cv_config_path: Path,
    results_path: Path,
    model_outdir: Path,
    k_folds_override: int | None,
) -> Dict[str, Any]:
    with open(cv_config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)
    train_val_defaults = DisentangledBetaVaeTuningConfig.default_train_val_params()
    config = {
        "n_gmm_components": train_val_defaults["n_gmm_components"],
        "vae_cont_lat_dim": train_val_defaults["vae_cont_lat_dim"],
        "hidden_size_1": train_val_defaults["hidden_size_1"],
        "hidden_size_2": train_val_defaults["hidden_size_2"],
        "num_feat_loss_metric": train_val_defaults["num_feat_loss_metric"],
        **config,
    }

    # Force MAE as requested for this step.
    config["halving_metric"] = "mae"
    config["results_path"] = str(results_path)
    config["model_outdir"] = str(model_outdir)
    if k_folds_override is not None:
        config["k_folds"] = int(k_folds_override)

    required = [
        "beta_range",
        "C_range",
        "learning_rate",
        "batch_size",
        "recycles",
        "m",
        "k_folds",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"cv config missing required keys: {missing!r}")
    if config["halving_metric"] not in BetaGausMixedDVAETrainer.CAND_METRIC_KEYS:
        raise ValueError(
            f"Unsupported halving_metric {config['halving_metric']!r}. "
            f"Permitted values: {sorted(BetaGausMixedDVAETrainer.CAND_METRIC_KEYS)!r}"
        )

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step1 beta/C halving CV tuner (preliminary)")
    parser.add_argument("--input_csv", type=Path, required=True, help="Raw tabular dataset CSV path")
    parser.add_argument("--feature_dict_json", type=Path, required=True, help="Feature-type JSON for FeaturesTypeDict")
    parser.add_argument(
        "--cv_config",
        type=Path,
        default=DisentangledBetaVaeTuningConfig.default_cv_beta_C_tuning_config_path(),
    )
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

    data_full, data_missing_nan, mask, model_feat_dict = build_cv_matrices(
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
    config["features_type_dict"] = model_feat_dict

    best_candidate = iterative_halving_search(
        config=config,
        data_full=data_full,
        data_missing_nan=data_missing_nan,
        data_X_mask=mask,
    )

    checkpoint_path = train_and_save_best_model(
        beta_val=float(best_candidate["beta"]),
        capacity_C=float(best_candidate["C"]),
        config=config,
        data_full=data_full,
        data_missing_nan=data_missing_nan,
        data_X_mask=mask,
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
