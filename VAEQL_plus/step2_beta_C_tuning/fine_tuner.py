#!/usr/bin/env python3
"""Fold-isolated grid-halving search for the disentangled beta-VAE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

# Ensure local imports work when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VAEQL_plus.beta_DVAE.lightning_mod import BetaGausMixedDVAETrainer
from VAEQL_plus.beta_DVAE.torch_nn import BetaGausMixedDVAE
from VAEQL_plus.beta_DVAE.utils import BetaGausMixedDVAEUtils
from VAEQL_plus.conf.config import DisentangledBetaVaeTuningConfig
from VAEQL_plus.conf.feat_types import FeaturesTypeDict
from VAEQL_plus.step1_preprocessing.feat_preprocessor import (
    FeaturePreprocessor,
    OrderedFeature,
)
from VAEQL_plus.util.dataset_loader import load_dataset


class ValidationView(NamedTuple):
    """Fold-local model input and the two masks used by validation."""

    model_input: np.ndarray
    validation_mask: np.ndarray
    imputation_mask: np.ndarray


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _available_devices() -> list[str]:
    """Use multiple GPUs when available; otherwise run CPU workers."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def _validate_aligned_data(
    X_ref: np.ndarray,
    X_in: np.ndarray,
    source_mask: np.ndarray,
    feature_specs: Sequence[OrderedFeature] | None,
    feat_type_dict: Dict[str, Any],
) -> None:
    if X_ref.ndim != 2 or X_in.ndim != 2 or source_mask.ndim != 2:
        raise ValueError("Reference/input matrices and source mask must all be two-dimensional.")
    if X_ref.shape != X_in.shape or X_ref.shape != source_mask.shape:
        raise ValueError(
            "Reference matrix, model-input matrix, and source mask must have identical shapes; "
            f"got {X_ref.shape}, {X_in.shape}, and {source_mask.shape}."
        )
    if not np.isfinite(X_ref).all() or not np.isfinite(X_in).all():
        raise ValueError("Preprocessed reference and model-input matrices must be finite.")
    if not np.isin(source_mask, (0, 1, 2)).all():
        raise ValueError("The source missingness mask must contain only values 0, 1, and 2.")

    expected_names = sorted(str(name) for name in feat_type_dict["all_feats"])
    if len(expected_names) != X_in.shape[1]:
        raise ValueError(
            "Feature schema does not match the matrix width: "
            f"{len(expected_names)} feature names versus {X_in.shape[1]} columns."
        )
    if feature_specs is not None:
        spec_names = [str(spec.name) for spec in feature_specs]
        if spec_names != expected_names:
            raise ValueError(
                "Ordered feature specifications must follow the same sorted column order "
                "as the matrices and feature schema."
            )


def _make_validation_view(
    validation_base: np.ndarray,
    source_mask: np.ndarray,
    feat_type_dict: Dict[str, Any],
) -> ValidationView:
    """Apply the existing grouped validation amputation and hide type-3 values.

    The existing utility remains responsible for selecting validation cells and
    assigning type 3/type 4. This adapter hides selected observed values and
    recodes type 3 as type 2 only in the imputation mask, so they are updated by
    ``impute_multiple`` while type-4 cells remain excluded from scoring. Rows
    with no type-3 targets are skipped without redrawing the other rows.
    """
    output = BetaGausMixedDVAEUtils.generate_validation_amputation(
        preimputed_data_arr=validation_base,
        X_mask=source_mask,
        feat_type_dict=feat_type_dict,
        prop_miss_rows=1.0,
        prop_miss_col=0.1,
    )
    validation_mask = _numpy(output.validation_mask).astype(np.int8, copy=True)
    score_cells = validation_mask == 3
    if not score_cells.any():
        raise ValueError(
            "The validation fold produced no type-3 cells in its grouped-amputation draw; "
            "it cannot be scored."
        )

    # A row with no type-3 target contributes nothing to the score.
    rows_without_score_cells = ~score_cells.any(axis=1)
    # Discard each such row's validation-time type-4 selections without redrawing other rows.
    validation_mask[rows_without_score_cells] = np.asarray(
        source_mask, dtype=np.int8
    )[rows_without_score_cells]

    model_input = _numpy(output.validation_input).astype(np.float32, copy=True)
    imputation_mask = _numpy(
        output.val_fold_baseline_amputation_mask
    ).astype(np.int8, copy=True)
    # Zero is only a placeholder: these entries are explicitly marked for
    # iterative model imputation and never used as scoring targets.
    model_input[score_cells] = 0.0
    imputation_mask[score_cells] = 2
    # Keep type 3 for scoring; mark the same cells as type 2 in the separate
    # imputation mask so ``impute_multiple`` updates them.
    return ValidationView(model_input, validation_mask, imputation_mask)


def _calculate_score(
    model: BetaGausMixedDVAE,
    view: ValidationView,
    validation_ref: np.ndarray,
    feat_type_dict: Dict[str, Any],
    num_imputations: int,
    imputation_cycles: int,
    metric_weights: Dict[str, float],
) -> tuple[float, Dict[str, float]]:
    """Compute the existing weighted composite score using only type-3 cells."""
    if not 2 <= num_imputations <= 100:
        raise ValueError(
            f"Number of imputation draws must be between 2 and 100, got {num_imputations}."
        )
    if not 2 <= imputation_cycles <= 20:
        raise ValueError(
            f"Imputation cycles must be between 2 and 20, got {imputation_cycles}."
        )
    score_cells = torch.as_tensor(view.validation_mask == 3, device=model.device)
    if not bool(score_cells.any().item()):
        raise ValueError("Cannot calculate a validation score without type-3 cells.")

    model_input = torch.as_tensor(view.model_input, dtype=torch.float32, device=model.device)
    imputation_mask = torch.as_tensor(
        view.imputation_mask, dtype=torch.int8, device=model.device
    )
    ref = torch.as_tensor(validation_ref, dtype=torch.float32, device=model.device)

    # Match the existing evaluation protocol: one iterative single imputation
    # and N stochastic multiple imputations, all compared against type-3 truth.
    single_imputation = model.impute_single(
        model_input,
        n_cycles=10,
        loss_option="BOTH",
        X_mask=imputation_mask,
        feat_type_dict=feat_type_dict,
    )
    single_errors = single_imputation.imputed_x[score_cells] - ref[score_cells]

    # Each call produces one VAE-based imputation; ``imputation_cycles`` sets
    # its internal reconstruction/update steps, not sklearn imputer iterations.
    draws = [
        model.impute_multiple(
            model_input,
            max_iter=imputation_cycles,
            X_mask=imputation_mask,
            feat_type_dict=feat_type_dict,
        )
        for _ in range(num_imputations)
    ]
    multiple_imputations = torch.stack(draws, dim=0)
    mean_imputation = multiple_imputations.mean(dim=0)
    multiple_errors = mean_imputation[score_cells] - ref[score_cells]
    metrics = {
        "mae": float(torch.abs(single_errors).mean().item()),
        "rmse": float(torch.sqrt(torch.square(single_errors).mean()).item()),
        "multi_mae": float(torch.abs(multiple_errors).mean().item()),
        "multi_rmse": float(torch.sqrt(torch.square(multiple_errors).mean()).item()),
    }
    metrics.update(
        BetaGausMixedDVAEUtils.evaluate_coverage_quantile(
            multi_imputes=multiple_imputations,
            ref_data_arr=ref,
            X_mask=view.validation_mask,
        )
    )
    result = float(
        BetaGausMixedDVAETrainer._composite_metric_score(metrics, metric_weights)
    )
    if not math.isfinite(result):
        raise ValueError(f"Validation score must be finite, got {result!r}.")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("All type-3 validation metrics must be finite.")
    return result, metrics


def _new_model(input_dim: int, config: Dict[str, Any], device: torch.device) -> BetaGausMixedDVAE:
    """Instantiate a fresh fold-specific VAE, including its learnable GMM prior."""
    return BetaGausMixedDVAE(
        input_dim=input_dim,
        latent_dim=int(config["vae_cont_lat_dim"]),
        hidden_dim1=int(config["hidden_size_1"]),
        hidden_dim2=int(config["hidden_size_2"]),
        n_gmm_components=int(config["n_gmm_components"]),
        num_feat_loss_metric=str(config["num_feat_loss_metric"]),
    ).to(device)


def _candidate_cv_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """Train/evaluate one beta-C candidate on disjoint K-fold splits."""
    config = job["config"]
    X_ref = np.asarray(job["X_ref"], dtype=np.float32)
    X_in = np.asarray(job["X_in"], dtype=np.float32)
    source_mask = np.asarray(job["source_mask"], dtype=np.int8)
    feature_specs = job.get("feature_specs")
    feat_type_dict = config["features_type_dict"]
    _validate_aligned_data(X_ref, X_in, source_mask, feature_specs, feat_type_dict)

    n_folds = int(config["k_folds"])
    if n_folds < 2 or n_folds > len(X_in):
        raise ValueError(f"k_folds must be between 2 and the row count; got {n_folds} for {len(X_in)} rows.")
    splits = KFold(n_splits=n_folds, shuffle=True, random_state=int(config.get("cv_random_seed", 1729)))
    device = torch.device(job.get("device", "cpu"))
    beta = float(job["beta"])
    capacity_C = float(job["C"])
    num_imputations = int(config.get("num_imputations", config.get("N", config.get("m", 5))))
    imputation_cycles = int(config.get("recycles", 5))
    selection_metric = "weighted_composite_metric"
    metric_weights = BetaGausMixedDVAETrainer._resolve_metric_weights(config)
    budgets = config.get("halving_epoch_budgets", (30, 300, 30))
    if len(budgets) != 3:
        raise ValueError("halving_epoch_budgets must be [first_eval_epochs, max_epochs, eval_interval].")
    first_eval_epochs, max_epochs, eval_interval = (int(value) for value in budgets)
    if min(first_eval_epochs, max_epochs, eval_interval) <= 0 or first_eval_epochs > max_epochs:
        raise ValueError(f"Invalid halving_epoch_budgets: {budgets!r}.")
    min_epochs = int(config.get("min_epochs_before_convergence", first_eval_epochs))
    patience = int(config.get("convergence_patience", 5))
    tolerance = float(config.get("convergence_tolerance", 1e-4))
    if min_epochs < 1 or patience < 1:
        raise ValueError("min_epochs_before_convergence and convergence_patience must be positive.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("convergence_tolerance must be finite and nonnegative.")
    use_adam = bool(config.get("use_adam_optimizer", True))
    learning_rate = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    fold_scores: list[float] = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits.split(X_in)):
        # Identical fold masks and initial weights across candidates make the
        # candidate comparison less sensitive to random variation.
        seed = int(config.get("cv_random_seed", 1729)) + fold_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        train_data = X_in[train_indices]
        validation_base = X_in[val_indices]
        validation_ref = X_ref[val_indices]
        validation_source_mask = source_mask[val_indices]
        view = _make_validation_view(
            validation_base,
            validation_source_mask,
            feat_type_dict,
        )

        train_tensor = torch.as_tensor(train_data, dtype=torch.float32)
        model = _new_model(X_in.shape[1], config, device)
        optimizer = (
            torch.optim.Adam(model.parameters(), lr=learning_rate)
            if use_adam
            else torch.optim.SGD(model.parameters(), lr=learning_rate)
        )

        best_fold_score = math.inf
        best_fold_metrics: Dict[str, float] = {}
        completed_epochs = 0
        rounds_without_improvement = 0
        while completed_epochs < max_epochs:
            epochs_this_round = (
                first_eval_epochs if completed_epochs == 0 else eval_interval
            )
            epochs_this_round = min(epochs_this_round, max_epochs - completed_epochs)
            BetaGausMixedDVAETrainer.train_one_fold(
                model=model,
                optimizer=optimizer,
                use_adam=use_adam,
                lr=learning_rate,
                train_tensor=train_tensor,
                epochs=epochs_this_round,
                beta=beta,
                capacity_C=capacity_C,
                device=device,
                feat_type_dict=feat_type_dict,
                batch_size=batch_size,
                starting_epoch=completed_epochs,
            )
            completed_epochs += epochs_this_round
            current_score, current_metrics = _calculate_score(
                model=model,
                view=view,
                validation_ref=validation_ref,
                feat_type_dict=feat_type_dict,
                num_imputations=num_imputations,
                imputation_cycles=imputation_cycles,
                metric_weights=metric_weights,
            )
            if best_fold_score - current_score > tolerance:
                best_fold_score = current_score
                best_fold_metrics = current_metrics
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1

            if (
                completed_epochs >= min_epochs
                and rounds_without_improvement >= patience
            ):
                break

        fold_scores.append(best_fold_score)
        if config.get("results_path"):
            BetaGausMixedDVAETrainer.save_results(
                {
                    "selection_metric_name": selection_metric,
                    "weighted_composite_score": best_fold_score,
                    **best_fold_metrics,
                    "fold_idx": fold_idx,
                },
                epoch=completed_epochs,
                beta=beta,
                capacity_C=capacity_C,
                results_path=str(config["results_path"]),
                lock_path=str(Path(config["results_path"]).with_suffix(".lock")),
            )

    return {
        "beta": beta,
        "C": capacity_C,
        "score": float(np.mean(fold_scores)),
        "fold_scores": tuple(float(score) for score in fold_scores),
        "metric": selection_metric,
        "metric_weights": metric_weights,
        "device": str(device),
    }


def _run_candidates_parallel(
    candidates: list[tuple[float, float]],
    config: Dict[str, Any],
    X_ref: np.ndarray,
    X_in: np.ndarray,
    source_mask: np.ndarray,
    feature_specs: Sequence[OrderedFeature] | None,
) -> list[dict[str, Any]]:
    devices = _available_devices()
    jobs = [
        {
            "beta": beta,
            "C": capacity_C,
            "config": config,
            "X_ref": X_ref,
            "X_in": X_in,
            "source_mask": source_mask,
            "feature_specs": feature_specs,
            "device": devices[index % len(devices)],
        }
        for index, (beta, capacity_C) in enumerate(candidates)
    ]
    if len(jobs) == 1:
        return [_candidate_cv_worker(jobs[0])]

    workers = min(len(jobs), cpu_count()) if devices == ["cpu"] else min(len(jobs), len(devices))
    with Pool(processes=workers) as pool:
        return list(pool.imap_unordered(_candidate_cv_worker, jobs))


def iterative_halving_search(
    config: Dict[str, Any],
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    data_X_mask: np.ndarray,
    feature_specs: Sequence[OrderedFeature] | None = None,
) -> Dict[str, Any]:
    """Evaluate each current grid's four corners and halve toward its winner.

    ``data_full``, ``data_missing_nan``, and ``data_X_mask`` are respectively
    ``X_ref``, ``X_in``, and the source mask ``M``. The legacy parameter names
    are retained for callers of the public tuning wrapper.
    """
    X_ref = np.asarray(data_full, dtype=np.float32)
    X_in = np.asarray(data_missing_nan, dtype=np.float32)
    source_mask = np.asarray(data_X_mask, dtype=np.int8)
    feat_type_dict = config.get("features_type_dict")
    if feat_type_dict is None:
        raise KeyError("Missing required config key: `features_type_dict`.")
    if feature_specs is None:
        feature_specs = config.get("ordered_feature_specs")
    _validate_aligned_data(X_ref, X_in, source_mask, feature_specs, feat_type_dict)

    beta_lo, beta_hi = (float(value) for value in config["beta_range"])
    C_lo, C_hi = (float(value) for value in config["C_range"])
    alpha0 = float(config.get("granularity", 0.01))
    if not beta_lo < beta_hi or not C_lo < C_hi:
        raise ValueError("Both beta_range and C_range must have strictly increasing bounds.")
    if not 0.0 < alpha0 < 1.0:
        raise ValueError(f"granularity (relative width alpha_0) must be in (0, 1), got {alpha0}.")

    beta_span0 = beta_hi - beta_lo
    C_span0 = C_hi - C_lo
    best_global: Dict[str, Any] | None = None
    iteration = 0
    while max((beta_hi - beta_lo) / beta_span0, (C_hi - C_lo) / C_span0) > alpha0:
        iteration += 1
        beta_mid = beta_lo + (beta_hi - beta_lo) / 2.0
        C_mid = C_lo + (C_hi - C_lo) / 2.0
        candidates = [
            (beta_lo, C_lo),
            (beta_lo, C_hi),
            (beta_hi, C_lo),
            (beta_hi, C_hi),
        ]
        print(
            f"[Round {iteration}] evaluating beta in [{beta_lo}, {beta_hi}] "
            f"and C in [{C_lo}, {C_hi}]"
        )
        results = _run_candidates_parallel(
            candidates,
            config,
            X_ref,
            X_in,
            source_mask,
            feature_specs,
        )
        # Candidate workers may finish out of order; deterministic tie-breaking
        # keeps equal-score corner selection reproducible.
        best_round = min(
            results,
            key=lambda result: (result["score"], result["beta"], result["C"]),
        )
        if best_global is None or best_round["score"] < best_global["score"]:
            best_global = best_round

        beta_t, C_t = float(best_round["beta"]), float(best_round["C"])
        if beta_t == beta_lo:
            beta_hi = beta_mid
        elif beta_t == beta_hi:
            beta_lo = beta_mid
        else:
            raise RuntimeError(f"Round winner beta={beta_t} is not a current grid endpoint.")
        if C_t == C_lo:
            C_hi = C_mid
        elif C_t == C_hi:
            C_lo = C_mid
        else:
            raise RuntimeError(f"Round winner C={C_t} is not a current grid endpoint.")

    if best_global is None:
        raise RuntimeError("Grid-halving search did not evaluate any candidates.")
    print(
        f"[Search done] best beta={best_global['beta']}, C={best_global['C']}, "
        f"score={best_global['score']}"
    )
    return best_global


def train_and_save_best_model(
    beta_val: float,
    capacity_C: float,
    config: Dict[str, Any],
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    data_X_mask: np.ndarray,
    feature_specs: Sequence[OrderedFeature] | None = None,
) -> str:
    """Train the selected model on all model-input rows and save its checkpoint.

    This is a post-search CLI action, separate from the cross-validation
    algorithm. CV folds themselves always train on fold-exclusive rows.
    """
    X_ref = np.asarray(data_full, dtype=np.float32)
    X_in = np.asarray(data_missing_nan, dtype=np.float32)
    source_mask = np.asarray(data_X_mask, dtype=np.int8)
    feat_type_dict = config.get("features_type_dict")
    if feat_type_dict is None:
        raise KeyError("Missing required config key: `features_type_dict`.")
    if feature_specs is None:
        feature_specs = config.get("ordered_feature_specs")
    _validate_aligned_data(X_ref, X_in, source_mask, feature_specs, feat_type_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _new_model(X_in.shape[1], config, device)
    use_adam = bool(config.get("use_adam_optimizer", True))
    learning_rate = float(config["learning_rate"])
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=learning_rate)
        if use_adam
        else torch.optim.SGD(model.parameters(), lr=learning_rate)
    )
    budgets = config.get("halving_epoch_budgets", (30, 300, 30))
    final_epochs = int(budgets[1]) if len(budgets) == 3 else int(config.get("max_epochs", 100))
    BetaGausMixedDVAETrainer.train_one_fold(
        model=model,
        optimizer=optimizer,
        use_adam=use_adam,
        lr=learning_rate,
        train_tensor=torch.as_tensor(X_in, dtype=torch.float32),
        epochs=final_epochs,
        beta=float(beta_val),
        capacity_C=float(capacity_C),
        device=device,
        feat_type_dict=feat_type_dict,
        batch_size=int(config["batch_size"]),
    )

    outdir = Path(config.get("model_outdir", "./trained_models"))
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = outdir / f"beta{beta_val}_C{capacity_C}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "beta": float(beta_val),
            "C": float(capacity_C),
            "epochs": final_epochs,
            "config": config,
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def _fill_preexisting_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Complete source-missing entries for a finite reference matrix."""
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if series.isna().any():
            fill_value = float(series.mean()) if series.notna().any() else 0.0
            out[col] = series.fillna(fill_value)
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
    preprocessing_engine: Literal["auto", "pandas", "pyspark"] = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[OrderedFeature, ...], Dict[str, Any]]:
    """Build aligned reference/input arrays, source mask, feature specs, schema."""
    feat_dict = FeaturesTypeDict.create(str(feature_dict_json))
    raw_df = load_dataset(str(input_csv), feat_dict, engine=preprocessing_engine)
    use_spark = not isinstance(raw_df, pd.DataFrame)
    preprocessor = FeaturePreprocessor(
        feat_dict=feat_dict,
        missing_mechanism=missing_mechanism,
        missing_rate=missing_rate,
        input_df=raw_df,
        use_spark=use_spark,
        pre_imputation_method=pre_imputation_method,
        pre_imputation_max_iter=pre_imputation_max_iter,
        mice_num_imputations=mice_num_imputations,
        random_forest_n_estimators=random_forest_n_estimators,
    )

    (
        pre_df,
        real_cols,
        pos_cols,
        count_cols,
        ord_groups,
        bi_groups,
        cat_groups,
    ) = preprocessor._build_preprocessed_dataframe()
    amputed_df, source_mask = preprocessor._apply_pyampute(
        pre_df,
        ord_groups,
        cat_groups,
    )
    model_input_df = preprocessor._impute_amputed_values(amputed_df, source_mask)

    # The validation-amputation utility indexes columns in sorted feature-name
    # order. Reorder every aligned object and Phi to that same order.
    ordered_columns = sorted(str(column) for column in pre_df.columns)
    reference_df = _fill_preexisting_nans(pre_df).loc[:, ordered_columns]
    model_input_df = model_input_df.loc[:, ordered_columns]
    column_indices = [list(pre_df.columns).index(column) for column in ordered_columns]
    source_mask = source_mask[:, column_indices].astype(np.int8, copy=False)
    feature_specs = preprocessor._build_ordered_feature_ordered_feats(
        ordered_cols=ordered_columns,
        real_cols=real_cols,
        pos_cols=pos_cols,
        count_cols=count_cols,
        ord_groups=ord_groups,
        bi_groups=bi_groups,
        cat_groups=cat_groups,
    )

    model_feat_dict: Dict[str, Any] = dict(feat_dict)
    model_feat_dict["all_feats"] = set(ordered_columns)
    X_ref = reference_df.to_numpy(dtype=np.float32, copy=True)
    X_in = model_input_df.to_numpy(dtype=np.float32, copy=True)
    _validate_aligned_data(X_ref, X_in, source_mask, feature_specs, model_feat_dict)
    return X_ref, X_in, source_mask, feature_specs, model_feat_dict


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
        "k_folds",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"cv config missing required keys: {missing!r}")
    if "N" in config:
        config["num_imputations"] = int(config["N"])
    elif "num_imputations" in config:
        config["num_imputations"] = int(config["num_imputations"])
    elif "m" in config:
        # Backward-compatible mapping from the repository's existing config.
        config["num_imputations"] = int(config["m"])
    else:
        raise KeyError("cv config must define the number of imputation draws as N or legacy m.")
    config.setdefault("halving_metric", "weighted_composite_metric")
    config["halving_metric"] = str(config["halving_metric"]).lower()
    if config["halving_metric"] != "weighted_composite_metric":
        raise ValueError(
            "Beta-C candidate selection uses the weighted composite validation metric; "
            "set halving_metric to 'weighted_composite_metric'."
        )
    BetaGausMixedDVAETrainer._resolve_metric_weights(config)
    if not 2 <= int(config["num_imputations"]) <= 100:
        raise ValueError("N, the number of imputation draws, must be between 2 and 100.")
    if int(config["k_folds"]) < 2:
        raise ValueError("k_folds must be at least 2.")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2 beta/C grid-halving CV tuner")
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
    parser.add_argument(
        "--preprocessing_engine",
        choices=["auto", "pandas", "pyspark"],
        default="auto",
        help="Use pandas, PySpark, or RAM-scaled automatic selection",
    )
    parser.add_argument("--k_folds", type=int, default=None)
    step_dir = REPO_ROOT / "VAEQL_plus" / "step2_beta_C_tuning"
    parser.add_argument("--results_path", type=Path, default=step_dir / "beta_C_halving.csv")
    parser.add_argument("--model_outdir", type=Path, default=step_dir / "trained_models")
    parser.add_argument("--summary_json", type=Path, default=step_dir / "best_beta_C_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.model_outdir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    X_ref, X_in, source_mask, feature_specs, feat_type_dict = build_cv_matrices(
        input_csv=args.input_csv,
        feature_dict_json=args.feature_dict_json,
        missing_mechanism=args.missing_mechanism,
        missing_rate=float(args.missing_rate),
        pre_imputation_method=args.pre_imputation_method,
        pre_imputation_max_iter=int(args.pre_imputation_max_iter),
        mice_num_imputations=int(args.mice_num_imputations),
        random_forest_n_estimators=int(args.random_forest_n_estimators),
        preprocessing_engine=args.preprocessing_engine,
    )
    config = load_and_patch_cv_config(
        cv_config_path=args.cv_config,
        results_path=args.results_path,
        model_outdir=args.model_outdir,
        k_folds_override=args.k_folds,
    )
    config["features_type_dict"] = feat_type_dict
    config["ordered_feature_specs"] = feature_specs

    best_candidate = iterative_halving_search(
        config=config,
        data_full=X_ref,
        data_missing_nan=X_in,
        data_X_mask=source_mask,
        feature_specs=feature_specs,
    )
    checkpoint_path = train_and_save_best_model(
        beta_val=float(best_candidate["beta"]),
        capacity_C=float(best_candidate["C"]),
        config=config,
        data_full=X_ref,
        data_missing_nan=X_in,
        data_X_mask=source_mask,
        feature_specs=feature_specs,
    )
    summary = {
        "best_beta": float(best_candidate["beta"]),
        "best_C": float(best_candidate["C"]),
        "best_score": float(best_candidate["score"]),
        "halving_metric": str(best_candidate["metric"]),
        "halving_metric_weights": best_candidate["metric_weights"],
        "num_imputations": int(config["num_imputations"]),
        "checkpoint_path": checkpoint_path,
        "results_path": str(args.results_path),
        "mask_counts": {
            "observed_0": int((source_mask == 0).sum()),
            "source_missing_1": int((source_mask == 1).sum()),
            "previously_amputed_2": int((source_mask == 2).sum()),
        },
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
