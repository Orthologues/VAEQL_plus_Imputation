#!/usr/bin/env python3
"""Fold-isolated grid-halving search for the disentangled beta-VAE."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Any, Dict, NamedTuple, Sequence, Tuple

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
from VAEQL_plus.conf.config import (
    CV_BETA_C_TUNING_PARAMS,
    DEFAULT_CONVERGENCE_TOLERANCE,
    DisentangledBetaVaeTuningConfig,
    _validate_convergence_tolerance,
    _validate_granularity,
    _validate_k_folds,
    _validate_search_ranges,
)
from VAEQL_plus.conf.feat_types import FeaturesTypeDict
from VAEQL_plus.step1_preprocessing.feat_preprocessor import OrderedFeature


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
    """Use visible GPUs when available; otherwise run CPU workers."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count:
            return [f"cuda:{i}" for i in range(device_count)]
    return ["cpu"]


def _validate_aligned_data(
    X_ref: np.ndarray,
    X_in: np.ndarray,
    source_mask: np.ndarray,
    feature_specs: Sequence[OrderedFeature],
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
    if feature_specs is None:
        raise ValueError("`ordered_feature_specs` is required to validate matrix columns.")
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
    score_cells = torch.as_tensor(view.validation_mask == 3, device=model.device)
    if not bool(score_cells.any().item()):
        raise ValueError("Cannot calculate a validation score without type-3 cells.")

    model_input = torch.as_tensor(view.model_input, dtype=torch.float32, device=model.device)
    imputation_mask = torch.as_tensor(
        view.imputation_mask, dtype=torch.int8, device=model.device
    )
    ref = torch.as_tensor(validation_ref, dtype=torch.float32, device=model.device)

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
    feature_specs = job["feature_specs"]
    feat_type_dict = config["features_type_dict"]
    _validate_aligned_data(X_ref, X_in, source_mask, feature_specs, feat_type_dict)

    n_folds = int(config["k_folds"])
    splits = KFold(n_splits=n_folds, shuffle=True, random_state=int(config.get("cv_random_seed", 42)))
    device = torch.device(job.get("device", "cpu"))
    beta = float(job["beta"])
    capacity_C = float(job["C"])
    num_imputations = int(config["num_imputations"])
    imputation_cycles = int(config.get("recycles", 10))
    if not 5 <= num_imputations <= 20:
        raise ValueError(
            f"Number of imputation draws must be between 5 and 20, got {num_imputations}."
        )
    if not 5 <= imputation_cycles <= 20:
        raise ValueError(
            f"Imputation cycles must be between 5 and 20, got {imputation_cycles}."
        )
    selection_metric = "weighted_composite_metric"
    metric_weights = BetaGausMixedDVAETrainer._resolve_metric_weights(config)
    budgets = config["halving_epoch_budgets"]
    first_eval_epochs, max_epochs, eval_interval = (int(value) for value in budgets)
    if min(first_eval_epochs, max_epochs, eval_interval) <= 0 or first_eval_epochs > max_epochs:
        raise ValueError(f"Invalid halving_epoch_budgets: {budgets!r}.")
    patience = int(config.get("convergence_patience", 5))
    tolerance = _validate_convergence_tolerance(
        config.get("convergence_tolerance", DEFAULT_CONVERGENCE_TOLERANCE)
    )
    if patience < 3:
        raise ValueError("`convergence_patience` must be at least 3.")
    use_adam = bool(config.get("use_adam_optimizer", True))
    learning_rate = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    fold_scores: list[float] = []
    fold_type3_counts: list[int] = []
    fold_type4_counts: list[int] = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits.split(X_in)):
        # Identical fold masks and initial weights across candidates make the
        # candidate comparison less sensitive to random variation.
        seed = int(config.get("cv_random_seed", 42)) + fold_idx
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # Train on the other K-1 folds; reserve this held-out fold for validation scoring.
        train_data = X_in[train_indices]
        validation_base = X_in[val_indices]
        validation_ref = X_ref[val_indices]
        validation_source_mask = source_mask[val_indices]
        view = _make_validation_view(
            validation_base,
            validation_source_mask,
            feat_type_dict,
        )
        fold_type3_counts.append(int(np.count_nonzero(view.validation_mask == 3)))
        fold_type4_counts.append(int(np.count_nonzero(view.validation_mask == 4)))

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

            if rounds_without_improvement >= patience:
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
        "mean_type3_cells_per_fold": float(np.mean(fold_type3_counts)),
        "mean_type4_cells_per_fold": float(np.mean(fold_type4_counts)),
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
    feature_specs: Sequence[OrderedFeature],
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

    use_cuda = devices != ["cpu"]
    cpu_limit = max(1, int(os.environ.get("VAEQL_MAX_CPU_WORKERS", cpu_count())))
    workers = min(len(jobs), cpu_limit)
    if use_cuda:
        workers = min(workers, len(devices))
    # CUDA subprocesses must use a clean interpreter rather than forked CUDA state.
    # `get_context("spawn")` explicitly starts each worker in a fresh Python interpreter
    context = get_context("spawn") if use_cuda else get_context()
    with context.Pool(processes=workers) as pool:
        return list(pool.imap_unordered(_candidate_cv_worker, jobs))


def iterative_halving_search(
    config: Dict[str, Any],
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    data_X_mask: np.ndarray,
    feature_specs: Sequence[OrderedFeature],
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
    _validate_aligned_data(X_ref, X_in, source_mask, feature_specs, feat_type_dict)
    if len(X_in) < 100:
        raise ValueError(
            f"Input data must contain at least 100 rows for cross-validation; got {len(X_in)}."
        )

    beta_lo, beta_hi = config["beta_range"]
    C_lo, C_hi = config["C_range"]
    alpha0 = float(config["granularity"])

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

        beta_t, C_t = best_round["beta"], best_round["C"]
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


def build_cv_matrices(
    preprocessed_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[OrderedFeature, ...], Dict[str, Any]]:
    """Load Step 1's output, i.e., X0, Phi, M, and feature schema for fold evaluation."""
    preprocessed_dir = Path(preprocessed_dir)
    data_df = pd.read_csv(preprocessed_dir / "preprocessed.csv")
    feature_dict = FeaturesTypeDict.create(str(preprocessed_dir / "feature_dict.json"))
    source_mask = np.load(preprocessed_dir / "missingness_mask.npy", allow_pickle=False)
    with open(preprocessed_dir / "ordered_feature_specs.json", "r", encoding="utf-8") as file:
        spec_records = json.load(file)

    columns = [str(column) for column in data_df.columns]
    if len(columns) != len(set(columns)):
        raise ValueError("Step 1 preprocessed data contains duplicate feature columns.")
    if source_mask.ndim != 2 or source_mask.shape != data_df.shape:
        raise ValueError(
            "Step 1 missingness mask must match the preprocessed matrix shape; "
            f"got {source_mask.shape} and {data_df.shape}."
        )
    if not np.isin(source_mask, (0, 1, 2)).all():
        raise ValueError("Step 1 missingness mask must contain only values 0, 1, and 2.")
    ordered_columns = sorted(columns)
    column_indices = [columns.index(column) for column in ordered_columns]
    if not isinstance(spec_records, list):
        raise TypeError("Step 1 ordered feature specifications must be a JSON list.")
    specs_by_name: Dict[str, OrderedFeature] = {}
    required_spec_fields = {"name", "feat_type", "base_feat"}
    for record in spec_records:
        if not isinstance(record, dict) or not required_spec_fields.issubset(record):
            raise ValueError(
                "Each ordered feature specification must contain `name`, `feat_type`, and `base_feat`."
            )
        name = str(record["name"])
        specs_by_name[name] = OrderedFeature(
            name=name,
            feat_type=str(record["feat_type"]),
            base_feat=str(record["base_feat"]),
        )
    if len(specs_by_name) != len(spec_records) or set(specs_by_name) != set(ordered_columns):
        raise ValueError(
            "Step 1 feature specifications must contain exactly one entry per preprocessed column."
        )

    # Keep the sorted order required by the grouped-amputation utility, while
    # preserving the aligned X0/M/Phi values emitted by Step 1.
    X0 = data_df.loc[:, ordered_columns].to_numpy(dtype=np.float32, copy=True)
    source_mask = source_mask[:, column_indices].astype(np.int8, copy=False)
    feature_specs = tuple(specs_by_name[column] for column in ordered_columns)
    model_feat_dict: Dict[str, Any] = dict(feature_dict)
    model_feat_dict["all_feats"] = set(ordered_columns)
    # Type-0 cells in X0 retain their observed values and serve as validation
    # references; the source mask excludes placeholder-filled type-1/2 cells.
    _validate_aligned_data(X0, X0, source_mask, feature_specs, model_feat_dict)
    return X0, X0, source_mask, feature_specs, model_feat_dict


def load_and_patch_cv_config(
    cv_config_path: Path,
    results_path: Path,
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
    budgets = config.get(
        "halving_epoch_budgets",
        CV_BETA_C_TUNING_PARAMS["halving_epoch_budgets"],
    )
    if not isinstance(budgets, (list, tuple)) or len(budgets) != 3:
        raise ValueError(
            "halving_epoch_budgets must contain [first_eval_epochs, max_epochs, eval_interval]."
        )
    config["halving_epoch_budgets"] = tuple(budgets)
    config["beta_range"], config["C_range"] = _validate_search_ranges(
        config["beta_range"], config["C_range"]
    )
    config["granularity"] = _validate_granularity(
        config.get("granularity", CV_BETA_C_TUNING_PARAMS["granularity"])
    )
    imputation_count_key = next(
        (key for key in ("num_imputations", "N", "m") if key in config),
        None
    )
    if imputation_count_key is None:
        raise KeyError(
            "cv config must define imputation draws using `num_imputations`, `N`, or `m`."
        )
    config["num_imputations"] = int(config[imputation_count_key])
    config.setdefault("halving_metric", "weighted_composite_metric")
    config["halving_metric"] = str(config["halving_metric"]).lower()
    if config["halving_metric"] != "weighted_composite_metric":
        raise ValueError(
            "Beta-C candidate selection uses the weighted composite validation metric; "
            "set halving_metric to 'weighted_composite_metric'."
        )
    BetaGausMixedDVAETrainer._resolve_metric_weights(config)
    config["recycles"] = int(config["recycles"])
    config["convergence_tolerance"] = _validate_convergence_tolerance(
        config.get("convergence_tolerance", DEFAULT_CONVERGENCE_TOLERANCE)
    )
    if not 5 <= int(config["num_imputations"]) <= 20:
        raise ValueError("N, the number of imputation draws, must be between 5 and 20.")
    if not 5 <= config["recycles"] <= 20:
        raise ValueError("recycles, the number of imputation cycles, must be between 5 and 20.")
    config["k_folds"] = _validate_k_folds(config["k_folds"])
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2 beta/C grid-halving CV tuner")
    parser.add_argument(
        "--preprocessed_dir",
        type=Path,
        required=True,
        help="Step 1 output bundle containing X0, ordered feature specs, mask, and schema",
    )
    parser.add_argument(
        "--cv_config",
        type=Path,
        default=DisentangledBetaVaeTuningConfig.default_cv_beta_C_tuning_config_path(),
    )
    step_dir = REPO_ROOT / "VAEQL_plus" / "step2_beta_C_tuning"
    parser.add_argument("--results_path", type=Path, default=step_dir / "beta_C_halving.csv")
    parser.add_argument("--summary_json", type=Path, default=step_dir / "best_beta_C_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    X_ref, X_in, source_mask, feature_specs, feat_type_dict = build_cv_matrices(
        preprocessed_dir=args.preprocessed_dir,
    )
    config = load_and_patch_cv_config(
        cv_config_path=args.cv_config,
        results_path=args.results_path,
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
    ## MARK
    summary = {
        "best_beta": float(best_candidate["beta"]),
        "best_C": float(best_candidate["C"]),
        "best_score": float(best_candidate["score"]),
        "halving_metric": str(best_candidate["metric"]),
        "halving_metric_weights": best_candidate["metric_weights"],
        "num_imputations": int(config["num_imputations"]),
        "results_path": str(args.results_path),
        "validation_amputation_counts": {
            "mean_type3_cells_per_fold": best_candidate["mean_type3_cells_per_fold"],
            "mean_type4_cells_per_fold": best_candidate["mean_type4_cells_per_fold"],
        },
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
