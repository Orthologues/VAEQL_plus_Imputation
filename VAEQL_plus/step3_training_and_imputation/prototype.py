"""Train the selected GMM beta-VAE and generate an initial Step 3 imputation.

This prototype consumes the Step 1 preprocessing bundle and the selected
beta/C pair from Step 2. It deliberately stops before Q-learning; the trained
model, source mask, and feature-aware aggregate are the handoff.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np
import torch

from VAEQL_plus.beta_DVAE.torch_nn import BetaGausMixedDVAE
from VAEQL_plus.conf.config import (
    CV_BETA_C_TUNING_PARAMS_PATH,
    DIS_BETAVAE_TRAIN_VAL_PARAMS_PATH,
)
from VAEQL_plus.conf.feat_types import FeaturesTypeDict
from VAEQL_plus.step1_preprocessing.feat_preprocessor import OrderedFeature


class Step3Inputs(NamedTuple):
    """Step 1 matrices and metadata in the feature order expected by the model."""

    X0: np.ndarray
    source_mask: np.ndarray
    feature_specs: Tuple[OrderedFeature, ...]
    feature_dict: FeaturesTypeDict


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {path}.")
    return value


def load_step3_inputs(preprocessed_dir: Path) -> Step3Inputs:
    """Reuse Step 2's validated Step 1-bundle loader to preserve column alignment."""
    from VAEQL_plus.step2_beta_C_tuning.fine_tuner import build_cv_matrices

    _, X0, source_mask, feature_specs, feature_dict = build_cv_matrices(
        preprocessed_dir=preprocessed_dir,
    )
    return Step3Inputs(X0, source_mask, feature_specs, feature_dict)


def load_selected_beta_C(beta_summary_path: Path) -> tuple[float, float, Dict[str, Any]]:
    """Read the selected beta/C pair from Step 2's summary file."""
    summary = _read_json(beta_summary_path)
    try:
        beta = float(summary["best_beta"])
        capacity_C = float(summary["best_C"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Step 2 summary must contain numeric `best_beta` and `best_C` values."
        ) from exc
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"Selected beta must be finite and > 0, got {beta!r}.")
    if not math.isfinite(capacity_C) or capacity_C < 0:
        raise ValueError(f"Selected C must be finite and >= 0, got {capacity_C!r}.")
    return beta, capacity_C, summary


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is unavailable in this runtime.")
    return device


#CHECKPOINT
def train_final_model(
    X0: np.ndarray,
    source_mask: np.ndarray,
    feature_dict: FeaturesTypeDict,
    beta: float,
    capacity_C: float,
    train_config: Dict[str, Any],
    device: torch.device,
) -> tuple[BetaGausMixedDVAE, Dict[str, float]]:
    """Train a fresh GMM beta-VAE on the full Step 1 model-input matrix."""
    from VAEQL_plus.beta_DVAE.lightning_mod import BetaGausMixedDVAETrainer

    model_config = {
        "input_dim": int(X0.shape[1]),
        "latent_dim": int(train_config["vae_cont_lat_dim"]),
        "hidden_dim1": int(train_config["hidden_size_1"]),
        "hidden_dim2": int(train_config["hidden_size_2"]),
        "n_gmm_components": int(train_config["n_gmm_components"]),
        "num_feat_loss_metric": str(train_config["num_feat_loss_metric"]),
    }
    learning_rate = float(train_config["vae_alpha"])
    batch_size = int(train_config["vae_batch_size"])
    epochs = int(train_config["vae_max_epochs"])
    use_adam = bool(train_config["vae_if_Adam"])
    if min(model_config[k] for k in ("input_dim", "latent_dim", "hidden_dim1", "hidden_dim2", "n_gmm_components")) <= 0:
        raise ValueError("Model dimensions and GMM component count must be positive.")
    if learning_rate <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("Learning rate, batch size, and training epochs must be positive.")

    seed = int(train_config.get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Match the existing Step 2 trainer: X0 is the training target matrix,
    # including its Step 1 placeholder values; the source mask excludes
    # type-1/2 placeholders from reconstruction loss and controls imputation.
    train_tensor = torch.as_tensor(X0, dtype=torch.float32)
    train_mask = torch.as_tensor(source_mask, dtype=torch.int8)
    model = BetaGausMixedDVAE(
        input_dim=model_config["input_dim"],
        latent_dim=model_config["latent_dim"],
        hidden_dim1=model_config["hidden_dim1"],
        hidden_dim2=model_config["hidden_dim2"],
        n_gmm_components=model_config["n_gmm_components"],
        batch_size=batch_size,
        num_feat_loss_metric=model_config["num_feat_loss_metric"],
    ).to(device)
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=learning_rate)
        if use_adam
        else torch.optim.SGD(model.parameters(), lr=learning_rate)
    )
    result = BetaGausMixedDVAETrainer.train_one_fold(
        model=model,
        optimizer=optimizer,
        use_adam=use_adam,
        lr=learning_rate,
        train_tensor=train_tensor,
        epochs=epochs,
        beta=beta,
        capacity_C=capacity_C,
        device=device,
        feat_type_dict=feature_dict,
        train_mask=train_mask,
        batch_size=batch_size,
        starting_epoch=0,
    )
    losses = {
        "last_loss": float(result.last_loss),
        "last_recon": float(result.last_recon),
        "last_kl": float(result.last_kl),
        "last_kl_disc": float(result.last_kl_disc),
        "last_kl_cont": float(result.last_kl_cont),
        "epochs": epochs,
    }
    if not all(math.isfinite(value) for value in losses.values()):
        raise ValueError("Training returned a non-finite loss value.")
    return model, {**model_config, **losses}


def _feature_aware_aggregate(
    mean_predictions: np.ndarray,
    X0: np.ndarray,
    source_mask: np.ndarray,
    feature_specs: Tuple[OrderedFeature, ...],
) -> np.ndarray:
    """Aggregate draws and restore valid binary, ordinal, and categorical encodings."""
    mean_predictions = np.asarray(mean_predictions, dtype=np.float64)
    X0 = np.asarray(X0, dtype=np.float32)
    source_mask = np.asarray(source_mask)
    if X0.ndim != 2 or source_mask.shape != X0.shape:
        raise ValueError("X0 and source_mask must be aligned two-dimensional matrices.")
    if mean_predictions.shape != X0.shape:
        raise ValueError(
            "Mean predictions must match X0's shape; "
            f"got {mean_predictions.shape} and {X0.shape}."
        )
    if not np.isfinite(X0).all() or not np.isfinite(mean_predictions).all():
        raise ValueError("X0 and mean predictions must contain only finite values.")
    if not np.isin(source_mask, (0, 1, 2)).all():
        raise ValueError("source_mask must contain only mask values 0, 1, and 2.")
    if len(feature_specs) != X0.shape[1]:
        raise ValueError(
            "feature_specs must contain one entry per column; "
            f"got {len(feature_specs)} specs for {X0.shape[1]} columns."
        )
    names = [spec.name for spec in feature_specs]
    if len(set(names)) != len(names):
        raise ValueError("feature_specs contains duplicate column names.")

    aggregated = X0.copy()
    grouped_columns: Dict[tuple[str, str], list[int]] = {}
    scalar_types = {"real_val", "pos_real_val", "count", "binary"}
    grouped_types = {"ordinal", "categorical"}

    for column_index, spec in enumerate(feature_specs):
        if spec.feat_type in scalar_types:
            target_rows = source_mask[:, column_index] != 0
            values = mean_predictions[target_rows, column_index]
            if spec.feat_type == "binary":
                # Binary outputs are probabilities; threshold to restore {0, 1}.
                aggregated[target_rows, column_index] = (values >= 0.5).astype(np.float32)
            else:
                # All numeric families, including standardized log-counts, use [-5, 5].
                aggregated[target_rows, column_index] = np.clip(values, -5.0, 5.0).astype(np.float32)
        elif spec.feat_type in grouped_types:
            grouped_columns.setdefault((spec.feat_type, spec.base_feat), []).append(column_index)
        else:
            raise ValueError(
                f"Unsupported feature type {spec.feat_type!r} for column {spec.name!r}."
            )

    for (feature_type, base_feature), column_indices in grouped_columns.items():
        if feature_type == "ordinal":
            prefix = f"{base_feature}-ge_"
            try:
                column_indices.sort(
                    key=lambda index: int(feature_specs[index].name.removeprefix(prefix))
                    if feature_specs[index].name.startswith(prefix)
                    else -1
                )
            except ValueError as exc:
                raise ValueError(
                    f"Ordinal columns for {base_feature!r} must end in numeric '-ge_' thresholds."
                ) from exc
            threshold_numbers = [
                int(feature_specs[index].name.removeprefix(prefix))
                if feature_specs[index].name.startswith(prefix)
                else -1
                for index in column_indices
            ]
            if threshold_numbers != list(range(1, len(column_indices) + 1)):
                raise ValueError(
                    f"Ordinal feature {base_feature!r} has invalid thresholds: {threshold_numbers}."
                )
        else:
            column_indices.sort(key=lambda index: feature_specs[index].name)
            prefix = f"{base_feature}-is_"
            if any(not feature_specs[index].name.startswith(prefix) for index in column_indices):
                raise ValueError(
                    f"Categorical columns for {base_feature!r} must use the '-is_' naming convention."
                )

        group_mask = source_mask[:, column_indices]
        if not np.all(group_mask == group_mask[:, :1]):
            raise ValueError(
                f"Encoded columns for {base_feature!r} must share the same missingness mask."
            )
        target_rows = group_mask[:, 0] != 0
        if not target_rows.any():
            continue

        group_mean = np.clip(mean_predictions[:, column_indices], 0.0, 1.0)
        if feature_type == "categorical":
            totals = group_mean.sum(axis=1, keepdims=True)
            if np.any(totals[target_rows] <= 0.0):
                raise ValueError(f"Categorical feature {base_feature!r} has zero probability mass.")
            probabilities = group_mean / np.maximum(totals, 1e-12)
            selected_levels = np.argmax(probabilities, axis=1)
            encoded = np.eye(len(column_indices), dtype=np.float32)[selected_levels]
        else:
            # Convert cumulative P(Y >= level) into probabilities for each ordinal level.
            level_probabilities = np.empty(
                (X0.shape[0], len(column_indices) + 1), dtype=np.float64
            )
            level_probabilities[:, 0] = 1.0 - group_mean[:, 0]
            if len(column_indices) > 1:
                level_probabilities[:, 1:-1] = group_mean[:, :-1] - group_mean[:, 1:]
            level_probabilities[:, -1] = group_mean[:, -1]
            level_probabilities = np.clip(level_probabilities, 0.0, None)
            totals = level_probabilities.sum(axis=1, keepdims=True)
            if np.any(totals[target_rows] <= 0.0):
                raise ValueError(f"Ordinal feature {base_feature!r} has zero probability mass.")
            probabilities = level_probabilities / np.maximum(totals, 1e-12)
            selected_levels = np.argmax(probabilities, axis=1)
            thresholds = np.arange(1, len(column_indices) + 1)
            encoded = (selected_levels[:, None] >= thresholds[None, :]).astype(np.float32)

        aggregated[np.ix_(target_rows, column_indices)] = encoded[target_rows]

    return aggregated


def generate_feature_aware_aggregate(
    model: BetaGausMixedDVAE,
    X0: np.ndarray,
    source_mask: np.ndarray,
    feature_dict: FeaturesTypeDict,
    feature_specs: Tuple[OrderedFeature, ...],
    num_imputations: int,
    imputation_cycles: int,
) -> np.ndarray:
    """Generate stochastic draws online and return their feature-aware aggregate."""
    if not 5 <= int(num_imputations) <= 20:
        raise ValueError("`num_imputations` must be between 5 and 20.")
    if not 2 <= int(imputation_cycles) <= 20:
        raise ValueError("`imputation_cycles` must satisfy 2 <= value <= 20.")
    if X0.shape != source_mask.shape:
        raise ValueError("X0 and source mask must have identical shapes.")
    if not np.isin(source_mask, (0, 1, 2)).all():
        raise ValueError("source_mask must contain only mask values 0, 1, and 2.")

    imputation_mask = np.where(source_mask == 0, 0, 2).astype(np.int8)
    prediction_sum = np.zeros(X0.shape, dtype=np.float64)
    device = next(model.parameters()).device
    model_input = torch.as_tensor(X0, dtype=torch.float32, device=device)
    mask_tensor = torch.as_tensor(imputation_mask, dtype=torch.int8, device=device)
    for draw_index in range(int(num_imputations)):
        print(f"[Initial imputation] draw {draw_index + 1}/{num_imputations}")
        draw = model.impute_multiple(
            model_input,
            max_iter=int(imputation_cycles),
            X_mask=mask_tensor,
            feat_type_dict=feature_dict,
        )
        draw_np = draw.detach().cpu().numpy()
        if not np.isfinite(draw_np).all():
            raise ValueError(f"Imputation draw {draw_index + 1} contains non-finite values.")
        prediction_sum += draw_np

    mean_predictions = prediction_sum / int(num_imputations)
    aggregate = _feature_aware_aggregate(mean_predictions, X0, source_mask, feature_specs)
    return aggregate


def _json_feature_dict(feature_dict: FeaturesTypeDict) -> Dict[str, Any]:
    """Convert set-valued feature metadata to JSON-safe lists."""
    result: Dict[str, Any] = {}
    for key, value in feature_dict.items():
        if isinstance(value, set):
            result[key] = sorted(value)
        elif isinstance(value, dict):
            result[key] = {
                name: sorted(items) if isinstance(items, set) else items
                for name, items in value.items()
            }
        else:
            result[key] = value
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the selected GMM beta-VAE and create initial imputations."
    )
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--beta-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, default=DIS_BETAVAE_TRAIN_VAL_PARAMS_PATH)
    parser.add_argument("--cv-config", type=Path, default=CV_BETA_C_TUNING_PARAMS_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    model_path = output_dir / "final_model" / "model.pt"
    config_path = output_dir / "run_config" / "config.json"
    imputations_dir = output_dir / "imputations"
    aggregate_path = imputations_dir / "feature_aware_initial_imputation.npy"
    source_mask_path = imputations_dir / "source_missingness_mask.npy"
    outputs = (model_path, config_path, aggregate_path, source_mask_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite Step 3 artifacts: " + ", ".join(map(str, existing)))

    inputs = load_step3_inputs(args.preprocessed_dir)
    beta, capacity_C, beta_summary = load_selected_beta_C(args.beta_summary)
    train_config = _read_json(args.training_config)
    cv_config = _read_json(args.cv_config)
    device = _resolve_device(args.device)
    model, training_metrics = train_final_model(
        inputs.X0, inputs.source_mask, inputs.feature_dict, beta, capacity_C, train_config, device
    )

    for directory in (model_path.parent, config_path.parent, imputations_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "selected_beta": beta,
        "selected_C": capacity_C,
        "model_config": {key: training_metrics[key] for key in (
            "input_dim", "latent_dim", "hidden_dim1", "hidden_dim2",
            "n_gmm_components", "num_feat_loss_metric",
        )},
    }
    torch.save(checkpoint, model_path)

    aggregate = generate_feature_aware_aggregate(
        model=model,
        X0=inputs.X0,
        source_mask=inputs.source_mask,
        feature_dict=inputs.feature_dict,
        feature_specs=inputs.feature_specs,
        num_imputations=int(cv_config["num_imputations"]),
        imputation_cycles=int(cv_config["recycles"]),
    )
    np.save(aggregate_path, aggregate, allow_pickle=False)
    np.save(source_mask_path, inputs.source_mask, allow_pickle=False)

    run_config = {
        "selected_beta": beta,
        "selected_C": capacity_C,
        "selected_cv_score": beta_summary.get("best_score"),
        "training": train_config,
        "training_metrics_last_batch": training_metrics,
        "imputation": {
            "num_imputations": int(cv_config["num_imputations"]),
            "cycles_per_draw": int(cv_config["recycles"]),
            "source_mask_semantics": "0 observed, 1 source-missing, 2 previously amputed",
            "aggregation": {
                "real_val": "mean across draws, clipped to [-5, 5]",
                "pos_real_val": "mean across draws, clipped to [-5, 5]",
                "count": "mean across draws in log1p-z-score space, clipped to [-5, 5]",
                "binary": "mean across draws, thresholded at 0.5",
                "ordinal": "MAP level from mean cumulative probabilities, unary encoded",
                "categorical": "argmax of mean category probabilities, one-hot encoded",
                "observed_cells": "preserved from the Step 1 input matrix",
            },
        },
        "feature_columns": [spec.name for spec in inputs.feature_specs],
        "ordered_feature_specs": [spec._asdict() for spec in inputs.feature_specs],
        "feature_dict": _json_feature_dict(inputs.feature_dict),
        "source_mask_counts": {
            str(mask_type): int(np.count_nonzero(inputs.source_mask == mask_type))
            for mask_type in (0, 1, 2)
        },
        "device": str(device),
    }
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2, allow_nan=False)

    # Q-learning handoff: model, source mask, and encoded aggregate are ready;
    # individual draws remain internal and this prototype creates no Q-table.
    print(json.dumps({
        "model_checkpoint": str(model_path),
        "feature_aware_initial_imputation": str(aggregate_path),
        "source_missingness_mask": str(source_mask_path),
        "mean_target_cells_per_row": float(
            np.mean(np.count_nonzero(inputs.source_mask != 0, axis=1))
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
