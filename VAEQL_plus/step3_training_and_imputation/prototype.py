"""Train the selected GMM beta-VAE and prepare a Q-learning handoff.

This prototype consumes the Step 1 preprocessing bundle and the selected
beta/C pair from Step 2. The returned handoff creates detached, feature-aware
multiple-imputation aggregates per Q-learning batch; the Q-learning algorithm
is implemented later.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, NamedTuple, Tuple

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


def train_vae_and_prepare_q_handoff(
    X0: np.ndarray,
    source_mask: np.ndarray,
    feature_dict: FeaturesTypeDict,
    feature_specs: Tuple[OrderedFeature, ...],
    beta: float,
    capacity_C: float,
    train_config: Dict[str, Any],
    device: torch.device,
    num_imputations: int,
    imputation_cycles: int,
) -> Step3Handoff:
    """Train the VAE and return a batch-lazy Q-learning handoff.

    Call ``handoff.impute_batch(row_indices)`` inside each Q-learning update.
    Each call returns a detached aggregate of fresh stochastic imputations; the
    Q-learning update consumes its values without backpropagating through the VAE.
    """
    model, training_metrics = train_vae_on_full_dataset(
        X0=X0,
        source_mask=source_mask,
        feature_dict=feature_dict,
        beta=beta,
        capacity_C=capacity_C,
        train_config=train_config,
        device=device,
    )
    return Step3Handoff(
        model=model,
        X0=torch.as_tensor(X0, dtype=torch.float32, device=device),
        source_mask=torch.as_tensor(source_mask, dtype=torch.int8, device=device),
        feature_specs=feature_specs,
        feature_dict=feature_dict,
        training_metrics=training_metrics,
        num_imputations=num_imputations,
        imputation_cycles=imputation_cycles,
    )


def train_vae_on_full_dataset(
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


class Step3Handoff(NamedTuple):
    """Trained VAE and inputs for batch-wise Q-learning imputation."""

    model: BetaGausMixedDVAE
    X0: torch.Tensor
    source_mask: torch.Tensor
    feature_specs: Tuple[OrderedFeature, ...]
    feature_dict: FeaturesTypeDict
    training_metrics: Dict[str, float]
    num_imputations: int
    imputation_cycles: int

    def impute_batch(self, row_indices: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Create a fresh detached imputation aggregate for selected rows."""
        rows = torch.as_tensor(row_indices, dtype=torch.long, device=self.X0.device).reshape(-1)
        return generate_feature_aware_MI_aggregate(
            model=self.model,
            X0=self.X0.index_select(0, rows),
            source_mask=self.source_mask.index_select(0, rows),
            feature_dict=self.feature_dict,
            feature_specs=self.feature_specs,
            num_imputations=self.num_imputations,
            imputation_cycles=self.imputation_cycles,
        )


def _feature_aware_MI_aggregate(
    imputation_draws: Iterable[torch.Tensor],
    X0: torch.Tensor,
    source_mask: torch.Tensor,
    feature_specs: Tuple[OrderedFeature, ...],
) -> torch.Tensor:
    """Average soft imputation draws feature-wise.

    Each draw's binary probabilities, categorical simplex vectors, and monotone
    ordinal cumulative probabilities remain valid under arithmetic averaging.
    Only numeric features are clipped; observed cells are copied from Step 1.
    """
    X0 = torch.as_tensor(X0, dtype=torch.float32)
    source_mask = torch.as_tensor(source_mask, device=X0.device)
    if X0.ndim != 2 or source_mask.shape != X0.shape:
        raise ValueError("X0 and source_mask must have the same two-dimensional shape.")
    if len(feature_specs) != X0.shape[1]:
        raise ValueError("feature_specs must have one entry per matrix column.")

    numeric_columns = torch.tensor(
        [spec.feat_type in {"real_val", "pos_real_val", "count"} for spec in feature_specs],
        dtype=torch.bool,
        device=X0.device,
    )
    draw_sum = None
    draw_count = 0
    for draw in imputation_draws:
        draw = torch.as_tensor(draw, dtype=X0.dtype, device=X0.device)
        if draw.shape != X0.shape:
            raise ValueError("Every imputation draw must have the same shape as X0.")
        draw_sum = draw if draw_sum is None else draw_sum + draw
        draw_count += 1

    if draw_sum is None:
        raise ValueError("At least one imputation draw is required.")

    mean_draw = draw_sum / draw_count
    mean_draw = torch.where(numeric_columns, mean_draw.clamp(-5.0, 5.0), mean_draw)
    return torch.where(source_mask == 0, X0, mean_draw)


def generate_feature_aware_MI_aggregate(
    model: BetaGausMixedDVAE,
    X0: np.ndarray | torch.Tensor,
    source_mask: np.ndarray | torch.Tensor,
    feature_dict: FeaturesTypeDict,
    feature_specs: Tuple[OrderedFeature, ...],
    num_imputations: int,
    imputation_cycles: int,
) -> torch.Tensor:
    """Generate soft draws and return a detached feature-aware aggregate."""
    num_imputations = int(num_imputations)
    imputation_cycles = int(imputation_cycles)
    device = next(model.parameters()).device
    model_input = torch.as_tensor(X0, dtype=torch.float32, device=device)
    source_mask = torch.as_tensor(source_mask, dtype=torch.int8, device=device)
    # Type-0 cells are observed; type-1/2 cells are missing in Step 3.
    obs_mask = source_mask == 0

    def generate_draws() -> Iterable[torch.Tensor]:
        for _ in range(num_imputations):
            # Stream detached draws into the aggregate instead of storing a list.
            yield model.impute_multiple(
                model_input,
                max_iter=imputation_cycles,
                obs_mask=obs_mask,
                feat_type_dict=feature_dict,
                differentiable=False,
            )

    with torch.no_grad():
        return _feature_aware_MI_aggregate(
            generate_draws(), model_input, source_mask, feature_specs
        )


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


#CHECKPOINT
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the selected GMM beta-VAE and prepare a Q-learning handoff."
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
    source_mask_path = imputations_dir / "source_missingness_mask.npy"
    outputs = (model_path, config_path, source_mask_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite Step 3 artifacts: " + ", ".join(map(str, existing)))

    inputs = load_step3_inputs(args.preprocessed_dir)
    beta, capacity_C, beta_summary = load_selected_beta_C(args.beta_summary)
    train_config = _read_json(args.training_config)
    cv_config = _read_json(args.cv_config)
    device = _resolve_device(args.device)
    handoff = train_vae_and_prepare_q_handoff(
        X0=inputs.X0,
        source_mask=inputs.source_mask,
        feature_dict=inputs.feature_dict,
        feature_specs=inputs.feature_specs,
        beta=beta,
        capacity_C=capacity_C,
        train_config=train_config,
        device=device,
        num_imputations=int(cv_config["num_imputations"]),
        imputation_cycles=int(cv_config["recycles"]),
    )
    model = handoff.model
    training_metrics = handoff.training_metrics

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

    np.save(source_mask_path, handoff.source_mask.detach().cpu().numpy(), allow_pickle=False)

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
                "binary": "mean sigmoid probability across draws; kept soft",
                "ordinal": "mean monotone cumulative probabilities across draws; kept soft",
                "categorical": "mean Gumbel-Softmax probability vector across draws; kept soft",
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

    # The standalone CLI writes model and metadata snapshots. For Q-learning,
    # call `train_vae_and_prepare_q_handoff` and impute each batch in-process.
    print(json.dumps({
        "model_checkpoint": str(model_path),
        "source_missingness_mask": str(source_mask_path),
        "q_learning_handoff": "Step3Handoff.impute_batch(row_indices)",
        "mean_target_cells_per_row": float(
            np.mean(np.count_nonzero(inputs.source_mask != 0, axis=1))
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
