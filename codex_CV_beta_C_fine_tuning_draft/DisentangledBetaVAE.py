"""
Disentangled Beta-VAE PyTorch implementation (torch port)

Author: Jiawei Zhao (jiz@imada.sdu.dk)
Date: 2025-11-27
"""

import os
import time
import json
import argparse
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from disentangledBetaVaeUtil import (
    get_scaled_data,
    evaluate_coverage,
    evaluate_coverage_quantile,
    DataMissingMaker
)


# ---- Documentations of crucial import  

#

# ---- imports assumed from your PyTorch util module ----
# from disentangledBetaVaeUtil import (
#     get_scaled_data,
#     evaluate_coverage,
#     evaluate_coverage_quantile,
#     DataMissingMaker,
# )
# ^ You’ll need to actually implement these in torch/numpy as discussed.
#   Below I’ll assume they return the exact same things your current
#   TensorFlow version returns.

#########################################################
# 1. Model wrapper / loss
#########################################################

class DisentangledBetaVaeTorchModule(nn.Module):
    """
    Minimal placeholder.
    You must replace encoder/decoder architectures with yours.

    forward(x) should return:
        recon_x: (B, D) reconstruction logits or values
        mu:      (B, z_dim)
        logvar:  (B, z_dim)
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim1: int, hidden_dim2: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim2, latent_dim)

        # decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            # assume we’re doing bounded [0,1] inputs, so decoder outputs logits
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return {
            "recon_logits": recon_logits,
            "mu": mu,
            "logvar": logvar
        }

    # ---------- these two must be implemented for MI ----------
    def impute_single(self, X_nan_np, X_complete_np, n_recycles: int, loss: str, scaler, return_losses: bool):
        """
        Single-imputation pass driven by the current model.
        Missing entries are iteratively replaced with decoder predictions; MAE is
        tracked on the masked cells (inverse-transformed when a scaler is given).
        """
        device = next(self.parameters()).device
        missing_mask = np.isnan(X_nan_np)
        has_missing = bool(missing_mask.any())

        imputed = np.copy(X_nan_np)
        # simple start: column means from the complete data, fallback to zeros
        if has_missing:
            col_means = np.nanmean(X_complete_np, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            imputed[missing_mask] = col_means[np.where(missing_mask)[1]]

        losses_hist: List[float] = []
        was_training = self.training
        self.eval()
        try:
            steps = max(1, n_recycles)
            for _ in range(steps):
                x_tensor = torch.tensor(imputed, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = self.forward(x_tensor)
                    recon = torch.sigmoid(out["recon_logits"]).cpu().numpy()
                # keep observed entries fixed, refresh missing with reconstructions
                if has_missing:
                    imputed[missing_mask] = recon[missing_mask]

                if return_losses:
                    if not has_missing:
                        losses_hist.append(0.0)
                    elif scaler is not None:
                        pred_orig = scaler.inverse_transform(imputed)
                        truth_orig = scaler.inverse_transform(X_complete_np)
                        mae = float(np.mean(np.abs(pred_orig[missing_mask] - truth_orig[missing_mask])))
                        losses_hist.append(mae)
                    else:
                        mae = float(np.mean(np.abs(imputed[missing_mask] - X_complete_np[missing_mask])))
                        losses_hist.append(mae)
        finally:
            self.train(was_training)

        return imputed, None, losses_hist

    def impute_multiple(self, X_nan_np, max_iter: int, method="Metropolis-within-Gibbs"):
        """
        Produce one stochastic imputation by repeatedly sampling through the VAE
        and replacing missing entries with the decoder outputs.
        """
        device = next(self.parameters()).device
        missing_mask = np.isnan(X_nan_np)
        imputed = np.copy(X_nan_np)

        if missing_mask.any():
            # initialize with per-column means from observed values
            col_means = np.nanmean(X_nan_np, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            imputed[missing_mask] = col_means[np.where(missing_mask)[1]]

        was_training = self.training
        self.eval()
        try:
            steps = max(1, max_iter)
            for _ in range(steps):
                x_tensor = torch.tensor(imputed, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = self.forward(x_tensor)
                    recon = torch.sigmoid(out["recon_logits"]).cpu().numpy()
                if missing_mask.any():
                    imputed[missing_mask] = recon[missing_mask]
        finally:
            self.train(was_training)

        return imputed, None


def kl_divergence(mu, logvar):
    # KL( q(z|x) || N(0,1) ) per-sample
    # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def beta_capacity_loss(
    recon_logits: torch.Tensor,
    x_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    C: float,
    observed_mask: torch.Tensor | None = None,
    eps: float = 1e-8
):
    """
    β * | KL - C | + BCE(recon, x_true on observed coords only)
    recon_logits: raw decoder logits (no sigmoid)
    x_true: in [0,1]
    observed_mask: 1 where feature observed, 0 where missing (optional)
    """

    # reconstruction loss (binary cross-entropy with logits)
    bce_per_feature = nn.functional.binary_cross_entropy_with_logits(
        recon_logits, x_true, reduction='none'
    )  # (B, D)

    if observed_mask is not None:
        bce_per_feature = bce_per_feature * observed_mask

    # mean over observed dims per batch
    recon_loss = bce_per_feature.sum() / (bce_per_feature.numel() if observed_mask is None
                                          else (observed_mask.sum() + eps))

    # KL
    kl_per_sample = kl_divergence(mu, logvar)  # (B,)
    kl_mean = kl_per_sample.mean()

    kl_obj = beta * torch.abs(kl_mean - C)

    total = recon_loss + kl_obj
    return total, recon_loss.detach(), kl_mean.detach()


#########################################################
# 2. Helpers from your TF script, Torch version
#########################################################

def create_lock(path='lock.txt'):
    with open(path, 'w') as f:
        f.write('temp lock\n')

def remove_lock(path='lock.txt'):
    if os.path.exists(path):
        os.remove(path)

def save_results(
    results: Dict,
    epoch: int,
    beta: float,
    Cval: float,
    results_path='beta_analysis.csv',
    lock_path='lock.txt'
):
    # block if someone else is writing
    while os.path.exists(lock_path):
        print('sleeping due to file lock')
        time.sleep(2)

    create_lock(lock_path)

    # prepare output df
    if not os.path.exists(results_path):
        with open(results_path, 'w') as fh:
            fh.write('beta,C,epoch,mae,multi_mae,average_variance,'
                     'prop_80,prop_90,prop_95,prop_99,'
                     'prop_80q,prop_90q,prop_95q,prop_99q,k\n')

    df_prev = pd.read_csv(results_path)

    row = {
        "beta": beta,
        "C": Cval,
        "epoch": epoch,
        "k": results.get("k", None),
        "mae": results.get("mae", None),
        "multi_mae": results.get("multi_mae", None),
        "average_variance": results.get("average_variance", None),
        "prop_80": results.get("prop_80", None),
        "prop_90": results.get("prop_90", None),
        "prop_95": results.get("prop_95", None),
        "prop_99": results.get("prop_99", None),
        "prop_80q": results.get("prop_80q", None),
        "prop_90q": results.get("prop_90q", None),
        "prop_95q": results.get("prop_95q", None),
        "prop_99q": results.get("prop_99q", None),
    }

    df_prev = pd.concat([df_prev, pd.DataFrame([row])], ignore_index=True)
    df_prev.to_csv(results_path, index=False)

    remove_lock(lock_path)


def evaluate_variance_torch(model, missing_w_nans_np, na_ind):
    """
    Rough analog of your evaluate_variance().
    We'll forward-sample latent noise multiple times and look at variance
    in recon logits on missing cells.
    """

    model.eval()
    with torch.no_grad():
        x_nan = torch.tensor(np.nan_to_num(missing_w_nans_np).astype(np.float32)).to(next(model.parameters()).device)
        out = model(x_nan)
        recon_logits = out["recon_logits"].cpu().numpy()  # (N,D)
        # treat logit variance proxy as squared diff from sigmoid mean
        # You can design something better. For now:
        preds = 1 / (1 + np.exp(-recon_logits))
        # take variance across features for missing positions only
        var_missing = np.var(preds[na_ind])
        return float(var_missing)


def evaluate_model(
    model,
    missing_w_nans_np: np.ndarray,
    missing_complete_np: np.ndarray,
    na_ind,
    scaler,
    recycles: int,
    m: int
):
    """
    Closely mirrors your TF evaluate_model():
      - multi-impute -> coverage, multi_mae
      - single-impute -> mae
      - average_variance
    NOTE: we assume you’ll finish model.impute_single / impute_multiple for torch.
    """

    # multiple imputations for coverage
    multi_imputes_list = []
    missing_row_ind = np.where(np.isnan(missing_w_nans_np).any(axis=1))
    subset_na = np.where(np.isnan(missing_w_nans_np[missing_row_ind]))
    for _ in range(m):
        imputeds, _ = model.impute_multiple(
            np.copy(missing_w_nans_np),
            max_iter=recycles,
            method="Metropolis-within-Gibbs"
        )
        multi_imputes_list.append(imputeds[subset_na])

    coverage_q = evaluate_coverage_quantile(
        multi_imputes_list,
        missing_complete_np,
        missing_w_nans_np,
        scaler
    )
    coverage_std = evaluate_coverage(
        multi_imputes_list,
        missing_complete_np,
        missing_w_nans_np,
        scaler
    )

    # single impute
    imput_single, _, losses_hist = model.impute_single(
        np.copy(missing_w_nans_np),
        missing_complete_np,
        n_recycles=6,
        loss='MAE',
        scaler=scaler,
        return_losses=True
    )
    # final MAE (already computed in impute_single in TF version)
    final_mae = losses_hist[-1] if len(losses_hist) > 0 else None

    # variance proxy
    avg_var = evaluate_variance_torch(model, missing_w_nans_np, na_ind)

    # merge
    out = dict(
        mae=final_mae,
        average_variance=avg_var
    )
    out.update(coverage_q)
    out.update(coverage_std)
    return out


def get_additional_masked_data(complete_w_nan: np.ndarray, prop_miss_rows=1, prop_miss_col=0.1):
    """
    Same logic as your TF version.
    We take rows that are fully observed, then randomly mask extra cells (for validation).
    """
    complete_row_index = np.where(np.isfinite(complete_w_nan).all(axis=1))[0]
    complete_only = complete_w_nan[complete_row_index]

    miss_maker = DataMissingMaker(complete_only, prop_miss_rows=prop_miss_rows, prop_miss_col=prop_miss_col)
    extra_missing_validation = miss_maker.generate_missing_data()

    val_na_ind = np.where(np.isnan(extra_missing_validation))
    return extra_missing_validation, complete_only, val_na_ind


def split_training_and_validation(config, k_fold_idx, data_missing_nan: np.ndarray, data_full: np.ndarray):
    """
    Reimplements your split_training_and_validation():
    - take kth fold slice
    - generate validation_w_nan
    - inject that masked version back into training_input
    """
    n_per_fold = len(data_full) // config['k_folds']
    start_index = k_fold_idx * n_per_fold
    end_index = len(data_full) if k_fold_idx == config['k_folds'] - 1 else start_index + n_per_fold

    current_fold = data_missing_nan[start_index:end_index]
    test_missing_row_ind = np.where(np.isnan(data_missing_nan).any(axis=1))[0]
    val_missing_row_ind = list(set(range(start_index, end_index)) - set(test_missing_row_ind))

    validation_w_nan, validation_complete, val_na_ind = get_additional_masked_data(
        current_fold,
        prop_miss_rows=1,
        prop_miss_col=0.1
    )

    training_input = np.copy(data_missing_nan)
    training_input[val_missing_row_ind] = validation_w_nan
    training_input = np.nan_to_num(training_input)

    return training_input, validation_w_nan, validation_complete, val_na_ind


#########################################################
# 3. Training loop for one fold
#########################################################

def train_one_fold(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_tensor: torch.Tensor,
    epochs: int,
    beta: float,
    Cval: float,
    device: torch.device,
    batch_size: int = 256
) -> Tuple[float, float, float]:
    """
    Train VAE for `epochs` on the given fold’s training data.
    Returns (last_total_loss, last_recon, last_kl)
    """

    ds = TensorDataset(train_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    last_total, last_recon, last_kl = None, None, None

    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()

            out = model(xb)
            recon_logits = out["recon_logits"]
            mu = out["mu"]
            logvar = out["logvar"]

            # observed mask is all 1s here (we already imputed NaNs w/ something)
            obs_mask = torch.ones_like(xb)

            total_loss, recon_l, kl_mean = beta_capacity_loss(
                recon_logits,
                xb,
                mu,
                logvar,
                beta=beta,
                C=Cval,
                observed_mask=obs_mask
            )

            total_loss.backward()
            optimizer.step()

            last_total = float(total_loss.detach().cpu())
            last_recon = float(recon_l.cpu())
            last_kl = float(kl_mean.cpu())

    return last_total, last_recon, last_kl


#########################################################
# 4. Optimizer / convergence helpers
#########################################################


def build_optimizer(model: nn.Module, lr: float, use_adam: bool) -> torch.optim.Optimizer:
    """
    Select optimizer based on config flag.
    """
    if use_adam:
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr)


def train_fold_with_eval(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_tensor: torch.Tensor,
    beta: float,
    Cval: float,
    device: torch.device,
    batch_size: int,
    epoch_chunk: int,
    final_epochs: int,
    metric: str,
    validation_w_nan_np: np.ndarray,
    validation_complete_np: np.ndarray,
    val_na_ind,
    scaler,
    recycles: int,
    m_multi: int,
    fold_idx: int,
    results_path: str,
    tolerance: float,
    patience: int,
    min_epochs: int,
    lock_path: str = "lock.txt"
) -> Tuple[float, int]:
    """
    Train one fold with periodic evaluation and simple convergence check.
    Returns (best_metric, epochs_ran).
    """
    best_metric = math.inf
    no_improve = 0
    completed_epochs = 0

    while completed_epochs < final_epochs:
        train_one_fold(
            model=model,
            optimizer=optimizer,
            train_tensor=train_tensor,
            epochs=epoch_chunk,
            beta=beta,
            Cval=Cval,
            device=device,
            batch_size=batch_size
        )
        completed_epochs += epoch_chunk

        results_eval = evaluate_model(
            model=model,
            missing_w_nans_np=np.copy(validation_w_nan_np),
            missing_complete_np=validation_complete_np,
            na_ind=val_na_ind,
            scaler=scaler,
            recycles=recycles,
            m=m_multi
        )
        results_eval["k"] = fold_idx
        metric_value = _select_metric(results_eval, metric)

        save_results(
            results_eval,
            epoch=completed_epochs,
            beta=beta,
            Cval=Cval,
            results_path=results_path,
            lock_path=lock_path
        )

        if best_metric - metric_value > tolerance:
            best_metric = metric_value
            no_improve = 0
        else:
            no_improve += 1

        if completed_epochs >= max(epoch_chunk, min_epochs) and no_improve >= patience:
            break

    return best_metric, completed_epochs


#########################################################
# 5. Halving grid search utilities
#########################################################


def _select_metric(results: Dict, metric: str) -> float:
    """
    Lower is better. Falls back to multi_mae / average_variance if needed.
    """
    primary = results.get(metric)
    if primary is not None:
        return primary
    for fallback in ("multi_mae", "average_variance"):
        if results.get(fallback) is not None:
            return results[fallback]
    return math.inf


def _epoch_settings_from_config(config: Dict) -> Tuple[int, int]:
    """
    Derive epoch chunk size and maximum epochs from config.
    Uses halving_epoch_budgets if provided, otherwise falls back to static defaults.
    """
    budgets = config.get("halving_epoch_budgets")
    if isinstance(budgets, (list, tuple)) and len(budgets) > 0:
        chunk = int(min(budgets))
        final_epochs = int(max(budgets))
    else:
        chunk = int(config.get("epoch_chunk", 30))
        final_epochs = int(config.get("max_epochs", 300))
    return chunk, final_epochs


def run_candidate_cv(job_kwargs: Dict) -> Dict:
    """
    Worker entry: train and evaluate one (beta, C) across all folds.
    """
    beta = job_kwargs["beta"]
    Cval = job_kwargs["C"]
    config = job_kwargs["config"]
    data_full = job_kwargs["data_full"]
    data_missing_nan = job_kwargs["data_missing_nan"]
    scaler = job_kwargs["scaler"]
    device = torch.device(job_kwargs.get("device", "cpu"))

    if device.type == "cuda":
        torch.cuda.set_device(device)

    metric = config.get("halving_metric", "mae")
    tolerance = config.get("convergence_tolerance", 1e-4)
    patience = config.get("convergence_patience", 2)
    use_adam = config.get("use_adam_optimizer", False)
    min_epochs = config.get("min_epochs_before_convergence", None)

    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    recycles = config["recycles"]
    m_multi = config["m"]
    k_folds = config["k_folds"]
    latent_dim = config["latent_size"]
    h1 = config["hidden_size_1"]
    h2 = config["hidden_size_2"]
    results_path = config["results_path"]

    epoch_chunk, final_epochs = _epoch_settings_from_config(config)
    min_epochs = epoch_chunk if min_epochs is None else int(min_epochs)

    fold_scores: List[float] = []
    for fold_idx in range(k_folds):
        (
            training_input_np,
            validation_w_nan_np,
            validation_complete_np,
            val_na_ind
        ) = split_training_and_validation(
            config,
            fold_idx,
            data_missing_nan,
            data_full
        )

        train_tensor = torch.tensor(training_input_np, dtype=torch.float32)
        model = DisentangledBetaVaeTorchModule(
            input_dim=train_tensor.shape[1],
            latent_dim=latent_dim,
            hidden_dim1=h1,
            hidden_dim2=h2
        ).to(device)
        optimizer = build_optimizer(model, lr=lr, use_adam=use_adam)

        best_metric, _ = train_fold_with_eval(
            model=model,
            optimizer=optimizer,
            train_tensor=train_tensor,
            beta=beta,
            Cval=Cval,
            device=device,
            batch_size=batch_size,
            epoch_chunk=epoch_chunk,
            final_epochs=final_epochs,
            metric=metric,
            validation_w_nan_np=validation_w_nan_np,
            validation_complete_np=validation_complete_np,
            val_na_ind=val_na_ind,
            scaler=scaler,
            recycles=recycles,
            m_multi=m_multi,
            fold_idx=fold_idx,
            results_path=results_path,
            tolerance=tolerance,
            patience=patience,
            min_epochs=min_epochs
        )
        fold_scores.append(best_metric)

    avg_score = float(np.mean(fold_scores))
    return {
        "beta": beta,
        "C": Cval,
        "score": avg_score,
        "fold_scores": fold_scores,
        "device": str(device)
    }


def _available_devices() -> List[str]:
    """
    Prefer multi-GPU. If only one or none, fall back to CPU for multiprocessing.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def _run_candidates_parallel(
    candidates: List[Tuple[float, float]],
    config: Dict,
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    scaler
) -> List[Dict]:
    """
    Execute candidate CV jobs in parallel using GPUs when available,
    otherwise CPU multiprocessing.
    """
    devices = _available_devices()
    jobs = []
    for idx, (b, c) in enumerate(candidates):
        jobs.append({
            "beta": b,
            "C": c,
            "config": config,
            "data_full": data_full,
            "data_missing_nan": data_missing_nan,
            "scaler": scaler,
            "device": devices[idx % len(devices)]
        })

    if len(jobs) == 1:
        return [run_candidate_cv(jobs[0])]

    if devices == ["cpu"]:
        workers = min(len(jobs), cpu_count())
    else:
        workers = min(len(jobs), len(devices))

    with Pool(processes=workers) as pool:
        results = list(pool.imap_unordered(run_candidate_cv, jobs))
    return results


def iterative_halving_search(
    config: Dict,
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    scaler
) -> Dict:
    """
    Iteratively shrink beta/C ranges by evaluating four corner combos
    until ranges are smaller than accuracy * original_span.
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
        print(f"[Iter {iteration}] evaluating beta in [{beta_min}, {beta_max}] "
              f"C in [{C_min}, {C_max}]")
        results = _run_candidates_parallel(combos, config, data_full, data_missing_nan, scaler)
        results = sorted(results, key=lambda r: r["score"])
        best_round = results[0]
        if best_global is None or best_round["score"] < best_global["score"]:
            best_global = best_round

        beta_m = (beta_min + beta_max) / 2.0
        C_m = (C_min + C_max) / 2.0

        if best_round["beta"] == beta_min and best_round["C"] == C_min:
            beta_max = beta_m
            C_max = C_m
        elif best_round["beta"] == beta_min and best_round["C"] == C_max:
            beta_max = beta_m
            C_min = C_m
        elif best_round["beta"] == beta_max and best_round["C"] == C_min:
            beta_min = beta_m
            C_max = C_m
        else:
            beta_min = beta_m
            C_min = C_m

        beta_span = beta_max - beta_min
        C_span = C_max - C_min
        if beta_span <= granularity * beta_span0 and C_span <= granularity * C_span0:
            break

    print(f"[Search done] best beta={best_global['beta']}, "
          f"C={best_global['C']}, score={best_global['score']}")
    return best_global


def train_and_save_best_model(
    beta_val: float,
    Cval: float,
    config: Dict,
    data_full: np.ndarray,
    data_missing_nan: np.ndarray,
    scaler
):
    """
    Train a single fold with the selected hyperparameters to persist a checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_chunk, final_epochs = _epoch_settings_from_config(config)
    metric = config.get("halving_metric", "mae")
    tolerance = config.get("convergence_tolerance", 1e-4)
    patience = config.get("convergence_patience", 2)
    min_epochs = config.get("min_epochs_before_convergence", epoch_chunk)

    (
        training_input_np,
        validation_w_nan_np,
        validation_complete_np,
        val_na_ind
    ) = split_training_and_validation(
        config,
        0,
        data_missing_nan,
        data_full
    )

    train_tensor = torch.tensor(training_input_np, dtype=torch.float32)
    model = DisentangledBetaVaeTorchModule(
        input_dim=train_tensor.shape[1],
        latent_dim=config["latent_size"],
        hidden_dim1=config["hidden_size_1"],
        hidden_dim2=config["hidden_size_2"]
    ).to(device)
    optimizer = build_optimizer(
        model,
        lr=config["learning_rate"],
        use_adam=config.get("use_adam_optimizer", False)
    )

    best_metric, completed_epochs = train_fold_with_eval(
        model=model,
        optimizer=optimizer,
        train_tensor=train_tensor,
        beta=beta_val,
        Cval=Cval,
        device=device,
        batch_size=config["batch_size"],
        epoch_chunk=epoch_chunk,
        final_epochs=final_epochs,
        metric=metric,
        validation_w_nan_np=validation_w_nan_np,
        validation_complete_np=validation_complete_np,
        val_na_ind=val_na_ind,
        scaler=scaler,
        recycles=config["recycles"],
        m_multi=config["m"],
        fold_idx=0,
        results_path=config["results_path"],
        tolerance=tolerance,
        patience=patience,
        min_epochs=min_epochs
    )

    outdir = config.get("model_outdir", "./trained_models")
    os.makedirs(outdir, exist_ok=True)
    checkpoint_path = os.path.join(outdir, f"beta{beta_val}_C{Cval}_fold0.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "beta": beta_val,
            "C": Cval,
            "fold": 0,
            "epochs": completed_epochs,
            "best_metric": best_metric,
            "config": config
        },
        checkpoint_path
    )
    return checkpoint_path


#########################################################
# 6. Main cross-validation driver
#########################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='cross_validation/cv_config.json',
                        help='path to configuration json')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if "beta_range" not in config or "C_range" not in config:
        raise ValueError("Config must include beta_range and C_range for halving search.")

    # Load & scale data once
    data_full, data_missing_nan, scaler = get_scaled_data(
        data_path=config["data_path"],
        corrupt_data_path=config["corrupt_data_path"],
        initial_imputation_strategy=config["initial_imputation_strategy"],
        return_scaler=True,
        put_nans_back=True,
        nextflow=False
    )

    best_candidate = iterative_halving_search(
        config=config,
        data_full=data_full,
        data_missing_nan=data_missing_nan,
        scaler=scaler
    )
    beta_val = best_candidate["beta"]
    Cval = best_candidate["C"]

    checkpoint_path = train_and_save_best_model(
        beta_val=beta_val,
        Cval=Cval,
        config=config,
        data_full=data_full,
        data_missing_nan=data_missing_nan,
        scaler=scaler
    )
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
