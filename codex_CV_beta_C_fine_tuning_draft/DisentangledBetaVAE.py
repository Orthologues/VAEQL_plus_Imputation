import os
import time
import json
import argparse
import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from pythae.models import DisentangledBetaVAE, DisentangledBetaVAEConfig


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
        TODO: implement your actual imputation loop.
        For now we'll just fill NaNs with the model’s forward pass mean.
        Must return (imputed, conv_loglik?, losses_per_iter?)
        """
        # dummy passthrough
        imputed = np.copy(X_nan_np)
        losses_hist = [0.0]
        return imputed, None, losses_hist

    def impute_multiple(self, X_nan_np, max_iter: int, method="Metropolis-within-Gibbs"):
        """
        TODO: draw multiple imputations stochastically from your latent sampler.
        Return (imputed_samples, conv_loglik?)
        """
        # dummy: return deterministic single imputation
        imputed = np.copy(X_nan_np)
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
# 4. Successive halving search utilities
#########################################################

def _linspace_range(values: List[float], points: int) -> List[float]:
    lo, hi = values
    if points <= 1:
        return [float(lo)]
    return np.linspace(lo, hi, points).tolist()


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


def successive_halving_search(
    beta_vals: List[float],
    C_vals: List[float],
    budgets: List[int],
    keep_ratio: float,
    metric: str,
    train_tensor: torch.Tensor,
    validation_w_nan_np: np.ndarray,
    validation_complete_np: np.ndarray,
    val_na_ind,
    scaler,
    recycles: int,
    m_multi: int,
    device: torch.device,
    latent_dim: int,
    h1: int,
    h2: int,
    lr: float,
    batch_size: int,
    results_path: str,
    fold_idx: int
):
    """
    Perform a coarse successive-halving search over (beta, C).
    Each candidate trains up to each `budget` (epochs) cumulatively;
    the worst-performing half is dropped each round.
    """
    candidates = [
        {
            "beta": b,
            "C": c,
            "model": None,
            "optimizer": None,
            "trained_epochs": 0,
            "score": math.inf,
        }
        for b in beta_vals
        for c in C_vals
    ]

    def init_state(candidate):
        model = DisentangledBetaVaeTorchModule(
            input_dim=train_tensor.shape[1],
            latent_dim=latent_dim,
            hidden_dim1=h1,
            hidden_dim2=h2
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        candidate["model"] = model
        candidate["optimizer"] = opt
        candidate["trained_epochs"] = 0

    for budget in budgets:
        if budget <= 0:
            continue

        print(f"[Halving] Starting budget {budget} with {len(candidates)} candidates")

        for cand in candidates:
            if cand["model"] is None:
                init_state(cand)

            extra_epochs = budget - cand["trained_epochs"]
            if extra_epochs > 0:
                train_one_fold(
                    model=cand["model"],
                    optimizer=cand["optimizer"],
                    train_tensor=train_tensor,
                    epochs=extra_epochs,
                    beta=cand["beta"],
                    Cval=cand["C"],
                    device=device,
                    batch_size=batch_size
                )
                cand["trained_epochs"] = budget

            results_eval = evaluate_model(
                model=cand["model"],
                missing_w_nans_np=np.copy(validation_w_nan_np),
                missing_complete_np=validation_complete_np,
                na_ind=val_na_ind,
                scaler=scaler,
                recycles=recycles,
                m=m_multi
            )
            results_eval["k"] = fold_idx
            cand["score"] = _select_metric(results_eval, metric)

            save_results(
                results_eval,
                epoch=budget,
                beta=cand["beta"],
                Cval=cand["C"],
                results_path=results_path,
                lock_path='lock.txt'
            )

        # rank and trim
        candidates = sorted(candidates, key=lambda x: x["score"])
        keep_count = max(1, int(math.ceil(len(candidates) * keep_ratio)))

        # free dropped models
        for drop in candidates[keep_count:]:
            drop["model"] = None
            drop["optimizer"] = None

        candidates = candidates[:keep_count]
        print(f"[Halving] Budget {budget} done; keeping {len(candidates)} candidates")

        if len(candidates) == 1:
            break

    best = sorted(candidates, key=lambda x: x["score"])[0]
    print(f"[Halving] Best beta={best['beta']} C={best['C']} score={best['score']}")
    return best


#########################################################
# 5. Main cross-validation driver
#########################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('d_index', type=int,
                        help='global job index: selects fold (and beta/C when not using halving)')
    parser.add_argument('--config', type=str,
                        default='cross_validation/cv_config.json',
                        help='path to configuration json')
    args = parser.parse_args()

    d_index = args.d_index - 1  # original script used 1-based

    with open(args.config) as f:
        config = json.load(f)

    # ------------------------------------------------------------------
    # hyperparams grid (like beta_rates and epoch_granularity before)
    # can be defined as ranges (for halving) or explicit lists.
    # ------------------------------------------------------------------
    use_halving = "beta_range" in config and "C_range" in config
    if use_halving:
        beta_grid = _linspace_range(
            config["beta_range"],
            config.get("beta_grid_points", 6)
        )
        C_grid = _linspace_range(
            config["C_range"],
            config.get("C_grid_points", 6)
        )
    else:
        beta_grid = config.get("beta_grid", [0.5, 1.0, 2.0, 4.0, 8.0])
        C_grid    = config.get("C_grid",   [0.0, 5.0, 10.0])

    # training schedule assumptions
    epoch_granularity = config.get("epoch_granularity", { "default": 30 })
    max_epochs_map    = config.get("max_epochs_map", { "default": 300 })

    k_folds = config['k_folds']
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    recycles = config["recycles"]
    m_multi = config["m"]
    results_path = config["results_path"]

    # Load & scale data once
    data_full, data_missing_nan, scaler = get_scaled_data(
        data_path=config["data_path"],
        corrupt_data_path=config["corrupt_data_path"],
        initial_imputation_strategy=config["initial_imputation_strategy"],
        return_scaler=True,
        put_nans_back=True,
        nextflow=False
    )

    fold_idx = d_index % k_folds

    # pick epoch granularity & max epochs for this beta (fallback "default")
    default_chunk = epoch_granularity.get("default", 30)
    default_final = max_epochs_map.get("default", 300)

    # prepare fold data
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = training_input_np.shape[1]
    latent_dim = config["latent_size"]
    h1 = config["hidden_size_1"]
    h2 = config["hidden_size_2"]

    train_tensor = torch.tensor(training_input_np, dtype=torch.float32)

    if use_halving:
        halving_budgets = config.get(
            "halving_epoch_budgets",
            [default_chunk, default_chunk * 2, default_final]
        )
        halving_budgets = sorted(set(int(b) for b in halving_budgets if b > 0))
        halving_keep_ratio = config.get("halving_keep_ratio", 0.5)
        halving_metric = config.get("halving_metric", "mae")

        best_candidate = successive_halving_search(
            beta_vals=beta_grid,
            C_vals=C_grid,
            budgets=halving_budgets,
            keep_ratio=halving_keep_ratio,
            metric=halving_metric,
            train_tensor=train_tensor,
            validation_w_nan_np=validation_w_nan_np,
            validation_complete_np=validation_complete_np,
            val_na_ind=val_na_ind,
            scaler=scaler,
            recycles=recycles,
            m_multi=m_multi,
            device=device,
            latent_dim=latent_dim,
            h1=h1,
            h2=h2,
            lr=lr,
            batch_size=batch_size,
            results_path=results_path,
            fold_idx=fold_idx
        )

        vae = best_candidate["model"]
        beta_val = best_candidate["beta"]
        Cval = best_candidate["C"]
        completed_epochs = best_candidate["trained_epochs"]
    else:
        beta_val = beta_grid[(d_index // k_folds) % len(beta_grid)]
        Cval = C_grid[(d_index // (k_folds * len(beta_grid))) % len(C_grid)]

        epochs_chunk = epoch_granularity.get(str(beta_val),
                         epoch_granularity.get("default", 30))
        final_epochs = max_epochs_map.get(str(beta_val),
                         max_epochs_map.get("default", 300))
        rounds = int(final_epochs / epochs_chunk) + 1

        print(f"[INFO] beta={beta_val}, C={Cval}, fold={fold_idx}, "
              f"epochs_chunk={epochs_chunk}, rounds={rounds}, final_epochs={final_epochs}")

        vae = DisentangledBetaVaeTorchModule(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim1=h1,
            hidden_dim2=h2
        ).to(device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

        for r in range(rounds):
            last_total, last_recon, last_kl = train_one_fold(
                model=vae,
                optimizer=optimizer,
                train_tensor=train_tensor,
                epochs=epochs_chunk,
                beta=beta_val,
                Cval=Cval,
                device=device,
                batch_size=batch_size
            )

            completed_epochs = (r + 1) * epochs_chunk
            print(f"[Fold {fold_idx}] after {completed_epochs} epochs "
                  f"loss={last_total:.4f} recon={last_recon:.4f} kl={last_kl:.4f}")

            results_eval = evaluate_model(
                model=vae,
                missing_w_nans_np=np.copy(validation_w_nan_np),
                missing_complete_np=validation_complete_np,
                na_ind=val_na_ind,
                scaler=scaler,
                recycles=recycles,
                m=m_multi
            )
            results_eval["k"] = fold_idx

            save_results(
                results_eval,
                epoch=completed_epochs,
                beta=beta_val,
                Cval=Cval,
                results_path=results_path,
                lock_path='lock.txt'
            )

    # optional: save final model weights for that (beta, C, fold)
    outdir = config.get("model_outdir", "./trained_models")
    os.makedirs(outdir, exist_ok=True)
    torch.save(
        {
            "state_dict": vae.state_dict() if vae is not None else None,
            "beta": beta_val,
            "C": Cval,
            "fold": fold_idx,
            "epochs": completed_epochs,
            "config": config
        },
        os.path.join(outdir, f"beta{beta_val}_C{Cval}_fold{fold_idx}.pt")
    )


if __name__ == "__main__":
    main()
