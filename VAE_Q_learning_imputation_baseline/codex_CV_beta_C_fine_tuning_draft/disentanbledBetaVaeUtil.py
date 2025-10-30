"""
disentangledBetaVaeUtil.py

Utilities for:
- scaling and initial imputation
- evaluating multiple-imputation quality (MAE, coverage)
- Monte Carlo log-likelihood-style diagnostics
- synthetically masking data

This version:
- removes TensorFlow / TFP
- uses NumPy, scikit-learn, PyTorch
- is meant for use with a capacity-controlled beta-VAE style objective:
      recon_loss + beta * | KL - C |
  where KL is the latent KL per-sample.

Assumptions:
- You are using pythae.autoencoders.DisentangledBetaVAE (or a subclass)
  which exposes .encoder and .decoder as nn.Modules.
- The decoder takes latent z -> reconstruction parameters (mean, logvar).
- You can adapt the "model interface" comments below to match your exact model.
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Literal

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn.functional as F


# -------------------------------------------------------
# Global defaults
# -------------------------------------------------------

param_imputation = {
    'strategy': 'mean',   # for simple imputer ('mean' or 'median')
    'n_neighbors': 5,     # for knn imputer
    'max_iter': 20,       # for iterative imputer
    'tol': 1e-3           # for iterative imputer
}


# -------------------------------------------------------
# Coverage / MAE evaluation (multiple imputation quality)
# -------------------------------------------------------

def evaluate_coverage_quantile(
    multi_imputes: np.ndarray,
    data: np.ndarray,
    data_missing: np.ndarray,
    scaler: StandardScaler
) -> Dict[str, float]:
    """
    multi_imputes : shape (M, K) where M = number of multiple imputations
                                   K = number of missing entries (flattened)
                    OR shape (M, n_rows, n_features) *but* then caller should
                    subset to only NA positions first.

    data          : ground truth complete data (scaled)
    data_missing  : data with NaNs in missing spots (scaled)
    scaler        : fitted StandardScaler used for inverse-transform elsewhere

    We compute empirical quantile coverage: does the true value fall
    in the [q_low, q_high] interval?

    Returns dict with prop_80q, prop_90q, prop_95q, prop_99q
    """
    na_ind = np.where(np.isnan(data_missing))
    true_values = data[na_ind]  # ground-truth on missing cells

    # multi_imputes expected shape: (M, K). We'll take quantiles over axis=0
    low_q80 = np.percentile(multi_imputes, 10, axis=0)
    up_q80  = np.percentile(multi_imputes, 90, axis=0)

    low_q90 = np.percentile(multi_imputes, 5, axis=0)
    up_q90  = np.percentile(multi_imputes, 95, axis=0)

    low_q95 = np.percentile(multi_imputes, 2.5, axis=0)
    up_q95  = np.percentile(multi_imputes, 97.5, axis=0)

    low_q99 = np.percentile(multi_imputes, 0.5, axis=0)
    up_q99  = np.percentile(multi_imputes, 99.5, axis=0)

    def prop_in_interval(lo, hi, truth):
        return np.mean([(lo[i] < truth[i] < hi[i]) for i in range(len(truth))])

    results = {
        'prop_80q': prop_in_interval(low_q80, up_q80, true_values),
        'prop_90q': prop_in_interval(low_q90, up_q90, true_values),
        'prop_95q': prop_in_interval(low_q95, up_q95, true_values),
        'prop_99q': prop_in_interval(low_q99, up_q99, true_values),
    }
    return results


def evaluate_coverage(
    multi_imputes: np.ndarray,
    data: np.ndarray,
    data_missing: np.ndarray,
    scaler: StandardScaler
) -> Dict[str, float]:
    """
    multi_imputes : shape (M, K_missing) = multiple draws for each originally-missing entry
    data          : ground truth complete data (scaled!)
    data_missing  : same shape as data but NaN in missing cells (scaled!)
    scaler        : StandardScaler (fit on observed rows), used later for MAE in original scale

    Returns prop_80, prop_90, prop_95, prop_99 and multi_mae
    """

    assert data_missing.shape == data.shape
    na_ind = np.where(np.isnan(data_missing))

    means = np.mean(multi_imputes, axis=0)              # per missing entry
    stdev = np.std(multi_imputes, axis=0)               # per missing entry
    diff  = np.abs(data[na_ind] - means)                # |truth - mean|
    n_dev = diff / (stdev + 1e-12)                      # #stdevs away

    ci_80 = 1.282
    ci_90 = 1.645
    ci_95 = 1.960
    ci_99 = 2.576

    prop_80 = np.mean(n_dev < ci_80)
    prop_90 = np.mean(n_dev < ci_90)
    prop_95 = np.mean(n_dev < ci_95)
    prop_99 = np.mean(n_dev < ci_99)

    results = {
        'prop_80': prop_80,
        'prop_90': prop_90,
        'prop_95': prop_95,
        'prop_99': prop_99
    }

    # Now compute MAE in ORIGINAL scale:
    data_unscaled = scaler.inverse_transform(data)
    # fill predicted means into the missing spots
    data_missing_filled = data_missing.copy()
    data_missing_filled[na_ind] = means
    data_missing_filled = scaler.inverse_transform(data_missing_filled)

    mae = np.mean(np.abs(
        data_unscaled[na_ind] - data_missing_filled[na_ind]
    ))
    results['multi_mae'] = mae

    return results


# -------------------------------------------------------
# Simple imputers for initialization
# -------------------------------------------------------

def impute_nas_with_zeros(data_missing: np.ndarray) -> np.ndarray:
    """Replace NaNs with 0 in a copy."""
    out = data_missing.copy()
    na_ind = np.where(np.isnan(out))
    out[na_ind] = 0.0
    return out


def impute_nas_with_iterative_imputer(
    data_missing,
    type_imputer: Literal["simple","knn","iterative_bayesridge","iterative_randomforest"],
    params: Dict
) -> np.ndarray:
    """
    Wrapper around scikit imputers.
    Returns a NumPy array.
    """
    if isinstance(data_missing, np.ndarray):
        df_with_nans = pd.DataFrame(data_missing)
    else:
        df_with_nans = data_missing.copy()

    if type_imputer == "simple":
        imp = SimpleImputer(missing_values=np.nan, strategy=params['strategy'])
    elif type_imputer == "knn":
        imp = KNNImputer(missing_values=np.nan, n_neighbors=params['n_neighbors'])
    elif type_imputer == "iterative_bayesridge":
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            missing_values=np.nan,
            max_iter=params['max_iter'],
            tol=params['tol']
        )
    elif type_imputer == "iterative_randomforest":
        imp = IterativeImputer(
            estimator=RandomForestRegressor(),
            missing_values=np.nan,
            max_iter=params['max_iter'],
            tol=params['tol'],
            verbose=1
        )
    else:
        raise ValueError(
            f"Invalid imputer '{type_imputer}'. "
            "Choose from 'simple','knn','iterative_bayesridge','iterative_randomforest'"
        )

    imputed = imp.fit_transform(df_with_nans)

    # always return np.ndarray
    return imputed


def perform_initial_imputation(
    data_missing: np.ndarray,
    type_imputer: Literal["zero","simple","knn","iterative_bayesridge","iterative_randomforest"],
    params: Dict = param_imputation
) -> np.ndarray:
    """
    Initial fill before VAE training.
    For large D, 'zero' is cheaper than IterativeImputer.
    """
    if type_imputer == "zero":
        return impute_nas_with_zeros(data_missing)
    elif type_imputer in ["simple","knn","iterative_bayesridge","iterative_randomforest"]:
        return impute_nas_with_iterative_imputer(data_missing, type_imputer, params)
    else:
        raise ValueError(
            "Invalid type_imputer. "
            "Choose 'zero','simple','knn','iterative_bayesridge','iterative_randomforest'."
        )


# -------------------------------------------------------
# Scaling, with optional NaN restoration
# -------------------------------------------------------

def get_scaled_data(
    data_path: str,
    corrupt_data_path: str,
    initial_imputation_strategy: str,
    return_scaler: bool = False,
    put_nans_back: bool = False,
    nextflow: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    1. Read "clean" data and "corrupt_data" (with NaNs).
    2. Fit StandardScaler on rows without any NaN.
    3. Initial-impute corrupt_data (so scaler can transform all rows).
    4. Scale both clean data and imputed corrupt_data.
    5. Optionally put NaNs back in the scaled corrupt_data.

    NOTE: you hard-coded dropping first 4 columns in `data = data[:,4:]`.
    I keep that behavior here.
    """
    data_fn = os.path.basename(data_path)
    corrupt_fn = os.path.basename(corrupt_data_path)

    if nextflow:
        full_data = pd.read_csv(os.path.join(os.getcwd(), data_fn)).values
        corrupt_data = pd.read_csv(os.path.join(os.getcwd(), corrupt_fn)).values
    else:
        full_data = pd.read_csv(data_path).values
        corrupt_data = pd.read_csv(corrupt_data_path).values

    non_missing_row_ind = np.where(np.isfinite(corrupt_data).all(axis=1))[0]
    na_ind = np.where(np.isnan(corrupt_data))

    scaler = StandardScaler()
    scaler.fit(corrupt_data[non_missing_row_ind, :])

    # initial impute corrupt_data for scaling
    corrupt_imputed = perform_initial_imputation(
        corrupt_data,
        type_imputer=initial_imputation_strategy
    )
    corrupt_scaled = scaler.transform(corrupt_imputed)

    # clean/full data -- you dropped first 4 cols
    data_scaled = scaler.transform(np.array(full_data[:, 4:], dtype="float64"))

    if put_nans_back:
        corrupt_scaled[na_ind] = np.nan

    if return_scaler:
        return data_scaled, corrupt_scaled, scaler
    else:
        return data_scaled, corrupt_scaled


def apply_scaler(
    data: np.ndarray,
    data_missing: np.ndarray,
    return_scaler: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    For test-time scaling:
    - fit scaler ONLY on rows with no NaN (in data_missing)
    - fill NaN with 0 just for transform, then restore NaN after scaling
    """
    non_missing_row_ind = np.where(np.isfinite(data_missing).all(axis=1))[0]
    na_ind = np.where(np.isnan(data_missing))

    scaler = StandardScaler()
    scaler.fit(data_missing[non_missing_row_ind, :])

    temp = data_missing.copy()
    temp[na_ind] = 0.0
    temp = scaler.transform(temp)
    temp[na_ind] = np.nan

    data_scaled = scaler.transform(data)

    if return_scaler:
        return data_scaled, temp, scaler
    else:
        return data_scaled, temp


# -------------------------------------------------------
# Monte Carlo utilities for approximate log-likelihood terms
# (porting from TF/TFP to PyTorch)
# -------------------------------------------------------

def _normal_logpdf(x, mean, log_var):
    """
    log N(x | mean, var) elementwise
    x, mean, log_var are torch tensors broadcastable to same shape.
    """
    # var = exp(log_var)
    # log N = -0.5 * ( log(2π) + log_var + (x-mean)^2 / var )
    return -0.5 * (
        np.log(2.0 * np.pi)
        + log_var
        + (x - mean) ** 2 / torch.exp(log_var)
    )


def _sample_standard_normal(num_samples, batch_size, latent_dim, device):
    """
    return shape [num_samples, batch_size, latent_dim]
    """
    return torch.randn(num_samples, batch_size, latent_dim, device=device)


@torch.no_grad()
def mc_wrt_p(
    data_complete: np.ndarray,
    num_samples_mc: int,
    model,
    want_log: bool = True,
    observed_indices_mask: Optional[np.ndarray] = None,
    device: str = "cuda"
) -> np.ndarray:
    """
    Approximate p(y) by integrating p(y|z)p(z) with z ~ N(0,I).
    Purely 'decoder-side' sampling.

    model is assumed to have:
        model.latent_dim
        model.beta  (optional if you scale sigma that way)
        model.decoder(z) -> (recon_mean, recon_log_var)

    data_complete: (N, D) torchable float
    observed_indices_mask: same shape (N,D) with 1 where observed (optional)

    Returns: np.array of shape (N,) giving log p(y_i) or p(y_i)
    """
    model.eval()
    x = torch.tensor(data_complete, dtype=torch.float32, device=device)
    N, D = x.shape
    latent_dim = model.latent_dim

    # sample z ~ N(0,I)
    z_samples = _sample_standard_normal(num_samples_mc, N, latent_dim, device=device)
    # decode all samples at once
    # EXPECTED: decoder accepts z -> (mean, log_var) with shape [num_samples_mc, N, D]
    recon_mean_list = []
    recon_logvar_list = []
    for s in range(num_samples_mc):
        mu_s, logvar_s = model.decoder(z_samples[s])  # shape (N, D) each
        recon_mean_list.append(mu_s.unsqueeze(0))
        recon_logvar_list.append(logvar_s.unsqueeze(0))

    recon_mean = torch.cat(recon_mean_list, dim=0)      # (S,N,D)
    recon_logvar = torch.cat(recon_logvar_list, dim=0)  # (S,N,D)

    # log p(x|z)
    # assume Gaussian likelihood with learned mean/var in decoder
    # log_probs: shape (S,N,D)
    log_probs = _normal_logpdf(
        x.unsqueeze(0),          # (1,N,D)
        recon_mean,              # (S,N,D)
        recon_logvar             # (S,N,D)
    )

    if observed_indices_mask is not None:
        mask_t = torch.tensor(
            observed_indices_mask, dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1,N,D)
        log_probs = log_probs * mask_t

    # sum over features D -> (S,N)
    log_probs_sum = torch.sum(log_probs, dim=2)

    # Now average over S samples
    # If want_log = True we do log(mean(exp(logp)))
    if want_log:
        # logsumexp-trick:
        m = torch.max(log_probs_sum, dim=0).values  # (N,)
        internal_mean = torch.mean(torch.exp(log_probs_sum - m), dim=0)  # (N,)
        out = m + torch.log(internal_mean + 1e-12)  # (N,)
    else:
        out = torch.mean(torch.exp(log_probs_sum), dim=0)  # (N,)

    return out.cpu().numpy()


@torch.no_grad()
def mc_wrt_q(
    data_complete: np.ndarray,
    num_samples_mc: int,
    model,
    proposal: str = 't',
    df: int = 3,
    want_log: bool = True,
    observed_indices_mask: Optional[np.ndarray] = None,
    device: str = "cuda"
) -> np.ndarray:
    """
    Importance sampling version:
    z ~ q(z|x) (or heavier-tailed proposal), weight by p(z)/q(z).

    You will have to implement model.encoder(x) -> (z_mean, z_logvar)
    and also a proposal sampler q_t(z|x) if you want heavy tails (e.g. Student-t).
    For now I'll assume Gaussian q(z|x).

    Returns np.array shape (N,) of approx log p(x) or p(x).
    """

    model.eval()
    x = torch.tensor(data_complete, dtype=torch.float32, device=device)
    N, D = x.shape
    latent_dim = model.latent_dim

    # encode
    # EXPECTED: encoder(x) -> (z_mean, z_logvar) each (N, latent_dim)
    z_mean, z_logvar = model.encoder(x)

    # sample S times from q(z|x) ~ Normal(z_mean, exp(0.5*z_logvar))
    eps = torch.randn(num_samples_mc, N, latent_dim, device=device)
    z_samples = z_mean.unsqueeze(0) + torch.exp(0.5 * z_logvar).unsqueeze(0) * eps
    # log p(z) for each sample
    log_pz = -0.5 * (
        latent_dim * np.log(2.0 * np.pi)
        + torch.sum(z_samples ** 2, dim=2)  # (S,N)
    )  # (S,N)

    # log q(z|x)
    # diag Gaussian log q = -0.5 * [ d*log(2π) + sum(logvar + (z-mean)^2/var ) ]
    log_var = z_logvar.unsqueeze(0)                   # (1,N,L)
    var = torch.exp(log_var)
    diff2 = (z_samples - z_mean.unsqueeze(0)) ** 2
    log_qz = -0.5 * (
        latent_dim * np.log(2.0 * np.pi)
        + torch.sum(
            log_var + diff2 / var,
            dim=2
        )  # (S,N)
    )  # (S,N)

    # decode each sampled z
    recon_mean_list = []
    recon_logvar_list = []
    for s in range(num_samples_mc):
        mu_s, logvar_s = model.decoder(z_samples[s])  # (N,D)
        recon_mean_list.append(mu_s.unsqueeze(0))
        recon_logvar_list.append(logvar_s.unsqueeze(0))
    recon_mean = torch.cat(recon_mean_list, dim=0)      # (S,N,D)
    recon_logvar = torch.cat(recon_logvar_list, dim=0)  # (S,N,D)

    # log p(x|z)
    log_px_given_z = _normal_logpdf(
        x.unsqueeze(0),
        recon_mean,
        recon_logvar
    )  # (S,N,D)

    if observed_indices_mask is not None:
        mask_t = torch.tensor(
            observed_indices_mask, dtype=torch.float32, device=device
        ).unsqueeze(0)
        log_px_given_z = log_px_given_z * mask_t

    log_px_given_z = torch.sum(log_px_given_z, dim=2)  # (S,N)

    # importance weight log [p(x|z)p(z)/q(z|x)]
    log_weight = log_px_given_z + log_pz - log_qz  # (S,N)

    if want_log:
        m = torch.max(log_weight, dim=0).values  # (N,)
        internal_mean = torch.mean(torch.exp(log_weight - m), dim=0)  # (N,)
        out = m + torch.log(internal_mean + 1e-12)
    else:
        out = torch.mean(torch.exp(log_weight), dim=0)

    return out.cpu().numpy()


def log_lik_ymis_given_obs_mcmc_q(
    data: np.ndarray,
    data_corrupt: np.ndarray,
    model,
    num_samples_mc: int = 500,
    device: str = "cuda"
) -> np.ndarray:
    """
    Approx log p(y_mis | y_obs).
    We'll approximate log p(y_full) - log p(y_obs)
    using mc_wrt_q with/without masking.

    data          = fully observed (scaled)
    data_corrupt  = same shape but NaNs where missing
    """

    missing_row_ind = np.where(np.isnan(data_corrupt).any(axis=1))[0]
    data_corrupt_sub = data_corrupt[missing_row_ind, :].copy()
    data_full_sub    = data[missing_row_ind, :].copy()

    compl_ind = np.where(np.isfinite(data_corrupt_sub))
    observed_mask = np.zeros_like(data_corrupt_sub)
    observed_mask[compl_ind] = 1.0

    log_p_y   = mc_wrt_q(
        data_full_sub, num_samples_mc, model,
        want_log=True, observed_indices_mask=None, device=device
    )
    log_p_yob = mc_wrt_q(
        data_full_sub, num_samples_mc, model,
        want_log=True, observed_indices_mask=observed_mask, device=device
    )
    return log_p_y - log_p_yob


def log_lik_ymis_given_obs_mcmc_p(
    data: np.ndarray,
    data_corrupt: np.ndarray,
    model,
    num_samples_mc: int = 500,
    device: str = "cuda"
) -> np.ndarray:
    """
    Same idea but samples z ~ p(z) instead of q(z|x).
    """
    missing_row_ind = np.where(np.isnan(data_corrupt).any(axis=1))[0]
    data_corrupt_sub = data_corrupt[missing_row_ind, :].copy()
    data_full_sub    = data[missing_row_ind, :].copy()

    compl_ind = np.where(np.isfinite(data_corrupt_sub))
    observed_mask = np.zeros_like(data_corrupt_sub)
    observed_mask[compl_ind] = 1.0

    log_p_y   = mc_wrt_p(
        data_full_sub, num_samples_mc, model,
        want_log=True, observed_indices_mask=None, device=device
    )
    log_p_yob = mc_wrt_p(
        data_full_sub, num_samples_mc, model,
        want_log=True, observed_indices_mask=observed_mask, device=device
    )

    return log_p_y - log_p_yob


# -------------------------------------------------------
# Synthetic missingness generator (same as before)
# -------------------------------------------------------

class DataMissingMaker:
    """
    Randomly knock out prop_miss_col fraction of columns in prop_miss_rows fraction of rows.
    """
    def __init__(self, complete_only: np.ndarray, prop_miss_rows: float = 1.0, prop_miss_col: float = 0.1):
        self.data = complete_only
        self.n_col = self.data.shape[1]
        self.prop_miss_rows = prop_miss_rows
        self.prop_miss_col = prop_miss_col
        self.n_rows_to_null = int(len(complete_only) * prop_miss_rows)

    def _get_random_col_selection(self):
        n_cols_to_null = np.random.binomial(n=self.n_col, p=self.prop_miss_col)
        return np.random.choice(range(self.n_col), n_cols_to_null, replace=False)

    def generate_missing_data(self) -> np.ndarray:
        random_rows = np.random.choice(range(len(self.data)), self.n_rows_to_null, replace=False)
        null_col_lists = [self._get_random_col_selection() for _ in range(self.n_rows_to_null)]
        null_row_lists = [np.repeat(row, repeats=len(null_col_lists[i])) for i, row in enumerate(random_rows)]

        null_cols = np.array([arr[j] for arr in null_col_lists for j in range(len(arr))])
        null_rows = np.array([arr[j] for arr in null_row_lists for j in range(len(arr))])

        masked = np.copy(self.data)
        masked[null_rows, null_cols] = np.nan
        return masked


# -------------------------------------------------------
# NOTES ON TRAINING WITH β * |KL - C|
# -------------------------------------------------------
#
# For your disentangled β-VAE with capacity C, your loss per batch will look like:
#
#   recon_loss = MSE(x_hat, x)  or BCE(x_hat, x) etc.
#   KL = D_KL( q_phi(z|x) || N(0,I) )  (sum over latent dim, mean over batch)
#
#   loss = recon_loss + beta * torch.abs(KL - C)
#
# You'd implement that in your training loop, not here in the utils.
#
# After training a model, you can:
#  - draw multiple imputations by sampling z ~ q(z|x_missing_filled) many times,
#    decoding, and then only using decoded values at missing indices;
#  - evaluate coverage / MAE with the helpers above.
#
# This file is ready to drop in as disentangledBetaVaeUtil.py
# You will still need to adapt:
#   - model.decoder(z) -> (mean, logvar)
#   - model.encoder(x) -> (z_mean, z_logvar)
#   - model.latent_dim
#   - model.beta
# so they match your pythae DisentangledBetaVAE subclass.
#
# End of module.
# -------------------------------------------------------

