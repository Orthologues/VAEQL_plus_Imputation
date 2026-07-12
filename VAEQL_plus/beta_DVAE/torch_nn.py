"""
Disentangled Beta-VAE PyTorch neural-network implementation.

This module contains the core Gaussian-mixture disentangled beta-VAE
`torch.nn.Module`, including encoder/decoder architecture, GMM-KL calculation,
beta-capacity loss, and iterative imputation methods.

Author: Jiawei Zhao (jiz@imada.sdu.dk)
Date: 2026-04-01
"""

import math
import re
import numpy as np
from typing import Dict, List, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from VAEQL_plus.conf.feat_types import FeaturesTypeDict
from .utils import BetaGausMixedDVAEUtils, ArrayLike

class BetaGausMixedDVAE(nn.Module):
    """
    Disentangled Gaussian-mixture Beta-VAE core module (PyTorch). Abbreviated as "beta-DVAE" in comments.

    This module owns:
    - encoder and decoder neural networks
    - discrete component bottleneck `K` in the encoder path
    - latent Gaussian heads (`posterior_z_component_mean`, `posterior_z_component_logvar`) conditioned on `K`
    - trainable GMM prior parameters (`pi_k`, `mu_k`, `logvar_k`)

    Shapes:
    - input `x`: `(B, D)` where `B` is batch size and `D` is feature dimension.
    - component logits/probabilities: `(B, K)`.
    - latent tensors: `(B, Z)`.
    - reconstruction logits: `(B, D)` in preprocessed feature space.
    """

    class EncodingOutput(NamedTuple):
        posterior_k_logits: torch.Tensor
        posterior_k_probs: torch.Tensor
        posterior_z_component_mean: torch.Tensor
        posterior_z_component_logvar: torch.Tensor
    
    class KLOutput(NamedTuple):
        kl_total: torch.Tensor
        kl_disc: torch.Tensor
        kl_cont: torch.Tensor
    
    class BetaCapacityLossOutput(NamedTuple):
        total: torch.Tensor
        recon_loss: torch.Tensor
        kl_mean: torch.Tensor
        kl_disc_mean: torch.Tensor
        kl_cont_mean: torch.Tensor
        beta_C_kl: torch.Tensor
        beta: float
        capacity_C: float
    
    class ImputeSingleOutput(NamedTuple):
        imputed_x: torch.Tensor
        losses: List[float]
        losses_mae: List[float] | None
        losses_rmse: List[float] | None

    def __init__(
        self,
        input_dim: int,
        latent_dim: int, # number of continuous latent dimensions, referred as Z
        hidden_dim1: int,
        hidden_dim2: int,
        # The KL-divergence recipe follows Nazabal et al., 2020:
        # https://www.sciencedirect.com/science/article/abs/pii/S0031320320303046
        n_gmm_components: int = 10, # number of discrete GMM prior components, referred as K
        batch_size: int = 64,
        tau_start: float = 0.5, # inferred from the preliminary SMOKE test
        tau_end: float = 0.25,
        anneal_rate: float = 0.01,
        num_feat_loss_metric: str = "RMSE",
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if not (1 < int(n_gmm_components) < 100):
            raise ValueError(
                f"`n_gmm_components` must satisfy 1 < `n_gmm_components` < 100, got {n_gmm_components}"
            )
        self.input_dim = input_dim
        self.z_dim = latent_dim
        self.k_gmm = int(n_gmm_components) # number of Gaussian Mixture prior components
        if float(tau_start) <= 0.0:
            raise ValueError(f"`tau_start` must be > 0, got {tau_start}")
        if float(tau_end) <= 0.0:
            raise ValueError(f"`tau_end` must be > 0, got {tau_end}")
        if float(tau_end) >= float(tau_start):
            raise ValueError(
                f"`tau_end` must be smaller than `tau_start`, got tau_end={tau_end} and tau_start={tau_start}"
            )
        if float(anneal_rate) <= 0.0:
            raise ValueError(f"`anneal_rate` must be > 0, got {anneal_rate}")
        if float(anneal_rate) > 0.1:
            raise ValueError(f"`anneal_rate` must be <= 0.1, got {anneal_rate}")
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.anneal_rate = float(anneal_rate)
        self.gumbel_softmax_tau = float(tau_start)
        self.num_feat_loss_metric = self._normalize_num_feat_loss_metric(num_feat_loss_metric)
        self.batch_size = batch_size

        # vanilla encoder trunk: D -> hidden
        vanilla_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
        )
        # gmm encoder heads: hidden -> K and hidden -> (K, Z), which is a diagonal matrix
        gmm_encoder = nn.ModuleDict({
            "k_logits_head": nn.Linear(hidden_dim2, self.k_gmm),
            "k_mu_head": nn.Linear(hidden_dim2, self.k_gmm * latent_dim),
            "k_logvar_head": nn.Linear(hidden_dim2, self.k_gmm * latent_dim),
        })
        self.encoder = nn.ModuleDict({
            "vanilla_encoder": vanilla_encoder,
            "gmm_encoder": gmm_encoder,
        })

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            # decoder outputs real-valued reconstructions in the preprocessed feature space
        )

        # GMM prior:
        # $p(z)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}\!\left(z\mid\mu_k,\operatorname{diag}(\sigma_k^2)\right)$
        # where K == n_gmm_components.
        self.mixture_logits = nn.Parameter(torch.zeros(self.k_gmm))
        self.mixture_means = nn.Parameter(torch.zeros(self.k_gmm, latent_dim))
        self.mixture_logvars = nn.Parameter(torch.zeros(self.k_gmm, latent_dim))
        resolved_device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(resolved_device)

    def get_gmm_prior_params(self):
        """Return trainable GMM prior parameters as a dict of torch tensors."""
        return {
            "logits": self.mixture_logits,
            "means": self.mixture_means,
            "logvars": self.mixture_logvars,
        }

    @property
    def device(self) -> torch.device:
        """Current device of this module, inferred from model parameters."""
        return self.mixture_logits.device

    def gumbel_softmax_tau_at_epoch(self, epoch: int) -> float:
        """Return exponentially annealed Gumbel-Softmax temperature for a training epoch."""
        if int(epoch) < 0:
            raise ValueError(f"`epoch` must be >= 0, got {epoch}")
        tempered_tau = max(
            self.tau_end,
            self.tau_start * math.exp(-self.anneal_rate * int(epoch)),
        )
        return float(tempered_tau)

    @staticmethod
    def _normalize_num_feat_loss_metric(num_feat_loss_metric: str) -> str:
        """Normalize and validate the numeric-feature reconstruction loss metric."""
        normalized = re.sub(r"[-_\.\s]+", "", str(num_feat_loss_metric)).upper()
        if normalized in {"RMSE", "MAE"}:
            return normalized
        raise ValueError(
            f"`num_feat_loss_metric` must be one of 'RMSE' or 'MAE', got {num_feat_loss_metric!r}"
        )

    # Encoding
    def encode(self, x: torch.Tensor) -> "BetaGausMixedDVAE.EncodingOutput":
        """
        Encode `x` through D -> K -> Z.

        Returns:
        - `posterior_k_logits`: `(B, K)`
        - `posterior_k_probs`: `(B, K)`
        - `posterior_z_component_mean`: `(B, K, Z)` K-conditioned means
        - `posterior_z_component_logvar`: `(B, K, Z)` K-conditioned log-variances
        """
        encoded_head = self.encoder["vanilla_encoder"](x)
        posterior_k_logits = self.encoder["gmm_encoder"]["k_logits_head"](encoded_head)  # (B, K)
        posterior_k_probs = torch.softmax(posterior_k_logits, dim=1)   # (B, K)

        # -1 means “infer its dimension automatically from the total number of elements.”
        k_mu = self.encoder["gmm_encoder"]["k_mu_head"](encoded_head).view(-1, self.k_gmm, self.z_dim)          # (B, K, Z)
        k_logvar = self.encoder["gmm_encoder"]["k_logvar_head"](encoded_head).view(-1, self.k_gmm, self.z_dim)  # (B, K, Z)

        return BetaGausMixedDVAE.EncodingOutput(
            posterior_k_logits=posterior_k_logits,
            posterior_k_probs=posterior_k_probs,
            posterior_z_component_mean=k_mu,
            posterior_z_component_logvar=k_logvar,
        )

    # Reparameterization methods
    def reparameterize_gaussian(
        self,
        gaussian_mean: torch.Tensor,
        gaussian_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from diagonal-Gaussian posterior parameters with the reparameterization trick."""
        std = torch.exp(0.5 * gaussian_logvar)
        eps = torch.randn_like(std)
        return gaussian_mean + eps * std

    def reparameterize_mixture(
        self,
        posterior_z_component_mean: torch.Tensor,
        posterior_z_component_logvar: torch.Tensor,
        posterior_k_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sample component Gaussians and combine them with a differentiable Gumbel-Softmax gate."""
        if posterior_z_component_mean.ndim != 3:
            raise ValueError(
                "`posterior_z_component_mean` must have shape (B, K, Z), "
                f"got {tuple(posterior_z_component_mean.shape)}"
            )
        if posterior_z_component_logvar.shape != posterior_z_component_mean.shape:
            raise ValueError(
                "`posterior_z_component_logvar` must match `posterior_z_component_mean`, "
                f"got {tuple(posterior_z_component_logvar.shape)} and {tuple(posterior_z_component_mean.shape)}"
            )
        batch_size, n_GMM_comps, latent_dim = posterior_z_component_mean.shape
        if n_GMM_comps != self.k_gmm or latent_dim != self.z_dim:
            raise ValueError(
                f"Expected component posterior shape (B, {self.k_gmm}, {self.z_dim}), "
                f"got {tuple(posterior_z_component_mean.shape)}"
            )
        if tuple(posterior_k_logits.shape) != (batch_size, n_GMM_comps):
            raise ValueError(
                f"`posterior_k_logits` must have shape {(batch_size, n_GMM_comps)}, "
                f"got {tuple(posterior_k_logits.shape)}"
            )

        relaxed_k = F.gumbel_softmax(
            posterior_k_logits,
            tau=self.gumbel_softmax_tau,
            # `hard=False` means the sampled Gumbel-Softmax vector is continuous/relaxed, not one-hot
            hard=False,
            dim=1,
        ).unsqueeze(-1)  # (B, K, 1)
        component_z = self.reparameterize_gaussian(
            posterior_z_component_mean,
            posterior_z_component_logvar,
        )  # (B, K, Z)
        # Weighted component sum removes the K axis and returns one latent sample per row: (B, Z).
        return torch.sum(relaxed_k * component_z, dim=1)

    # Decoding and Reconstruction
    def decode(self, posterior_z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples `z` into reconstruction logits in feature space."""
        return self.decoder(posterior_z)

    @staticmethod
    def _ordered_ordinal_logits(ordinal_raw: torch.Tensor) -> torch.Tensor:
        """
        Convert one ordinal raw decoder group into ordered cumulative logits.

        Monotone-logit flow:
            z -> decoder -> ordinal_raw
            ordinal_raw[:, :1] -> eta(z)
            ordinal_raw[:, 1:] -> raw logit gaps
            softplus(raw logit gaps) -> positive gaps
            eta(z) - cumsum(positive gaps) -> ordered logits for later `feat-ge_*`

        Thus logit_1 >= logit_2 >= ... and sigmoid preserves:
            P(Y >= 1 | z) >= P(Y >= 2 | z) >= ...
        """
        # Expected shape is (B, R-1): one row per batch item and one column per
        # unary ordinal threshold (`feat-ge_1`, `feat-ge_2`, ...).
        if ordinal_raw.ndim != 2:
            raise ValueError(
                f"`ordinal_raw` must have shape (B, R-1), got {tuple(ordinal_raw.shape)}"
            )
        if ordinal_raw.shape[1] < 1:
            raise ValueError(
                f"`ordinal_raw` must include at least one ordinal threshold column, got {tuple(ordinal_raw.shape)}"
            )
        # eta(z) is the decoder-derived base logit for the first unary ordinal
        # threshold; later threshold logits are obtained by subtracting positive gaps.
        eta = ordinal_raw[:, 0].unsqueeze(1)
        if ordinal_raw.shape[1] == 1:
            return eta
        positive_gaps = F.softplus(ordinal_raw[:, 1:])
        ordered_tail_logits = eta - torch.cumsum(positive_gaps, dim=1)
        return torch.cat((eta, ordered_tail_logits), dim=1)

    # To implement the hybrid type-aware reconstruction objective:
    # transformed-space numeric surrogates plus grouped discrete heads/losses.
    def activate_reconstruction(
        self,
        recon_logits: torch.Tensor,
        feat_type_dict: FeaturesTypeDict | None,
    ) -> torch.Tensor:
        """
        Map decoder logits to the valid preprocessed reconstruction space.

        Numeric transformed columns remain real-valued. Binary columns are
        sigmoid-activated, ordinal unary groups use monotone cumulative logits
        parameterized by positive softplus gaps, and categorical one-hot groups
        are stochastically sampled with Gumbel-Softmax using the current scheduled
        temperature.
        """
        if feat_type_dict is None:
            return recon_logits

        recon = recon_logits.clone()
        D = recon.shape[1]
        ordered_feat_names_raw = feat_type_dict["all_feats"]
        if isinstance(ordered_feat_names_raw, set):
            ordered_feat_names = sorted(str(n) for n in ordered_feat_names_raw)
        else:
            ordered_feat_names = [str(n) for n in ordered_feat_names_raw]
        if len(ordered_feat_names) != D:
            raise ValueError(
                f"Feature dimension mismatch: recon dim D={D}, but `len(all_feats)`={len(ordered_feat_names)}."
            )
        name_to_index = {name: idx for idx, name in enumerate(ordered_feat_names)}

        for feat, n_orders in feat_type_dict.get("ord_feats", {}).items():
            # Ordinal preprocessing expands one source feature into threshold
            # columns such as `feat-ge_1`, `feat-ge_2`, ... . Activate each
            # group together so the cumulative probabilities stay monotone.
            ordinal_col_names = [
                f"{feat}-ge_{order}" for order in range(1, int(n_orders))
            ]
            grp_idx = [
                name_to_index[col_name]
                for col_name in ordinal_col_names
                if col_name in name_to_index
            ]
            if grp_idx:
                ordinal_logits = self._ordered_ordinal_logits(recon_logits[:, grp_idx])
                recon[:, grp_idx] = torch.sigmoid(ordinal_logits)
        for feat in feat_type_dict.get("bi_feats", {}).keys():
            if feat in name_to_index:
                recon[:, name_to_index[feat]] = torch.sigmoid(recon_logits[:, name_to_index[feat]])
        for feat, values in feat_type_dict.get("cat_feats", {}).items():
            grp_idx = [
                name_to_index[col_name]
                for col_name in [f"{feat}-is_{value}" for value in sorted(values)]
                if col_name in name_to_index
            ]
            if grp_idx:
                recon[:, grp_idx] = F.gumbel_softmax(
                    recon_logits[:, grp_idx],
                    tau=self.gumbel_softmax_tau,
                    hard=False,
                    dim=1,
                )
        return recon

    # Implementation of the mandatory `forward` method in a `torch.nn` module
    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
        - `recon_logits`: `(B, D)`
        - `posterior_k_logits`: `(B, K)`
        - `posterior_k_probs`: `(B, K)`
        - `posterior_z_component_mean`: `(B, K, Z)`
        - `posterior_z_component_logvar`: `(B, K, Z)`
        """
        encoded = self.encode(x)
        z = self.reparameterize_mixture(
            encoded.posterior_z_component_mean,
            encoded.posterior_z_component_logvar,
            encoded.posterior_k_logits,
        )
        recon_logits = self.decode(z)
        return {
            "recon_logits": recon_logits,
            "posterior_k_logits": encoded.posterior_k_logits,
            "posterior_k_probs": encoded.posterior_k_probs,
            "posterior_z_component_mean": encoded.posterior_z_component_mean,
            "posterior_z_component_logvar": encoded.posterior_z_component_logvar,
        }


    # Recipes of the Loss Function (K-component-wise KL divergence Plus the Disentangled-Beta-tuned Reconstruction Loss)
    def gmm_kl_decomposed(
        self,
        posterior_z_component_mean: torch.Tensor,
        posterior_z_component_logvar: torch.Tensor,
        posterior_k_probs: torch.Tensor,
        prior_k_logits: torch.Tensor,
        prior_z_mixture_means: torch.Tensor,
        prior_z_mixture_logvars: torch.Tensor,
        eps: float = 1e-12,
    ) -> "BetaGausMixedDVAE.KLOutput":
        """
        Decompose KL(q(z,k|x) || p(z,k)) into discrete and continuous terms.

        Inputs:
        - `posterior_z_component_mean`: `(B, K, Z)` component posterior means from encoder.
        - `posterior_z_component_logvar`: `(B, K, Z)` component posterior log-variances from encoder.
        - `posterior_k_probs`: `(B, K)` posterior probabilities over GMM components.
        - `prior_k_logits`: `(K,)` prior logits over mixture components.
        - `prior_z_mixture_means`: `(K, Z)` prior component means.
        - `prior_z_mixture_logvars`: `(K, Z)` prior component log-variances.
        - `eps`: positive scalar for numerical stability.

        Returns:
        - `KLOutput` with:
          - `kl_total`: `(B,)`
          - `kl_disc`: `(B,)`
          - `kl_cont`: `(B,)`
        """
        if posterior_z_component_mean.ndim == 3:
            batch_size, n_GMM_comps, latent_dim = posterior_z_component_mean.shape
            if n_GMM_comps != self.k_gmm:
                raise ValueError(f"`posterior_z_component_mean` K mismatch: expected {self.k_gmm}, got {n_GMM_comps}")
            if latent_dim != self.z_dim:
                raise ValueError(f"`posterior_z_component_mean` Z mismatch: expected {self.z_dim}, got {latent_dim}")
        else:
            raise ValueError(f"`posterior_z_component_mean` must have three dimensions, got {posterior_z_component_mean.ndim}")

        if posterior_k_probs.ndim == 2:
            probs_batch_size, n_GMM_comps = posterior_k_probs.shape
            if n_GMM_comps != self.k_gmm:
                raise ValueError(f"`posterior_k_probs` K mismatch: expected {self.k_gmm}, got {n_GMM_comps}")
            if probs_batch_size != batch_size:
                raise ValueError(f"`posterior_k_probs` B mismatch: expected {batch_size}, got {probs_batch_size}")
        else:
            raise ValueError(f"`posterior_k_probs` must have two dimensions, got {posterior_k_probs.ndim}")

        if prior_k_logits.ndim == 1:
            if tuple(prior_k_logits.shape) != (self.k_gmm,):
                raise ValueError(f"`prior_k_logits` must have shape {(self.k_gmm,)}, got {tuple(prior_k_logits.shape)}")
        else:
            raise ValueError(f"`prior_k_logits` must have one dimension, got {prior_k_logits.ndim}")

        if posterior_z_component_logvar.ndim == 3:
            if tuple(posterior_z_component_logvar.shape) != (batch_size, self.k_gmm, self.z_dim):
                raise ValueError(
                    f"`posterior_z_component_logvar` must have shape {(batch_size, self.k_gmm, self.z_dim)}, got {tuple(posterior_z_component_logvar.shape)}"
                )
        else:
            raise ValueError(f"`posterior_z_component_logvar` must have three dimensions, got {posterior_z_component_logvar.ndim}")

        if prior_z_mixture_means.ndim == 2:
            if tuple(prior_z_mixture_means.shape) != (self.k_gmm, self.z_dim):
                raise ValueError(
                    f"`prior_z_mixture_means` must have shape {(self.k_gmm, self.z_dim)}, got {tuple(prior_z_mixture_means.shape)}"
                )
        else:
            raise ValueError(
                f"`prior_z_mixture_means` must have two dimensions, got {prior_z_mixture_means.ndim}"
            )

        if prior_z_mixture_logvars.ndim == 2:
            if tuple(prior_z_mixture_logvars.shape) != (self.k_gmm, self.z_dim):
                raise ValueError(
                    f"`prior_z_mixture_logvars` must have shape {(self.k_gmm, self.z_dim)}, got {tuple(prior_z_mixture_logvars.shape)}"
                )
        else:
            raise ValueError(
                f"`prior_z_mixture_logvars` must have two dimensions, got {prior_z_mixture_logvars.ndim}"
            )

        if eps <= 0.0:
            raise ValueError(f"`eps` must be > 0, got {eps}")

        log_prior_k_probs = torch.log_softmax(prior_k_logits, dim=0)  # (K,)
        prior_z_mixture_var = torch.exp(prior_z_mixture_logvars) + eps  # (K, Z)
        posterior_z_var = torch.exp(posterior_z_component_logvar)  # (B, K, Z)
        posterior_k_probs = posterior_k_probs.clamp_min(eps) # Clamp to avoid log(0)
        posterior_k_probs = posterior_k_probs / posterior_k_probs.sum(dim=1, keepdim=True)
        log_posterior_k_probs = torch.log(posterior_k_probs) # (B, K)

        kl_disc = torch.sum(posterior_k_probs * (log_posterior_k_probs - log_prior_k_probs.unsqueeze(0)), dim=1)  # (B,)
        delta_sqr = (
            posterior_z_component_mean - prior_z_mixture_means.unsqueeze(0)
        ).pow(2)  # (B, K, Z)
        kl_z_mixture = 0.5 * torch.sum(
            prior_z_mixture_logvars.unsqueeze(0)
            - posterior_z_component_logvar
            + (posterior_z_var + delta_sqr) / prior_z_mixture_var.unsqueeze(0)
            - 1.0,
            dim=2,
        )  # (B, K)
        kl_cont = torch.sum(posterior_k_probs * kl_z_mixture, dim=1)  # (B,)
        kl_total = kl_disc + kl_cont  # (B,)
        return BetaGausMixedDVAE.KLOutput(
            kl_total=kl_total,
            kl_disc=kl_disc,
            kl_cont=kl_cont,
        )
        
    # Training loss intentionally consumes raw decoder logits:
    # BCE-with-logits and Categorical Cross-Entropy are numerically stable on logits,
    # while numeric features reconstruct standardized values directly.
    # `activate_reconstruction` is reserved for imputation/evaluation outputs.
    def beta_capacity_loss(
        self,
        recon_logits: torch.Tensor,
        x_obs_processed: torch.Tensor,
        posterior_z_component_mean: torch.Tensor,
        posterior_z_component_logvar: torch.Tensor,
        posterior_k_probs: torch.Tensor,
        beta: float,
        capacity_C: float,
        gmm_prior: dict,
        feat_type_dict: FeaturesTypeDict,
        obs_mask: ArrayLike,
        eps: float = 1e-12,
    ) -> "BetaGausMixedDVAE.BetaCapacityLossOutput":
        """
        Compute the beta-capacity loss with a hybrid type-aware reconstruction objective.

        Inputs:
        - `recon_logits`: `(B, D)` decoder outputs in preprocessed feature space.
        - `x_obs_processed`: `(B, D)` processed targets aligned with `recon_logits`.
        - `posterior_z_component_mean`, `posterior_z_component_logvar`: `(B, K, Z)` component Gaussian posterior parameters.
        - `posterior_k_probs`: `(B, K)` posterior probabilities over GMM components.
        - `beta`, `capacity_C`: KL-capacity objective hyperparameters.
        - `gmm_prior`: dict with keys `logits`, `means`, `logvars` for prior parameters.
        - `feat_type_dict`: feature-type metadata including `all_feats` and type groups.
        - `obs_mask`: `(B, D)` mask; `0` denotes observed entries used for reconstruction weighting.
        - `eps`: positive scalar for numerical stability in weighted reductions.

        Behavior:
        - reconstruction loss weighting uses only type-0 cells (`obs_mask == 0`),
        - uses transformed-space RMSE/MAE on numeric/count-style columns,
        - uses BCE-with-logits on binary columns and monotone ordinal cumulative logits,
        - uses categorical CE on each categorical one-hot group with per-row observed-weighting,
        - combines reconstruction with KL-capacity term:
          `total = recon_loss + beta * |mean(KL) - C|`.

        Returns:
        - `BetaCapacityLossOutput` with `total`, `recon_loss`, `kl_mean`, `beta_C_kl`, `beta`, and `capacity_C`.
        """
        B, D = recon_logits.shape
        device = recon_logits.device
        obs_mask_t = (
            obs_mask.to(device=device)
            if isinstance(obs_mask, torch.Tensor)
            else torch.tensor(obs_mask, device=device)
        )
        if obs_mask_t.shape != recon_logits.shape:
            raise ValueError(
                f"`obs_mask` must have shape {(B, D)}, got {tuple(obs_mask_t.shape)}"
            )
        x_obs_processed_t = x_obs_processed.to(device=device, dtype=recon_logits.dtype)
        if x_obs_processed_t.shape != recon_logits.shape:
            raise ValueError(
                f"`x_obs_processed` must have shape {(B, D)}, got {tuple(x_obs_processed_t.shape)}"
            )
        obs_mask_unique_vals = set(int(v) for v in torch.unique(obs_mask_t).detach().tolist())
        if not obs_mask_unique_vals.issubset({0, 1, 2}):
            raise ValueError(
                f"`obs_mask` can only contain values from {{0, 1, 2}}, got {sorted(obs_mask_unique_vals)}"
            )
        if 0 not in obs_mask_unique_vals:
            raise ValueError("`obs_mask` must contain value 0 for reconstruction-loss weighting.")
        
        num_feat_loss_type = self.num_feat_loss_metric
        
        ord_feats = feat_type_dict.get("ord_feats", {})
        bi_feats = feat_type_dict.get("bi_feats", {})
        cat_feats = feat_type_dict.get("cat_feats", {})
        ordered_feat_names_raw = feat_type_dict["all_feats"]
        if not isinstance(ordered_feat_names_raw, (set, list, tuple)):
            raise ValueError(
                "`feat_type_dict['all_feats']` must be an iterable of preprocessed feature names."
            )
        if isinstance(ordered_feat_names_raw, set):
            ordered_feat_names = sorted(str(n) for n in ordered_feat_names_raw)
        else: # $ordered_feat_names is a List or Tuple
            ordered_feat_names = [str(n) for n in ordered_feat_names_raw]
        if len(ordered_feat_names) != D:
            raise ValueError(
                f"Feature dimension mismatch: recon dim D={D}, but `len(all_feats)`={len(ordered_feat_names)}."
            )
        name_to_index = {name: idx for idx, name in enumerate(ordered_feat_names)}

        loss_obs_mask = (obs_mask_t == 0).to(dtype=recon_logits.dtype) # True -> 1.0, False -> 0.0
        loss_terms: List[torch.Tensor] = []
        encoded_feature_indices: set[int] = set()

        def append_binary_cross_entropy_group_loss(
            indices: Tuple[int, ...],
            logits_override: torch.Tensor | None = None,
        ) -> None:
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            logits_grp = (
                logits_override
                if logits_override is not None
                else recon_logits.index_select(1, idx)
            )
            target_grp = x_obs_processed_t.index_select(1, idx).clamp(0.0, 1.0)
            obs_grp = loss_obs_mask.index_select(1, idx).bool()
            bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_grp,
                target_grp,
                reduction="none",
            )
            if bool(obs_grp.any().item()):
                loss_terms.append(bce_raw[obs_grp].mean())

        def append_cross_entropy_group_loss(indices: Tuple[int, ...]) -> None:
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            logits_grp = recon_logits.index_select(1, idx)
            target_grp = x_obs_processed_t.index_select(1, idx).clamp(0.0, 1.0)
            ce_grp = -(target_grp * torch.log_softmax(logits_grp, dim=1)).sum(dim=1)
            row_type0_mask = loss_obs_mask.index_select(1, idx).all(dim=1).bool()
            if bool(row_type0_mask.any().item()):
                loss_terms.append(ce_grp[row_type0_mask].mean())

        # Treat each original ordinal feature as one loss unit, no matter how many
        # unary threshold columns it expands into. The raw decoder group is first
        # converted into ordered logits through positive softplus gaps.
        for feat in sorted(ord_feats.keys()):
            n_orders = int(ord_feats[feat])
            grp_idx = tuple(
                name_to_index[f"{feat}-ge_{order}"]
                for order in range(1, n_orders)
            )
            encoded_feature_indices.update(grp_idx)
            ordinal_logits = self._ordered_ordinal_logits(recon_logits[:, grp_idx])
            append_binary_cross_entropy_group_loss(grp_idx, ordinal_logits)

        # Binary features are already one original feature per encoded column.
        for feat in sorted(bi_feats.keys()):
            feat_idx = name_to_index[feat]
            encoded_feature_indices.add(feat_idx)
            append_binary_cross_entropy_group_loss((feat_idx,))

        # Treat each original categorical feature as one loss unit, no matter how
        # many one-hot category columns it expands into.
        for feat in sorted(cat_feats.keys()):
            grp_idx = tuple(
                name_to_index[f"{feat}-is_{value}"]
                for value in sorted(cat_feats[feat])
            )
            encoded_feature_indices.update(grp_idx)
            append_cross_entropy_group_loss(grp_idx)

        # Remaining columns are numeric features in preprocessed space. Each numeric
        # feature contributes one loss unit.
        numeric_indices = [
            idx for idx in range(D) if idx not in encoded_feature_indices
        ]
        for idx in numeric_indices:
            type0_mask = loss_obs_mask[:, idx].bool()
            if not bool(type0_mask.any().item()):
                continue
            abs_err_raw = torch.abs(recon_logits[:, idx] - x_obs_processed_t[:, idx])
            sqr_err_raw = abs_err_raw.pow(2)
            # Statistical coefficient under standardized Gaussian assumption:
            # larger |z| gets smaller weight via exp(-0.5 * z^2).
            # References:
            #   1. Bishop, Pattern Recognition and Machine Learning (2006), Gaussian density form.
            #   2. Murphy, Machine Learning: A Probabilistic Perspective (2012), Normal distribution and quadratic exponent.
            z_score_disc_coef = torch.exp(-0.5 * x_obs_processed_t[:, idx].pow(2))
            mse_vals = (sqr_err_raw * z_score_disc_coef)[type0_mask]
            mae_vals = (abs_err_raw * z_score_disc_coef)[type0_mask]
            weighted_mse = mse_vals.mean()
            weighted_mae = mae_vals.mean()
            loss_terms.append(
                torch.sqrt(weighted_mse + eps)
                if num_feat_loss_type == "RMSE"
                else weighted_mae
            )

        if not loss_terms:
            raise ValueError("No observed feature groups were available for reconstruction loss.")
        # hybrid type-aware reconstruction objective: ordinal/categorical expansions are
        # averaged inside their original feature group before this final average.
        recon_loss = torch.stack(loss_terms).mean()

        kl_output = self.gmm_kl_decomposed(
            posterior_z_component_mean=posterior_z_component_mean,
            posterior_z_component_logvar=posterior_z_component_logvar,
            posterior_k_probs=posterior_k_probs,
            prior_k_logits=gmm_prior["logits"],
            prior_z_mixture_means=gmm_prior["means"],
            prior_z_mixture_logvars=gmm_prior["logvars"],
            eps=eps,
        )
        #  kl_total is per-sample with a shape of (B,)
        kl_mean = kl_output.kl_total.mean()
        kl_disc_mean = kl_output.kl_disc.mean()
        kl_cont_mean = kl_output.kl_cont.mean()
        beta_C_kl = beta * torch.abs(kl_mean - capacity_C)
        total = recon_loss + beta_C_kl
        return BetaGausMixedDVAE.BetaCapacityLossOutput(
            total=total,
            recon_loss=recon_loss.detach(),
            kl_mean=kl_mean.detach(),
            kl_disc_mean=kl_disc_mean.detach(),
            kl_cont_mean=kl_cont_mean.detach(),
            beta_C_kl=beta_C_kl.detach(),
            beta=float(beta),
            capacity_C=float(capacity_C),
        )


    # ---------- these two must be implemented for MI ----------
    def impute_single(
        self,
        X_incomplete: ArrayLike,
        n_cycles: int,
        loss_option: str,
        X_mask: ArrayLike,
        feat_type_dict: FeaturesTypeDict | None = None,
    ) -> "BetaGausMixedDVAE.ImputeSingleOutput":
        """
        Perform one iterative stochastic imputation pass.

        Inputs:
        - `X_incomplete`: `(N, D)` pre-imputed input with no NaN values at `VAEQL_plus.step1_preprocessing.feat_preprocessor`
        - `X_mask`: `(N, D)` mask with semantics:
          `0` = observed after amputation,
          `1` = pre-existing NaN values in raw data,
          `2` = newly amputated values.
        - `n_cycles`: number of iterative refresh cycles (min effectively 5).
        - `loss_type`: `"MAE"`, `"RMSE"`, or `"BOTH"` for tracked missing-cell error.

        Behavior:
        - expects `X_incomplete` to be already pre-imputed (no NaN values allowed)
        - repeatedly run model forward (with stochastic latent sampling)
        - for each cycle, compute reconstruction loss on type-0 cells using `X_incomplete` as reference
        - reset type-0 cells to their original observed values after each cycle loss is computed
        - keep type-1 (pre-existing missing, already pre-imputed) cells unchanged
        - retain iterative updates only for type-2 cells (`X_mask == 2`)
        - use `X_mask` to identify type-0 cells for per-cycle error tracking and type-2 cells for updates.

        Returns:
        - imputed matrix `temp_x_t` of shape `(N, D)` as torch Tensor on model device
        - `losses_list`: list of float errors per cycle (or empty when disabled)
        """
        device = self.device
        x_incomplete_t = (
            X_incomplete.to(device=device, dtype=torch.float32)
            if isinstance(X_incomplete, torch.Tensor)
            else torch.tensor(X_incomplete, dtype=torch.float32, device=device)
        )
        x_mask_t = (
            X_mask.to(device=device, dtype=torch.int64)
            if isinstance(X_mask, torch.Tensor)
            else torch.tensor(X_mask, dtype=torch.int64, device=device)
        )
        if x_incomplete_t.shape != x_mask_t.shape:
            raise ValueError(
                f"Shape mismatch: `X_incomplete` {tuple(x_incomplete_t.shape)}, "
                f"`X_mask` {tuple(x_mask_t.shape)}"
            )
        BetaGausMixedDVAEUtils.mask_type2_indices(x_mask_t, tuple(x_incomplete_t.shape))

        if bool(torch.isnan(x_incomplete_t).any().item()):
            raise ValueError("`X_incomplete` must be pre-imputed and contain no NaN values in `impute_single`.")
        amputated_mask_type2 = (x_mask_t == 2)
        if not bool(amputated_mask_type2.any().item()):
            raise ValueError("`X_mask` contains no type-2 entries; `impute_single` has no cells to update.")
        eval_mask_type0 = (x_mask_t == 0)
        if int(n_cycles) < 5 or int(n_cycles) > 100:
            raise ValueError(f"`n_cycles` must be <= 100 and >= 5, got {n_cycles}")

        loss_type = re.sub(r"[-_\.\s]+", "", str(loss_option)).upper()
        if loss_type not in {"MAE", "RMSE", "BOTH"}:
            raise ValueError(f"Unsupported `loss_option` '{loss_type}'. Expected one of: 'MAE', 'RMSE', 'BOTH'.")

        # stores the temporary imputed matrix as we iteratively refresh masked entries with reconstructions
        temp_x_t = x_incomplete_t.clone()
        eval_mask = eval_mask_type0 & torch.isfinite(x_incomplete_t)
        can_track_supervised_loss = bool(eval_mask.any().item())
        if not can_track_supervised_loss:
            raise ValueError(
                "Cannot track supervised loss: no evaluable cells found where "
                "(`X_mask` == 0) & isfinite(`X_incomplete`)."
            )

        losses_list: List[float] | None = [] if not loss_type == "BOTH" else None
        losses_mae_list: List[float] | None = [] if loss_type == "BOTH" else None
        losses_rmse_list: List[float] | None = [] if loss_type == "BOTH" else None
        # `self.training` is a built-in nn.Module flag toggled by `.train()` / `.eval()`.
        # Save and restore it so this method does not alter the caller's mode.
        was_training = self.training
        self.eval()
        try:
            for cycle_idx in range(n_cycles):
                try:
                    # since we do not apply gradient descents here, forward can be run under `torch.no_grad()` context to save memory
                    with torch.no_grad():
                        # `temp_x_t` is already pre-imputed, so forward pass can proceed without NaN issues
                        output_t = self.forward(temp_x_t) 
                        recon = self.activate_reconstruction(output_t["recon_logits"], feat_type_dict)

                    pred_vals = recon[eval_mask]
                    true_vals = x_incomplete_t[eval_mask]
                    mae_error = float(torch.mean(torch.abs(pred_vals - true_vals)).item())
                    rmse_error = float(torch.sqrt(torch.mean((pred_vals - true_vals) ** 2)).item())
                    if loss_type == "MAE":
                        assert losses_list is not None
                        losses_list.append(mae_error)
                    elif loss_type == "RMSE":
                        assert losses_list is not None
                        losses_list.append(rmse_error)
                    else:
                        assert losses_mae_list is not None and losses_rmse_list is not None
                        losses_mae_list.append(mae_error)
                        losses_rmse_list.append(rmse_error)
                    # After cycle loss: reset type-0 to `X_incomplete`, keep type-1 unchanged,
                    # and retain updates only on type-2 cells.
                    temp_x_t = torch.where(eval_mask_type0, x_incomplete_t, temp_x_t)
                    temp_x_t = torch.where(amputated_mask_type2, recon, temp_x_t)
                except Exception as exc:
                    raise RuntimeError(
                        f"`impute_single` failed at cycle {cycle_idx + 1}/{n_cycles}"
                    ) from exc
        finally:
            self.train(was_training)

        return BetaGausMixedDVAE.ImputeSingleOutput(
            imputed_x=temp_x_t.detach(),
            losses=losses_list,
            losses_mae=losses_mae_list,
            losses_rmse=losses_rmse_list,
        )


    def impute_multiple(
        self,
        X_incomplete: ArrayLike,
        max_iter: int,
        X_mask: ArrayLike,
        feat_type_dict: FeaturesTypeDict | None = None,
    ) -> torch.Tensor:
        """
        Produce one iterative stochastic imputation sample.

        Inputs:
        - `X_incomplete`: `(N, D)` pre-imputed input with no NaN values at `VAEQL_plus.step1_preprocessing.feat_preprocessor`.
        - `max_iter`: number of iterative update cycles; must satisfy `2 <= max_iter <= 20`.
        - `X_mask`: `(N, D)` mask with semantics:
          `0` = observed after amputation,
          `1` = pre-existing NaN values in raw data,
          `2` = newly amputated values.

        Behavior:
        - expects `X_incomplete` to be already pre-imputed (no NaN values allowed)
        - repeatedly run model forward (with stochastic latent sampling)
        - keep type-0 and type-1 cells fixed at their `X_incomplete` values
        - retain iterative updates only for type-2 cells (`X_mask == 2`)

        Returns:
        - imputed matrix of shape `(N, D)` as torch Tensor on model device
        """
        device = self.device
        x_incomplete_t = (
            X_incomplete.to(device=device, dtype=torch.float32)
            if isinstance(X_incomplete, torch.Tensor)
            else torch.tensor(X_incomplete, dtype=torch.float32, device=device)
        )
        if int(max_iter) < 2 or int(max_iter) > 20:
            raise ValueError(f"`max_iter` must satisfy 2 <= `max_iter` <= 20, got {max_iter}")
        x_mask_t = (
            X_mask.to(device=device, dtype=torch.int64)
            if isinstance(X_mask, torch.Tensor)
            else torch.tensor(X_mask, dtype=torch.int64, device=device)
        )
        if x_incomplete_t.shape != x_mask_t.shape:
            raise ValueError(
                f"Shape mismatch: `X_incomplete` {tuple(x_incomplete_t.shape)}, "
                f"`X_mask` {tuple(x_mask_t.shape)}"
            )
        BetaGausMixedDVAEUtils.mask_type2_indices(x_mask_t, tuple(x_incomplete_t.shape))
        if bool(torch.isnan(x_incomplete_t).any().item()):
            raise ValueError("`X_incomplete` must be pre-imputed and contain no NaN values in `impute_multiple`.")

        update_mask = (x_mask_t == 2)
        temp_x_t = x_incomplete_t.clone()

        was_training = self.training
        self.eval()
        try:
            for iter_idx in range(max_iter):
                try:
                    with torch.no_grad():
                        output_t = self.forward(temp_x_t)
                        recon = self.activate_reconstruction(output_t["recon_logits"], feat_type_dict)
                    temp_x_t = torch.where(update_mask, recon, x_incomplete_t)
                except Exception as exc:
                    raise RuntimeError(
                        f"`impute_multiple` failed at iteration {iter_idx + 1}/{max_iter}"
                    ) from exc
        finally:
            self.train(was_training)

        return temp_x_t.detach()
