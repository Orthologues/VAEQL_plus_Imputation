#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Description: This file is used to define the configuration classes for the VAEQL imputation pipeline.
#########################################################

from typing import TypedDict, Tuple

from .feat_types import FeaturesTypeDict

# pre-training of the disentangled beta-VAE
class DisentangledBetaVaeTuningConfig(TypedDict):
    """Configuration for tuning the beta and C parameters of a disentangled VAE."""

    dataset_name: str
    vae_latent_dim: int
    beta_min: float
    beta_max: float
    C_min: float
    C_max: float
    max_granularity: float  # between 0 and 0.5, checks if the fine-tuned beta and C values are within this fraction of the original range (beta_min to beta_max, C_min to C_max)
    num_epochs: int  # number of epochs per cross-validation tuning round
    batch_size: int

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        vae_latent_dim: int,
        beta_min: float,
        beta_max: float,
        C_min: float,
        C_max: float,
        max_granularity: float,
        num_epochs: int,
        batch_size: int,
    ) -> Tuple["DisentangledBetaVaeTuningConfig", int]:
        """Construct a validated config and compute minimum CV rounds.

        Enforces:
        - 0 < beta_min < beta_max
        - 0 <= C_min < C_max
        - 0 < max_granularity <= 0.5

        Computes the minimum cross-validation rounds as
        ``ceil(log2(1 / max_granularity))``.
        """

        import math

        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be > 0.")
        if beta_min >= beta_max:
            raise ValueError("Require 0 < beta_min < beta_max.")

        if C_min < 0 or C_max < 0:
            raise ValueError("C_min and C_max must be >= 0.")
        if C_min >= C_max:
            raise ValueError("Require 0 <= C_min < C_max.")

        if not (0 < max_granularity <= 0.5):
            raise ValueError("max_granularity must satisfy 0 < value <= 0.5.")

        min_cv_rounds = math.ceil(math.log2(1.0 / max_granularity))
        if min_cv_rounds <= 0:
            raise ValueError(
                "Computed minimum cross-validation rounds must be positive; check max_granularity."
            )

        cfg: "DisentangledBetaVaeTuningConfig" = cls(
            dataset_name=dataset_name,
            vae_latent_dim=vae_latent_dim,
            beta_min=beta_min,
            beta_max=beta_max,
            C_min=C_min,
            C_max=C_max,
            max_granularity=max_granularity,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        return cfg, min_cv_rounds


# tranining of the dientangled beta-VAEQL imputation pipeline
class VaeQlConfig(TypedDict):
    """Configuration for VAE-Q learning imputation baseline."""
    # Dataset configs
    dataset_name: str
    dataset_features: FeaturesTypeDict
    # VAE configs
    vae_disc_lat_dim: int # dimension of the discrete latent space S (number of mixed Gaussian distribution components)
    vae_cont_lat_dim: int # dimension of the continuous latent space Z (each composed of the weighted disc_lat_dim Gaussian components)
    vae_alpha: float # learning rate for the VAE training
    vae_beta: float # beta parameter for the KL term of the VAE loss (reconstruction error + KL divergence)
    vae_C: float # capacity parameter for the KL term of the VAE loss (reconstruction error + KL divergence)
    vae_batch_size: int
    vae_layers: int # number of layers for the VAE encoder and decoder (symmetric architecture)
    vae_layer_size: int # number of neurons for each layer in the VAE encoder and decoder (symmetric architecture)
    vae_max_epochs: int # maximum number of epochs for the VAE training
    vae_if_Adam: bool # whether to use the Adam optimizer for the VAE training (if False, will use vanilla SGD)
    # Q-learning configs
    ql_alpha: float # learning rate for the Q-value update
    ql_gamma: float # discount factor for the max Q-value of the transition's destination state
    ql_epsilon: float # parameter for the epsilon-greedy policy (off-policy exploration)
    ql_max_episodes: int # maximum number of episodes for the Q-learning training
    ql_max_time_steps: int # maximum number of time steps per episode for the Q-learning training
    # shared configs
    replay_buffer_size: int
    
    