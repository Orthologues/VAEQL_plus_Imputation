#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Description: This file is used to define the configuration classes for the VAE-Q learning imputation baseline (non-cloud baseline solution run at on-premises servers).
#########################################################

from typing import TypedDict
from .types import FeaturesTypeDict

class VaeQlConfig(TypedDict):
    """Configuration for VAE-Q learning imputation baseline."""
    dataset_name: str
    dataset_features: FeaturesTypeDict
    vae_latent_dim: int
    q_learning_alpha: float
    q_learning_gamma: float
    q_learning_epsilon: float
    num_episodes: int
    max_steps_per_episode: int
    

class DisentangledBetaVaeTuningConfig(TypedDict):
    """Configuration for tuning the beta and C parameters of a disentangled VAE."""
    dataset_name: str
    vae_latent_dim: int
    beta_values: list[float]
    num_epochs: int
    batch_size: int
    