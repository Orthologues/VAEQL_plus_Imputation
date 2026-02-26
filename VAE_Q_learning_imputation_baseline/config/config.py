#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Description: This file is used to define the configuration classes for the VAEQL imputation pipeline.
#########################################################

# general imports
import math
from typing import TypedDict, Tuple

# local imports
from .feat_types import FeaturesTypeDict

# pre-training of the disentangled beta-VAE
class DisentangledBetaVaeTuningConfig(TypedDict):
    """Configuration for tuning the beta and C parameters of a disentangled VAE."""

    dataset_name: str
    vae_disc_lat_dim: int # dimension of the discrete latent space S (number of mixed Gaussian distribution components)
    vae_cont_lat_dim: int # dimension of the continuous latent space Z (each composed of the weighted disc_lat_dim Gaussian components)
    beta_min: float
    beta_max: float
    C_min: float
    C_max: float
    max_granularity: float  # between 0 and 0.25, checks if the fine-tuned beta and C values are within this fraction of the original range (beta_min to beta_max, C_min to C_max)
    num_epochs: int  # number of epochs per cross-validation tuning round
    cross_val_rounds: int # number of cross-validation rounds for tuning beta and C (computed from max_granularity as ceil(log2(1 / max_granularity)))
    batch_size: int

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        vae_disc_lat_dim: int,
        vae_cont_lat_dim: int,
        beta_min: float,
        beta_max: float,
        C_min: float,
        C_max: float,
        max_granularity: float,
        num_epochs: int,
        batch_size: int,
    ) -> "DisentangledBetaVaeTuningConfig":
        """Construct a validated config and compute minimum CV rounds.

        Enforces:
        - 0 < beta_min < beta_max
        - 0 <= C_min < C_max
        - 0 < max_granularity <= 0.25

        Computes the minimum cross-validation rounds as
        ``ceil(log2(1 / max_granularity))``.
        """

        if beta_min <= 0 or beta_max <= 0:
            raise ValueError("beta_min and beta_max must be > 0.")
        if beta_min >= beta_max:
            raise ValueError("Require 0 < beta_min < beta_max.")

        if C_min < 0 or C_max < 0:
            raise ValueError("C_min and C_max must be >= 0.")
        if C_min >= C_max:
            raise ValueError("Require 0 <= C_min < C_max.")

        if not (0 < max_granularity <= 0.25):
            raise ValueError("max_granularity must satisfy 0 < value <= 0.25.")

        min_cv_rounds = math.ceil(math.log2(1.0 / max_granularity))
        if min_cv_rounds < 2:
            raise ValueError(
                "Computed minimum cross-validation rounds must be at least two; check max_granularity."
            )

        cfg: "DisentangledBetaVaeTuningConfig" = cls(
            dataset_name=dataset_name,
            vae_disc_lat_dim=vae_disc_lat_dim,
            vae_cont_lat_dim=vae_cont_lat_dim,
            beta_min=beta_min,
            beta_max=beta_max,
            C_min=C_min,
            C_max=C_max,
            max_granularity=max_granularity,
            num_epochs=num_epochs,
            batch_size=batch_size,
            cross_val_rounds=min_cv_rounds
        )

        return cfg


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
    num_of_q_agents: int # number of Q-learning agents to train in parallel (with shared experience replay buffer)
    replay_buffer_size: int # maximum size of the experience replay buffer (must be >= ql_max_time_steps * num_of_q_agents to allow for at least one full episode per agent in the buffer)
    
    
    def _gt_zero(name: str, value: int | float) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be > 0; got {value}.")

    def _in_unit_interval(name: str, value: float, *, inclusive_zero: bool = False) -> None:
        low_ok = value >= 0 if inclusive_zero else value > 0
        if not (low_ok and value <= 1):
            bound = "0 <= x <= 1" if inclusive_zero else "0 < x <= 1"
            raise ValueError(f"{name} must satisfy {bound}; got {value}.")


    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        dataset_features: FeaturesTypeDict,
        vae_disc_lat_dim: int,
        vae_cont_lat_dim: int,
        vae_alpha: float,
        vae_beta: float,
        vae_C: float,
        vae_batch_size: int,
        vae_layers: int,
        vae_layer_size: int,
        vae_max_epochs: int,
        vae_if_Adam: bool,
        ql_alpha: float,
        ql_gamma: float,
        ql_epsilon: float,
        ql_max_episodes: int,
        ql_max_time_steps: int,
        num_of_q_agents: int,
        replay_buffer_size: int,
    ) -> "VaeQlConfig":
        """Construct and validate a VAE-QL pipeline config.

        Basic validation rules:
        - Latent dims, layers, layer_size, batch_size, epochs, episodes, time steps, buffer: strictly > 0
        - Learning rates/coefficients: 0 < vae_alpha, ql_alpha <= 1; 0 < vae_beta; vae_C >= 0
        - Discount factor: 0 < ql_gamma <= 1
        - Epsilon: 0 <= ql_epsilon <= 1
        - Replay Buffer Size: replay_buffer_size >= ql_max_time_steps * num_of_q_agents
        """
        
        # Disentangled beta-VAE hyperparameters
        if vae_beta <= 0:
            raise ValueError(f"vae_beta must be > 0; got {vae_beta}.")
        if vae_C < 0:
            raise ValueError(f"vae_C must be >= 0; got {vae_C}.")

        # Dimensionalities and counts
        for n, v in [
            ("vae_disc_lat_dim", vae_disc_lat_dim),
            ("vae_cont_lat_dim", vae_cont_lat_dim),
            ("vae_layers", vae_layers),
            ("vae_layer_size", vae_layer_size),
            ("vae_batch_size", vae_batch_size),
            ("vae_max_epochs", vae_max_epochs),
            ("ql_max_episodes", ql_max_episodes),
            ("ql_max_time_steps", ql_max_time_steps),
            ("num_of_q_agents", num_of_q_agents),
        ]:
            cls._gt_zero(n, v)
            
        if replay_buffer_size < ql_max_time_steps * num_of_q_agents:
            raise ValueError(f"replay_buffer_size must be >= ql_max_time_steps X num_of_q_agents: {ql_max_time_steps * num_of_q_agents}.")

        # Learning rates / coefficients
        cls._in_unit_interval("vae_alpha", vae_alpha)
        cls._in_unit_interval("ql_alpha", ql_alpha)
        cls._in_unit_interval("ql_gamma", ql_gamma)
        cls._in_unit_interval("ql_epsilon", ql_epsilon, inclusive_zero=True)
        

        cfg: "VaeQlConfig" = cls(
            dataset_name=dataset_name,
            dataset_features=dataset_features,
            vae_disc_lat_dim=vae_disc_lat_dim,
            vae_cont_lat_dim=vae_cont_lat_dim,
            vae_alpha=vae_alpha,
            vae_beta=vae_beta,
            vae_C=vae_C,
            vae_batch_size=vae_batch_size,
            vae_layers=vae_layers,
            vae_layer_size=vae_layer_size,
            vae_max_epochs=vae_max_epochs,
            vae_if_Adam=vae_if_Adam,
            ql_alpha=ql_alpha,
            ql_gamma=ql_gamma,
            ql_epsilon=ql_epsilon,
            ql_max_episodes=ql_max_episodes,
            ql_max_time_steps=ql_max_time_steps,
            num_of_q_agents=num_of_q_agents,
            replay_buffer_size=replay_buffer_size,
        )

        return cfg
    