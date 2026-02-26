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

# Defaults for the VAE-QL pipeline (excluding dataset-specific values)
VAE_DISC_LAT_DIM = 10
VAE_CONT_LAT_DIM = 10
VAE_ALPHA = 1e-3
VAE_BETA = 4.0
VAE_C = 0.0
VAE_BATCH_SIZE = 128
VAE_LAYERS = 2
VAE_LAYER_SIZE = 256
VAE_MAX_EPOCHS = 100
VAE_IF_ADAM = True
QL_ALPHA = 0.1
QL_GAMMA = 0.9
QL_EPSILON = 0.1
QL_MAX_EPISODES = 1000
QL_MAX_TIME_STEPS = 100
NUM_OF_Q_AGENTS = 4
REPLAY_BUFFER_SIZE = 10000

DEFAULT_VAEQL_CONFIG: dict = {
    "vae_disc_lat_dim": VAE_DISC_LAT_DIM,
    "vae_cont_lat_dim": VAE_CONT_LAT_DIM,
    "vae_alpha": VAE_ALPHA,
    "vae_beta": VAE_BETA,
    "vae_C": VAE_C,
    "vae_batch_size": VAE_BATCH_SIZE,
    "vae_layers": VAE_LAYERS,
    "vae_layer_size": VAE_LAYER_SIZE,
    "vae_max_epochs": VAE_MAX_EPOCHS,
    "vae_if_Adam": VAE_IF_ADAM,
    "ql_alpha": QL_ALPHA,
    "ql_gamma": QL_GAMMA,
    "ql_epsilon": QL_EPSILON,
    "ql_max_episodes": QL_MAX_EPISODES,
    "ql_max_time_steps": QL_MAX_TIME_STEPS,
    "num_of_q_agents": NUM_OF_Q_AGENTS,
    "replay_buffer_size": REPLAY_BUFFER_SIZE,
}


# pre-training of the disentangled beta-VAE (corner-halving search)
class DisentangledBetaVaeTuningConfig(TypedDict):
    """Configuration aligned with the corner-halving CV driver (DisentangledBetaVAE.py)."""

    # dataset / architecture basics
    dataset_name: str
    vae_disc_lat_dim: int  # discrete latent dim
    vae_cont_lat_dim: int  # continuous latent dim
    latent_size: int
    hidden_size_1: int
    hidden_size_2: int

    # search ranges and stopping criterion
    beta_range: Tuple[float, float]
    C_range: Tuple[float, float]
    granularity: float  # shrink until spans <= granularity * initial_span
    halving_min_rounds: int  # derived: ceil(log2(1 / granularity))

    # training / optimization knobs
    learning_rate: float
    batch_size: int
    use_adam_optimizer: bool
    epoch_chunk: int | None
    max_epochs: int | None
    halving_epoch_budgets: Tuple[int, ...]
    halving_metric: str
    convergence_tolerance: float
    convergence_patience: int
    min_epochs_before_convergence: int

    # CV + evaluation
    k_folds: int
    recycles: int
    m: int

    # I/O paths and misc
    results_path: str
    data_path: str
    corrupt_data_path: str
    initial_imputation_strategy: str
    model_outdir: str

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        vae_disc_lat_dim: int,
        vae_cont_lat_dim: int,
        latent_size: int,
        hidden_size_1: int,
        hidden_size_2: int,
        beta_range: Tuple[float, float],
        C_range: Tuple[float, float],
        granularity: float,
        learning_rate: float,
        batch_size: int,
        use_adam_optimizer: bool,
        k_folds: int,
        recycles: int,
        m: int,
        results_path: str,
        data_path: str,
        corrupt_data_path: str,
        initial_imputation_strategy: str,
        model_outdir: str,
        halving_epoch_budgets: Tuple[int, ...] | list[int] | None = None,
        halving_metric: str = "mae",
        convergence_tolerance: float = 1e-4,
        convergence_patience: int = 2,
        min_epochs_before_convergence: int = 30,
        epoch_chunk: int | None = None,
        max_epochs: int | None = None,
    ) -> "DisentangledBetaVaeTuningConfig":
        """Validate inputs and compute the minimum halving rounds from ``granularity``."""

        b_min, b_max = beta_range
        c_min, c_max = C_range

        if b_min <= 0 or b_max <= 0:
            raise ValueError("beta_range entries must be > 0.")
        if b_min >= b_max:
            raise ValueError("Require beta_range[0] < beta_range[1].")

        if c_min < 0 or c_max < 0:
            raise ValueError("C_range entries must be >= 0.")
        if c_min >= c_max:
            raise ValueError("Require C_range[0] < C_range[1].")

        if not (0 < granularity <= 0.25):
            raise ValueError("granularity must satisfy 0 < value <= 0.25.")

        halving_min_rounds = math.ceil(math.log2(1.0 / granularity))
        if halving_min_rounds < 2:
            raise ValueError("Computed halving_min_rounds must be at least two; check granularity.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if k_folds <= 0:
            raise ValueError("k_folds must be > 0.")
        if recycles <= 0:
            raise ValueError("recycles must be > 0.")
        if m <= 0:
            raise ValueError("m must be > 0.")
        if latent_size <= 0 or hidden_size_1 <= 0 or hidden_size_2 <= 0:
            raise ValueError("latent_size and hidden sizes must be > 0.")
        if vae_disc_lat_dim <= 0 or vae_cont_lat_dim <= 0:
            raise ValueError("Latent dims must be > 0.")

        if epoch_chunk is not None and epoch_chunk <= 0:
            raise ValueError("epoch_chunk must be > 0 when provided.")
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError("max_epochs must be > 0 when provided.")

        budgets: Tuple[int, ...] = tuple(halving_epoch_budgets) if halving_epoch_budgets else tuple()
        for b in budgets:
            if b <= 0:
                raise ValueError("halving_epoch_budgets entries must be > 0.")

        cfg: "DisentangledBetaVaeTuningConfig" = cls(
            dataset_name=dataset_name,
            vae_disc_lat_dim=vae_disc_lat_dim,
            vae_cont_lat_dim=vae_cont_lat_dim,
            latent_size=latent_size,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            beta_range=beta_range,
            C_range=C_range,
            granularity=granularity,
            halving_min_rounds=halving_min_rounds,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_adam_optimizer=use_adam_optimizer,
            epoch_chunk=epoch_chunk,
            max_epochs=max_epochs,
            halving_epoch_budgets=budgets,
            halving_metric=halving_metric,
            convergence_tolerance=convergence_tolerance,
            convergence_patience=convergence_patience,
            min_epochs_before_convergence=min_epochs_before_convergence,
            k_folds=k_folds,
            recycles=recycles,
            m=m,
            results_path=results_path,
            data_path=data_path,
            corrupt_data_path=corrupt_data_path,
            initial_imputation_strategy=initial_imputation_strategy,
            model_outdir=model_outdir,
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


    # define the default config values for the VAE-QL imputation pipeline; these can be overridden by the user when creating a config instance
    @classmethod
    def initiate(
        cls,
        *,
        dataset_name: str,
        dataset_features: FeaturesTypeDict,
        **overrides: dict,
    ) -> "VaeQlConfig":
        """Return defaults union_dict with required dataset info and optional overrides."""

        union_dict = {
            "dataset_name": dataset_name,
            "dataset_features": dataset_features,
            **DEFAULT_VAEQL_CONFIG,
            **overrides,
        }
        return cls.create(**union_dict)


    @classmethod
    def create(
        cls,
        **config: dict,
    ) -> "VaeQlConfig":
        """Construct and validate a VAE-QL pipeline config.

        Basic validation rules:
        - Latent dims, layers, layer_size, batch_size, epochs, episodes, time steps, buffer: strictly > 0
        - Learning rates/coefficients: 0 < vae_alpha, ql_alpha <= 1; 0 < vae_beta; vae_C >= 0
        - Discount factor: 0 < ql_gamma <= 1
        - Epsilon: 0 <= ql_epsilon <= 1
        - Replay Buffer Size: replay_buffer_size >= ql_max_time_steps * num_of_q_agents
        """
        
        union_dict = {**DEFAULT_VAEQL_CONFIG, **config}

        dataset_name = union_dict.get("dataset_name")
        dataset_features = union_dict.get("dataset_features")
        vae_disc_lat_dim = union_dict.get("vae_disc_lat_dim")
        vae_cont_lat_dim = union_dict.get("vae_cont_lat_dim")
        vae_alpha = union_dict.get("vae_alpha")
        vae_beta = union_dict.get("vae_beta")
        vae_C = union_dict.get("vae_C")
        vae_batch_size = union_dict.get("vae_batch_size")
        vae_layers = union_dict.get("vae_layers")
        vae_layer_size = union_dict.get("vae_layer_size")
        vae_max_epochs = union_dict.get("vae_max_epochs")
        vae_if_Adam = union_dict.get("vae_if_Adam")
        ql_alpha = union_dict.get("ql_alpha")
        ql_gamma = union_dict.get("ql_gamma")
        ql_epsilon = union_dict.get("ql_epsilon")
        ql_max_episodes = union_dict.get("ql_max_episodes")
        ql_max_time_steps = union_dict.get("ql_max_time_steps")
        num_of_q_agents = union_dict.get("num_of_q_agents")
        replay_buffer_size = union_dict.get("replay_buffer_size")

        if dataset_name is None:
            raise ValueError("dataset_name must be provided.")
        if dataset_features is None:
            raise ValueError("dataset_features must be provided.")

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
            replay_buffer_size=replay_buffer_size
        )

        return cfg
    