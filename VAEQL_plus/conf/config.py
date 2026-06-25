#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-02-19
# Description: This file is used to define the configuration classes for the VAEQL imputation pipeline.
#########################################################

# general imports
import importlib
import json
import math
import re
from pathlib import Path
from typing import TypedDict, Tuple, Any

# local imports
from .feat_types import FeaturesTypeDict

DIS_BETAVAE_TRAIN_VAL_PARAMS_PATH = Path(__file__).with_name(
    "DisBetaVAE_train_val_params.json"
)
with open(DIS_BETAVAE_TRAIN_VAL_PARAMS_PATH, "r", encoding="utf-8") as _f:
    DIS_BETAVAE_TRAIN_VAL_PARAMS: dict[str, Any] = json.load(_f)

CV_BETA_C_TUNING_PARAMS_PATH = Path(__file__).with_name(
    "CV_beta_C_tuning_params.json"
)
with open(CV_BETA_C_TUNING_PARAMS_PATH, "r", encoding="utf-8") as _f:
    CV_BETA_C_TUNING_PARAMS: dict[str, Any] = json.load(_f)


def _resolve_device():
    """Resolve torch device lazily and fail-safe to CPU if torch is unavailable/broken."""
    try:
        torch = importlib.import_module("torch")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return "cpu"


# pre-training of the disentangled beta-VAE (corner-halving search)
class DisentangledBetaVaeTuningConfig(TypedDict):
    """Configuration aligned with the beta-DVAE corner-halving CV driver."""

    # dataset / architecture basics
    dataset_name: str
    n_gmm_components: int  # number of GMM prior components
    vae_cont_lat_dim: int  # continuous latent dim
    vae_hidden_size_1: int
    vae_hidden_size_2: int

    # search ranges and stopping criterion
    beta_range: Tuple[float, float]
    C_range: Tuple[float, float]
    granularity: float  # shrink until spans <= granularity * initial_span
    halving_rounds: int  # derived: ceil(log2(1 / granularity))

    # training / optimization knobs
    learning_rate: float
    batch_size: int
    use_adam_optimizer: bool
    num_feat_loss_metric: str
    epoch_chunk: int | None
    max_epochs: int | None
    halving_epoch_budgets: Tuple[int, ...]
    halving_metric: str
    convergence_tolerance: float
    convergence_patience: int
    min_epochs_before_convergence: int
    device: Any = _resolve_device()
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
    def default_cv_beta_C_tuning_config_path(cls) -> Path:
        return CV_BETA_C_TUNING_PARAMS_PATH

    @classmethod
    def default_train_val_params(cls) -> dict[str, Any]:
        return dict(DIS_BETAVAE_TRAIN_VAL_PARAMS)

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        n_gmm_components: int = int(DIS_BETAVAE_TRAIN_VAL_PARAMS["n_gmm_components"]),
        vae_cont_lat_dim: int = int(DIS_BETAVAE_TRAIN_VAL_PARAMS["vae_cont_lat_dim"]),
        vae_hidden_size_1: int = int(DIS_BETAVAE_TRAIN_VAL_PARAMS["hidden_size_1"]),
        vae_hidden_size_2: int = int(DIS_BETAVAE_TRAIN_VAL_PARAMS["hidden_size_2"]),
        num_feat_loss_metric: str = str(DIS_BETAVAE_TRAIN_VAL_PARAMS["num_feat_loss_metric"]),
        m: int = int(CV_BETA_C_TUNING_PARAMS["m"]),
        beta_range: Tuple[float, float] = tuple(CV_BETA_C_TUNING_PARAMS["beta_range"]),
        C_range: Tuple[float, float] = tuple(CV_BETA_C_TUNING_PARAMS["C_range"]),
        granularity: float = float(CV_BETA_C_TUNING_PARAMS["granularity"]),
        learning_rate: float = float(CV_BETA_C_TUNING_PARAMS["learning_rate"]),
        batch_size: int = int(CV_BETA_C_TUNING_PARAMS["batch_size"]),
        use_adam_optimizer: bool = bool(CV_BETA_C_TUNING_PARAMS["use_adam_optimizer"]),
        k_folds: int = int(CV_BETA_C_TUNING_PARAMS["k_folds"]),
        recycles: int = int(CV_BETA_C_TUNING_PARAMS["recycles"]),
        results_path: str = str(CV_BETA_C_TUNING_PARAMS["results_path"]),
        data_path: str = str(CV_BETA_C_TUNING_PARAMS["data_path"]),
        corrupt_data_path: str = str(CV_BETA_C_TUNING_PARAMS["corrupt_data_path"]),
        initial_imputation_strategy: str = str(CV_BETA_C_TUNING_PARAMS["initial_imputation_strategy"]),
        model_outdir: str = str(CV_BETA_C_TUNING_PARAMS["model_outdir"]),
        halving_epoch_budgets: Tuple[int, ...] | list[int] | None = tuple(CV_BETA_C_TUNING_PARAMS["halving_epoch_budgets"]),
        halving_metric: str = str(CV_BETA_C_TUNING_PARAMS["halving_metric"]),
        convergence_tolerance: float = float(CV_BETA_C_TUNING_PARAMS["convergence_tolerance"]),
        convergence_patience: int = int(CV_BETA_C_TUNING_PARAMS["convergence_patience"]),
        min_epochs_before_convergence: int = int(CV_BETA_C_TUNING_PARAMS["min_epochs_before_convergence"]),
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

        halving_rounds = math.ceil(math.log2(1.0 / granularity))
        if halving_rounds < 2:
            raise ValueError("Computed halving_rounds must be at least two; check granularity.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        normalized_num_feat_loss_metric = re.sub(
            r"[_\-\.\s]+", "", str(num_feat_loss_metric).upper()
        )
        if normalized_num_feat_loss_metric not in {"RMSE", "MAE"}:
            raise ValueError(
                "num_feat_loss_metric must resolve to 'RMSE' or 'MAE', "
                f"got {num_feat_loss_metric!r}."
            )
        if k_folds <= 0:
            raise ValueError("k_folds must be > 0.")
        if recycles <= 0:
            raise ValueError("recycles must be > 0.")
        if m <= 0:
            raise ValueError("m must be > 0.")
        if vae_hidden_size_1 <= 0 or vae_hidden_size_2 <= 0:
            raise ValueError("hidden sizes must be > 0.")
        if n_gmm_components <= 0 or vae_cont_lat_dim <= 0:
            raise ValueError("Latent dims must be > 0.")

        if epoch_chunk is not None and epoch_chunk <= 0:
            raise ValueError("epoch_chunk must be > 0 when provided.")
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError("max_epochs must be > 0 when provided.")

        budgets: Tuple[int, ...] = tuple(halving_epoch_budgets) if halving_epoch_budgets else tuple()
        if budgets:
            if len(budgets) != 3:
                raise ValueError(
                    "halving_epoch_budgets must be a 3-tuple/list: "
                    "[start_val, max_val, step_size]."
                )
            start_val, max_val, step_size = (int(b) for b in budgets)
            if start_val <= 0 or max_val <= 0 or step_size <= 0:
                raise ValueError("halving_epoch_budgets entries must be > 0.")
            if start_val > max_val:
                raise ValueError(
                    "halving_epoch_budgets must satisfy start_val <= max_val."
                )

        cfg: "DisentangledBetaVaeTuningConfig" = cls(
            dataset_name=dataset_name,
            n_gmm_components=n_gmm_components,
            vae_cont_lat_dim=vae_cont_lat_dim,
            vae_hidden_size_1=vae_hidden_size_1,
            vae_hidden_size_2=vae_hidden_size_2,
            beta_range=beta_range,
            C_range=C_range,
            granularity=granularity,
            halving_rounds=halving_rounds,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_adam_optimizer=use_adam_optimizer,
            num_feat_loss_metric=normalized_num_feat_loss_metric,
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
    data_path: str
    input_dim: int
    # VAE configs
    n_gmm_components: int # number of mixed Gaussian prior components in the discrete latent space S
    vae_cont_lat_dim: int # dimension of the continuous latent space Z (each composed of the weighted disc_lat_dim Gaussian components)
    vae_alpha: float # learning rate for the VAE training
    vae_beta: float # beta parameter for the KL term of the VAE loss (reconstruction error + KL divergence)
    vae_C: float # capacity parameter for the KL term of the VAE loss (reconstruction error + KL divergence)
    vae_batch_size: int
    vae_layers: int # number of layers for the VAE encoder and decoder (symmetric architecture)
    vae_hidden_size1: int # number of neurons in the first hidden layer for the VAE encoder and decoder (symmetric architecture)
    vae_hidden_size2: int # number of neurons in the second hidden layer for the VAE encoder and decoder (symmetric architecture)
    vae_max_epochs: int # maximum number of epochs for the VAE training
    vae_if_Adam: bool # whether to use the Adam optimizer for the VAE training (if False, will use vanilla SGD)
    # Q-learning configs
    ql_alpha: float # learning rate for the Q-value update
    ql_gamma: float # discount factor for the max Q-value of the transition's destination state
    ql_epsilon_start: float # starting epsilon for the epsilon-greedy policy (off-policy exploration)
    ql_epsilon_end: float   # final epsilon value after decay
    ql_epsilon_decay: float # multiplicative decay factor applied each step/episode
    ql_episodes_per_cycle: int # number of episodes for the Q-learning training per "Ensemble Cycle of Q-agent-reinforced VAE models" 
    ql_max_time_steps: int # maximum number of time steps per episode for the Q-learning training
    ## reward calculation
    ql_local_reward_eta: float # reward coefficient for the local imputation quality (e.g., negative delta of MAE of the same patient compared to the initial imputation for the current state)
    # Shared configs
    num_of_q_agents: int # number of Q-learning agents to train in parallel (with shared experience replay buffer)
    vaeql_cycles: int # number of "Ensemble Cycles of Q-agent-reinforced VAE models" to run; each cycle consists of training num_of_q_agents in parallel with shared experience replay, then retraining the VAEs on the public replay buffer (experience), then ensemble of the VAEs into one "consensus model" for the next cycle
    replay_buffer_size: int # maximum size of the experience replay buffer (must be >= ql_max_time_steps * num_of_q_agents to allow for at least one full episode per agent in the buffer)
    gaussian_outlier_sigma: float # sigma parameter for the Gaussian outlier penalty in the reward function (e.g., negative exp(- (imputation_error^2) / (2 * gaussian_outlier_sigma^2))) to penalize large imputation errors less heavily on values deviated from the mean
    device: Any = _resolve_device()


    
    @staticmethod
    def _gt_zero(name: str, value: int | float) -> None:
        if value <= 0:
            raise ValueError(f"{name!r} must be > 0; got {value!r}.")

    @staticmethod
    def _in_unit_interval(name: str, value: float, *, inclusive_zero: bool = False) -> None:
        low_ok = value >= 0 if inclusive_zero else value > 0
        if not (low_ok and value <= 1):
            bound = "0 <= x <= 1" if inclusive_zero else "0 < x <= 1"
            raise ValueError(f"{name!r} must satisfy {bound!r}; got {value!r}.")


    # define the default config values for the VAE-QL imputation pipeline; these can be overridden by the user when creating a config instance
    @classmethod
    def initiate(
        cls,
        *,
        dataset_name: str,
        dataset_features: FeaturesTypeDict,
        data_path: str,
        vae_beta: float, #  tuned via the corner-halving search in the DisentangledBetaVaeTuningConfig; no default since this is a key hyperparameter for controlling the disentanglement of the VAE latent space, which is critical for the quality of the learned representations and thus the downstream Q-learning performance
        vae_C: float, #  tuned via the corner-halving search in the DisentangledBetaVaeTuningConfig as well
        **overrides: dict, # optional overrides for any of the other config values (e.g., vae_alpha, ql_alpha, ql_gamma, ql_epsilon_start, ql_epsilon_end, ql_epsilon_decay, etc.); these will be merged with the defaults defined in DIS_BETAVAE_TRAIN_VAL_PARAMS
    ) -> "VaeQlConfig":
        """Return defaults union_dict with required dataset info and optional overrides."""

        union_dict = {
            "dataset_name": dataset_name,
            "dataset_features": dataset_features,
            "vae_beta": vae_beta,
            "vae_C": vae_C,
            "data_path": data_path,
            **DIS_BETAVAE_TRAIN_VAL_PARAMS,
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
        - Epsilon schedule: 0 < ql_epsilon_start <= 1; 0 <= ql_epsilon_end <= 1; 0 < ql_epsilon_decay <= 1; ql_epsilon_start >= ql_epsilon_end
        - Replay Buffer Size: replay_buffer_size >= ql_max_time_steps * num_of_q_agents
        """
        
        union_dict = {**DIS_BETAVAE_TRAIN_VAL_PARAMS, **config}

        dataset_name = union_dict.get("dataset_name")
        dataset_features = union_dict.get("dataset_features")
        n_gmm_components = union_dict.get("n_gmm_components")
        vae_cont_lat_dim = union_dict.get("vae_cont_lat_dim")
        vae_alpha = union_dict.get("vae_alpha")
        vae_beta = union_dict.get("vae_beta")
        vae_C = union_dict.get("vae_C")
        vae_batch_size = union_dict.get("vae_batch_size")
        vae_layers = union_dict.get("vae_layers")
        vae_hidden_size1 = union_dict.get("vae_hidden_size1")
        vae_hidden_size2 = union_dict.get("vae_hidden_size2")
        vae_max_epochs = union_dict.get("vae_max_epochs")
        vae_if_Adam = union_dict.get("vae_if_Adam")
        ql_alpha = union_dict.get("ql_alpha")
        ql_gamma = union_dict.get("ql_gamma")
        ql_epsilon_start = union_dict.get("ql_epsilon_start")
        ql_epsilon_end = union_dict.get("ql_epsilon_end")
        ql_epsilon_decay = union_dict.get("ql_epsilon_decay")
        ql_max_episodes = union_dict.get("ql_max_episodes")
        ql_max_time_steps = union_dict.get("ql_max_time_steps")
        num_of_q_agents = union_dict.get("num_of_q_agents")
        replay_buffer_size = union_dict.get("replay_buffer_size")
        
        if dataset_name is None:
            raise ValueError("dataset_name must be provided.")
        if dataset_features is None:
            raise ValueError("dataset_features must be provided.")
        
        try:
            input_dim = dataset_features["all_feats"].__len__()  # returns to the length of a set
        except Exception as e:
            raise ValueError(f"dataset_features must contain 'all_feats' key, which refers to a set of all feature names; error: {e!r}")

        # Disentangled beta-VAE hyperparameters
        if vae_beta <= 0:
            raise ValueError(f"vae_beta must be > 0; got {vae_beta!r}.")
        if vae_C < 0:
            raise ValueError(f"vae_C must be >= 0; got {vae_C!r}.")

        # Dimensionalities and counts
        for n, v in [
            ("n_gmm_components", n_gmm_components),
            ("vae_cont_lat_dim", vae_cont_lat_dim),
            ("vae_layers", vae_layers),
            ("vae_hidden_size1", vae_hidden_size1),
            ("vae_hidden_size2", vae_hidden_size2),
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
        cls._in_unit_interval("ql_epsilon_start", ql_epsilon_start)
        cls._in_unit_interval("ql_epsilon_end", ql_epsilon_end, inclusive_zero=True)
        cls._in_unit_interval("ql_epsilon_decay", ql_epsilon_decay)

        if ql_epsilon_start < ql_epsilon_end:
            raise ValueError(f"ql_epsilon_start must be >= ql_epsilon_end; got {ql_epsilon_start!r} < {ql_epsilon_end!r}.")
        

        cfg: "VaeQlConfig" = cls(
            dataset_name=dataset_name,
            dataset_features=dataset_features,
            n_gmm_components=n_gmm_components,
            vae_cont_lat_dim=vae_cont_lat_dim,
            vae_alpha=vae_alpha,
            vae_beta=vae_beta,
            vae_C=vae_C,
            vae_batch_size=vae_batch_size,
            vae_layers=vae_layers,
            vae_hidden_size1=vae_hidden_size1,
            vae_hidden_size2=vae_hidden_size2,
            vae_max_epochs=vae_max_epochs,
            vae_if_Adam=vae_if_Adam,
            ql_alpha=ql_alpha,
            ql_gamma=ql_gamma,
            ql_epsilon_start=ql_epsilon_start,
            ql_epsilon_end=ql_epsilon_end,
            ql_epsilon_decay=ql_epsilon_decay,
            ql_max_episodes=ql_max_episodes,
            ql_max_time_steps=ql_max_time_steps,
            num_of_q_agents=num_of_q_agents,
            replay_buffer_size=replay_buffer_size,
            input_dim=input_dim
        )

        return cfg
    
