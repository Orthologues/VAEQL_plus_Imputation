# Defaults for the VAE-QL pipeline (excluding dataset-specific values)
VAE_DISC_LAT_DIM = 10
VAE_CONT_LAT_DIM = 10
VAE_ALPHA = 1e-3
VAE_BATCH_SIZE = 128
VAE_LAYERS = 2
VAE_HIDDEN_SIZE1 = 256
VAE_HIDDEN_SIZE2 = 256
VAE_MAX_EPOCHS = 100
VAE_IF_ADAM = True
QL_ALPHA = 0.1
QL_GAMMA = 0.9
QL_EPSILON_START = 1.0
QL_EPSILON_END = 0.1
QL_EPSILON_DECAY = 0.995
QL_MAX_EPISODES = 1000
QL_MAX_TIME_STEPS = 100
NUM_OF_Q_AGENTS = 4
REPLAY_BUFFER_SIZE = 10000

DEFAULT_VAEQL_CONFIG: dict = {
    "vae_disc_lat_dim": VAE_DISC_LAT_DIM,
    "vae_cont_lat_dim": VAE_CONT_LAT_DIM,
    "vae_alpha": VAE_ALPHA,
    "vae_batch_size": VAE_BATCH_SIZE,
    "vae_layers": VAE_LAYERS,
    "vae_hidden_size1": VAE_HIDDEN_SIZE1,
    "vae_hidden_size2": VAE_HIDDEN_SIZE2,
    "vae_max_epochs": VAE_MAX_EPOCHS,
    "vae_if_Adam": VAE_IF_ADAM,
    "ql_alpha": QL_ALPHA,
    "ql_gamma": QL_GAMMA,
    "ql_epsilon_start": QL_EPSILON_START,
    "ql_epsilon_end": QL_EPSILON_END,
    "ql_epsilon_decay": QL_EPSILON_DECAY,
    "ql_max_episodes": QL_MAX_EPISODES,
    "ql_max_time_steps": QL_MAX_TIME_STEPS,
    "num_of_q_agents": NUM_OF_Q_AGENTS,
    "replay_buffer_size": REPLAY_BUFFER_SIZE,
}
