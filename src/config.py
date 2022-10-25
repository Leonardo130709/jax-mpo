import dataclasses
from rltools.config import Config


@dataclasses.dataclass
class MPOConfig(Config):
    # Algorithm
    discount: float = .99
    action_repeat: int = 1
    n_step: int = 1
    num_actions: int = 20
    num_actor_quantiles: int = 32
    num_critic_quantiles: int = 1
    tv_constraint: float = 1.
    hubber_delta: float = 1.
    epsilon_eta: float = .1
    epsilon_mean: float = 1e-1
    epsilon_std: float = 1e-3
    init_log_temperature: float = 10.
    init_log_alpha_mean: float = 10.
    init_log_alpha_std: float = 1000.

    # Architecture
    activation: str = "elu"
    normalization: str = "layer"
    stop_actor_grad: bool = True
    #   Encoder
    keys: str = r"observations"
    encoder_emb_dim: int = 64
    mlp_layers: tuple[int] = (256, 256)
    pn_layers: tuple[int] = (64, 256, 512)
    cnn_kernels: tuple[int] = (4, 4, 4, 4)
    cnn_depth: int = 48
    feature_fusion: str = "cnn"
    #   Actor
    actor_layers: tuple[int] = (256, 256, 256)
    min_std: float = .1
    init_std: float = 1.
    #   Critic
    use_iqn: bool = False
    num_critic_heads: int = 2
    critic_layers: tuple[int] = (512, 512, 512)
    quantile_embedding_dim: int = 64

    # reverb
    min_replay_size: int = 1e3
    samples_per_insert: int = 64
    batch_size: int = 256
    buffer_capacity: int = 1e6
    actor_update_every: int = 1
    learner_dump_every: int = 1
    reverb_port: int = 4446

    # training
    learning_rate: float = 5e-4
    dual_lr: float = 1e-2
    adam_b1: float = .9
    adam_b2: float = .999
    adam_eps: float = 1e-6
    targets_update_period: int = 50
    max_seq_len: int = 50
    eval_every: int = 1e4
    log_every: int = 1e2
    eval_times: int = 5
    grad_norm: float = 40.
    mp_policy: str = "p=f32,c=f32,o=f32"
    jit: bool = True

    # task
    seed: int = 0
    task: str = "dmc_walker_walk"
    logdir: str = "/dev/null"
    total_steps: int = 1e6
    time_limit: int = 1e3
