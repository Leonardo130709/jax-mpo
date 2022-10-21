import dataclasses
from rltools.config import Config


@dataclasses.dataclass
class MPOConfig(Config):
    # Algorithm
    discount: float = .99
    action_repeat: int = 1
    n_step: int = 3
    num_actions: int = 20
    num_actor_quantiles: int = 8
    num_critic_quantiles: int = 4
    epsilon_eta: float = .1
    epsilon_mean: float = 1e-1
    epsilon_std: float = 1e-4
    tv_constraint: float = float("inf")
    hubber_delta: float = 0.
    init_log_temperature: float = 1.
    init_log_alpha_mean: float = 1.
    init_log_alpha_std: float = 100.

    # Architecture
    activation: str = "relu"
    normalization: str = "none"
    stop_actor_grad: bool = True
    #   Encoder
    keys: str = r".*"
    mlp_layers: tuple[int] = ()
    pn_layers: tuple[int] = (64, 256, 512)
    cnn_kernels: tuple[int] = (4, 4, 4, 4)
    cnn_depth: int = 48
    feature_fusion: str = "none"
    #   Actor
    actor_layers: tuple[int] = (256, 256)
    min_std: float = .1
    init_std: float = 1.
    #   Critic
    num_critic_heads: int = 1
    critic_layers: tuple[int] = (512, 512)
    quantile_embedding_dim: int = 64

    # reverb
    min_replay_size: int = 1e4
    samples_per_insert: int = 256
    batch_size: int = 256
    buffer_capacity: int = 1e6
    actor_update_every: int = 1
    learner_dump_every: int = 100
    reverb_port: int = 4446

    # training
    learning_rate: float = 1e-4
    dual_lr: float = 1e-2
    adam_b1: float = .9
    adam_b2: float = .999
    adam_eps: float = 1e-8
    target_actor_update_period: int = 25
    target_critic_update_period: int = 100
    seq_len: int = 100
    eval_every: int = 1e4
    log_every: int = 1e2
    eval_times: int = 5
    grad_norm: float = float("inf")
    mp_policy: str = "p=f32,c=f32,o=f32"
    jit: bool = True

    # task
    seed: int = 0
    task: str = "dmc_walker_walk"
    logdir: str = "logdir"
    total_steps: int = 1e6
    time_limit: int = 1e3
