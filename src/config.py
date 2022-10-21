import dataclasses
from rltools.config import Config


@dataclasses.dataclass
class MPOConfig(Config):
    # Algorithm
    discount: float = .99
    action_repeat: int = 1
    n_step: int = 1
    num_actions: int = 20
    num_actor_quantiles: int = 8
    num_critic_quantiles: int = 4
    epsilon_eta: float = .1
    epsilon_mean: float = 2.5e-3
    epsilon_std: float = 1e-6
    tv_constraint: float = 1.
    hubber_delta: float = 0.
    init_log_temperature: float = 10.
    init_log_alpha_mean: float = 10.
    init_log_alpha_std: float = 1000.

    # Architecture
    activation: str = "elu"
    normalization: str = "layer"
    stop_actor_grad: bool = True
    #   Encoder
    keys: str = r"image"
    mlp_layers: tuple[int] = ()
    pn_layers: tuple[int] = (64, 256, 512)
    cnn_kernels: tuple[int] = (4, 4, 4, 4)
    cnn_depth: int = 48
    feature_fusion: str = "cnn"
    #   Actor
    actor_layers: tuple[int] = (256, 256)
    min_std: float = .1
    init_std: float = 1.
    #   Critic
    num_critic_heads: int = 1
    critic_layers: tuple[int] = (512, 512)
    quantile_embedding_dim: int = 64

    # reverb
    min_replay_size: int = 1e3
    samples_per_insert: int = 512
    batch_size: int = 256
    buffer_capacity: int = 1e6
    actor_update_every: int = 1
    learner_dump_every: int = 50
    reverb_port: int = 4446

    # training
    learning_rate: float = 1e-4
    dual_lr: float = 1e-2
    adam_b1: float = .9
    adam_b2: float = .999
    adam_eps: float = 1e-8
    target_actor_update_period: int = 25
    target_critic_update_period: int = 100
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
    logdir: str = "logdir"
    total_steps: int = 1e6
    time_limit: int = 1e3
