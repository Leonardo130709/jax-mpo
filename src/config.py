import dataclasses
from rltools.config import Config


@dataclasses.dataclass
class MPOConfig(Config):
    # Algorithm
    discount: float = .99
    action_repeat: int = 2
    n_step: int = 3
    #  IQN.
    num_actions: int = 20
    num_actor_quantiles: int = 32
    num_critic_quantiles: int = 4
    hubber_delta: float = 1.
    #  MPO.
    tv_constraint: float = 1.
    epsilon_eta: float = .1
    epsilon_mean: float = 1e-2
    epsilon_std: float = 1e-4
    init_log_temperature: float = 10.
    init_log_alpha_mean: float = 10.
    init_log_alpha_std: float = 1000.
    #  HER.
    hindsight_goal_key: str = "jaco_arm/jaco_hand/pinch_site_pos"
    augmentation_strategy: str = "none"
    num_augmentations: int = 1

    # Architecture
    activation: str = "elu"
    normalization: str = "layer"
    stop_actor_grad: bool = True
    #   Encoder
    keys: str = "observations"
    mlp_layers: tuple[int] = ()
    pn_number: int = 500
    pn_layers: tuple[int] = (256, 256, 256)
    cnn_kernels: tuple[int] = (4, 4, 4, 4)
    cnn_depth: tuple[int] = (32, 32, 32, 32)
    feature_fusion: str = r"$^"
    #   Actor
    actor_backend: str = "gpu"
    actor_layers: tuple[int] = (256, 256)
    min_std: float = 0.
    init_std: float = .5
    #   Critic
    use_iqn: bool = False
    num_critic_heads: int = 2
    critic_layers: tuple[int] = (512, 512)
    quantile_embedding_dim: int = 64

    # reverb
    min_replay_size: int = 1e3
    samples_per_insert: int = 128
    batch_size: int = 256
    buffer_capacity: int = 1e6
    actor_update_every: int = 1
    learner_dump_every: int = 1
    reverb_port: int = 4445

    # training
    learning_rate: float = 1e-4
    dual_lr: float = 1e-2
    adam_b1: float = .9
    adam_b2: float = .999
    adam_eps: float = 1e-5
    weight_decay: float = 0.
    target_actor_update_period: int = 25
    target_critic_update_period: int = 100
    max_seq_len: int = 25
    eval_every: int = 1e4
    log_every: int = 1e2
    eval_times: int = 7
    grad_norm: float = 40.
    mp_policy: str = "p=f32,c=f32,o=f32"
    jit: bool = True

    # task
    seed: int = 0
    task: str = "dmc_walker_walk"
    logdir: str = "logdir/walker_walk_image"
    total_steps: int = 1e6
    time_limit: int = 1e3
