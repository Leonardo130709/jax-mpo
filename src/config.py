import dataclasses
from rltools.config import Config


@dataclasses.dataclass
class MPOConfig(Config):
    # Algorithm
    discount: float = .99
    action_repeat: int = 1
    n_step: int = 1
    #  IQN.
    num_actions: int = 20
    num_actor_quantiles: int = 32
    num_critic_quantiles: int = 4
    hubber_delta: float = 1.
    #  MPO.
    tv_constraint: float = 1.
    epsilon_eta: float = .1
    epsilon_mean: float = 1e-1
    epsilon_std: float = 1e-3
    init_log_temperature: float = 10.
    init_log_alpha_mean: float = 10.
    init_log_alpha_std: float = 100.
    #  HER.
    hindsight_goal_key: str = "jaco_arm/jaco_hand/pinch_site_pos"
    augmentation_strategy: str = "none"
    num_augmentations: int = 1

    # Architecture
    activation: str = "elu"
    normalization: str = "layer"
    stop_actor_grad: bool = True
    #   Encoder
    keys: str = r".*"
    encoder_emb_dim: int = 64
    mlp_layers: tuple[int] = (256,)
    pn_layers: tuple[int] = (64, 256, 512, 512)
    cnn_kernels: tuple[int] = (4, 4, 4, 4)
    cnn_depth: tuple[int] = (48, 48, 48, 48)
    feature_fusion: str = r"$^"
    #   Actor
    actor_backend: str = "cpu"
    actor_layers: tuple[int] = (256, 256)
    min_std: float = 1e-2
    init_std: float = 1.
    #   Critic
    use_iqn: bool = False
    num_critic_heads: int = 2
    critic_layers: tuple[int] = (1024, 1024)
    quantile_embedding_dim: int = 64

    # reverb
    min_replay_size: int = 1e3
    samples_per_insert: int = 512
    batch_size: int = 256
    buffer_capacity: int = 1e6
    actor_update_every: int = 1
    learner_dump_every: int = 1
    reverb_port: int = 4445

    # training
    learning_rate: float = 5e-4
    dual_lr: float = 1e-2
    adam_b1: float = .9
    adam_b2: float = .999
    adam_eps: float = 1e-6
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
    task: str = "dmc_manip_reach_site_features"
    logdir: str = "logdir/manip_reach_wo_aug"
    time_limit: int = 1e3
