import dataclasses
from rltools.config import Config


@dataclasses.dataclass
class MPOConfig(Config):
    # Algorithm
    discount: float = .99
    action_repeat: int = 1
    n_step: int = 1
    num_actions: int = 20
    num_quantiles: int = 8
    epsilon_eta: float = .1
    epsilon_mean: float = 2.5e-3
    epsilon_std: float = 1e-6
    init_duals: float = 10.
    hubber_delta: float = 1.

    # Model
    activation: str = 'relu'
    normalization: str = 'none'
    stop_actor_grad: bool = True
    #   Encoder
    cnn_keys: str = r'.*'
    mlp_keys: str = r'.*'
    cnn_kernels: tuple = (48, 48, 48, 48)
    cnn_depth: int = 48
    mlp_layers: tuple = (256, 256, 256, 256)
    feature_fusion: bool = False
    #   Actor
    actor_layers: tuple = (256, 256)
    min_std: float = .1
    #   Critic
    critic_layers: tuple = (512, 512)
    quantile_embedding_dim: int = 64

    # reverb
    min_replay_size: int = 1e4
    samples_per_insert: int = 64
    batch_size: int = 256
    buffer_capacity: int = int(1e7)

    # training
    learning_rate: float = 1e-4
    dual_lr: float = 1e-2
    adam_b1: float = .9
    adam_b2: float = .999
    adam_eps: float = 1e-6
    target_update_period: int = 100
    grad_norm: float = 40.
    mp_policy: str = 'p=f32,c=f32,o=f32'

    # task
    seed: int = 0
    task: str = 'cartpole_balance'
    logdir: str = 'logdir'
    time_limit: int = float('inf')  # :)
    total_steps: int = 1e6
