import dataclasses


@dataclasses.dataclass
class MPOConfig:
    # alg
    discount: float = .99
    num_actions: int = 20
    num_quantiles: int = 4
    epsilon_eta: float = .1
    epsilon_alpha: float = 1e-3
    init_duals: float = .01
    huber_kappa: float = 1.

    # model
    actor_layers: tuple = (200, 200)
    mean_scale: float = 1.
    critic_layers: tuple = (256, 256)
    quantile_embedding_dim: int = 64
    hidden_dim: int = 256

    # reverb
    min_replay_size: int = 10000
    samples_per_insert: int = 128
    batch_size: int = 64
    buffer_capacity: int = int(1e6)

    # training
    seq_len: int = 1
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    encoder_lr: float = 5e-4
    dual_lr: float = 1e-2
    actor_polyak: float = 5e-3
    critic_polyak: float = 5e-3
    encoder_polyak: float = 5e-3
    max_grad: float = 10.

    # task
    seed: int = 0
    task: str = 'cartpole_balance'
    total_episodes: int = -1
    mp_policy: str = 'p=f32,c=f32,o=f32'
