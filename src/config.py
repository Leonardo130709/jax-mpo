import dataclasses


@dataclasses.dataclass
class Config:
    # alg
    discount: float = .99
    num_actions: int = 20
    epsilon_eta: float = .1
    epsilon_alpha_mean: float = .1
    epsilon_alpha_std: float = 1e-4
    init_duals: float = .01

    buffer_capacity: int = 10**6

    # model
    actor_layers: tuple = (200, 200)
    critic_layers: tuple = (256, 256)
    hidden_dim: int = 256

    # training
    batch_size: int = 128
    spi: int = 128
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    dual_lr: float = 1e-2
    actor_tau: float = .01
    critic_tau: float = .01
    max_grad: float = 20.

    # task
    seed: int = 0
    task: str = 'cartpole_balance'
