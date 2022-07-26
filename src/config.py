import dataclasses


@dataclasses.dataclass
class Config:
    discount: float = .99
    num_actions: int = 20
    epsilon_eta: float = .1
    epsilon_alpha_mean: float = .1
    epsilon_alpha_std: float = 1e-4
    init_duals: float = .01

    buffer_capacity: int = 10**6
    batch_size: int = 256

    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    dual_lr: float = 1e-2
    actor_layers: tuple = (64, 64)
    critic_layers: tuple = (100, 50)
    actor_tau: float = .01
    critic_tau: float = .01
    max_grad: float = 20.

    seed: int = 0
    task: str = 'cartpole_balance'
