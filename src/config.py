import dataclasses

@dataclasses.dataclass
class Config:
    discount: float = .99
    num_actions: int = 16
    epsilon_eta: float = .1
    epsilon_alpha: float = .01
    init_duals: float = .01

    buffer_capacity: int = 10**6
    batch_size: int = 128

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    dual_lr: float = 1e-2
    actor_layers: tuple = (64, 64)
    critic_layers: tuple = (100, 50)
    actor_tau: float = .01
    critic_tau: float = .01
    max_grad: float = 10.

    seed: int = 0
    task: str = 'cartpole_balance'
