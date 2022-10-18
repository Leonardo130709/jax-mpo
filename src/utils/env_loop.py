from typing import Callable, Tuple, Dict, Optional, TypedDict
from collections import deque, defaultdict

import dm_env
import numpy as np

Array = np.ndarray
Observation = Dict[str, Array]


class Trajectory(TypedDict):
    observations: list[Observation]
    actions: list[Array]
    rewards: list[float]
    discounts: list[bool]


def environment_loop(env: dm_env.Environment,
                     policy: Callable[[Observation], Array],
                     prev_timestep: dm_env.TimeStep,
                     max_timesteps: int = float('inf'),
                     ) -> Tuple[Trajectory, dm_env.TimeStep]:
    steps = 0
    trajectory = defaultdict(list)
    timestep = env.reset() if prev_timestep.last() else prev_timestep
    done = False
    while not done:
        steps += 1
        obs = timestep.observation
        action = policy(obs)
        timestep = env.step(action)
        done = timestep.last() or (steps >= max_timesteps)
        # o_tm1, a_tm1, r_t, discount_t
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(timestep.reward)
        trajectory['discounts'].append(not timestep.last())
    # Last o_t is in the timestep.
    return trajectory, timestep


class NStep:
    def __init__(self, n_step: int, discount: float):
        self.n_step = n_step
        self.discount = discount
        self._discount_n = discount ** n_step

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        # Use scipy.signals?
        rewards, disc = map(trajectory.get, ('rewards', 'discounts'))
        assert not np.all(disc[:-1])
        n_step_rewards = []
        reward = 0
        prev_rewards = deque(self.n_step * [0.], maxlen=self.n_step)
        for r in reversed(rewards):
            stale_reward = prev_rewards.pop()
            reward =\
                r + self.discount * reward - self._discount_n * stale_reward
            prev_rewards.appendleft(r)
            n_step_rewards.append(reward)

        trajectory['rewards'] = n_step_rewards[::-1]
        # trajectory['discounts'] = ?
        return trajectory


#
# def n_step(n_step, gamma, rewards, dones):
#     assert not np.all(dones[:-1])
#     n_step_rewards = []
#     reward = 0
#     gamma_n = gamma ** n_step
#     prev_rewards = deque(n_step * [0.], maxlen=n_step)
#     for r in reversed(rewards):
#         stale_reward = prev_rewards.pop()
#         reward = r + gamma * reward - gamma_n * stale_reward
#         prev_rewards.appendleft(r)
#         n_step_rewards.append(reward)
#
#     return n_step_rewards[::-1]
#
