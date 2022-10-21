from typing import Callable, Tuple, Dict, TypedDict
from collections import deque, defaultdict

import dm_env
import numpy as np
import reverb
from jax.tree_util import tree_map

Array = np.ndarray
Observation = Dict[str, Array]


class Trajectory(TypedDict, total=False):
    observations: list[Observation]
    actions: list[Array]
    rewards: list[float]
    discounts: list[float]
    next_observations: list[Observation]


def environment_loop(env: dm_env.Environment,
                     policy: Callable[[Observation], Array],
                     prev_timestep: dm_env.TimeStep,
                     max_timesteps: int = float("inf"),
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
        trajectory["observations"].append(obs)
        trajectory["actions"].append(action)
        trajectory["rewards"].append(timestep.reward)
        trajectory["discounts"].append(1. - timestep.last())

    trajectory["observations"].append(timestep.observation)
    return trajectory, timestep


class NStep:
    def __init__(self, n_step: int, discount: float):
        self.n_step = n_step
        self.discount = discount
        self._discount_n = discount ** n_step

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        obs, rewards, disc = map(
            trajectory.get,
            ("observations", "rewards", "discounts")
        )
        assert np.all(disc[:-1]), "Unhandled early termination."

        length = len(rewards)
        next_obs = obs[self.n_step:] + self.n_step * [obs[-1]]
        discounts = \
            (length - self.n_step) * [self._discount_n] + \
            [self.discount ** i for i in range(self.n_step, 0, -1)]

        res = trajectory.copy()
        res["next_observations"] = next_obs
        res["discounts"] = discounts

        if self.n_step == 1:
            return res

        n_step_rewards = []
        reward = 0
        prev_rewards = deque(self.n_step * [0.], maxlen=self.n_step)
        for r in reversed(rewards):
            stale_reward = prev_rewards.pop()
            reward = \
                r + self.discount * reward - self._discount_n * stale_reward
            prev_rewards.appendleft(r)
            n_step_rewards.append(reward)

        res["rewards"] = n_step_rewards[::-1]
        return res


class GoalSampler:
    def __init__(self, strategy: str):
        self.strategy = strategy

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        length = len(trajectory["actions"])
        last_obs = trajectory["observations"][-1]

        res = trajectory.copy()
        res["goals"] = length * [last_obs]
        res["rewards"][-1] = 1.

        return res


class Adder:
    def __init__(self,
                 client: reverb.Client,
                 n_step: int = 1,
                 discount: float = .99
                 ):
        self._client = client
        self._n_step = NStep(n_step, discount)

    def __call__(self, trajectory: Trajectory):
        trajectory = self._n_step(trajectory)
        tr_length = len(trajectory["actions"])

        with self._client.trajectory_writer(num_keep_alive_refs=1) as writer:
            for i in range(tr_length):
                writer.append(tree_slice(
                    i, trajectory,
                    is_leaf=lambda x: isinstance(x, list)
                ))

                if i < 1:
                    continue

                # o_tm1, a_tm1, r_t, d_t, o_t
                writer.create_item(
                    table="replay_buffer",
                    priority=1.,
                    trajectory=tree_slice(-1, writer.history)
                )
                # writer.flush(block_until_num_items=10)


def tree_slice(sl, tree, is_leaf=None):
    return tree_map(
        lambda t: t[sl], tree, is_leaf=is_leaf
    )
