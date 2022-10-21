from typing import Callable, Tuple, Dict, TypedDict
from collections import deque, defaultdict

import dm_env
import numpy as np
import reverb
import jax

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
        trajectory["discounts"].append(not timestep.last())
    # Last o_t is in the timestep.
    return trajectory, timestep


class NStep:
    def __init__(self, n_step: int, discount: float):
        assert n_step == 1
        self.n_step = n_step
        self.discount = discount
        self._discount_n = discount ** n_step

    def __call__(self, trajectory: Trajectory) -> Trajectory:
        # Use scipy.signals?
        if self.n_step == 1:
            trajectory["discounts"] *= self.discount
            return trajectory

        rewards, disc = map(trajectory.get, ("rewards", "discounts"))
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

        trajectory["rewards"] = n_step_rewards[::-1]
        # trajectory["discounts"] = ?
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


class Adder:
    def __init__(self, client: reverb.Client, n_step: int = 1):
        self._client = client
        self._n_step = NStep(n_step, discount=.99)

    def __call__(self, trajectory: Trajectory):
        tr_length = len(trajectory["actions"])
        assert len(trajectory["observations"]) == tr_length + 1

        with self._client.trajectory_writer(num_keep_alive_refs=2) as writer:
            for i in range(tr_length + 1):
                if i < tr_length:
                    writer.append(tree_slice(
                        i, trajectory,
                        is_leaf=lambda x: isinstance(x, list)
                    ))
                else:
                    # Here is the terminal obs.
                    writer.append({
                        "observations": trajectory["observations"][i]
                    })

                if i < 1:
                    continue

                def slice_writer(sl, key):
                    return tree_slice(sl, writer.history[key])

                # o_tm1, a_tm1, r_t, d_t, o_t
                writer.create_item(
                    table="replay_buffer",
                    priority=1.,
                    trajectory={
                        "observations": slice_writer(-2, "observations"),
                        "actions": slice_writer(-2, "actions"),
                        "rewards": slice_writer(-2, "rewards"),
                        "discounts": slice_writer(-2, "discounts"),
                        "next_observations": slice_writer(-1, "observations")
                    }
                )
                writer.flush(block_until_num_items=10)


def tree_slice(sl, tree, is_leaf=None):
    return jax.tree_util.tree_map(
        lambda t: t[sl], tree, is_leaf=is_leaf
    )
