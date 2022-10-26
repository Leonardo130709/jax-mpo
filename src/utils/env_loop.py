from typing import Callable, Tuple, Dict, TypedDict
from collections import deque, defaultdict
import re
import copy

import jax
import dm_env
import numpy as np
import reverb

from src.utils.ops import sample_from_geometrical

Action = Array = np.ndarray
Observation = Dict[str, Array]


class Trajectory(TypedDict, total=False):
    observations: list[Observation]
    actions: list[Action]
    rewards: list[float]
    discounts: list[float]
    next_observations: list[Observation]


def environment_loop(env: dm_env.Environment,
                     policy: Callable[[Observation], Action],
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


def n_step_fn(trajectory: Trajectory,
              n_step: int = 1,
              discount: float = .99
              ) -> Trajectory:
    """Computes N-step rewards for the trajectory."""
    obs, rewards, disc = map(
        trajectory.get,
        ("observations", "rewards", "discounts")
    )
    assert np.all(disc[:-1]), "Unhandled early termination."

    length = len(rewards)
    discount_n = discount ** n_step
    next_obs = obs[n_step:] + n_step * [obs[-1]]
    discounts = \
        (length - n_step) * [discount_n] + \
        [discount ** i for i in range(n_step, 0, -1)]

    res = trajectory.copy()
    res["next_observations"] = next_obs
    res["discounts"] = discounts

    if n_step == 1:
        return res

    n_step_rewards = []
    reward = 0
    prev_rewards = deque(n_step * [0.], maxlen=n_step)
    for r in reversed(rewards):
        stale_reward = prev_rewards.pop()
        reward = \
            r + discount * reward - discount_n * stale_reward
        prev_rewards.appendleft(r)
        n_step_rewards.append(reward)

    res["rewards"] = n_step_rewards[::-1]
    return res


def goal_augmentation(trajectory: Trajectory,
                      rng: jax.random.PRNGKey,
                      goal_key: str,
                      strategy: str = "none",
                      discount: float = 1.,
                      amount: int = 1,
                      ) -> list[Trajectory]:
    """Augments source trajectory with additional goals."""
    test_obs = trajectory["observations"][0]
    assert np.all(trajectory["discounts"][:-1] > 0.), "Early termination."
    if goal_key not in test_obs.keys():
        candidates = list(filter(
            lambda key: re.match(goal_key, key),
            test_obs.keys()
        ))
        assert len(candidates) == 1,\
            f"Wrong key regex {goal_key!r}: {candidates}"
        goal_key = candidates.pop()

    trajectories = [trajectory]
    if strategy == "last":
        hindsight_goal = trajectory["observations"][-1][goal_key]
        aug = copy.deepcopy(trajectory)
        for obs in aug["observations"]:
            obs[GOAL_KEY] = hindsight_goal
        aug["rewards"][-1] = 1.
        trajectories.extend(amount * [aug])
    elif strategy == "future":
        discounts = discount * jax.numpy.array(trajectory["discounts"])
        term_idx = sample_from_geometrical(rng, discounts, (amount,))
        for i in term_idx.tolist():
            tr = tree_slice(
                slice(0, i), trajectory,
                is_leaf=lambda x: isinstance(x, list)
            )
            aug = goal_augmentation(tr, rng, goal_key, "last", 1)
            trajectories.append(aug[-1])

    return trajectories


class Adder:
    def __init__(self,
                 client: reverb.Client,
                 rng: jax.random.PRNGKey,
                 n_step: int = 1,
                 discount: float = .99,
                 ):
        self._client = client
        self._rng = rng
        self.n_step = n_step
        self.discount = discount
        self._n_step_fn = lambda tr: n_step_fn(tr, n_step, discount)
        self._augmentation_fn = lambda tr, *args: [tr]

    def __call__(self, trajectory: Trajectory):
        self._rng, rng = jax.random.split(self._rng)
        trajectories = self._augmentation_fn(trajectory, rng)
        trajectories = map(self._n_step_fn, trajectories)
        for tr in trajectories:
            self._insert(tr)

    def _insert(self, trajectory: Trajectory):
        with self._client.trajectory_writer(num_keep_alive_refs=1) as writer:
            for i in range(len(trajectory["actions"])):
                writer.append(
                    tree_slice(
                        i, trajectory,
                        is_leaf=lambda x: isinstance(x, list)
                    )
                )
                # o_tm1, a_tm1, r_t, d_t, o_t
                writer.create_item(
                    table="replay_buffer",
                    priority=1.,
                    trajectory=tree_slice(-1, writer.history)
                )
                writer.flush(block_until_num_items=20)


def tree_slice(sl, tree, is_leaf=None):
    return jax.tree_util.tree_map(
        lambda t: t[sl], tree, is_leaf=is_leaf
    )
