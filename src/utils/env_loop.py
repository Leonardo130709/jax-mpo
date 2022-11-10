from typing import Callable, Tuple, Dict, TypedDict, List
from collections import deque, defaultdict
import copy

import jax
import dm_env
import numpy as np
import reverb

from src import GOAL_KEY

Action = Array = np.ndarray
Observation = Dict[str, Array]


class Trajectory(TypedDict, total=False):
    observations: List[Observation]
    actions: List[Action]
    rewards: List[float]
    discounts: List[float]
    next_observations: List[Observation]


def environment_loop(env: dm_env.Environment,
                     policy: Callable[[Observation], Action],
                     prev_timestep: dm_env.TimeStep,
                     max_timesteps: int = float("inf"),
                     ) -> Tuple[Trajectory, dm_env.TimeStep]:
    step = 0
    trajectory = defaultdict(list)
    timestep = env.reset() if prev_timestep.last() else prev_timestep
    done = False
    while not done:
        step += 1
        obs = timestep.observation
        action = policy(obs)
        timestep = env.step(action)
        done = timestep.last() or (step >= max_timesteps)
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
    res = copy.deepcopy(trajectory)
    obs, rewards, disc = map(
        res.get,
        ("observations", "rewards", "discounts")
    )
    length = len(rewards)
    discount_n = discount ** n_step
    is_not_terminal = res['discounts'][-1]
    next_obs = obs[n_step:] + n_step * [obs[-1]]
    discounts = \
        (length - n_step) * [discount_n] + \
        [is_not_terminal * discount ** i for i in range(n_step, 0, -1)]

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
                      rng: np.random.Generator,
                      goal_source: str,
                      strategy: str = "none",
                      discount: float = 1.,
                      amount: int = 1,
                      ) -> List[Trajectory]:
    """Augments source trajectory with additional goals."""
    if strategy == "none":
        return [trajectory]

    trajectories = [trajectory]
    if strategy == "final":
        hindsight_goal = trajectory["observations"][-1][goal_source]
        aug = copy.deepcopy(trajectory)
        for obs in aug["observations"]:
            obs[GOAL_KEY] = hindsight_goal
        aug["rewards"][-1] = 1.
        trajectories.extend(amount * [aug])
    elif strategy == "future":
        discounts = discount * np.asarray(trajectory["discounts"])
        term_idx = sample_from_geometrical(rng, discounts, amount)
        for i in term_idx.tolist():
            tr = tree_slice(
                slice(0, i), trajectory,
                is_leaf=lambda x: isinstance(x, list)
            )
            aug = goal_augmentation(tr, rng, goal_source, "final", 1)
            trajectories.append(aug[-1])
    else:
        raise ValueError(strategy)

    return trajectories


class Adder:
    def __init__(self,
                 client: reverb.Client,
                 rng: np.random.Generator,
                 n_step: int = 1,
                 discount: float = .99,
                 goal_source: str = r"$^",
                 aug_strategy: str = "none",
                 amount: int = 1
                 ):
        self._client = client
        self._n_step_fn = lambda tr: n_step_fn(tr, n_step, discount)

        self._augmentation_fn = lambda tr: goal_augmentation(
            tr, rng, goal_source,
            aug_strategy,
            discount, amount
        )

    def __call__(self, trajectory: Trajectory):
        trajectories = self._augmentation_fn(trajectory)
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


def sample_from_geometrical(rng: np.random.Generator,
                            discount_t: np.ndarray,
                            size: tuple = ()
                            ) -> np.ndarray:
    # P(t) ~ \prod^t_0 d_i * (1 - d_t)
    cont_prob_t = np.concatenate([
        np.ones_like(discount_t[:1]),
        discount_t
    ])
    term_prob_t = np.concatenate([
        1. - discount_t,
        np.ones_like(discount_t[-1:])
    ])
    cumprod_t = np.cumprod(cont_prob_t)
    prob_t = cumprod_t * term_prob_t
    return rng.choice(prob_t.size, size=size, p=prob_t)
