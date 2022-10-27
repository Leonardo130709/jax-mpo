import re
import time
from functools import partial

import dm_env
import numpy as np
import jax
import jax.numpy as jnp
import chex
import haiku as hk
import reverb

from rltools.loggers import JSONLogger
from src.networks import MPONetworks
from src.config import MPOConfig
from src.utils import env_loop


class Actor:
    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 env: dm_env.Environment,
                 cfg: MPOConfig,
                 networks: MPONetworks,
                 client: reverb.Client
                 ):

        @jax.jit
        @chex.assert_max_traces(n=1)
        def _act(params: hk.Params,
                 key: jax.random.PRNGKey,
                 observation: jnp.ndarray,
                 training: bool
                 ) -> jnp.ndarray:

            observation = networks.preprocess(observation)
            state = networks.encoder(params, observation)
            policy_params = networks.actor(params, state)
            dist = networks.make_policy(*policy_params)
            action = jax.lax.select(
                training,
                dist.sample(seed=key),
                dist.distribution.mean()
            )
            return jnp.clip(action, a_min=-1, a_max=1)

        self._env = env
        self._act_prec = env.action_spec().dtype
        self._rng_seq = hk.PRNGSequence(rng_key)
        self._act = _act
        self.cfg = cfg
        self._client = client
        self._weights_ds = reverb.TimestepDataset.from_table_signature(
            client.server_address,
            table="weights",
            max_in_flight_samples_per_worker=1,
            num_workers_per_iterator=1,
        ).as_numpy_iterator()

        goal_key = _match_key(cfg.hindsight_goal_key,
                              env.observation_spec().keys()
                              )
        self._adder = env_loop.Adder(client,
                                     next(self._rng_seq),
                                     cfg.n_step,
                                     cfg.discount,
                                     goal_key,
                                     cfg.augmentation_strategy,
                                     cfg.num_augmentations
                                     )
        self._params = None
        self.update_params()

    def act(self, observation, training):
        rng = next(self._rng_seq)
        action = self._act(self._params, rng, observation, training)
        return np.asarray(action, dtype=self._act_prec)

    def update_params(self):
        params = next(self._weights_ds).data
        self._params = jax.device_put(params)

    def run(self):
        step = 0
        start = time.time()
        should_update = Every(self.cfg.actor_update_every)
        should_eval = Every(self.cfg.eval_every)
        timestep = self._env.reset()
        eval_policy = partial(self.act, training=False)
        train_policy = partial(self.act, training=True)
        log = JSONLogger(self.cfg.logdir + "/eval_metrics.jsonl")

        while step < self.cfg.total_steps:
            if should_update(step):
                self.update_params()

            trajectory, timestep = env_loop.environment_loop(
                self._env,
                train_policy,
                timestep,
                self.cfg.max_seq_len,
            )
            tr_length = len(trajectory["actions"])
            step += self.cfg.action_repeat * tr_length
            self._adder(trajectory)

            if should_eval(step):
                returns = []
                dur = []
                for _ in range(self.cfg.eval_times):
                    tr, timestep = env_loop.environment_loop(
                        self._env,
                        eval_policy,
                        self._env.reset()
                    )
                    returns.append(sum(tr["rewards"]))
                    dur.append(len(tr["actions"]))

                metrics = {
                    "step": step,
                    "time_expired": time.time() - start,
                    "train_return": sum(trajectory["rewards"]),
                    "eval_return_mean": np.mean(returns),
                    "eval_return_std": np.std(returns),
                    "eval_duration_mean": np.mean(dur),
                }
                reverb_info = _get_reverb_metrics(self._client)
                metrics.update(reverb_info)
                # Should eval steps also add to total_steps?
                log.write(metrics)


class Every:
    def __init__(self, interval: int):
        self.interval = interval
        self._prev_step = 0

    def __call__(self, step: int) -> bool:
        assert step >= self._prev_step
        diff = step - self._prev_step
        if diff >= self.interval:
            self._prev_step = step
            return True
        return False


def _get_reverb_metrics(client: reverb.Client) -> dict[str, float]:
    info = client.server_info()["replay_buffer"]
    info = info.table_worker_time
    stats = (
        "sampling_ms",
        "inserting_ms",
        "sleeping_ms",
        "waiting_for_sampling_ms",
        "waiting_for_inserts_ms",
    )
    reverb_info = {}
    for key in stats:
        if hasattr(info, key):
            reverb_info[f"reverb_{key}"] = getattr(info, key)
    return reverb_info


def _match_key(pattern, candidates):
    candidates = [k for k in candidates if re.match(pattern, k)]
    assert len(candidates) == 1, \
        f"Invalid or ambiguous key: {pattern!r} -- {candidates}."
    return candidates.pop()


