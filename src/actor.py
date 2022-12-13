import time
import pickle
import multiprocessing as mp
from functools import partial

import dm_env
import numpy as np
import jax
import jax.numpy as jnp
import chex
import haiku as hk
import reverb

from rltools.loggers import JSONLogger, TFSummaryLogger

from src.networks import MPONetworks
from src.config import MPOConfig
from src.utils import env_loop


class Actor:

    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 env: dm_env.Environment,
                 cfg: MPOConfig,
                 networks: MPONetworks,
                 client: reverb.Client,
                 total_steps: mp.Value
                 ):

        @partial(jax.jit, backend=cfg.actor_backend)
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
            if cfg.discretize:
                logits = dist.distribution.logits
                action = jax.lax.select(
                    training,
                    dist.sample(seed=key),
                    jax.nn.one_hot(logits.argmax(-1),
                                   logits.shape[-1],
                                   dtype=jnp.int32)
                )
            else:
                action = jax.lax.select(
                    training,
                    dist.sample(seed=key),
                    dist.distribution.mean()
                )
                action = jnp.clip(action, a_min=-1, a_max=1)
            return action

        self._env = env
        self._act_prec = env.action_spec().dtype
        self._act = _act
        self.cfg = cfg
        self._client = client
        self._total_steps = total_steps
        self._device = jax.devices(cfg.actor_backend)[0]

        rng_key = jax.device_put(rng_key, self._device)
        self._rng_seq = hk.PRNGSequence(rng_key)
        self._weights_ds = reverb.TimestepDataset.from_table_signature(
            client.server_address,
            table="weights",
            max_in_flight_samples_per_worker=1,
            num_workers_per_iterator=1,
        ).as_numpy_iterator()

        np_rng = np.asarray(next(self._rng_seq))
        self._adder = env_loop.Adder(client,
                                     np.random.default_rng(np_rng),
                                     cfg.n_step,
                                     cfg.discount,
                                     cfg.goal_sources,
                                     cfg.goal_targets,
                                     env.task.compute_reward,
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
        self._params = jax.device_put(params, self._device)

    def run(self):
        step = 0
        start = time.time()
        should_update = env_loop.Every(self.cfg.actor_update_every)
        should_eval = env_loop.Every(self.cfg.eval_every)
        timestep = self._env.reset()
        eval_policy = partial(self.act, training=False)
        train_policy = partial(self.act, training=True)
        json_log = JSONLogger(self.cfg.logdir + "/eval_metrics.jsonl")
        tf_log = TFSummaryLogger(self.cfg.logdir, "eval", "step")

        while step < self.cfg.total_steps:
            if should_update(step):
                self.update_params()

            self._env.task.eval_flag = False
            trajectory, timestep = env_loop.environment_loop(
                self._env,
                train_policy,
                timestep,
                self.cfg.max_seq_len,
            )
            tr_length = len(trajectory["actions"])
            env_steps = self.cfg.action_repeat * tr_length
            step += env_steps
            with self._total_steps.get_lock():
                self._total_steps.value += env_steps
            self._adder(trajectory)

            if should_eval(step):
                lock = self._total_steps.get_lock()
                if lock.acquire(False):
                    returns = []
                    dur = []
                    self._env.task.eval_flag = False
                    for _ in range(self.cfg.eval_times):
                        tr, timestep = env_loop.environment_loop(
                            self._env,
                            eval_policy,
                            self._env.reset()
                        )
                        returns.append(sum(tr["rewards"]))
                        dur.append(len(tr["actions"]))
    
                    metrics = {
                        "step": self._total_steps.value,
                        "time_expired": time.time() - start,
                        "train_return": sum(trajectory["rewards"]),
                        "eval_return_mean": np.mean(returns),
                        "eval_return_std": np.std(returns),
                        "eval_duration_mean": np.mean(dur),
                    }
                    reverb_info = _get_reverb_metrics(self._client)
                    metrics.update(reverb_info)
                    json_log.write(metrics)
                    tf_log.write(metrics)
                    path = self.cfg.logdir + "/total_steps.pickle"
                    with open(path, "wb") as f:
                        pickle.dump(self._total_steps.value, f)
                    lock.release()


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
