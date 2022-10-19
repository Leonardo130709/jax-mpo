import time
from functools import partial

import dm_env
import numpy as np
import jax
import jax.numpy as jnp
import chex
import haiku as hk
import reverb

from src.networks import MPONetworks
from src.config import MPOConfig
from src.utils import env_loop
from rltools.loggers import TerminalOutput

CPU = jax.devices("cpu")[0]


class Actor:
    def __init__(self,
                 rng_key: jax.random.PRNGKey,
                 env: dm_env.Environment,
                 cfg: MPOConfig,
                 networks: MPONetworks,
                 client: reverb.Client
                 ):

        jax.config.update("jax_disable_jit", not cfg.jit)

        @partial(jax.jit, backend="cpu", static_argnums=(3,))
        @chex.assert_max_traces(n=2)
        def _act(params: hk.Params,
                 key: jax.random.PRNGKey,
                 observation: jnp.ndarray,
                 training: bool
                 ) -> jnp.ndarray:
            state = networks.encoder(params, observation)
            policy_params = networks.actor(params, state)
            dist = networks.make_policy(*policy_params)
            if training:
                action = dist.sample(seed=key)
            else:
                action = dist.distribution.mean()
            action = jnp.clip(action, a_min=-1, a_max=1)
            return action

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
        self._params = None
        self.update_params()

    def act(self, observation, training):
        rng = next(self._rng_seq)
        action = self._act(self._params, rng, observation, training)
        return np.asarray(action, dtype=self._act_prec)

    def update_params(self):
        params = next(self._weights_ds).data
        self._params = jax.device_put(params, CPU)

    def run(self):
        step = 0
        start = time.time()
        should_update = Every(self.cfg.actor_update_every)
        should_eval = Every(self.cfg.eval_every)
        timestep = self._env.reset()
        eval_policy = partial(self.act, training=False)
        train_policy = partial(self.act, training=True)
        log = TerminalOutput()

        def tree_slice(sl, tree, is_leaf=None):
            return jax.tree_util.tree_map(
                lambda t: t[sl], tree, is_leaf=is_leaf
            )

        while step < self.cfg.total_steps:
            if should_update(step):
                self.update_params()

            trajectory, timestep = env_loop.environment_loop(
                self._env,
                train_policy,
                timestep,
                self.cfg.seq_len,
            )
            tr_length = len(trajectory["actions"])
            step += self.cfg.action_repeat * tr_length

            with self._client.trajectory_writer(2) as writer:
                def slice_writer(sl, key):
                    return tree_slice(sl, writer.history[key])

                for i in range(tr_length+1):
                    if i < tr_length:
                        writer.append(tree_slice(
                            i, trajectory,
                            is_leaf=lambda x: isinstance(x, list)
                        ))
                    else:
                        # Here is the terminal obs.
                        writer.append({"observations": timestep.observation})

                    if i < 1:
                        continue

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

                now = time.strftime("%H:%M", time.gmtime(time.time() - start))
                eval_metrics = {
                    "step": step,
                    'time_expired': now,
                    "train_return": sum(trajectory['rewards']),
                    "eval_returns_mean": np.mean(returns),
                    "eval_return_std": np.std(returns),
                    "eval_duration_mean": np.mean(dur)
                }
                # Should eval steps also count in step?
                log.write(eval_metrics)


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
