from typing import NamedTuple
from dm_control import suite
from dm_env import specs

import numpy as np
import reverb
import tensorflow as tf
import jax

from rltools import dmc_wrappers
from .config import MPOConfig
from .actor import Actor
from .networks import MPONetworks, make_networks
from .learner import MPOLearner


class EnvironmentSpecs(NamedTuple):
    observation_spec: specs.Array
    action_spec: specs.Array
    reward_spec: specs.Array


class Builder:
    def __init__(self, config: MPOConfig):
        self._c = config

    def make_reverb_tables(self, env_specs, networks):
        # todo: remove networks from init
        def from_specs(spec):
            tf_spec = lambda x: tf.TensorSpec((self._c.seq_len,) + tuple(x.shape), dtype=x.dtype)
            return jax.tree_util.tree_map(tf_spec, spec)

        trajectory_signature = dict(
            observations=from_specs(env_specs.observation_spec),
            actions=from_specs(env_specs.action_spec),
            rewards=from_specs(env_specs.reward_spec),
            next_observations=from_specs(env_specs.observation_spec)
        )
        weights_signature = jax.tree_util.tree_map(
            lambda x: tf.TensorSpec(x.shape, x.dtype),
            networks.init(jax.random.PRNGKey(0))
        )
        return [
            reverb.Table(
                name="replay_buffer",
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self._c.buffer_capacity,
                rate_limiter=reverb.rate_limiters.SampleToInsertRatio(
                    min_size_to_sample=self._c.min_replay_size,
                    samples_per_insert=self._c.samples_per_insert,
                    error_buffer=self._c.batch_size * self._c.samples_per_insert,
                ),
                signature=trajectory_signature
            ),
            reverb.Table(
                name="weights",
                sampler=reverb.selectors.Lifo(),
                remover=reverb.selectors.Fifo(),
                max_size=1,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=weights_signature
            )
        ]

    def make_dataset_iterator(self, server_address):
        ds: tf.data.Dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=server_address,
            table="replay_buffer",
            max_in_flight_samples_per_worker=4*self._c.batch_size,
            get_signature_timeout_secs=10
        )
        ds = ds.batch(self._c.batch_size, drop_remainder=True)
        ds = ds.prefetch(-1)
        return ds.as_numpy_iterator()

    def make_networks(self, env_specs: EnvironmentSpecs):
        return make_networks(self._c,
                             env_specs.observation_spec,
                             env_specs.action_spec)

    def make_learner(self,
                     rng_key: jax.random.PRNGKey,
                     env_spec: EnvironmentSpecs,
                     iterator: tf.data.Dataset,
                     networks: MPONetworks,
                     client: reverb.Client
                     ):

        return MPOLearner(rng_key, self._c, env_spec,
                          networks, iterator, client)

    def make_actor(self, rng_key, env, networks, client):
        return Actor(rng_key, env, self._c, networks, client)

    def make_env(self):
        domain, task = self._c.task.split("_", maxsplit=1)
        if domain == "ball":
            domain = "bal_in_cup"
            task = "catch"
        env = suite.load(domain, task,
                         task_kwargs={"random": self._c.seed},
                         environment_kwargs=None)

        env = dmc_wrappers.StatesWrapper(env)
        env = dmc_wrappers.ActionRepeat(env, self._c.action_repeat)
        env = dmc_wrappers.TypesCast(env,
                                     observation_dtype=np.float32,
                                     action_dtype=np.float32,
                                     reward_dtype=np.float32)

        return env, env.environment_specs
