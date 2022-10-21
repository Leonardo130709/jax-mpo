from typing import NamedTuple

import dm_env
import jax
import reverb
import numpy as np
import tensorflow as tf
from dm_env import specs

from rltools import dmc_wrappers
from src.config import MPOConfig
from src.actor import Actor
from src.networks import make_networks
from src.learner import MPOLearner
from src.utils import envs


class EnvironmentSpecs(NamedTuple):
    observation_spec: specs.Array
    action_spec: specs.Array
    reward_spec: specs.Array
    discount_spec: specs.Array


class Builder:
    def __init__(self, config: MPOConfig):
        self.cfg = config
        rng = jax.random.PRNGKey(config.seed)
        self._actor_rng, self._learner_rng, self._env_rng = \
            jax.random.split(rng, 3)
        self._env_specs = None

    def make_server(self, env_specs: EnvironmentSpecs):
        networks = self.make_networks(env_specs)
        self._actor_rng, rng = jax.random.split(self._actor_rng)
        params = networks.init(rng)

        def to_tf_spec(spec):
            fn = lambda sp: tf.TensorSpec(tuple(sp.shape), dtype=sp.dtype)
            return jax.tree_util.tree_map(fn, spec)

        trajectory_signature = dict(
            observations=env_specs.observation_spec,
            actions=env_specs.action_spec,
            rewards=env_specs.reward_spec,
            discounts=env_specs.discount_spec,
            next_observations=env_specs.observation_spec
        )
        trajectory_signature = to_tf_spec(trajectory_signature)
        weights_signature = to_tf_spec(params)

        tables = [
            reverb.Table(
                name="replay_buffer",
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self.cfg.buffer_capacity,
                rate_limiter=reverb.rate_limiters.SampleToInsertRatio(
                    min_size_to_sample=self.cfg.min_replay_size,
                    samples_per_insert=self.cfg.samples_per_insert,
                    error_buffer=.1 * self.cfg.min_replay_size *
                                 self.cfg.samples_per_insert,
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
        # TODO: use reverb checkpoint.
        server = reverb.Server(tables, self.cfg.reverb_port)
        client = reverb.Client(f"localhost:{self.cfg.reverb_port}")
        client.insert(params, priorities={"weights": 1})

        return server

    def make_dataset_iterator(self, server_address: str):
        ds: tf.data.Dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=server_address,
            table="replay_buffer",
            max_in_flight_samples_per_worker=2 * self.cfg.batch_size,
            get_signature_timeout_secs=100
        )
        ds = ds.batch(self.cfg.batch_size, drop_remainder=True)
        ds = ds.prefetch(-1)
        return ds.as_numpy_iterator()

    def make_networks(self, env_specs: EnvironmentSpecs):
        return make_networks(self.cfg,
                             env_specs.observation_spec,
                             env_specs.action_spec)

    def make_learner(self,
                     env_spec: EnvironmentSpecs,
                     iterator: tf.data.Dataset,
                     client: reverb.Client
                     ):
        self._learner_rng, rng_key = jax.random.split(self._learner_rng)
        networks = self.make_networks(env_spec)
        return MPOLearner(rng_key, self.cfg, env_spec,
                          networks, iterator, client)

    def make_actor(self,
                   env: dm_env.Environment,
                   env_specs: EnvironmentSpecs,
                   client: reverb.Client
                   ):
        self._actor_rng, rng_key = jax.random.split(self._actor_rng)
        networks = self.make_networks(env_specs)
        return Actor(rng_key, env, self.cfg, networks, client)

    def make_env(self):
        self._env_rng, seed = jax.random.split(self._env_rng)
        seed = np.random.RandomState(seed)
        domain, task = self.cfg.task.split('_', 1)
        if domain == "dmc":
            env = envs.DMC(task, seed, 1, 1)
        else:
            raise NotImplementedError

        env = dmc_wrappers.ActionRepeat(env, self.cfg.action_repeat)
        env_specs = EnvironmentSpecs(
            observation_spec=env.observation_spec(),
            action_spec=env.action_spec(),
            reward_spec=env.reward_spec(),
            discount_spec=env.discount_spec()
        )
        return env, env_specs
