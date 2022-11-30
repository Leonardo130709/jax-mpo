import os
import pickle
import multiprocessing as mp

import dm_env
import jax
import reverb
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from rltools import dmc_wrappers
from rltools.dmc_wrappers import EnvironmentSpecs
from src.config import MPOConfig
from src.actor import Actor
from src.networks import make_networks
from src.learner import MPOLearner
from src.utils import envs


class Builder:

    def __init__(self, config: MPOConfig):
        path = os.path.expanduser(config.logdir)
        if not os.path.exists(path):
            os.makedirs(path)
        path = config.logdir + "/config.yaml"
        if os.path.exists(path):
            config = MPOConfig.load(path)
        else:
            config.save(path)
        self.cfg = config

        rng = jax.random.PRNGKey(config.seed)
        self.rng = jax.device_get(rng)

        path = self.cfg.logdir + "/total_steps.pickle"
        if os.path.exists(path):
            with open(path, "rb") as f:
                val = pickle.load(f)
        else:
            val = 0
        self._total_steps = mp.Value("i", val)

    def make_server(self,
                    random_key: jax.random.PRNGKey,
                    env_specs: EnvironmentSpecs,
                    ):
        networks = self.make_networks(env_specs)
        params = networks.init(random_key)

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
        checkpointer = reverb.checkpointers.DefaultCheckpointer(
            self.cfg.logdir + "/reverb")
        server = reverb.Server(tables,
                               self.cfg.reverb_port,
                               checkpointer)
        return server

    def make_dataset_iterator(self, server_address: str):
        ds: tf.data.Dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=server_address,
            table="replay_buffer",
            max_in_flight_samples_per_worker=2 * self.cfg.batch_size,
        )
        ds = ds.batch(self.cfg.batch_size, drop_remainder=True)
        ds = ds.prefetch(5)
        return ds.as_numpy_iterator()

    def make_networks(self, env_specs: EnvironmentSpecs):
        return make_networks(self.cfg,
                             env_specs.observation_spec,
                             env_specs.action_spec)

    def make_learner(self,
                     random_key: jax.random.PRNGKey,
                     env_specs: EnvironmentSpecs,
                     iterator: tf.data.Dataset,
                     client: reverb.Client
                     ):
        networks = self.make_networks(env_specs)
        return MPOLearner(random_key, self.cfg, env_specs,
                          networks, iterator, client)

    def make_actor(self,
                   random_key: jax.random.PRNGKey,
                   env: dm_env.Environment,
                   env_specs: EnvironmentSpecs,
                   client: reverb.Client
                   ):
        networks = self.make_networks(env_specs)
        return Actor(random_key, env, self.cfg,
                     networks, client, self._total_steps)

    def make_env(self, random_key: jax.random.PRNGKey):

        seed = jax.random.randint(random_key, (), 0, 2**16).item()
        domain, task = self.cfg.task.split("_", 1)
        if domain == "dmc":
            env = envs.DMC(task, seed, self.cfg.img_size, 0, self.cfg.pn_number)
        elif domain == "ur":
            assert self.cfg.action_repeat == 1
            address = ("10.201.2.136", 5553)
            env = envs.UR5(address, self.cfg.img_size, self.cfg.pn_number)
        else:
            raise NotImplementedError

        env = dmc_wrappers.ActionRepeat(env, self.cfg.action_repeat)
        if self.cfg.discretize:
            env = dmc_wrappers.DiscreteActionWrapper(env, self.cfg.nbins)
        else:
            env = dmc_wrappers.ActionRescale(env)
        return env, env.environment_specs
