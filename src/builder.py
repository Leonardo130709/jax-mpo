from typing import NamedTuple
from dm_control import suite
from dm_env import specs
from dmc_wrappers import StatesWrapper

import reverb
import tensorflow as tf
import jax
import optax

from .config import MPOConfig
from .actor import Transitions, Actor
from .networks import MPONetworks, make_networks
from .learner import MPOLearner, MPOState


class EnvironmentSpecs(NamedTuple):
    observation: specs.Array
    action: specs.Array
    reward: specs.Array


class Builder:
    def __init__(self, config: MPOConfig):
        self._c = config

    def make_reverb_tables(self, env_specs):
        def from_specs(spec):
            tf_spec = lambda x: tf.TensorSpec((self._c.seq_len,) + tuple(x.shape), dtype=x.dtype)
            return jax.tree_util.tree_map(tf_spec, spec)
        signature = Transitions(
            observations=from_specs(env_specs.observation),
            actions=from_specs(env_specs.action),
            rewards=from_specs(env_specs.reward),
            next_observations=from_specs(env_specs.observation)
        )
        return [
            reverb.Table(
                name='replay_buffer',
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self._c.buffer_capacity,
                rate_limiter=reverb.rate_limiters.SampleToInsertRatio(
                    min_size_to_sample=self._c.min_replay_size,
                    samples_per_insert=self._c.samples_per_insert,
                    error_buffer=self._c.samples_per_insert,
                ),
                signature=signature
            ),
            reverb.Table(
                name='weights',
                sampler=reverb.selectors.Lifo(),
                remover=reverb.selectors.Fifo(),
                max_size=1,
                rate_limiter=reverb.rate_limiters.MinSize(1)
            )
        ]

    def make_dataset_iterator(self, server_address):
        ds: tf.data.Dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=server_address,
            table='replay_buffer',
            max_in_flight_samples_per_worker=2*self._c.batch_size,
        )
        ds = ds.batch(self._c.batch_size)
        ds = ds.prefetch(5)
        return ds.as_numpy_iterator()

    def make_networks(self, env_specs):
        return make_networks(self._c, env_specs.observation, env_specs.action)

    def make_learner(self,
                     key: jax.random.PRNGKey,
                     dataset: tf.data.Dataset,
                     networks: MPONetworks
                     ):

        key1, key2 = jax.random.split(key)

        optim = optax.multi_transform({
            'encoder': optax.adam(self._c.encoder_lr),
            'critic': optax.adam(self._c.critic_lr),
            'actor': optax.adam(self._c.actor_lr),
            'dual_params': optax.adam(self._c.dual_lr),
        },
            ('encoder', 'critic', 'actor', 'dual_params')
        )

        params = networks.init(key1)
        optim_state = optim.init(params)
        encoder_params, critic_params, actor_params, dual_params = params

        state = MPOState(
            actor_params=actor_params,
            target_actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=critic_params,
            encoder_params=encoder_params,
            target_encoder_params=encoder_params,
            optim_state=optim_state,
            dual_params=dual_params,
            key=key2
        )

        return MPOLearner(
            self._c,
            networks,
            optim,
            dataset,
            state
        )

    def make_actor(self, rng_key, env, networks, client):
        return Actor(rng_key, env, self._c, networks, client)

    def make_env(self):
        domain, task = self._c.task.split('_', maxsplit=1)
        if domain == "ball":
            domain = "bal_in_cup"
            task = "catch"
        env = suite.load(domain, task,
                         task_kwargs={'random': self._c.seed},
                         environment_kwargs=None)

        env_specs = EnvironmentSpecs(
            observation=env.observation_spec(),
            action=env.action_spec(),
            reward=env.reward_spec(),
        )
        return StatesWrapper(env), env_specs
