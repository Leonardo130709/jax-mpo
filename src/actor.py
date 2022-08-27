from typing import NamedTuple, Any
import functools
import itertools

import chex
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import reverb

from .networks import MPONetworks
from .config import MPOConfig


class Transitions(NamedTuple):
    observations: Any
    actions: Any
    rewards: Any
    next_observations: Any


CPU = jax.devices('cpu')[0]


class Actor:
    def __init__(self,
                 rng_key,
                 env,
                 config: MPOConfig,
                 networks: MPONetworks,
                 client: reverb.Client):

        @functools.partial(jax.jit, backend='cpu', static_argnums=(3,))
        @chex.assert_max_traces(n=2)
        def _act(params, key, observation, training):
            state = networks.encoder(params, observation)
            policy_params = networks.actor(params, state)
            dist = networks.make_policy(*policy_params)
            if training:
                action = dist.sample(seed=key)
            else:
                action = dist.bijector(dist.distribution.mean())
            return action

        self._env = env
        self._rng_seq = hk.PRNGSequence(rng_key)
        self._act = _act
        self.config = config
        self._client = client
        self.num_interactions = 0
        self._params = None
        self._weights_ds = reverb.TimestepDataset.from_table_signature(
            client.server_address,
            table='weights',
            max_in_flight_samples_per_worker=1,
            num_workers_per_iterator=1,
        ).as_numpy_iterator()

    def simulate(self, env, training):
        step = 0
        seq_len = self.config.seq_len
        self._rng_seq.reserve(1000) # // self.config.action_repeat)
        nested_slice = lambda x: jax.tree_util.tree_map(lambda t: t[-seq_len:], x)
        timestep = env.reset()
        with self._client.trajectory_writer(seq_len) as writer:
            while not timestep.last():
                observation = timestep.observation
                action = self.act(observation, training)
                timestep = env.step(action)
                writer.append(
                    dict(
                        observations=observation,
                        actions=action,
                        rewards=timestep.reward,
                        next_observations=timestep.observation
                    )
                )
                step += 1

                if step >= seq_len:
                    writer.create_item(
                        table='replay_buffer',
                        priority=1.,
                        trajectory=nested_slice(writer.history)
                    )
                    writer.flush()
            writer.end_episode()

        return step

    def act(self, observation, training):
        rng = next(self._rng_seq)
        action = self._act(self._params, rng, observation, training)
        return np.asarray(action)

    def get_params(self):
        params = next(self._weights_ds).data
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, CPU), params)

    def interact(self):
        if self.config.total_episodes > 0:
            episodes = range(self.config.total_episodes)
        else:
            episodes = itertools.count()

        for _ in episodes:
            self._params = self.get_params()
            steps = self.simulate(self._env, training=True)
            self.num_interactions += steps
            print(f'Actor interactions:{self.num_interactions}')
