from typing import NamedTuple, Any
import functools

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

        @functools.partial(jax.jit, backend='cpu')
        def _act(params, key, observation, training):
            state = networks.encoder(params, observation)
            policy_params = networks.actor(params, state)
            dist = networks.make_policy(*policy_params)
            action = jax.lax.switch(
                training,
                dist.sample(seed=key),
                dist.bijector(dist.distribution.mean())
            )
            return action

        self._env = env
        self._rng_seq = hk.PRNGSequence(rng_key)
        self._act = _act
        self.config = config
        self._client = client
        self.num_interactions = 0
        self._params = self.get_params()

    def simulate(self, env, training):
        step = 0
        seq_len = self.config.seq_len
        nested_slice = lambda x: jax.tree_util.tree_map(lambda t: t[:seq_len], x)
        timestep = env.reset()
        writer: reverb.Writer = self._client.trajectory_writer(
            num_keep_alive_refs=seq_len)

        while not timestep.last():
            step += 1
            observation = timestep.observation
            action = self.act(observation, training)
            timestep = env.step(action)
            transition = dict(
                observation=observation,
                action=action,
                reward=np.array(timestep.reward, dtype=np.float32)[np.newaxis],
                next_observation=timestep.observation
            )
            writer.append(transition)

            if step > seq_len:
                writer.create_item(
                    table='replay_buffer',
                    priority=1.,
                    trajectory=nested_slice(writer.history)
                )
        return step

    def act(self, observation, training):
        rng = next(self._rng_seq)
        action = self._act(self._params, rng, observation, training)
        return np.asarray(action)

    def get_params(self):
        params = self._client.sample(table='params')
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, CPU), params)

    def interact(self):
        while True:
            self._params = self.get_params()
            steps = self.simulate(self._env, training=True)
            self.num_interactions += steps


