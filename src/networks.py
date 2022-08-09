from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


class Actor(hk.Module):
    def __init__(self, act_dim, layers, mean_scale, activation):
        super().__init__(name='actor')
        self.mean_scale = mean_scale
        self.mlp = hk.nets.MLP(
            output_sizes=list(layers)+[2*act_dim],
            w_init=hk.initializers.Orthogonal(),
            b_init=jnp.zeros,
            activation=activation,
        )

    def __call__(self, state):
        out = self.mlp(state)
        mean, stddev = jnp.split(out, 2, -1)
        mean = self.mean_scale * jnp.tanh(mean / self.mean_scale)
        stddev = jax.nn.softplus(stddev) + 1e-3
        dist = tfd.Normal(mean, stddev)
        return tfp.bijectors.Tanh()(dist)


class MPOModel(hk.Module):
    def __init__(self, config, act_dim):
        super().__init__(name="mpo_spr_networks")
        self.config = config
        self.encoder = lambda x: x
        self.actor = Actor(act_dim, config.actor_layers, 1., jax.nn.elu)
        self.critic = hk.nets.MLP(list(config.critic_layers) + [1], name='critic')
        self.proj = hk.Linear(
            config.hidden_dim, name='projection_head')
        self.prediction_head = hk.Linear(
            config.hidden_dim, name='prediction_head')
        self.dynamics = hk.VanillaRNN(config.hidden_dim)

    def projection(self, observation):
        state = self.encoder(observation)
        return self.proj(state)

    def q_value(self, observation, action):
        state = self.projection(observation)
        x = jnp.concatenate([state, action], axis=-1)
        return self.critic(x)

    def policy(self, observation):
        state = self.projection(observation)
        state = jax.lax.stop_gradient(state)
        return self.actor(state)

    def prediction(self, observations, actions):
        raise NotImplementedError


class MPONetworks(NamedTuple):
    init: Any
    q_value: Any
    policy: Any


def make_networks(config, act_dim):
    @hk.without_apply_rng
    @hk.transform
    def q_value(observation, action):
        model = MPOModel(config, act_dim)
        return jnp.squeeze(model.q_value(observation, action), axis=-1)

    @hk.without_apply_rng
    @hk.transform
    def policy(observation):
        model = MPOModel(config, act_dim)
        return model.policy(observation)

    def init(rng, observation, action):
        p1 = policy.init(rng, observation)
        p2 = q_value.init(rng, observation, action)
        return hk.data_structures.merge(p1, p2)

    return MPONetworks(init=init, q_value=q_value.apply, policy=policy.apply)
