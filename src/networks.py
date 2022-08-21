from typing import NamedTuple, Callable, Iterable

import jax
import jax.numpy as jnp
import haiku as hk
import jmp
import tensorflow_probability.substrates.jax as tfp

from src.config import MPOConfig
from src.utils import TruncatedTanh


tfd = tfp.distributions
_MIN_STDDEV = 1e-5


class Actor(hk.Module):
    def __init__(self,
                 act_dim: int,
                 layers: Iterable[int],
                 mean_scale: int,
                 activation: Callable,
                 ):
        super().__init__(name="actor")
        self.mean_scale = mean_scale
        self._mlp = hk.nets.MLP(
            output_sizes=list(layers)+[2*act_dim],
            w_init=hk.initializers.Orthogonal(),
            b_init=jnp.zeros,
            activation=activation,
        )

    def __call__(self, state):
        """Actor returns params instead of a distribution itself
        since tfp.Distribution doesn't play nicely with jax.vmap."""
        out = self._mlp(state)
        mean, stddev = jnp.split(out, 2, -1)
        mean = self.mean_scale * jnp.tanh(mean / self.mean_scale)
        stddev = jax.nn.softplus(stddev) + _MIN_STDDEV
        return mean, stddev


class QuantileNetwork(hk.Module):
    """Quantile network from IQN (arxiv/)."""
    def __init__(self, output_dim: int, embedding_dim: int):
        super().__init__(name="quantile_embedding")
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

    def __call__(self, tau):
        x = jnp.arange(self.embedding_dim, dtype=tau.dtype)
        x = jnp.expand_dims(x, axis=0)
        x = jnp.cos(jnp.pi * tau @ x)
        x = hk.Linear(self.output_dim)(x)
        return jax.nn.relu(x)


class DistributionalCritic(hk.Module):
    """IQN"""
    def __init__(self,
                 layers: Iterable[int],
                 quantile_embedding_dim: int
                 ):
        super().__init__(name="critic")
        self._net = hk.nets.MLP(list(layers) + [1])
        self.quantile_embedding_dim = quantile_embedding_dim

    def __call__(self, observation, action, taus):
        x = jnp.concatenate([observation, action], axis=-1)
        x = jnp.expand_dims(x, axis=-2)
        taus = QuantileNetwork(x.shape[-1], self.quantile_embedding_dim)(taus)
        x = self._net(x * taus)
        return jnp.squeeze(x, axis=-1)


class MultimodalEncoder(hk.Module):
    def __init__(self, config: MPOConfig):
        super().__init__(name="encoder")
        self._c = config

    def __call__(self, observations):
        embeddings = jax.tree_util.tree_map(self.encode, observations)
        values, _ = jax.tree_util.tree_flatten(embeddings)
        return jnp.concatenate(values, axis=-1)

    def encode(self, value):
        ndim = jnp.ndim(value)
        encoders = [self._mlp, self._pn, self._cnn]
        return jax.lax.switch(ndim - 1, encoders, value)

    def _mlp(self, x):
        return x

    def _cnn(self, x):
        return x

    def _pn(self, x):
        return x


class MPONetworks(NamedTuple):
    init: Callable
    actor: Callable
    make_policy: Callable
    critic: Callable
    encoder: Callable


def make_networks(config, observation_spec, action_spec):
    dummy_obs = jax.tree_util.tree_map(
        lambda t: jnp.zeros(t.shape), observation_spec)
    act_dim = action_spec.shape[0]
    prec = jmp.get_policy(config.mp_policy)

    hk.mixed_precision.set_policy(Actor, prec)
    hk.mixed_precision.set_policy(DistributionalCritic, prec)
    hk.mixed_precision.set_policy(MultimodalEncoder, prec)

    @hk.without_apply_rng
    @hk.transform
    def encode_fn(observation):
        return MultimodalEncoder(config)(observation)

    @hk.without_apply_rng
    @hk.transform
    def policy_fn(state):
        actor = Actor(
            act_dim,
            config.actor_layers,
            config.mean_scale,
            activation=jax.nn.relu
        )
        return actor(state)

    @hk.without_apply_rng
    @hk.transform
    def critic_fn(state, action, quantile):
        critic = DistributionalCritic(
            config.critic_layers,
            config.quantile_embedding_dim
        )
        return critic(state, action, quantile)

    def make_policy(mean, stddev):
        return tfd.TransformedDistribution(
            tfd.Normal(mean, stddev),
            TruncatedTanh()
        )

    def init(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        encoder_params = encode_fn.init(k1, dummy_obs)
        state = encode_fn.apply(encoder_params, dummy_obs)
        actor_params = policy_fn.init(k2, state)
        action, _ = policy_fn.apply(actor_params, state)
        tau = jnp.array([1.])
        critic_params = critic_fn.init(k3, state, action, tau)
        dual_params = (jnp.zeros(()), jnp.zeros(act_dim), jnp.zeros(act_dim))
        return encoder_params, critic_params, actor_params, dual_params

    return MPONetworks(
        init=init,
        actor=policy_fn.apply,
        make_policy=make_policy,
        critic=critic_fn.apply,
        encoder=encode_fn.apply,
    )

