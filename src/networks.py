from typing import NamedTuple, Callable, Iterable
import re

import jax
import jmp
import jax.numpy as jnp
import chex
import haiku as hk
import tensorflow_probability.substrates.jax as tfp
from dm_env import specs

from src.config import MPOConfig

tfd = tfp.distributions


def get_act(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name == 'identity':
        return lambda x: x
    elif hasattr(jax.nn, name):
        return getattr(jax.nn, name)
    elif hasattr(jax, name):
        return getattr(jax, name)
    else:
        raise ValueError


class NormLayer(hk.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def __call__(self, x):
        if self.norm == 'layer':
            return hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(x)
        return x


class MLP(hk.Module):
    """MLP w/o dropout but with preactivation normalization."""

    def __init__(self,
                 output_size: int,
                 layers: Iterable[int],
                 act: str = 'elu',
                 norm: str = 'none',
                 activate_final: bool = False,
                 name: str = None
                 ):
        super().__init__(name=name)
        output_sizes = tuple(layers) + (output_size,)
        num_layers = len(output_sizes)
        layers = []

        for i, output_size in enumerate(output_sizes):
            layers.append(hk.Linear(output_size))
            if i < num_layers - 1 or activate_final:
                layers.append(NormLayer(norm))
                layers.append(get_act(act))

        self._mlp = hk.Sequential(layers)

    def __call__(self, x):
        return self._mlp(x)


class Actor(hk.Module):
    def __init__(self,
                 act_dim,
                 layers: Iterable[int],
                 act: str = 'elu',
                 norm: str = 'none',
                 min_std: float = 0.1
                 ):
        super().__init__(name='actor')
        # Add discrete params.
        self.min_std = min_std
        self._net = MLP(2 * act_dim, layers, act, norm)

    def __call__(self, state):
        """Actor returns params instead of a distribution itself
        since tfd.Distribution doesn't play nicely with jax.vmap."""
        out = self._net(state)
        mean, stddev = jnp.split(out, 2, -1)
        mean = jnp.tanh(mean)
        stddev = jax.nn.softplus(stddev) + self.min_std
        return mean, stddev


class QuantileNetwork(hk.Module):
    """Quantile network from the IQN paper (1806.06923)."""

    def __init__(self, output_dim: int, embedding_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

    def __call__(self, tau):
        chex.assert_tree_shape_suffix(tau, (1,))
        x = jnp.arange(self.embedding_dim, dtype=tau.dtype)
        x = jnp.expand_dims(x, 0)
        x = jnp.cos(jnp.pi * tau @ x)
        x = hk.Linear(self.output_dim)(x)
        return jax.nn.relu(x)


class DistributionalCritic(hk.Module):
    def __init__(self,
                 layers: Iterable[int],
                 quantile_embedding_dim: int,
                 act: str = 'elu',
                 norm: str = 'none',
                 ):
        super().__init__(name='critic')
        self._net = MLP(1, layers, act, norm)
        self.quantile_embedding_dim = quantile_embedding_dim

    def __call__(self, state, action, tau):
        chex.assert_equal_rank([state, action, tau])

        x = jnp.concatenate([state, action], -1)
        x = jnp.expand_dims(x, -2)
        tau = QuantileNetwork(x.shape[-1], self.quantile_embedding_dim)(tau)
        x = self._net(x * tau)
        return jnp.squeeze(x, -1)


class Encoder(hk.Module):
    def __init__(self,
                 mlp_keys: str,
                 cnn_keys: str,
                 mlp_layers: Iterable[int],
                 cnn_kernels: Iterable[int],
                 cnn_depth: int,
                 act: str,
                 norm: str,
                 feature_fusion: bool = False,
                 ):
        super().__init__(name='encoder')
        self.mlp_keys = mlp_keys
        self.mlp_layers = tuple(mlp_layers)
        self.cnn_keys = cnn_keys
        self.cnn_kernels = tuple(cnn_kernels)
        self.cnn_depth = cnn_depth
        self.act = act
        self.norm = norm
        self.feature_fusion = feature_fusion

    def __call__(self, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        def match_concat(pattern, ndim):
            values = [
                val for key, val in obs.items()
                if re.match(pattern, key) and val.ndim == ndim
            ]
            if values:
                return jnp.concatenate(values, -1)
            return None

        mlp_features = match_concat(self.mlp_keys, 1)
        cnn_features = match_concat(self.cnn_keys, 3)

        if self.feature_fusion and cnn_features is not None:
            if mlp_features is not None:
                mlp_features = jnp.tile(
                    mlp_features,
                    reps=cnn_features.shape[:2] + (1,)
                )
                cnn_features = jnp.concatenate([cnn_features, mlp_features], -1)
            return self._cnn(cnn_features)

        outputs = []
        if mlp_features is not None:
            outputs.append(self._mlp(mlp_features))
        if cnn_features is not None:
            outputs.append(self._cnn(cnn_features))
        return jnp.concatenate(outputs, -1)

    def _cnn(self, x):
        for kernel in self.cnn_kernels:
            x = hk.Conv2D(self.cnn_depth, kernel, 2)(x)
            x = NormLayer(self.norm)(x)
            x = get_act(self.act)(x)
        return x.reshape(-1)

    def _mlp(self, x):
        *layers, out_dim = self.mlp_layers
        return MLP(
            out_dim,
            layers,
            self.act,
            self.norm,
            activate_final=True
        )(x)


class MPONetworks(NamedTuple):
    init: Callable
    encoder: Callable
    actor: Callable
    critic: Callable
    make_policy: Callable


def make_networks(cfg: MPOConfig, observation_spec, action_spec):
    prec = jmp.get_policy(cfg.mp_policy)
    hk.mixed_precision.set_policy(Encoder, prec)
    hk.mixed_precision.set_policy(Actor, prec)
    hk.mixed_precision.set_policy(DistributionalCritic, prec)

    if isinstance(action_spec, specs.BoundedArray):
        act_dim = action_spec.shape[0]
    elif isinstance(action_spec, specs.DiscreteArray):
        act_dim = action_spec.num_values
    else:
        raise ValueError

    obs = {
        k: spec.generate_value().astype(prec.compute_dtype)
        for k, spec in observation_spec.items()
    }

    @hk.without_apply_rng
    @hk.multi_transform
    def _():
        encoder = Encoder(
            cfg.mlp_keys,
            cfg.cnn_keys,
            cfg.mlp_layers,
            cfg.cnn_kernels,
            cfg.cnn_depth,
            cfg.activation,
            cfg.normalization,
            cfg.feature_fusion
        )
        actor = Actor(
            act_dim,
            cfg.actor_layers,
            cfg.activation,
            cfg.normalization,
            cfg.min_std
        )
        critic = DistributionalCritic(
            cfg.critic_layers,
            cfg.quantile_embedding_dim,
            cfg.activation,
            cfg.normalization
        )

        def init():
            state = encoder(obs)
            policy_params = actor(state)
            dist = make_policy(*policy_params)
            key = hk.next_rng_key()
            action = dist.sample(seed=key)
            tau = jnp.ones(1)
            value = critic(state, action, tau)
            return state, action, value

        return init, (encoder, actor, critic)

    def make_policy(mean, stddev):
        return tfd.TruncatedNormal(mean, stddev, -1, 1)

    encoder_fn, actor_fn, critic_fn = _.apply
    return MPONetworks(
        init=_.init,
        encoder=encoder_fn,
        actor=actor_fn,
        critic=critic_fn,
        make_policy=make_policy,
    )

