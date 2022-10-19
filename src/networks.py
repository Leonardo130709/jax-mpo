from typing import NamedTuple, Callable, Iterable, Dict
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
    if name == "none":
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
        if self.norm == "layer":
            return hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(x)
        return x


class MLP(hk.Module):
    """MLP w/o dropout but with preactivation normalization."""

    def __init__(self,
                 layers: Iterable[int],
                 act: str = "elu",
                 norm: str = "none",
                 activate_final: bool = False,
                 name: str = None
                 ):
        super().__init__(name=name)
        output_sizes = tuple(layers)
        num_layers = len(output_sizes)
        layers = []

        for i, output_size in enumerate(output_sizes):
            layers.append(hk.Linear(output_size))
            if i < (num_layers - 1) or activate_final:
                layers.append(NormLayer(norm))
                layers.append(get_act(act))

        self._mlp = hk.Sequential(layers)

    def __call__(self, x):
        return self._mlp(x)


class Actor(hk.Module):
    def __init__(self,
                 action_spec,
                 layers: Iterable[int],
                 act: str = "elu",
                 norm: str = "none",
                 min_std: float = 0.1
                 ):
        super().__init__(name="actor")

        self._discrete = isinstance(action_spec, specs.DiscreteArray)
        if self._discrete:
            # act_dim = action_spec.num_values
            raise NotImplementedError
        else:
            output_dim = 2 * action_spec.shape[0]

        self._net = MLP(tuple(layers) + (output_dim,), act, norm)
        self.min_std = min_std

    def __call__(self, state):
        """Actor returns params instead of a distribution itself
        since tfd.Distribution doesn't play nicely with jax.vmap."""
        out = self._net(state)
        if not self._discrete:
            mean, std = jnp.split(out, 2, -1)
            mean = jnp.tanh(mean)
            std = jax.nn.softplus(std) + self.min_std
            return mean, std

        raise NotImplementedError


class QuantileNetwork(hk.Module):
    """Quantile network from the IQN paper (1806.06923)."""

    def __init__(self, output_dim: int, embedding_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim

    def __call__(self, tau):
        x = jnp.arange(self.embedding_dim, dtype=tau.dtype)
        x = jnp.expand_dims(x, 0)
        tau = jnp.expand_dims(tau, -1)
        x = jnp.cos(jnp.pi * tau @ x)
        x = hk.Linear(self.output_dim)(x)
        return jax.nn.relu(x)


class DistributionalCritic(hk.Module):
    def __init__(self,
                 layers: Iterable[int],
                 quantile_embedding_dim: int,
                 act: str = "elu",
                 norm: str = "none",
                 ):
        super().__init__(name="critic")
        self.layers = tuple(layers)
        self.act = act
        self.norm = norm
        self.quantile_embedding_dim = quantile_embedding_dim

    def __call__(self, state, action, tau):
        chex.assert_equal_rank([state, action, tau])

        x = jnp.concatenate([state, action], -1)
        tau = QuantileNetwork(x.shape[-1], self.quantile_embedding_dim)(tau)
        # w: implicit broadcasting
        x = jnp.expand_dims(x, -2)
        x = MLP(self.layers + (1,), self.act, self.norm)(x * tau)
        return jnp.squeeze(x, -1)


class DDCritic(DistributionalCritic):
    def __call__(self, *args, **kwargs):
        z1 = super().__call__(*args, **kwargs)
        z2 = super().__call__(*args, **kwargs)
        return jnp.stack([z1, z2], -1)


class Encoder(hk.Module):
    def __init__(self,
                 mlp_keys: str,
                 pn_keys: str,
                 cnn_keys: str,
                 mlp_layers: Iterable[int],
                 pn_layers: Iterable[int],
                 cnn_kernels: Iterable[int],
                 cnn_depth: int,
                 act: str,
                 norm: str,
                 feature_fusion: bool = False,
                 ):
        super().__init__(name="encoder")
        self.mlp_keys = mlp_keys
        self.pn_keys = pn_keys
        self.cnn_keys = cnn_keys
        self.mlp_layers = tuple(mlp_layers)
        self.pn_layers = tuple(pn_layers)
        self.cnn_kernels = tuple(cnn_kernels)
        self.cnn_depth = cnn_depth
        self.act = act
        self.norm = norm
        self.feature_fusion = feature_fusion

    def __call__(self, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Works with unbatched inputs,
        since there is jax.vmap and hk.BatchApply."""
        def match_concat(pattern, ndim):
            values = [
                val for key, val in obs.items()
                if re.match(pattern, key) and val.ndim == ndim
            ]
            if values:
                return jnp.concatenate(values, -1)
            return None

        mlp_features = match_concat(self.mlp_keys, 1)
        pn_features = match_concat(self.pn_keys, 2)
        cnn_features = match_concat(self.cnn_keys, 3)

        if self.feature_fusion and cnn_features is not None:
            if mlp_features is not None:
                mlp_features = jnp.tile(
                    mlp_features,
                    reps=cnn_features.shape[:2] + (1,)  # HWC
                )
                cnn_features = jnp.concatenate([cnn_features, mlp_features], -1)
                mlp_features = None

        outputs = []
        if mlp_features is not None:
            outputs.append(self._mlp(mlp_features))
        if pn_features is not None:
            outputs.append(self._pn(pn_features))
        if cnn_features is not None:
            outputs.append(self._cnn(cnn_features))
        if not outputs:
            raise ValueError(f"No valid key: {obs.keys()}")

        return jnp.concatenate(outputs, -1)

    def _cnn(self, x):
        for kernel in self.cnn_kernels:
            x = hk.Conv2D(self.cnn_depth, kernel, 2)(x)
            x = NormLayer(self.norm)(x)
            x = get_act(self.act)(x)
        return x.reshape(-1)

    def _mlp(self, x):
        return MLP(
            self.mlp_layers,
            self.act,
            self.norm,
            activate_final=True
        )(x)

    def _pn(self, x):
        *layers, out_dim = self.pn_layers
        x = MLP(layers,
                self.act,
                self.norm,
                activate_final=True
                )(x)
        x = jnp.max(x, axis=-2)
        return hk.Linear(out_dim)(x)


class MPONetworks(NamedTuple):
    init: Callable
    encoder: Callable
    actor: Callable
    critic: Callable
    make_policy: Callable


def make_networks(cfg: MPOConfig,
                  observation_spec: Dict[str, specs.Array],
                  action_spec: specs.BoundedArray | specs.DiscreteArray
                  ) -> MPONetworks:
    prec = jmp.get_policy(cfg.mp_policy)
    hk.mixed_precision.set_policy(Encoder, prec)
    hk.mixed_precision.set_policy(Actor, prec)
    hk.mixed_precision.set_policy(DistributionalCritic, prec)

    is_discrete = isinstance(action_spec, specs.DiscreteArray)
    obs = {
        k: spec.generate_value()
        for k, spec in observation_spec.items()
    }
    obs = prec.cast_to_compute(obs)

    @hk.without_apply_rng
    @hk.multi_transform
    def model():
        encoder = Encoder(
            cfg.mlp_keys,
            cfg.pn_keys,
            cfg.cnn_keys,
            cfg.mlp_layers,
            cfg.pn_layers,
            cfg.cnn_kernels,
            cfg.cnn_depth,
            cfg.activation,
            cfg.normalization,
            cfg.feature_fusion
        )
        actor = Actor(
            action_spec,
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

    def make_policy(*params: chex.Array) -> tfd.Distribution:
        if is_discrete:
            logits = params[0]
            dist = tfd.OneHotCategorical(logits)
        else:
            mean, std = params
            # TruncNormal gives wrong kl.
            # dist = tfd.TruncatedNormal(mean, std, -1, 1)
            dist = tfd.Normal(mean, std)
        return tfd.Independent(dist, 1)

    encoder_fn, actor_fn, critic_fn = model.apply
    return MPONetworks(
        init=model.init,
        encoder=encoder_fn,
        actor=actor_fn,
        critic=critic_fn,
        make_policy=make_policy,
    )


class _MultiHeadAttentionEncoder(hk.Module):
    def __init__(self,
                 emb_dim: int,
                 keys: str,
                 mlp_layers: Iterable[int],
                 pn_layers: Iterable[int],
                 cnn_kernels: Iterable[int],
                 cnn_depth: Iterable[int],
                 act: str,
                 norm: str,
                 name: str = None,
                 ):
        super().__init__(name=name)
        self.emb_dim = emb_dim
        self.keys = keys
        self.mlp_layers = mlp_layers
        self.pn_layers = pn_layers
        self.cnn_kernels = cnn_kernels
        self.cnn_depth = cnn_depth
        self.act = get_act(act)
        self.norm = norm

    def __call__(self, obs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        obs = {
            k: v for k, v in obs.items()
            if re.match(self.keys, k)
        }
        chex.assert_type(list(obs.values()), float)
        chex.assert_type(list(obs.values()), {1, 2, 3})

        def call(feat):
            return jax.lax.switch(
                feat.ndim - 1,
                [self._mlp, self._pn, self._cnn],
                feat
            )
        outputs = jax.tree_util.tree_map(call, obs)
        values = jnp.stack(jax.tree_util.tree_leaves(outputs))
        qkv = jnp.split(values, 3, -1)
        mha = hk.MultiHeadAttention(1, self.emb_dim, 1.)(*qkv)

        return hk.Linear(self.emb_dim)(mha.reshape(-1))

    def _mlp(self, x):
        return x

    def _pn(self, x):
        pass

    def _cnn(self, x):
        return x
