from typing import NamedTuple, Callable, Iterable, Dict, Tuple, Optional
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
Layers = Iterable[int]


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
    """MLP w/o dropout but with preactivation normalization.
    Weights and biases initialization are the haiku default: trunc_normal.
    """

    def __init__(self,
                 output_sizes: Layers,
                 act: str,
                 norm: str,
                 activate_final: bool = False,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        output_sizes = tuple(output_sizes)
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
                 action_spec: specs.BoundedArray,
                 layers: Layers,
                 act: str = "elu",
                 norm: str = "none",
                 min_std: float = 0.1,
                 init_std: float = 1.,
                 name: str = "actor"
                 ):
        super().__init__(name=name)
        self.act_dim = action_spec.shape[0]
        self.layers = tuple(layers)
        self.act = act
        self.norm = norm
        self.min_std = min_std
        self._log_init_std = jnp.log(jnp.exp(init_std - min_std) - 1.)

    def __call__(self, state):
        """Actor returns params instead of a distribution itself
        since tfd.Distribution doesn't play nicely with jax.vmap."""
        mlp = MLP(self.layers, self.act, self.norm, activate_final=True)
        out = hk.Linear(2 * self.act_dim,
                        w_init=hk.initializers.RandomNormal(stddev=1e-2),
                        b_init=jnp.zeros
                        )
        x = mlp(state)
        x = out(x)
        mean, std = jnp.split(x, 2, -1)
        mean = jnp.tanh(mean)
        std = jax.nn.softplus(std + self._log_init_std) + self.min_std
        return mean, std


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
                 name: str = "critic"
                 ):
        super().__init__(name=name)
        self.layers = tuple(layers)
        self.act = act
        self.norm = norm
        self.quantile_embedding_dim = quantile_embedding_dim

    def __call__(self, state, action, tau):
        chex.assert_equal_rank([state, action, tau])

        x = jnp.concatenate([state, action], -1)
        tau = QuantileNetwork(x.shape[-1], self.quantile_embedding_dim)(tau)
        x = jnp.expand_dims(x, -2)
        # warn: implicit broadcast.
        return MLP(self.layers + (1,), self.act, self.norm)(x * tau)


class Critic(DistributionalCritic):
    """Ordinary MLP critic."""
    def __call__(self, state, action, tau=None):
        chex.assert_equal_rank([state, action])
        x = jnp.concatenate([state, action], -1)
        mlp = MLP(self.layers + (1,), self.act, self.norm)
        return mlp(x)


class CriticsEnsemble(hk.Module):

    def __init__(self,
                 num_heads: int,
                 use_iqn: bool,
                 *args,
                 name="critic",
                 **kwargs):
        super().__init__(name=name)
        self.num_heads = num_heads
        critic_cls = DistributionalCritic if use_iqn else Critic
        self._factory = lambda n: critic_cls(*args, name=n, **kwargs)

    def __call__(self, *args, **kwargs):
        values = []
        for i in range(self.num_heads):
            critic = self._factory(f"critic_{i}")
            values.append(critic(*args, **kwargs))

        return jnp.concatenate(values, -1)


class Encoder(hk.Module):
    def __init__(self,
                 keys: str,
                 emb_dim: int,
                 mlp_layers: Layers,
                 pn_layers: Layers,
                 cnn_kernels: Layers,
                 cnn_depths: Layers,
                 act: str,
                 norm: str,
                 feature_fusion: str = "none",
                 name: str = "encoder"
                 ):
        super().__init__(name=name)
        self.keys = keys
        self.emb_dim = emb_dim
        self.mlp_layers = tuple(mlp_layers)
        self.pn_layers = tuple(pn_layers)
        self.cnn_kernels = tuple(cnn_kernels)
        self.cnn_depths = tuple(cnn_depths)
        self.act = act
        self.norm = norm
        self.feature_fusion = feature_fusion

    def __call__(self, obs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Works with unbatched inputs,
        since there are jax.vmap and hk.BatchApply."""
        chex.assert_rank(list(obs.values()), {1, 2, 3})
        mlp_features, pn_features, cnn_features = _ndim_partition(
            {k: v for k, v in obs.items() if re.match(self.keys, k)}
        )
        outputs = []

        if mlp_features:
            mlp_features = list(mlp_features.values())
            mlp_features = jnp.concatenate(mlp_features, -1)
            return mlp_features
            # Also fuse these features with the multidimensional inputs.
            fuse_with = filter(
                lambda k: re.match(self.feature_fusion, k), obs.keys()
            )
            fuse_with = tuple(fuse_with)
            sources = [pn_features, cnn_features]
            for key in fuse_with:
                source_ndim = obs[key].ndim
                chex.assert_scalar_non_negative(source_ndim - 2)
                source = sources[source_ndim - 2]
                source_feat = source[key]
                tiled_mlp_feat = jnp.tile(
                    mlp_features,
                    reps=source_feat.shape[:-1] + (1,)
                )
                source[key] = jnp.concatenate(
                    [source_feat, tiled_mlp_feat], -1
                )

            if not fuse_with:
                outputs.append(self._mlp(mlp_features))

        for pcd in pn_features.values():
            outputs.append(self._pn(pcd))
        for image in cnn_features.values():
            outputs.append(self._cnn(image))
        if not outputs:
            raise ValueError(f"No valid {self.keys!r} in {obs.keys()}")

        state = jnp.concatenate(outputs, -1)
        emb = MLP((self.emb_dim,),
                  #TODO: remove tanh
                  act="tanh",
                  norm=self.norm,
                  activate_final=True)

        return emb(state)

    def _cnn(self, x):
        for depth, kernel in zip(self.cnn_depths, self.cnn_kernels):
            # TODO: don't forget 2 ** i * self._depth
            x = hk.Conv2D(depth, kernel, 2)(x)
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
        x = MLP(
            self.pn_layers,
            self.act,
            self.norm,
            activate_final=True
        )(x)
        return jnp.max(x, -2)


def _ndim_partition(items: Dict[str, jnp.ndarray],
                    n: int = 3
                    ) -> Tuple[Dict[str, jnp.ndarray]]:
    """Splits inputs in groups by number of dimensions."""
    structures = tuple(type(items)() for _ in range(n))
    for key, value in items.items():
        struct = structures[value.ndim - 1]
        struct[key] = value

    return structures


class MPONetworks(NamedTuple):
    init: Callable
    actor: Callable
    encoder: Callable
    critic: Callable
    make_policy: Callable
    split_params: Callable
    preprocess: Callable


def make_networks(cfg: MPOConfig,
                  observation_spec: Dict[str, specs.Array],
                  action_spec: specs.BoundedArray
                  ) -> MPONetworks:
    prec = jmp.get_policy(cfg.mp_policy)
    hk.mixed_precision.set_policy(Encoder, prec)
    hk.mixed_precision.set_policy(Actor, prec)
    hk.mixed_precision.set_policy(CriticsEnsemble, prec)

    dummy_obs = jax.tree_util.tree_map(
        lambda sp: sp.generate_value(),
        observation_spec,
        is_leaf=lambda x: isinstance(x, specs.Array)
    )

    def preprocess(data: dict[str, jnp.ndarray]):
        data = data.copy()
        for key, val in data.items():
            if val.dtype == jnp.uint8:
                data[key] = val / 255. - 0.5
            if "depth" in key:
                data[key] = jnp.tanh(val / 10.)

        return prec.cast_to_compute(data)

    @hk.without_apply_rng
    @hk.multi_transform
    def model():
        actor = Actor(
            action_spec,
            cfg.actor_layers,
            cfg.activation,
            cfg.normalization,
            cfg.min_std,
            cfg.init_std
        )
        encoder = Encoder(
            cfg.keys,
            cfg.encoder_emb_dim,
            cfg.mlp_layers,
            cfg.pn_layers,
            cfg.cnn_kernels,
            cfg.cnn_depth,
            cfg.activation,
            cfg.normalization,
            cfg.feature_fusion
        )
        critic = CriticsEnsemble(
            cfg.num_critic_heads,
            cfg.use_iqn,
            cfg.critic_layers,
            cfg.quantile_embedding_dim,
            cfg.activation,
            cfg.normalization
        )

        def init():
            obs = preprocess(dummy_obs)
            state = encoder(obs)
            policy_params = actor(state)
            dist = make_policy(*policy_params)
            key = hk.next_rng_key()
            action = dist.sample(seed=key)
            tau = jnp.ones(1)
            value = critic(state, action, tau)
            return state, action, value

        return init, (actor, encoder, critic)

    def make_policy(mean, std) -> tfd.Distribution:
        dist = tfd.Normal(mean, std)
        return tfd.Independent(dist, 1)

    def split_params(params: hk.Params) -> Tuple[hk.Params]:
        modules = ("actor", "encoder", "critic")

        def fn(module, name, value):
            name = module.split("/")[0]
            return modules.index(name)

        return hk.data_structures.partition_n(fn, params, len(modules))

    actor_fn, encoder_fn, critic_fn = model.apply
    return MPONetworks(
        init=model.init,
        encoder=encoder_fn,
        actor=actor_fn,
        critic=critic_fn,
        make_policy=make_policy,
        split_params=split_params,
        preprocess=preprocess
    )
