import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


class Actor(hk.Module):
    def __init__(self, act_dim, layers, mean_scale, activation, name='actor'):
        super().__init__(name=name)
        self.mean_scale = mean_scale
        self.mlp = hk.nets.MLP(
            output_sizes=[*layers, 2*act_dim],
            w_init=hk.initializers.Orthogonal(),
            b_init=jnp.zeros,
            activation=activation,
        )

    def __call__(self, state):
        out = self.mlp(state)
        mean, stddev = jnp.split(out, 2, -1)
        mean = self.mean_scale * jnp.tanh(mean / self.mean_scale)
        stddev = jax.nn.softplus(stddev) + 1e-4
        dist = tfd.Normal(mean, stddev)
        return tfp.bijectors.Tanh()(dist)


def networks_factory(
        act_dim,
        actor_layers,
        critic_layers,
        mean_scale=1.,

):

    @hk.without_apply_rng
    @hk.transform
    def actor_fn(x):
        actor = Actor(act_dim, actor_layers, mean_scale, jnp.tanh)
        return actor(x)

    @hk.without_apply_rng
    @hk.transform
    def critic_fn(x):
        critic = hk.nets.MLP([*critic_layers, 1], name='critic')
        return jnp.squeeze(critic(x), axis=-1)

    return actor_fn, critic_fn
