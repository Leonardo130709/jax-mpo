from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import jax.scipy
import haiku as hk
import optax
import chex
import reverb
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

from .networks import MPONetworks
from .config import MPOConfig


class MPOState(NamedTuple):
    actor_params: Any
    target_actor_params: Any
    critic_params: Any
    target_critic_params: Any
    encoder_params: Any
    target_encoder_params: Any
    dual_params: Any
    optim_state: Any
    key: Any


class MPOLearner:
    def __init__(
            self,
            rng_key: jax.random.PRNGKey,
            config: MPOConfig,
            networks: MPONetworks,
            optim: optax.GradientTransformation,
            dataset: 'Dataset',
            client: reverb.Client
    ):

        key, subkey = jax.random.split(rng_key)
        self._dataset = dataset
        self._client = client
        self._step = 0

        params = networks.init(subkey)
        optim_state = optim.init(params)
        encoder_params, critic_params, actor_params, dual_params = params

        self._state = MPOState(
            actor_params=actor_params,
            target_actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=critic_params,
            encoder_params=encoder_params,
            target_encoder_params=encoder_params,
            optim_state=optim_state,
            dual_params=dual_params,
            key=key
        )

        def scaled_and_dual_loss(dual_params, loss, epsilon):
            scaled_loss = jax.lax.stop_gradient(dual_params) * loss
            loss = dual_params * jax.lax.stop_gradient(loss - epsilon)
            return jnp.sum(scaled_loss, axis=-1), jnp.sum(loss, axis=-1)

        def z_learning(critic_params, state, action, taus, target_z_values):
            z_values = networks.critic(critic_params, state, action, taus)
            chex.assert_equal_shape([z_values, target_z_values])

            # Pairwise difference
            target_z_values = jnp.expand_dims(target_z_values, axis=-2)
            z_values = jnp.expand_dims(z_values, axis=-1)
            deltas = target_z_values - z_values

            ind = jnp.where(
                deltas < 0,
                jnp.ones_like(deltas),
                jnp.zeros_like(deltas)
            )
            weight = jnp.abs(taus - ind)
            loss = optax.huber_loss(deltas, delta=config.huber_kappa)
            return weight * loss / config.huber_kappa

        def cross_entropy_loss(distribution, actions, normalized_weights):
            log_probs = distribution.log_prob(actions)
            log_probs = jnp.sum(log_probs, axis=-1)
            chex.assert_equal_shape([log_probs, normalized_weights])
            cross_entropy = jnp.sum(log_probs * normalized_weights, axis=0)
            return - cross_entropy

        def weight_and_temperature_loss(temperature, q_values):
            chex.assert_rank(temperature, 0)
            tempered_q_values = q_values / temperature
            normalized_weights = jax.nn.softmax(tempered_q_values, axis=0)

            log_num_actions = jnp.log(config.num_actions)
            q_logsumexp = jax.scipy.special.logsumexp(tempered_q_values, axis=0)
            temperature_loss = config.epsilon_eta + q_logsumexp - log_num_actions
            temperature_loss = temperature_loss * temperature
            return temperature_loss, normalized_weights

        def policy_improvement(actor_params, dual_params, target_dist, states, actions, q_values):
            online_mean, online_stddev = networks.actor(actor_params, states)

            fixed_mean, fixed_stddev = map(jax.lax.stop_gradient,
                                           (online_mean, online_stddev))

            fixed_mean_dist, fixed_stddev_dist = map(
                lambda params: networks.make_policy(*params),
                (
                    (fixed_mean, online_stddev),
                    (online_mean, fixed_stddev),
                )
            )
            temperature, alpha_mean, alpha_stddev = map(
                jax.nn.softplus,
                dual_params
            )

            temperature_loss, normalized_weights = weight_and_temperature_loss(temperature, q_values)

            ce_mean_loss = cross_entropy_loss(
                fixed_mean_dist, actions, normalized_weights)
            ce_stddev_loss = cross_entropy_loss(
                fixed_stddev_dist, actions, normalized_weights)
            ce_loss = ce_mean_loss + ce_stddev_loss

            kl_mean = tfd.kl_divergence(target_dist, fixed_stddev_dist)
            kl_stddev = tfd.kl_divergence(target_dist, fixed_mean_dist)

            kl_mean_loss, alpha_mean_loss = scaled_and_dual_loss(
                alpha_mean, kl_mean, config.epsilon_alpha_mean)
            kl_stddev_loss, alpha_stddev_loss = scaled_and_dual_loss(
                alpha_stddev, kl_stddev, config.epsilon_alpha_stddev)
            kl_loss = kl_mean_loss + kl_stddev_loss

            chex.assert_equal_shape(
                [
                    temperature_loss, ce_mean_loss, ce_stddev_loss,
                    kl_mean_loss, alpha_mean_loss,
                    kl_stddev_loss, alpha_stddev_loss
                 ]
            )
            dual_loss = temperature_loss + alpha_mean_loss + alpha_stddev_loss

            metrics = None
            return jnp.mean(ce_loss + kl_loss + dual_loss), metrics

        def step(mpostate, data):
            actor_params, target_actor_params, \
                critic_params, target_critic_params, \
                encoder_params, target_encoder_params, \
                dual_params, optim_state, key = mpostate

            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

            observations, actions, rewards, next_observations = data

            next_states = networks.encoder(target_encoder_params, next_observations)

            target_policy_params = networks.actor(target_actor_params, next_states)
            target_policy = networks.make_policy(*target_policy_params)
            sampled_actions = target_policy.sample(config.num_actions, seed=subkey1)
            next_taus = jax.random.uniform(
                subkey2,
                shape=tuple(sampled_actions.shape[:-1]) + (config.num_quantiles, 1), # ?
                dtype=next_observations.dtype
            )
            add_dim = lambda x: jnp.expand_dims(x, axis=0)

            repeated_next_states = jnp.repeat(
                add_dim(next_states),
                config.num_actions,
                axis=0
            )

            rewards = add_dim(rewards)
            chex.assert_rank([repeated_next_states, sampled_actions, next_taus, rewards], [3, 3, 4, 3])

            target_z_values = networks.critic(target_critic_params, repeated_next_states, sampled_actions, next_taus)
            q_values = jnp.mean(target_z_values, axis=-1)
            target_z_values = jnp.mean(rewards + config.discount * target_z_values, axis=0)
            target_z_values = jax.lax.stop_gradient(target_z_values)

            taus = jax.random.uniform(
                subkey3,
                shape=tuple(actions.shape[:-1]) + (config.num_quantiles, 1),
                dtype=actions.dtype)

            def critic_loss_fn(encoder_params, critic_params):
                states = networks.encoder(encoder_params, observations)
                loss = z_learning(critic_params, states, actions, taus, target_z_values)
                return jnp.mean(loss), states

            critic_and_encoder_loss = jax.grad(critic_loss_fn, argnums=(0, 1), has_aux=True)
            actor_and_dual_loss = jax.grad(policy_improvement, argnums=(0, 1), has_aux=True)

            (encoder_grads, critic_grads), states = critic_and_encoder_loss(
                encoder_params, critic_params)
            (actor_grads, dual_grads), metrics = actor_and_dual_loss(
                actor_params, dual_params, target_policy, next_states, sampled_actions, q_values)

            updates, optim_state = optim.update((encoder_grads, critic_grads, actor_grads, dual_grads), optim_state)
            (encoder_params, critic_params, actor_params, dual_params) = optax.apply_updates(updates, (encoder_params, critic_params, actor_params, dual_params))

            target_encoder_params = optax.incremental_update(encoder_params, target_encoder_params, config.encoder_polyak)
            target_critic_params = optax.incremental_update(critic_params, target_critic_params, config.critic_polyak)
            target_actor_params = optax.incremental_update(actor_params, target_actor_params, config.actor_polyak)

            return MPOState(
                actor_params=actor_params,
                target_actor_params=target_actor_params,
                critic_params=critic_params,
                target_critic_params=target_critic_params,
                encoder_params=encoder_params,
                target_encoder_params=target_encoder_params,
                dual_params=dual_params,
                optim_state=optim_state,
                key=key
            ), metrics

        self._step = step

    def learn(self):
        while True:
            data = next(self._dataset)
            if data:
                self._step += 1
                self._state, metrics = self._step(self._state, data)



class Transition(NamedTuple):
    observations: Any
    actions: Any
    rewards: Any
    next_observations: Any

k = 30
what = Transition(
    observations=jnp.zeros((k, 5)),
    actions=jnp.zeros((k, 1)),
    rewards=jnp.zeros((k, 1)),
    next_observations=jnp.zeros((k, 5))
)