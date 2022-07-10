import functools
from typing import NamedTuple, Any

import chex
import jax
import jax.numpy as jnp
import jax.scipy
import rlax
import optax
import haiku as hk
import tensorflow_probability.substrates.jax as tfp
from .networks import networks_factory
tfd = tfp.distributions


class MPOState(NamedTuple):
    actor_params: Any
    target_actor_params: Any
    actor_optim_state: Any
    critic_params: Any
    target_critic_params: Any
    critic_optim_state: Any
    dual_params: Any
    dual_optim_state: Any
    random_key: Any


class MPO:
    def __init__(self,
                 config,
                 env,
                 rng_key
                 ):
        self.env = env
        self.config = config
        self._build(rng_key)

        def policy_learning(params, states, actions, target_values):
            chex.assert_rank([target_values, states, actions], [1, 2, 2])

            x = jnp.concatenate([states, actions], -1)
            values = self.critic(params, x)
            return jnp.mean(rlax.l2_loss(values, target_values))

        def weight_and_temperature_loss(temperature, q_values):
            chex.assert_rank(q_values, 2)

            tempered_q_value = q_values / temperature
            normalized_weights = jax.nn.softmax(tempered_q_value, axis=0)
            normalized_weights = jax.lax.stop_gradient(normalized_weights)

            q_logsumexp = jax.scipy.special.logsumexp(tempered_q_value, axis=0)
            log_num_actions = jnp.log(q_values.shape[0])
            temperature_loss = self.config.epsilon_eta + jnp.mean(q_logsumexp) - log_num_actions
            temperature_loss = temperature * temperature_loss
            return temperature_loss, normalized_weights

        def cross_entropy_loss(online_dist, sampled_actions, normalized_weights):
            chex.assert_rank([sampled_actions, normalized_weights], [3, 2])
            log_probs = jnp.sum(online_dist.log_prob(sampled_actions), axis=-1)
            loss = -jnp.sum(normalized_weights * log_probs, axis=0)
            return jnp.mean(loss)

        def kl_and_dual_loss(alpha, kl):
            chex.assert_rank(kl, 2)

            loss_kl = jnp.sum(jax.lax.stop_gradient(alpha)*kl, axis=-1)
            loss_alpha = jnp.sum(alpha*(self.config.epsilon_alpha - jax.lax.stop_gradient(kl)), axis=-1)
            return jnp.mean(loss_alpha), jnp.mean(loss_kl)

        def policy_improvement(actor_params, dual_params, states, actions, target_dist, q_values):
            chex.assert_rank([actions, q_values], [3, 2])

            online_dist = self.actor(actor_params, states)
            temperature, alpha = jax.tree_map(jax.nn.softplus, dual_params)

            temperature_loss, normalized_weights = weight_and_temperature_loss(temperature, q_values)
            ce_loss = cross_entropy_loss(online_dist, actions, normalized_weights)

            kl = tfd.kl_divergence(online_dist, target_dist)
            alpha_loss, kl_loss = kl_and_dual_loss(alpha, kl)

            total_loss = ce_loss + kl_loss + alpha_loss + temperature_loss

            metrics = dict(
                ce_loss=ce_loss,
                kl=jnp.mean(jnp.sum(kl, axis=-1)),
                alpha_loss=alpha_loss,
                temperature_loss=temperature_loss,
            )
            return total_loss, metrics

        @jax.jit
        @chex.assert_max_traces(n=1)
        def _step(mpostate, states, actions, rewards, done_flags, next_states):
            chex.assert_rank(
                [states, actions, rewards, done_flags, next_states],
                [2, 2, 1, 1, 2]
            )
            actor_params, target_actor_params, actor_optim_state, \
                critic_params, target_critic_params, critic_optim_state, \
                dual_params, dual_optim_state, key = mpostate

            key, subkey = jax.random.split(key, 2)

            target_dist = self.actor(target_actor_params, next_states)
            sampled_actions = target_dist.sample(seed=subkey, sample_shape=self.config.num_actions)

            sa = jnp.repeat(next_states[None], repeats=self.config.num_actions, axis=0)
            sa = jnp.concatenate([sa, sampled_actions], axis=-1)
            q_values = self.critic(target_critic_params, sa)
            q_values = jax.lax.stop_gradient(q_values)
            discounts = self.config.discount * (1. - done_flags.astype(jnp.float32))
            target_values = rewards + discounts * jnp.mean(q_values, axis=0)

            critic_loss, critic_grads = jax.value_and_grad(policy_learning)(critic_params, states, actions, target_values)

            (actor_grads, dual_grads), metrics = jax.grad(policy_improvement, argnums=(0, 1), has_aux=True)\
                (actor_params, dual_params, next_states, sampled_actions, target_dist, q_values)
            metrics.update(critic_loss=critic_loss)

            actor_updates, actor_optim_state = self.actor_optim.update(actor_grads, actor_optim_state)
            actor_params = optax.apply_updates(actor_params, actor_updates)
            target_actor_params = optax.incremental_update(actor_params, target_actor_params, self.config.actor_tau)

            critic_updates, critic_optim_state = self.critic_optim.update(critic_grads, critic_optim_state)
            critic_params = optax.apply_updates(critic_params, critic_updates)
            target_critic_params = optax.incremental_update(critic_params, target_critic_params, self.config.critic_tau)

            dual_updates, dual_optim_state = self.dual_optim.update(dual_grads, dual_optim_state)
            dual_params = optax.apply_updates(dual_params, dual_updates)

            new_state = MPOState(
                actor_params=actor_params,
                target_actor_params=target_actor_params,
                actor_optim_state=actor_optim_state,
                critic_params=critic_params,
                target_critic_params=target_critic_params,
                critic_optim_state=critic_optim_state,
                dual_params=dual_params,
                dual_optim_state=dual_optim_state,
                random_key=key
            )
            return new_state, metrics

        self.step = _step

        @jax.jit
        @chex.assert_max_traces(n=1)
        def _select_action(actor_params, rng_key, observation, training):
            dist = self.actor(actor_params, observation)
            action = jax.lax.select(
                training,
                dist.sample(seed=rng_key),
                dist.bijector(dist.distribution.mean())
            )
            return action

        def _act(observation, training):
            key = self._state.random_key
            actor_params = self._state.actor_params
            key, subkey = jax.random.split(key, 2)
            action = _select_action(actor_params, subkey, observation, training)

            self._state = self._state._replace(random_key=key)
            return action

        self.act = _act

    def _build(self, rng_key):
        if isinstance(rng_key, int):
            rng_key = jax.random.PRNGKey(rng_key)
        self.act_dim = self.env.action_spec().shape[0]
        self.obs_dim = self.env.observation_spec().shape[0]
        self.actor, self.critic = networks_factory(
            self.act_dim,
            self.config.actor_layers,
            self.config.critic_layers
        )

        rng_key, actor_key, critic_key = jax.random.split(rng_key, 3)
        dummy_state = jnp.zeros(self.obs_dim)
        dummy_state_action = jnp.zeros(self.obs_dim + self.act_dim)
        actor_params = self.actor.init(actor_key, dummy_state)
        critic_params = self.critic.init(critic_key, dummy_state_action)
        dual_params = (jnp.zeros([]), jnp.zeros(self.act_dim))

        self.actor_optim = optax.adam(learning_rate=self.config.actor_lr)
        self.critic_optim = optax.adam(learning_rate=self.config.critic_lr)
        self.dual_optim = optax.adam(learning_rate=self.config.dual_lr)
        actor_optim_state = self.actor_optim.init(actor_params)
        critic_optim_state = self.critic_optim.init(critic_params)
        dual_optim_state = self.dual_optim.init(dual_params)

        self.actor = self.actor.apply
        self.critic = self.critic.apply

        self._state = MPOState(
            actor_params=actor_params,
            target_actor_params=actor_params.copy(),
            actor_optim_state=actor_optim_state,
            critic_params=critic_params,
            target_critic_params=critic_params.copy(),
            critic_optim_state=critic_optim_state,
            dual_params=dual_params,
            dual_optim_state=dual_optim_state,
            random_key=rng_key
        )
