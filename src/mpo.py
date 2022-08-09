from typing import NamedTuple, Any
import re

import jax
import jax.numpy as jnp
import jax.scipy
import chex
import optax
import haiku as hk
import tensorflow_probability.substrates.jax as tfp
from .networks import make_networks
tfd = tfp.distributions


class MPOState(NamedTuple):
    params: Any
    target_params: Any
    optim_state: Any
    rng_key: Any


class MPO:
    def __init__(self, config, env, rng_key):

        self.action_spec = env.action_spec()
        chex.disable_asserts()
        self.observation_spec = env.observation_spec()
        self.config = config
        self._build(rng_key)

        def policy_learning(params, states, actions, target_values):
            chex.assert_rank([target_values, states, actions], [1, 2, 2])

            values = self.networks.q_value(params, states, actions)
            chex.assert_equal_shape([values, target_values])
            loss = .5 * jnp.square(values - target_values)
            return jnp.mean(loss)

        def weight_and_temperature_loss(temperature, q_values):
            chex.assert_rank([temperature, q_values], [0, 2])

            tempered_q_value = q_values / temperature
            normalized_weights = jax.nn.softmax(tempered_q_value, axis=0)
            normalized_weights = jax.lax.stop_gradient(normalized_weights)

            q_logsumexp = jax.scipy.special.logsumexp(tempered_q_value, axis=0)
            log_num_actions = jnp.log(q_values.shape[0])
            temperature_loss = self.config.epsilon_eta + jnp.mean(q_logsumexp) - log_num_actions
            temperature_loss = temperature * temperature_loss
            return jnp.mean(temperature_loss), normalized_weights

        def cross_entropy_loss(online_dist, sampled_actions, normalized_weights):
            chex.assert_rank([sampled_actions, normalized_weights], [3, 2])

            log_probs = jnp.sum(online_dist.log_prob(sampled_actions), axis=-1)
            loss = -jnp.sum(normalized_weights * log_probs, axis=0)
            return jnp.mean(loss)

        def kl_and_dual_loss(kl, alpha, epsilon):
            chex.assert_rank([alpha, kl], [1, 2])

            loss_kl = jnp.sum(jax.lax.stop_gradient(alpha)*kl, axis=-1)
            loss_alpha = jnp.sum(alpha * jax.lax.stop_gradient(epsilon - kl), axis=-1)
            return jnp.mean(loss_alpha), jnp.mean(loss_kl)

        def policy_improvement(params, states, actions, target_dist, q_values):
            chex.assert_rank([states, actions, q_values], [2, 3, 2])

            states = jax.lax.stop_gradient(states)
            online_dist = self.networks.policy(params, states)
            temperature, alpha_mean, alpha_std = jax.tree_map(jax.nn.softplus, params['~']['dual_params'])
            alpha_mean, alpha_std = map(lambda x: jnp.expand_dims(x, axis=0), (alpha_mean, alpha_std))

            mean, std = online_dist.distribution.loc, online_dist.distribution.scale
            fixed_mean = tfd.Normal(jax.lax.stop_gradient(mean), std)
            fixed_std = tfd.Normal(mean, jax.lax.stop_gradient(std))
            fixed_mean_online, fixed_std_online = map(lambda d: tfp.bijectors.Tanh()(d),
                                                         (fixed_mean, fixed_std))

            temperature_loss, normalized_weights = weight_and_temperature_loss(
                temperature, q_values)

            ce_mean_loss = cross_entropy_loss(fixed_std_online, actions, normalized_weights)
            ce_std_loss = cross_entropy_loss(fixed_mean_online, actions, normalized_weights)

            kl_mean = tfd.kl_divergence(fixed_std_online, target_dist)
            kl_std = tfd.kl_divergence(fixed_mean_online, target_dist)

            alpha_mean_loss, kl_mean_loss = kl_and_dual_loss(
                kl_mean, alpha_mean, self.config.epsilon_alpha_mean)
            alpha_std_loss, kl_std_loss = kl_and_dual_loss(
                kl_std, alpha_std, self.config.epsilon_alpha_std)

            chex.assert_rank([ce_mean_loss, ce_std_loss, kl_mean_loss, kl_std_loss,
                              alpha_mean_loss, alpha_std_loss, temperature_loss], 0)
            ce_loss = ce_mean_loss + ce_std_loss
            kl_loss = kl_mean_loss + kl_std_loss
            dual_loss = alpha_mean_loss + alpha_std_loss + temperature_loss
            total_loss = ce_loss + kl_loss + dual_loss

            metrics = dict(
                ce_loss=ce_loss,
                kl_loss=kl_loss,
                dual_loss=dual_loss,
                kl_mean=jnp.mean(jnp.sum(kl_mean, axis=-1)),
                kl_std=jnp.mean(jnp.sum(kl_std, axis=-1)),
                temperature=temperature,
                alpha_mean=jnp.mean(alpha_mean),
                alpha_std=jnp.mean(alpha_std)
            )
            return total_loss, metrics

        @chex.assert_max_traces(n=3)
        def _step(mpostate, states, actions, rewards, done_flags, next_states):
            chex.assert_rank(
                [states, actions, rewards, done_flags, next_states],
                [2, 2, 1, 1, 2]
            )
            params, target_params, optim_state, rng_key = mpostate

            rng_key, subkey = jax.random.split(rng_key, 2)

            target_dist = self.networks.policy(target_params, next_states)
            sampled_actions = target_dist.sample(seed=subkey, sample_shape=self.config.num_actions)
            log_probs = target_dist.log_prob(sampled_actions)

            tiled_next_states = jnp.repeat(next_states[None], repeats=self.config.num_actions, axis=0)
            q_values = self.networks.q_value(target_params, tiled_next_states, sampled_actions)
            discounts = self.config.discount * (1. - done_flags.astype(jnp.float32))
            target_values = rewards + discounts * jnp.mean(q_values, axis=0)

            critic_loss, critic_grads = jax.value_and_grad(policy_learning)(params, states, actions, target_values)

            actor_grads, metrics = jax.grad(policy_improvement, has_aux=True)\
                (params, next_states, sampled_actions, target_dist, q_values)
            metrics.update(critic_loss=critic_loss,
                           mean_reward=jnp.mean(rewards),
                           mean_value=jnp.mean(target_values),
                           entropy=-jnp.mean(log_probs)
                           )

            total_grads = jax.tree_map(lambda g1, g2: g1 + g2, critic_grads, actor_grads)
            updates, optim_state = self.optim.update(total_grads, optim_state)
            params = optax.apply_updates(params, updates)

            split = lambda x: hk.data_structures.partition(lambda module, n, p: 'actor' in module, x)
            (actor_params, critic_params), (target_actor_params, target_critic_params) = map(split, (params, target_params))

            target_actor_params = optax.incremental_update(actor_params, target_actor_params, self.config.actor_tau)
            target_critic_params = optax.incremental_update(critic_params, target_critic_params, self.config.critic_tau)
            target_params = hk.data_structures.merge(target_actor_params, target_critic_params)

            new_state = MPOState(
                params=params,
                target_params=target_params,
                optim_state=optim_state,
                rng_key=rng_key)
            return new_state, metrics

        self._step = jax.jit(_step)
        self._step = _step

        @chex.assert_max_traces(n=2)
        def _select_action(params, rng_key, observation, training):
            dist = self.networks.policy(params, observation)
            action = jax.lax.select(
                training,
                dist.sample(seed=rng_key),
                dist.bijector(dist.distribution.mean())
            )
            return action

        self._select_action = _select_action

    def act(self, observation, training):
        key = self.state.rng_key
        params = self.state.params
        key, subkey = jax.random.split(key, 2)
        action = jax.jit(self._select_action)(params, subkey, observation['observations'], training)
        self.state = self.state._replace(rng_key=key)
        return action

    def step(self, states, actions, rewards, done_flags, next_states):
        states, next_states = map(lambda d: d['observations'], (states, next_states))
        self.state, metrics = self._step(self.state, states, actions, rewards, done_flags, next_states)
        return metrics

    def _build(self, rng_key):
        if isinstance(rng_key, int):
            rng_key = jax.random.PRNGKey(rng_key)
        rng_key, subkey = jax.random.split(rng_key, 2)
        obs_dim = self.observation_spec.shape[0]
        act_dim = self.action_spec.shape[0]

        self.networks = make_networks(self.config, act_dim)
        dummy_obs = jnp.zeros(obs_dim)
        dummy_action = jnp.zeros(act_dim)
        params = self.networks.init(subkey, dummy_obs, dummy_action)
        init_duals = jnp.log(jnp.exp(self.config.init_duals) - 1.)
        params['~'] = {'dual_params': jnp.full(1 + 2 * act_dim, init_duals)}

        def _module_fn(p):

            def fn(m, n, _):
                match = re.match(r'.*?~/(\w+)', m)
                if match:
                    return 'actor' if match[1] == 'actor' else 'critic'
                else:
                    return 'dual'
            return hk.data_structures.map(fn, p)

        self.optim = optax.multi_transform(
            {
                'actor': optax.adam(self.config.actor_lr),
                'critic': optax.chain(
                    optax.clip_by_global_norm(self.config.max_grad),
                    optax.adam(self.config.critic_lr)),
                'dual': optax.adam(self.config.dual_lr)
            },
            _module_fn
        )
        optim_state = self.optim.init(params)

        self.state = MPOState(
            params=params,
            target_params=params,
            optim_state=optim_state,
            rng_key=rng_key
        )
