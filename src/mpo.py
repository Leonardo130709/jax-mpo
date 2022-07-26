from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
import jax.scipy
import chex
import rlax
import optax
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

        def policy_improvement(actor_params, dual_params, states, actions, target_dist, q_values):
            chex.assert_rank([actions, q_values], [3, 2])

            states = jax.lax.stop_gradient(states)
            online_dist = self.actor(actor_params, states)
            temperature, alpha_mean, alpha_std = jax.tree_map(jax.nn.softplus, dual_params)

            mean, std = online_dist.distribution.loc, online_dist.distribution.scale
            fixed_mean = tfd.Normal(jax.lax.stop_gradient(mean), std)
            fixed_std = tfd.Normal(mean, jax.lax.stop_gradient(std))
            fixed_mean_online, fixed_std_online = map(tfp.bijectors.Tanh(),
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
            actor_params, target_actor_params, actor_optim_state, \
                critic_params, target_critic_params, critic_optim_state, \
                dual_params, dual_optim_state, key = mpostate

            key, subkey = jax.random.split(key, 2)

            target_dist = self.actor(target_actor_params, next_states)
            sampled_actions = target_dist.sample(seed=subkey, sample_shape=self.config.num_actions)
            log_probs = target_dist.log_prob(sampled_actions)

            sa = jnp.repeat(next_states[None], repeats=self.config.num_actions, axis=0)
            sa = jnp.concatenate([sa, sampled_actions], axis=-1)
            q_values = self.critic(target_critic_params, sa)
            # q_values = jax.lax.stop_gradient(q_values)
            discounts = self.config.discount * (1. - done_flags.astype(jnp.float32))
            target_values = rewards + discounts * jnp.mean(q_values, axis=0)

            critic_loss, critic_grads = jax.value_and_grad(policy_learning)(critic_params, states, actions, target_values)

            (actor_grads, dual_grads), metrics = jax.grad(policy_improvement, argnums=(0, 1), has_aux=True)\
                (actor_params, dual_params, next_states, sampled_actions, target_dist, q_values)
            metrics.update(critic_loss=critic_loss,
                           mean_reward=jnp.mean(rewards),
                           mean_value=jnp.mean(target_values),
                           entropy=-jnp.mean(log_probs)
                           )

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

        self._step = jax.jit(_step)
        #self._step = _step

        @jax.jit
        @chex.assert_max_traces(n=2)
        def _select_action(actor_params, rng_key, observation, training):
            dist = self.actor(actor_params, observation)
            action = jax.lax.select(
                training,
                dist.sample(seed=rng_key),
                dist.bijector(dist.distribution.mean())
            )
            return action

        self._select_action = _select_action

    def act(self, observation, training):
        key = self._state.random_key
        actor_params = self._state.actor_params
        key, subkey = jax.random.split(key, 2)
        action = self._select_action(actor_params, subkey, observation, training)
        self._state = self._state._replace(random_key=key)
        return action

    def step(self, states, actions, rewards, done_flags, next_states):
        self._state, metrics = self._step(self._state, states, actions, rewards, done_flags, next_states)
        return metrics

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
        init_duals = jnp.log(jnp.exp(self.config.init_duals) - 1.)
        dual_params = (init_duals,
                       jnp.full(self.act_dim, init_duals),
                       jnp.full(self.act_dim, init_duals))

        self.actor_optim = optax.chain(optax.clip_by_global_norm(self.config.max_grad),
                                       optax.adam(learning_rate=self.config.actor_lr))
        self.critic_optim = optax.chain(optax.clip_by_global_norm(self.config.max_grad),
                                        optax.adam(learning_rate=self.config.critic_lr))
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
