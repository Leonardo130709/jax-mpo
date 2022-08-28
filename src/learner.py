from typing import NamedTuple, Any
import functools

import jax
import jax.scipy
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
import chex
import reverb
import tensorflow_probability.substrates.jax as tfp

from rltools.loggers import TFSummaryLogger, JSONLogger
from src.networks import MPONetworks
from src.config import MPOConfig
tfd = tfp.distributions


class MPOState(NamedTuple):
    params: Any
    target_params: Any
    dual_params: Any
    optim_state: Any
    rng_key: jax.random.PRNGKey


class MPOLearner:
    def __init__(
            self,
            rng_key: jax.random.PRNGKey,
            config: MPOConfig,
            env_spec: 'EnvironmentSpec',
            networks: MPONetworks,
            train_dataset: 'Dataset',
            client: reverb.Client
    ):

        key, subkey = jax.random.split(rng_key)
        self._data_iterator = train_dataset
        self._client = client
        self._logger1 = TFSummaryLogger(logdir='tut', label='train', step_key='step')
        self._logger2 = JSONLogger('metrics.jsonl')
        self.learning_steps = 0

        params = networks.init(subkey)
        encoder_params, actor_params, critic_params = networks.split_params(params)

        init_duals = jnp.log(jnp.exp(config.init_duals) - 1)
        dual_params = (
            init_duals,
            jnp.full((env_spec.action_spec.shape[0]), init_duals)
        )

        optim = optax.multi_transform({
            'encoder': optax.adam(config.encoder_lr),
            'critic': optax.adam(config.critic_lr),
            'actor': optax.adam(config.actor_lr),
            'dual_params': optax.adam(config.dual_lr),
        },
            ('encoder', 'critic', 'actor', 'dual_params')
        )
        
        optim_state = optim.init((
            encoder_params,
            critic_params,
            actor_params,
            dual_params))
        self._client.insert(params, {'weights': 1.})

        self._state = MPOState(
            params=params,
            target_params=hk.data_structures.to_haiku_dict(params),
            dual_params=dual_params,
            optim_state=optim_state,
            rng_key=key
        )

        def scaled_and_dual_loss(dual_params, loss, epsilon):
            scaled_loss = jax.lax.stop_gradient(dual_params) * loss
            loss = dual_params * jax.lax.stop_gradient(loss - epsilon)
            return jnp.sum(scaled_loss, axis=-1), jnp.sum(loss, axis=-1)

        def z_learning(critic_params, encoder_params,
                       observation, action, taus, target_z_values):
            state = networks.encoder(encoder_params, observation)
            chex.assert_equal_rank([state, action])
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
            loss = weight * loss
            return jnp.mean(loss) / config.huber_kappa, state

        def cross_entropy_loss(policy, actions, normalized_weights):
            log_probs = policy.log_prob(actions)
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

        def policy_improvement(actor_params, dual_params, target_policy, states, actions, q_values):
            # todo: test with gaussian penalized policy
            policy_params = networks.actor(actor_params, states)
            online_policy = networks.make_policy(*policy_params)

            temperature, alpha = map(jax.nn.softplus, dual_params)

            temperature_loss, normalized_weights = weight_and_temperature_loss(temperature, q_values)
            ce_loss = cross_entropy_loss(online_policy, actions, normalized_weights)
            kl_loss = tfd.kl_divergence(target_policy, online_policy)
            scaled_kl_loss, alpha_loss = scaled_and_dual_loss(alpha, kl_loss, config.epsilon_alpha)

            chex.assert_equal_shape([temperature_loss, alpha_loss, ce_loss, scaled_kl_loss])
            policy_loss = ce_loss + scaled_kl_loss
            dual_loss = temperature_loss + alpha_loss

            metrics = dict(
                kl_loss=jnp.mean(jnp.sum(kl_loss, axis=-1)),
                ce_loss=jnp.mean(ce_loss),
                temperature=temperature,
                alpha_mean=jnp.mean(alpha),
            )
            return jnp.mean(policy_loss + dual_loss), metrics

        @chex.assert_max_traces(n=3)
        def step(mpostate, data):
            params, target_params, dual_params, \
                optim_state, rng_key = mpostate

            rng_key, subkey1, subkey2, subkey3 = jax.random.split(rng_key, 4)

            def _reshape(data_key):
                val = data.get(data_key)
                return jnp.reshape(val, (-1,) + tuple(val.shape[2:]))

            observations, actions, rewards, next_observations = map(
                _reshape,
                ('observations', 'actions', 'rewards', 'next_observations')
            )
            (encoder_params, actor_params, critic_params),\
                (target_encoder_params, target_actor_params,
                 target_critic_params) =\
                map(networks.split_params, (params, target_params))
            
            next_states = networks.encoder(encoder_params, next_observations)
            target_next_states = networks.encoder(target_encoder_params, next_observations)
            target_policy_params = networks.actor(target_actor_params, target_next_states)
            target_policy = networks.make_policy(*target_policy_params)


            sampled_actions = target_policy.sample(config.num_actions, seed=subkey1)

            log_probs = jnp.sum(target_policy.log_prob(sampled_actions), axis=-1)
            entropy = -jnp.mean(log_probs)

            next_taus = jax.random.uniform(
                subkey2,
                shape=tuple(sampled_actions.shape[:-1]) + (config.num_quantiles, 1), # batch consistence?
                dtype=next_observations.dtype
            )
            repeated_next_states = jnp.repeat(
                jnp.expand_dims(target_next_states, axis=0),
                config.num_actions,
                axis=0
            )
            rewards = jnp.reshape(rewards, (1,) + tuple(rewards.shape) + (1,))

            chex.assert_rank([repeated_next_states, sampled_actions, next_taus, rewards], [3, 3, 4, 3])
            target_z_values = networks.critic(target_critic_params, repeated_next_states, sampled_actions, next_taus)
            q_values = jnp.mean(target_z_values, axis=-1)
            target_z_values = jnp.mean(rewards + config.discount * target_z_values, axis=0)

            taus = jax.random.uniform(
                subkey3,
                shape=tuple(actions.shape[:-1]) + (config.num_quantiles, 1),
                dtype=actions.dtype)

            critic_and_encoder_loss = jax.value_and_grad(z_learning, argnums=(0, 1), has_aux=True)
            actor_and_dual_loss = jax.grad(policy_improvement, argnums=(0, 1), has_aux=True)

            (critic_loss, states), (critic_grads, encoder_grads) = critic_and_encoder_loss(
                critic_params, encoder_params, observations, actions, taus, target_z_values)

            (actor_grads, dual_grads), metrics = actor_and_dual_loss(
                actor_params, dual_params, target_policy, next_states, sampled_actions, q_values)

            updates, optim_state = optim.update((encoder_grads, critic_grads, actor_grads, dual_grads), optim_state)
            (encoder_params, critic_params, actor_params, dual_params) = optax.apply_updates(updates, (encoder_params, critic_params, actor_params, dual_params))

            target_encoder_params = optax.incremental_update(encoder_params, target_encoder_params, config.encoder_polyak)
            target_critic_params = optax.incremental_update(critic_params, target_critic_params, config.critic_polyak)
            target_actor_params = optax.incremental_update(actor_params, target_actor_params, config.actor_polyak)

            params = hk.data_structures.merge(encoder_params,
                                              actor_params,
                                              critic_params,
                                              check_duplicates=True
                                              )
            target_params = hk.data_structures.merge(target_encoder_params,
                                                     target_actor_params,
                                                     target_critic_params,
                                                     check_duplicates=True
                                                     )

            metrics.update(
                reward=jnp.mean(rewards),
                critic_loss=critic_loss,
                entropy=entropy,
                mean_value=jnp.mean(q_values),
            )

            return MPOState(
                params=params,
                target_params=target_params,
                dual_params=dual_params,
                optim_state=optim_state,
                rng_key=rng_key
            ), metrics

        self._step = jax.jit(step)

    def run(self):
        while True:
            data = next(self._data_iterator)
            info, data = data
            data = self._preprocess(data)
            self.learning_steps += 1
            self._state, metrics = self._step(self._state, data)
            self._client.insert(self._state.params, {'weights': 1.})
            metrics.update(step=self.learning_steps)
            self._logger1.write(metrics)
            metrics = jax.tree_util.tree_map(float, metrics)
            self._logger2.write(metrics)

    @functools.partial(jax.jit, static_argnums=0)
    @chex.assert_max_traces(n=1)
    def _preprocess(self, data):
        return jax.tree_util.tree_map(jnp.asarray, data)
