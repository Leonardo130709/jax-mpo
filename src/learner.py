from typing import NamedTuple
import functools

import jax
import jax.scipy
import jax.numpy as jnp
import jmp
import haiku as hk
import optax
import chex
import reverb
import tensorflow_probability.substrates.jax as tfp

from rltools.loggers import TFSummaryLogger
from src.networks import MPONetworks
from src.config import MPOConfig
from src.utils import ops

tfd = tfp.distributions
Array = jnp.ndarray


class Duals(NamedTuple):
    log_temperature: Array
    log_alpha_mean: Array
    log_alpha_std: Array


class MPOState(NamedTuple):
    # __slots__
    params: hk.Params
    target_params: hk.Params
    dual_params: Duals
    optim_state: optax.OptState
    dual_optim_state: optax.OptState
    rng_key: jax.random.PRNGKey
    loss_scale: jmp.LossScale
    step: int


class MPOLearner:
    def __init__(
            self,
            rng_key: jax.random.PRNGKey,
            cfg: MPOConfig,
            env_spec: "EnvironmentSpec",
            networks: MPONetworks,
            train_dataset: "Dataset",
            client: reverb.Client
    ):
        jax.config.update("jax_disable_jit", not cfg.jit)
        key, subkey = jax.random.split(rng_key)
        self._cfg = cfg
        self._data_iterator = train_dataset
        self._client = client
        self._prec = jmp.get_policy(cfg.mp_policy)

        _params = networks.init(subkey)
        act_dim = env_spec.action_spec.shape[0]
        _dual_params = Duals(
            log_temperature=cfg.init_log_temperature,
            log_alpha_mean=jnp.full(act_dim, cfg.init_log_alpha_mean),
            log_alpha_std=jnp.full(act_dim, cfg.init_log_alpha_std)
        )
        # to param or to compute?
        _dual_params = self._prec.cast_to_compute(_dual_params)

        adam = functools.partial(optax.adam,
                                 b1=cfg.adam_b1,
                                 b2=cfg.adam_b2,
                                 eps=cfg.adam_eps
                                 )
        optim = optax.chain(
            optax.clip_by_global_norm(cfg.grad_norm),
            adam(cfg.learning_rate)
        )
        dual_optim = adam(cfg.dual_lr)
        _optim_state = optim.init(_params)
        _dual_optim_state = dual_optim.init(_dual_params)

        if "16" in cfg.mp_policy:
            _loss_scale = jmp.DynamicLossScale(
                self._prec.cast_to_output(2 ** 15)
            )
        else:
            _loss_scale = jmp.NoOpLossScale()

        self._state = MPOState(
            params=_params,
            target_params=_params,
            dual_params=_dual_params,
            optim_state=_optim_state,
            dual_optim_state=_dual_optim_state,
            rng_key=key,
            loss_scale=_loss_scale,
            step=0
        )

        def mpo_loss(params,
                     dual_params,
                     target_params,
                     loss_scale,
                     rng,
                     o_tm1, a_tm1, r_t, discount_t, o_t
                     ):
            sg = jax.lax.stop_gradient
            k1, k2, k3 = jax.random.split(rng, 3)
            # Sample quantiles.
            tau_tm1 = jax.random.uniform(
                k1,
                (cfg.num_critic_quantiles,),
                dtype=a_tm1.dtype
            )
            tau_t = jax.random.uniform(
                k2,
                (cfg.num_actions, cfg.num_actor_quantiles),
                dtype=a_tm1.dtype
            )

            s_tm1 = networks.encoder(params, o_tm1)
            target_s_t = networks.encoder(target_params, o_t)
            s_t = jax.lax.select(
                cfg.stop_actor_grad,
                target_s_t,
                networks.encoder(params, o_t)
            )
            target_policy_params = networks.actor(target_params, target_s_t)
            target_dist = networks.make_policy(*target_policy_params)
            a_t = target_dist.sample(seed=k3, sample_shape=cfg.num_actions)
            target_s_t = jnp.repeat(target_s_t[jnp.newaxis],
                                    cfg.num_actions, axis=0)

            z_t = networks.critic(target_params, target_s_t, a_t, tau_t)
            chex.assert_shape(
                z_t, (cfg.num_actions,
                      cfg.num_actor_quantiles,
                      cfg.num_critic_heads)
            )
            z_t = jnp.min(z_t, axis=2)
            z_tm1 = networks.critic(params, s_tm1, a_tm1, tau_tm1)
            chex.assert_shape(
                z_tm1, (cfg.num_critic_quantiles,
                        cfg.num_critic_heads)
            )
            v_t = jnp.mean(z_t, axis=0)
            q_t = jnp.mean(z_t, axis=1)  # risk-neutral agent
            target_z_tm1 = sg(r_t + discount_t * v_t)
            critic_loss_fn = jax.vmap(
                ops.quantile_regression_loss,
                in_axes=(1, None, None, None)
            )
            critic_loss = critic_loss_fn(
                z_tm1, tau_tm1, target_z_tm1, cfg.hubber_delta)
            critic_loss = jnp.mean(critic_loss)

            temperature, alpha_mean, alpha_std = \
                jax.tree_util.tree_map(ops.softplus, dual_params)
            temperature_loss, normalized_weights = \
                ops.temperature_loss_and_normalized_weights(
                    temperature,
                    q_t,
                    cfg.epsilon_eta,
                    cfg.tv_constraint
                )

            mean, std = networks.actor(params, s_t)
            fixed_mean, fixed_std = map(sg, (mean, std))
            fixed_mean = networks.make_policy(fixed_mean, std)
            fixed_std = networks.make_policy(mean, fixed_std)

            def policy_loss(online_dist,
                            duals,
                            epsilon,
                            prefix
                            ):
                ce_loss = ops.cross_entropy_loss(
                    online_dist, a_t, normalized_weights)
                kl = tfd.kl_divergence(
                    target_dist.distribution,
                    online_dist.distribution
                )
                kl_loss, dual_loss = ops.scaled_and_dual_loss(
                    kl, duals, epsilon, per_dimension=True
                )
                chex.assert_rank([ce_loss, kl_loss, dual_loss], 0)
                loss = ce_loss + kl_loss + dual_loss
                pfx = lambda k: f"{prefix}_{k}"
                met = {
                    pfx("ce_loss"): ce_loss,
                    pfx("kl_rel"): jnp.mean(kl) / epsilon,
                    pfx("kl_loss"): kl_loss,
                    pfx("dual_loss"): dual_loss,
                }
                return .5 * loss, met

            mean_loss, metrics_mean = policy_loss(
                fixed_std, alpha_mean, cfg.epsilon_mean, prefix="mean")
            std_loss, metrics_std = policy_loss(
                fixed_mean, alpha_std, cfg.epsilon_std, prefix="std")
            chex.assert_rank([critic_loss, mean_loss, std_loss, temperature_loss], 0)
            total_loss = critic_loss + mean_loss + std_loss + temperature_loss

            metrics = dict(
                critic_loss=critic_loss,
                mean_value=jnp.mean(v_t),
                value_std=jnp.std(v_t),
                q_value_std=jnp.std(q_t),
                entropy=fixed_std.entropy(),
                temperature=temperature,
                alpha_mean=jnp.mean(alpha_mean),
                alpha_std=jnp.mean(alpha_std),
                temperature_loss=temperature_loss,
                total_loss=total_loss,
                mean_reward=r_t,
            )
            metrics.update(metrics_mean)
            metrics.update(metrics_std)

            return loss_scale.scale(total_loss), metrics

        def update_step(mpostate: MPOState, grads) -> MPOState:
            params = mpostate.params
            target_params = mpostate.target_params
            dual_params = mpostate.dual_params
            optim_state = mpostate.optim_state
            dual_optim_state = mpostate.dual_optim_state
            step = mpostate.step
            params_grads, dual_grads = grads

            params_update, optim_state = optim.update(
                params_grads, optim_state)
            dual_update, dual_optim_state = dual_optim.update(
                dual_grads, dual_optim_state)
            params = optax.apply_updates(params, params_update)
            dual_params = optax.apply_updates(dual_params, dual_update)

            actor_params, encoder_params, critic_params =\
                networks.split_params(params)
            target_actor_params, target_encoder_params, target_critic_params = \
                networks.split_params(target_params)

            target_actor_params = optax.periodic_update(
                actor_params,
                target_actor_params,
                step,
                cfg.target_actor_update_period
            )
            (target_encoder_params, target_critic_params) =\
                optax.periodic_update(
                    (encoder_params, critic_params),
                    (target_encoder_params, target_critic_params),
                    step,
                    cfg.target_critic_update_period
                )
            target_params.update(target_actor_params)
            target_params.update(target_encoder_params)
            target_params.update(target_critic_params)

            metrics = dict(
                params_grad_norm=optax.global_norm(params_grads),
                dual_grad_norm=optax.global_norm(dual_grads),
            )
            return mpostate._replace(
                params=params,
                target_params=target_params,
                dual_params=dual_params,
                optim_state=optim_state,
                dual_optim_state=dual_optim_state,
                step=step+1
            ), metrics

        @chex.assert_max_traces(n=2)  # As first optim state is EmptyState.
        def _step(mpostate: MPOState, data):
            params = mpostate.params
            target_params = mpostate.target_params
            dual_params = mpostate.dual_params
            rng_key = mpostate.rng_key
            loss_scale = mpostate.loss_scale

            observations, actions, rewards, discounts, next_observations =\
                map(
                    data.get,
                    ("observations",
                     "actions",
                     "rewards",
                     "discounts",
                     "next_observations")
                )
            chex.assert_tree_shape_prefix(actions, (cfg.batch_size,))
            keys = jax.random.split(rng_key, num=cfg.batch_size+1)
            rng_key, subkeys = keys[0], keys[1:]

            in_axes = 4 * (None,) + 6 * (0,)
            args = (
                params, dual_params, target_params, loss_scale, subkeys,
                observations, actions, rewards, discounts, next_observations
            )
            in_axes = tuple(
                jax.tree_util.tree_map(lambda t: axis, val)
                for axis, val in zip(in_axes, args)
            )

            grad_fn = jax.grad(mpo_loss, argnums=(0, 1), has_aux=True)
            grads, metrics = jax.vmap(grad_fn, in_axes=in_axes)(*args)
            grads, metrics = jax.tree_util.tree_map(
                lambda t: jnp.mean(t, axis=0),
                (grads, metrics)
            )
            grads = loss_scale.unscale(grads)
            grads_finite = jmp.all_finite(grads)
            loss_scale = loss_scale.adjust(grads_finite)

            met_placeholder = {
                "params_grad_norm": float("inf"),
                "dual_grad_norm": float("inf"),
            }
            mpostate, met = jmp.select_tree(
                grads_finite,
                update_step(mpostate, grads),
                (mpostate, met_placeholder)
            )
            metrics.update(loss_scale=loss_scale.loss_scale,
                           grads_finite=grads_finite,
                           **met)

            return mpostate._replace(
                rng_key=rng_key,
                loss_scale=loss_scale,
            ), metrics

        self._step = jax.jit(_step)

        param_groups = networks.split_params(_params)
        names = ("Actor", "Encoder", "Critic")
        for pg, name in zip(param_groups, names):
            print(f"{name} params: {hk.data_structures.tree_size(pg)}")

    def run(self):
        logger = TFSummaryLogger(logdir=self._cfg.logdir,
                                 label="train",
                                 step_key="step")
        while True:
            data = next(self._data_iterator)
            info, data = data
            data = self._preprocess(data)
            self._state, metrics = self._step(self._state, data)

            step = self._state.step
            if step % self._cfg.learner_dump_every == 0:
                self._client.insert(self._state.params, {"weights": 1.})

            if step % self._cfg.log_every == 0:
                chex.assert_rank(jax.tree_leaves(metrics), 0)
                metrics = jax.tree_util.tree_map(float, metrics)
                metrics["step"] = step
                logger.write(metrics)

    def _preprocess(self, data):
        data = jax.device_put(data)
        return self._prec.cast_to_compute(data)
