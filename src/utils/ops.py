"""There is a well known rlax, but as a matter of practice..."""
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import chex
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

Array = jnp.ndarray


def scaled_and_dual_loss(loss: Array,
                         duals: Array,
                         epsilon: float,
                         per_dimension: bool
                         ) -> Tuple[Array, Array]:
    """Lagrange multiplier loss."""
    chex.assert_rank(epsilon, 0)
    chex.assert_type([loss, duals], float)

    sg = jax.lax.stop_gradient
    scaled_loss = sg(duals) * loss
    dual_loss = duals * sg(epsilon - loss)

    if per_dimension:
        scaled_loss = jnp.sum(scaled_loss, axis=-1)
        dual_loss = jnp.sum(dual_loss, axis=-1)

    return scaled_loss, dual_loss


def quantile_regression_loss(predictions: Array,
                             pred_quantiles: Array,
                             targets: Array,
                             hubber_delta: float
                             ) -> Array:
    chex.assert_type([predictions, pred_quantiles, targets], float)
    chex.assert_rank([predictions, pred_quantiles, targets], 1)
    sg = jax.lax.stop_gradient
    targets = sg(targets)

    resids = targets[jnp.newaxis, :] - predictions[:, jnp.newaxis]
    ind = (resids < 0).astype(pred_quantiles.dtype)
    weight = jnp.abs(pred_quantiles[:, jnp.newaxis] - ind)
    loss = optax.huber_loss(resids, delta=hubber_delta) / hubber_delta
    loss *= sg(weight)

    return jnp.sum(jnp.mean(loss, axis=-1))


def cross_entropy_loss(dist: tfd.Distribution,
                       actions: Array,
                       normalized_weights: Array
                       ) -> Array:
    chex.assert_type([actions, normalized_weights], float)
    chex.assert_rank([actions, normalized_weights], [2, 1])
    log_probs = dist.log_prob(actions)
    return - jnp.sum(normalized_weights * log_probs)


def temperature_loss_and_normalized_weights(
        temperature: Array,
        q_values: Array,
        epsilon: float,
        tv_constraint: float
) -> Tuple[Array, Array]:
    """Direct dual constraint as a part of MPO loss."""
    chex.assert_type([temperature, q_values, epsilon, tv_constraint], float)
    chex.assert_rank(
        [temperature, q_values, epsilon, tv_constraint], [0, 1, 0, 0]
    )
    sg = jax.lax.stop_gradient
    # q_values = sg(q_values)
    # adv = q_values - jnp.mean(q_values)
    # tempered_q_values = jnp.clip(
    #     adv / temperature,
    #     a_min=-tv_constraint,
    #     a_max=tv_constraint
    # )
    tempered_q_values = sg(q_values) / temperature
    tempered_q_values = tempered_q_values.astype(jnp.float32)
    normalized_weights = jax.nn.softmax(tempered_q_values)
    normalized_weights = sg(normalized_weights)

    log_num_actions = jnp.log(q_values.size / 1.)
    q_logsumexp = logsumexp(tempered_q_values)
    temperature_loss = epsilon + q_logsumexp - log_num_actions
    temperature_loss *= temperature

    return temperature_loss.astype(q_values.dtype),\
           normalized_weights.astype(q_values.dtype)


def softplus(param):
    param = jnp.maximum(param, -18.)
    return jax.nn.softplus(param) + 1e-8


@tfd.RegisterKL(tfd.TruncatedNormal, tfd.TruncatedNormal)
def _kl_trunc_trunc(dist_a, dist_b, name=None):
    # Duct tape.
    norm_kl = tfd.kullback_leibler._DIVERGENCES[(tfd.Normal, tfd.Normal)]
    return norm_kl(dist_a, dist_b, name=name)


def sample_from_geometrical(rng, discount_t, shape=()):
    # P(t) ~ \prod^t_0 (1 - d_i) * d_t
    chex.assert_type(discount_t, float)
    chex.assert_rank(discount_t, 1)

    cont_prob_t = jnp.concatenate([
        jnp.ones_like(discount_t[:1]),
        discount_t
    ])
    term_prob_t = jnp.concatenate([
        1. - discount_t,
        jnp.ones_like(discount_t[-1:])
    ])
    cumprod_t = jnp.cumprod(cont_prob_t)
    prob_t = cumprod_t * term_prob_t
    return jax.random.choice(
        rng, shape=shape, a=prob_t.size, p=prob_t)
