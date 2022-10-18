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
    dual_loss = duals * sg(loss - epsilon)

    if per_dimension:
        scaled_loss = jnp.sum(scaled_loss, axis=-1)
        dual_loss = jnp.sum(dual_loss, axis=-1)

    return scaled_loss, dual_loss


def quantile_regression_loss(predictions: Array,
                             taus: Array,
                             targets: Array,
                             hubber_delta: float
                             ) -> Array:
    chex.assert_type([predictions, taus, targets], float)
    chex.assert_rank([predictions, taus, targets], 1)
    sg = jax.lax.stop_gradient
    targets = sg(targets)

    resids = targets[jnp.newaxis, :] - predictions[:, jnp.newaxis]
    ind = (resids < 0).astype(taus.dtype)
    weight = jnp.abs(taus[:, jnp.newaxis] - ind)
    loss = optax.huber_loss(resids, delta=hubber_delta)
    loss *= sg(weight)

    return jnp.sum(jnp.mean(loss, axis=-1))


def cross_entropy_loss(dist: tfd.Distribution,
                       actions: Array,
                       normalized_weights: Array
                       ) -> Array:
    chex.assert_type([actions, normalized_weights], float)
    log_probs = dist.log_prob(actions)
    return - jnp.sum(normalized_weights * log_probs)


def temperature_loss_and_normalized_weights(
        temperature: Array,
        q_values: Array,
        epsilon: float
) -> Tuple[Array, Array]:
    """Direct dual constraint as a part of MPO loss."""
    chex.assert_type([temperature, q_values, epsilon], float)
    chex.assert_rank([temperature, q_values, epsilon], [0, 1, 0])
    sg = jax.lax.stop_gradient

    tempered_q_values = sg(q_values) / temperature
    tempered_q_values = tempered_q_values.astype(jnp.float32)
    normalized_weights = jax.nn.softmax(tempered_q_values)
    normalized_weights = sg(normalized_weights)

    log_num_actions = jnp.log(q_values.size)
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
