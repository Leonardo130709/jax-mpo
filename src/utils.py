from typing import Any, NamedTuple
from collections import defaultdict, deque

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfd


class TruncatedTanh(tfd.bijectors.Tanh):
    _lim = .999

    def _inverse(self, y):
        y = jnp.clip(y, a_min=-self._lim, a_max=self._lim)
        return super()._inverse(y)
