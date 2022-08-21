from typing import Any, NamedTuple
from collections import defaultdict, deque

import jax
import jax.numpy as jnp
import numpy as np
from dm_control import suite
from dmc_wrappers import StatesWrapper
import tensorflow_probability.substrates.jax as tfd


def make_env(task_name, task_kwargs=None, environment_kwargs=None):
    domain, task = task_name.split('_', maxsplit=1)
    if domain == "ball":
        domain = "bal_in_cup"
        task = "catch"
    env = suite.load(domain, task,
                     task_kwargs=task_kwargs,
                     environment_kwargs=environment_kwargs)
    return StatesWrapper(env)


class TruncatedTanh(tfd.bijectors.Tanh):
    _lim = .999

    def _inverse(self, y):
        y = jnp.clip(y, a_min=-self._lim, a_max=self._lim)
        return super()._inverse(y)
