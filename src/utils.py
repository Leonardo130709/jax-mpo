from typing import Any, NamedTuple
from collections import defaultdict, deque

import dm_env
from dm_env import specs
from dm_control import suite
import dm_control.rl.control
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


class Wrapper(dm_env.Environment):
    """This allows to modify attributes which agent observes and to pack it back."""
    def __init__(self, env: dm_env.Environment):
        self.env = env

    @staticmethod
    def observation(timestep: dm_env.TimeStep) -> Any:
        return timestep.observation

    @staticmethod
    def reward(timestep: dm_env.TimeStep) -> float:
        return timestep.reward

    @staticmethod
    def done(timestep: dm_env.TimeStep) -> bool:
        return timestep.last()

    @staticmethod
    def step_type(timestep: dm_env.TimeStep) -> dm_env.StepType:
        return timestep.step_type

    @staticmethod
    def discount(timestep: dm_env.TimeStep) -> float:
        return timestep.discount

    def step(self, action) -> dm_env.TimeStep:
        timestep = self.env.step(action)
        return self._wrap_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        return self._wrap_timestep(self.env.reset())

    def _wrap_timestep(self, timestep) -> dm_env.TimeStep:
        # pylint: disable-next=protected-access
        return timestep._replace(
            step_type=self.step_type(timestep),
            reward=self.reward(timestep),
            discount=self.discount(timestep),
            observation=self.observation(timestep)
        )

    def action_spec(self) -> dm_env.specs.Array:
        return self.env.action_spec()

    def observation_spec(self) -> dm_env.specs.Array:
        return self.env.observation_spec()

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self) -> dm_env.Environment:
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped
        else:
            return self.env


class StatesWrapper(Wrapper):
    """Converts OrderedDict obs to 1-dim np.ndarray[np.float32].
    Does a similar thing as dm_control/flatten_observation."""
    def observation(self, timestep):
        return dm_control.rl.control.flatten_observation(timestep.observation)

    def observation_spec(self):
        dim = sum(
            np.prod(ar.shape) for ar in self.env.observation_spec().values()
        )
        return dm_env.specs.Array(shape=(dim,), dtype=np.float32, name='states')


def make_env(task_name, task_kwargs=None, environment_kwargs=None):
    domain, task = task_name.split('_', maxsplit=1)
    if domain == "ball":
        domain = "bal_in_cup"
        task = "catch"
    env = suite.load(domain, task,
                     task_kwargs=task_kwargs,
                     environment_kwargs=environment_kwargs)
    return StatesWrapper(env)


def simulate(env, policy, training=False):
    done = False
    obs = env.reset().observation
    tr = defaultdict(list)
    while not done:
        obs = jax.tree_util.tree_map(jnp.asarray, obs)
        action = policy(obs, training)
        ts = env.step(action)
        next_obs = ts.observation
        done = ts.last()
        reward = ts.reward
        tr['observations'].append(obs)
        tr['actions'].append(action)
        tr['rewards'].append(reward)
        tr['done_flags'].append(done)
        tr['next_observation'].append(next_obs)
        obs = next_obs

    return {k: np.array(v) for k, v in tr.items()}


class Transition(NamedTuple):
    state: Any
    action: Any
    reward: Any
    done: Any
    next_state: Any


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self._data = deque(maxlen=capacity)
        self._batch_size = batch_size

    def add(self, tr):
        transitions = map(tr.get, ('observations', 'actions', 'rewards',
                                   'done_flags', 'next_observation'))
        for state, action, reward, done, next_state in zip(*transitions):
            self._data.append(
                Transition(state, action, reward, done, next_state))

    def sample(self, size):
        idx = np.random.randint(len(self._data), size=size)
        transitions = [self._data[i] for i in idx]
        return jax.tree_map(lambda *t: jax.numpy.stack(t), *transitions)

    def as_dataset(self, size):
        samples = self.sample(size)
        ds = tf.data.Dataset.from_tensor_slices(samples)
        ds = ds.batch(self._batch_size,
                      drop_remainder=True,
                      num_parallel_calls=tf.data.AUTOTUNE)
        # ds = ds.map(lambda x: jax.tree_map(lambda t: jnp.asarray(memoryview(t)), x))
        return ds.prefetch(tf.data.AUTOTUNE)\
            .as_numpy_iterator()


def evaluate(env, policy):
    total_reward = 0
    done = False
    obs = env.reset().observation
    while not done:
        action = policy(obs, training=False)
        ts = env.step(action)
        obs = ts.observation
        done = ts.last()
        total_reward += ts.reward
    return total_reward
