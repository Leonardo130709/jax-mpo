import numpy as np
from dm_env import specs
from typing import Any
import dm_env
from dm_control import suite
from collections import defaultdict, deque
from typing import NamedTuple
import jax


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

    # TODO: explicit declaration
    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self) -> dm_env.Environment:
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped
        else:
            return self.env


class StatesWrapper(Wrapper):
    """ Converts OrderedDict obs to 1-dim np.ndarray[np.float32]. """
    def observation(self, timestamp):
        obs = []
        for v in timestamp.observation.values():
            if v.ndim == 0:
                v = v[None]
            obs.append(v.flatten())
        obs = np.concatenate(obs)
        return obs.astype(np.float32)

    def observation_spec(self):
        dim = sum((np.prod(ar.shape) for ar in self.env.observation_spec().values()))
        return specs.Array(shape=(dim,), dtype=np.float32, name='states')


def make_env(task_name):
    domain, task = task_name.split('_', maxsplit=1)
    env = suite.load(domain, task)
    return StatesWrapper(env)


def simulate(env, policy, training=False, init_obs=None, num_actions=50):
    if init_obs is None:
        ts = env.reset()
        obs = ts.observation
    else:
        obs = init_obs
    tr = defaultdict(list)
    for i in range(num_actions):
        action = policy(obs, training)
        ts = env.step(action)
        next_obs = ts.observation
        done = ts.last()
        reward = ts.reward
        tr['observations'].append(obs)
        tr['actions'].append(action)
        tr['rewards'].append(reward)
        tr['done_flags'].append(done)
        obs = next_obs
        if done:
            obs = None

    for k, v in tr.items():
        tr[k] = np.array(v)
    return tr, obs


class Transition(NamedTuple):
    state: Any
    action: Any
    reward: Any
    done: Any
    next_state: Any


class ReplayBuffer:
    def __init__(self, capacity):
        self._data = deque(maxlen=capacity)

    def add(self, state, action, reward, done, next_state):
        tr = Transition(state, action, reward, done, next_state)
        self._data.append(tr)

    def sample(self, size):
        idx = np.random.randint(len(self._data), size=size)
        transitions = [self._data[i] for i in idx]
        return jax.tree_map(lambda *t: jax.numpy.stack(t), *transitions)


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
