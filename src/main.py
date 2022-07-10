import jax
import jax.numpy as jnp
from statistics import mean
from .mpo import MPO
from . import utils


class RLAlg:
    def __init__(self, config):
        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)
        self.env = utils.make_env(config.task)
        self.agent = MPO(config, self.env, self.rng)
        self.buffer = utils.ReplayBuffer(config.buffer_capacity)
        self.interaction_count = 0

    def learn(self):
        obs = None
        while True:
            if obs is None:
                obs = self.env.reset().observation

            action = self.agent.act(obs, training=True)
            ts = self.env.step(action)
            reward = ts.reward
            done = ts.last()
            next_obs = ts.observation
            self.buffer.add(obs, action, reward, done, next_obs)
            self.interaction_count += 1

            if done:
                obs = None
            else:
                obs = next_obs

            transitions = self.buffer.sample(self.config.batch_size)
            self.agent._state, metrics = self.agent.step(self.agent._state, *transitions)

            if self.interaction_count % 1000 == 0:
                scores = [utils.evaluate(utils.make_env(self.config.task), self.agent.act) for _ in range(3)]
                print(self.interaction_count, mean(scores))
