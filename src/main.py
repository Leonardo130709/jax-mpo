import jax
import jax.numpy as jnp
from statistics import mean
from collections import defaultdict
from .mpo import MPO
from . import utils


class RLAlg:
    def __init__(self, config):
        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)
        self.env = utils.make_env(config.task)
        self.agent = MPO(config, self.env, self.rng)
        self.buffer = utils.ReplayBuffer(config.buffer_capacity)
        self.callback = defaultdict(list)
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
            transitions = jax.tree_map(jnp.asarray, transitions)
            metrics = self.agent.step(*transitions)

            for k, v in metrics.items():
                self.callback[k].append(v)

            if self.interaction_count % 10000 == 0:
                scores = [utils.evaluate(utils.make_env(self.config.task), self.agent.act) for _ in range(10)]
                print(self.interaction_count, mean(scores))
                self.callback['scores'].append(mean(scores))
                # import matplotlib.pyplot as plt
                # from IPython.display import clear_output
                # clear_output(wait=True)
                # for k, v in self.callback.items():
                #     plt.plot(v, label=k)
                #     plt.legend()
                #     plt.show()

