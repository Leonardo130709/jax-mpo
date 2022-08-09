from statistics import mean
from collections import defaultdict

import jax
import jax.numpy as jnp
import jax.profiler

from .mpo import MPO
from . import utils

jax.config.update('jax_disable_jit', True)


class RLAlg:
    def __init__(self, config):
        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)
        self.env = utils.make_env(config.task,
                                  task_kwargs={'random': config.seed})
        self.agent = MPO(config, self.env, self.rng)
        self.buffer = utils.ReplayBuffer(config.buffer_capacity,
                                         config.batch_size)
        self.callback = defaultdict(list)
        self.interactions_count = 0

    def learn(self):
        jax.profiler.start_trace('/tmp/tensorboard')
        while True:
            tr = utils.simulate(self.env, self.agent.act, training=True)
            int_count = len(tr['actions'])
            self.interactions_count += int_count
            self.buffer.add(tr)

            ds = self.buffer.as_dataset(self.config.spi * len(tr['actions']))
            for transitions in ds:
                metrics = self.agent.step(*transitions)
                for k, v in metrics.items():
                    self.callback[k].append(v)

            jax.profiler.stop_trace()

            if self.interactions_count % 10000 == 0:
                scores = [
                    utils.simulate(
                        utils.make_env(self.config.task),
                        self.agent.act,
                        training=False
                    ) for _ in range(10)]
                print(f'Interactions{self.interactions_count}, '
                      f'score= {mean(scores)}')
                self.callback['scores'].append(mean(scores))
                # import matplotlib.pyplot as plt
                # from IPython.display import clear_output
                # clear_output(wait=True)
                # for k, v in self.callback.items():
                #     plt.plot(v, label=k)
                #     plt.legend()
                #     plt.show()

