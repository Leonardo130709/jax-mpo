import multiprocessing

import jax
import reverb

from src.builder import Builder
from src.config import MPOConfig
jax.config.update('jax_disable_jit', True)


def run_actor(config, server_address):
    builder = Builder(config)
    client = reverb.Client(server_address)
    key = jax.random.PRNGKey(0)
    env, env_specs = builder.make_env()
    networks = builder.make_networks(env_specs)
    actor = builder.make_actor(key, env, networks, client)
    actor.run()


def run_learner(config, server_address):
    builder = Builder(config)
    key = jax.random.PRNGKey(0)
    client = reverb.Client(server_address)
    env, env_specs = builder.make_env()
    networks = builder.make_networks(env_specs)
    ds = builder.make_dataset_iterator(server_address)
    learner = builder.make_learner(key, env_specs, ds, networks, client)
    learner.run()


def run_server(builder, env_specs):
    server = builder.make_server(env_specs)
    server.wait()


def main():
    config = MPOConfig()
    builder = Builder(config)
    env, env_specs = builder.make_env()
    address = f'localhost:{config.reverb_port}'
    server = multiprocessing.Process(target=run_server,
                                     args=(builder, env_specs))
    server.start()
    client = reverb.Client(address)
    actor = builder.make_actor(env, env_specs, client)
    ds = builder.make_dataset_iterator(address)
    learner = builder.make_learner(env_specs, ds, client)
    actor.run()
    learner.run()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
