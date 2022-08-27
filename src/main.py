import multiprocessing
import time

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
    actor.interact()


def run_learner(config, server_address):
    builder = Builder(config)
    key = jax.random.PRNGKey(0)
    client = reverb.Client(server_address)
    env, env_specs = builder.make_env()
    networks = builder.make_networks(env_specs)
    ds = builder.make_dataset_iterator(server_address)
    learner = builder.make_learner(key, env_specs, ds, networks, client)
    learner.learn()


def main():
    config = MPOConfig()
    builder = Builder(config)
    env, env_specs = builder.make_env()

    networks = builder.make_networks(env_specs)
    port = 41905
    server_address = f'localhost:{port}'
    tables = builder.make_reverb_tables(env_specs, networks)
    server = reverb.Server(tables, port=port)

    learner_process = multiprocessing.Process(
        target=run_learner,
        args=(config, server_address)
    )
    actor_process = multiprocessing.Process(
        target=run_actor,
        args=(config, server_address)
    )
    learner_process.start()
    actor_process.start()
    # learner_process.join()
    server.wait()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
