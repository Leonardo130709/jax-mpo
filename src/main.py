import os
import multiprocessing as mp
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import jax
import reverb
import chex

from src.builder import Builder
from src.config import MPOConfig
jax.config.update("jax_disable_jit", True)


def run_actor(builder, server_address):
    env, env_specs = builder.make_env()
    client = reverb.Client(server_address)
    actor = builder.make_actor(env, env_specs, client)
    actor.run()


def run_learner(builder, server_address, env_specs):
    ds = builder.make_dataset_iterator(server_address)
    client = reverb.Client(server_address)
    learner = builder.make_learner(env_specs, ds, client)
    learner.run()


def run_server(builder, env_specs):
    server = builder.make_server(env_specs)
    server.wait()


def main():
    config = MPOConfig()
    builder = Builder(config)
    chex.disable_asserts()
    env, env_specs = builder.make_env()
    server_address = f"localhost:{config.reverb_port}"
    server = mp.Process(target=run_server,
                        args=(builder, env_specs))
    actor = mp.Process(target=run_actor,
                       args=(builder, server_address))
    learner = mp.Process(target=run_learner,
                         args=(builder, server_address, env_specs)
                         )
    server.start()
    actor.start()
    learner.start()

    actor.join()
    learner.join()
    server.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
