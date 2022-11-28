import os
import multiprocessing as mp
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import reverb

from src.builder import Builder
from src.config import MPOConfig


def run_actor(builder, env_rng, actor_rng, server_address):
    client = reverb.Client(server_address)
    env, env_specs = builder.make_env(env_rng)
    actor = builder.make_actor(actor_rng, env, env_specs, client)
    actor.run()


def run_learner(builder, learner_rng, server_address, env_specs):
    ds = builder.make_dataset_iterator(server_address)
    client = reverb.Client(server_address)
    learner = builder.make_learner(learner_rng, env_specs, ds, client)
    learner.run()


def run_server(builder, server_rng, env_specs):
    server = builder.make_server(server_rng, env_specs)
    server.wait()


def main(config):
    builder = Builder(config)
    rngs = jax.random.split(builder.rng, builder.cfg.num_actors + 4)
    rngs = jax.device_get(rngs)
    _, env_specs = builder.make_env(rngs[-1])
    server_address = f"localhost:{config.reverb_port}"
    server = mp.Process(target=run_server,
                        args=(builder, rngs[-2], env_specs)
                        )
    learner = mp.Process(target=run_learner,
                         args=(builder, rngs[-3], server_address, env_specs)
                         )
    server.start()
    learner.start()
    actors = []
    for i in range(config.num_actors):
        actor = mp.Process(
            target=run_actor,
            args=(builder, rngs[-4], rngs[i], server_address)
        )
        actor.start()
        actors.append(actor)

    for actor in actors:
        actor.join()
    learner.join()
    server.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    config = MPOConfig.from_entrypoint()
    main(config)
