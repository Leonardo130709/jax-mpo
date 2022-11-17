import os
import multiprocessing as mp
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import reverb

from src.builder import Builder
from src.config import MPOConfig


def run_actor(builder, server_address):
    prepare_logdir(builder.cfg)
    client = reverb.Client(server_address)
    env, env_specs = builder.make_env()
    actor = builder.make_actor(env, env_specs, client)
    actor.run()


def run_learner(builder, server_address, env_specs):
    prepare_logdir(builder.cfg)
    ds = builder.make_dataset_iterator(server_address)
    client = reverb.Client(server_address)
    learner = builder.make_learner(env_specs, ds, client)
    learner.run()


def run_server(builder, env_specs):
    server = builder.make_server(env_specs)
    server.wait()


def prepare_logdir(cfg: MPOConfig):
    path = os.path.expanduser(cfg.logdir)
    if not os.path.exists(path):
        os.makedirs(path)
    cfg.save(path + "/config.yaml")


def main(config):
    builder = Builder(config)
    env, env_specs = builder.make_env()
    server_address = f"localhost:{config.reverb_port}"
    server = mp.Process(target=run_server,
                        args=(builder, env_specs)
                        )
    learner = mp.Process(target=run_learner,
                         args=(builder, server_address, env_specs)
                         )
    server.start()
    learner.start()
    actors = []
    for _ in range(config.num_actors):
        actor = mp.Process(target=run_actor,
                           args=(builder, server_address)
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
