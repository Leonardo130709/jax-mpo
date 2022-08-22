import multiprocessing

import jax
import jax.numpy as jnp
import reverb

from src.builder import Builder
from src.config import MPOConfig

key = jax.random.PRNGKey(0)

config = MPOConfig()
builder = Builder(config)
env, env_specs = builder.make_env()

networks = builder.make_networks(env_specs)
tables = builder.make_reverb_tables(env_specs, networks)
port = 41905
server = reverb.Server(tables, port=port)
client = server.localhost_client()
server_address = client.server_address
ds = builder.make_dataset_iterator(server_address)
learner = builder.make_learner(key, ds, networks, client)
actor = builder.make_actor(key, env, networks, client)

p = multiprocessing.Process(target=actor.interact,)
p.start()
# p.join()
import pdb; pdb.set_trace()
learner.learn()
# print(next(ds))