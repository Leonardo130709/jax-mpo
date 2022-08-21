import jax.random
import jax.numpy as jnp

from .config import MPOConfig
from .learner import MPOLearner
from .networks import make_networks
from .learner import MPOState

from dm_control import suite
from dmc_wrappers import StatesWrapper

env = suite.load('cartpole', 'balance')
env = StatesWrapper(env)
config = MPOConfig()

networks = make_networks(config, env.observation_spec(), env.action_spec())
import optax

optim = optax.multi_transform({
    'encoder': optax.sgd(1),
    'critic': optax.sgd(1),
    'actor': optax.sgd(1),
    'dual_params': optax.sgd(1)
}, ('encoder', 'critic', 'actor', 'dual_params'))

key = jax.random.PRNGKey(0)
encoder_params, critic_params, actor_params = networks.init(key)
dual_params = (jnp.zeros(()), jnp.zeros((1,)), jnp.zeros((1,)))

opt_state = optim.init((encoder_params, critic_params, actor_params, dual_params))

state = MPOState(actor_params=actor_params,
                 target_actor_params=actor_params,
                 critic_params=critic_params,
                 target_critic_params=critic_params,
                 encoder_params=encoder_params,
                 target_encoder_params=encoder_params,
                 dual_params=dual_params,
                 optim_state=opt_state,
                 key=key)

learner = MPOLearner(config, networks, optim, None, state)
learner.learn()
