import chex

from jax.random import PRNGKey
import jax.numpy as jnp

import stanza.envs as envs
from stanza.envs import Actions
from stanza.util.logging import logger

# create an environment
logger.info("Creating environment")
pendulum = envs.create("pendulum")

# Will rollout from a specified input array
logger.info("Rolling out trajectory")
rollout = envs.rollout(
            model=pendulum.step,
            state0=pendulum.reset(PRNGKey(0)),
            policy=Actions(jnp.ones((10,)))
        )
# we can see the rollout results
logger.info('Rollout Results')
logger.info('states: {}', rollout.states)
logger.info('actions: {}', rollout.actions)
logger.info('final_policy_state: {}', rollout.final_policy_state)