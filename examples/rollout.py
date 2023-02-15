import chex

from jax.random import PRNGKey
import jax.numpy as jnp

import stanza.envs as envs
from stanza.envs import Actions
from stanza.logging import logger

# create an environment
pendulum = envs.create("pendulum")

# Will rollout from a specified input array
rollout = envs.rollout(
            model=pendulum.step,
            state0=pendulum.reset(PRNGKey(0)),
            policy=Actions(jnp.ones((10,)))
        )
# we can see the rollout results
logger.info('rollout', 'Rollout Results')
logger.info('rollout', 'states', rollout.states)
logger.info('rollout', 'actions', rollout.actions)
logger.info('rollout', 'final_policy_state', rollout.final_policy_state)
