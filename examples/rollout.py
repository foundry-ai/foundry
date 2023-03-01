import chex

from jax.random import PRNGKey
import jax.numpy as jnp

import stanza.envs as envs
import stanza.policy as policy
from stanza.policy import Actions
from stanza.policy.mpc import MPC
from stanza.util.logging import logger
from stanza.solver.newton import NewtonSolver

# create an environment
logger.info("Creating environment")
pendulum = envs.create("pendulum")

# Will rollout from a specified input array
logger.info("Rolling out trajectory")
rollout = policy.rollout(
            model=pendulum.step,
            state0=pendulum.reset(PRNGKey(0)),
            policy=Actions(jnp.ones((10,)))
        )
# we can see the rollout results
logger.info('Actions Rollout Results')
logger.info('states: {}', rollout.states)
logger.info('actions: {}', rollout.actions)

# An MPC policy
rollout = policy.rollout(
    model=pendulum.step,
    state0=pendulum.reset(PRNGKey(0)),
    policy=MPC(
        # Sample action
        action_sample=pendulum.sample_action(PRNGKey(0)),
        cost_fn=pendulum.cost, 
        model_fn=pendulum.step,
        horizon_length=20,
        solver=NewtonSolver()
    ),
    length=50
)
logger.info('MPC Rollout Results')
logger.info('states: {}', rollout.states)
logger.info('actions: {}', rollout.actions)

# Rolling out an MPC with log-barrier functions
logger.info('Barrier MPC Rollout Results')