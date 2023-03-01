import chex

from jax.random import PRNGKey
import jax.numpy as jnp

import stanza.envs as envs
import stanza.policy as policy
from stanza.policy import Actions
from stanza.policy.mpc import MPC
from stanza.util.logging import logger
from stanza.solver.newton import NewtonSolver
from stanza.solver.optax import OptaxSolver

import optax

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

# rollout_inputs is an alias for the above
rollout = policy.rollout_inputs(
            model=pendulum.step,
            state0=pendulum.reset(PRNGKey(0)),
            actions=jnp.ones((10,))
        )

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
logger.info('MPC Rollout with Newton solver results')
logger.info('states: {}', rollout.states)
logger.info('actions: {}', rollout.actions)
optimizer = optax.chain(
    # Set the parameters of Adam. Note the learning_rate is not here.
    optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                            1000, alpha=0.01)),
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    # Put a minus sign to *minimise* the loss.
    optax.scale(-0.01)
)
rollout = policy.rollout(
    model=pendulum.step,
    state0=pendulum.reset(PRNGKey(0)),
    policy=MPC(
        # Sample action
        action_sample=pendulum.sample_action(PRNGKey(0)),
        cost_fn=pendulum.cost, 
        model_fn=pendulum.step,
        horizon_length=20,
        solver=OptaxSolver(optimizer=optimizer)
    ),
    length=50
)
logger.info('MPC Rollout with Optax solver results')
logger.info('states: {}', rollout.states)
logger.info('actions: {}', rollout.actions)