import jax
import jax.numpy as jnp
import stanza.envs as envs
import stanza.policies as policies
import optax

from jax.random import PRNGKey
from stanza.policies import Actions
from stanza.policies.mpc import MPC, BarrierMPC
from stanza.policies.iterative import FeedbackRollout
from stanza.util.logging import logger

from stanza.solver.newton import NewtonSolver
from stanza.solver.optax import OptaxSolver
from stanza.solver.ilqr import iLQRSolver

optimizer = optax.chain(
    # Set the parameters of Adam. Note the learning_rate is not here.
    optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                            1000, alpha=0.1)),
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    # Put a minus sign to *minimise* the loss.
    optax.scale(-0.05)
)

# create an environment
logger.info("Creating environment")
env = envs.create("pendulum")

# rollout_inputs is an alias for the above

def rollout_inputs():
    rollout = policies.rollout(
                model=env.step,
                state0=env.reset(PRNGKey(0)),
                policy=Actions(jnp.ones((10,)))
            )
    logger.info('states: {}', rollout.states)
    logger.info('actions: {}', rollout.actions)
    
    # This is equivalent to the above:
    rollout = policies.rollout_inputs(
        model=env.step,
        state0=env.reset(PRNGKey(0)),
        actions=jnp.ones((10,))
    )

def rollout_mpc(solver='newton'):
    if solver == 'newton':
        solver_t = NewtonSolver()
    elif solver == 'optax':
        iterations = 10000
        optimizer = optax.chain(
            # Set the parameters of Adam. Note the learning_rate is not here.
            optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                    iterations, alpha=0.1)),
            optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
            # Put a minus sign to *minimise* the loss.
            optax.scale(-0.05)
        )
        solver_t = OptaxSolver(optimizer=optimizer, max_iterations=iterations)
    elif solver == 'ilqr':
        solver_t = iLQRSolver()
    # An MPC policy
    rollout = policies.rollout(
        model=env.step,
        state0=env.reset(PRNGKey(0)),
        policy=MPC(
            # Sample action
            action_sample=env.sample_action(PRNGKey(42)),
            cost_fn=env.cost, 
            model_fn=env.step,
            horizon_length=50,
            solver=solver_t,
            receed=False
        ),
        length=50
    )
    logger.info(f'MPC Rollout with {solver} solver results')
    logger.info('states: {}', rollout.states)
    logger.info('actions: {}', rollout.actions)
    cost = env.cost(rollout.states, rollout.actions)
    logger.info('cost: {}', cost)

# rollout_mpc(solver='newton')
rollout_mpc(solver='optax')
rollout_mpc(solver='ilqr')