import jax
import jax.numpy as jnp
import stanza.envs as envs
import stanza.policy as policy
import optax

from jax.random import PRNGKey
from stanza.policy import Actions
from stanza.policy.mpc import MPC, BarrierMPC
from stanza.util.logging import logger

from stanza.solver.newton import NewtonSolver
from stanza.solver.optax import OptaxSolver
from stanza.solver.ilqr import iLQRSolver


# create an environment
logger.info("Creating environment")
env = envs.create("pendulum")

# rollout_inputs is an alias for the above

def rollout_inputs():
    rollout = policy.rollout(
                model=env.step,
                state0=env.reset(PRNGKey(0)),
                policy=Actions(jnp.ones((10,)))
            )
    logger.info('states: {}', rollout.states)
    logger.info('actions: {}', rollout.actions)
    
    # This is equivalent to the above:
    rollout = policy.rollout_inputs(
        model=env.step,
        state0=env.reset(PRNGKey(0)),
        actions=jnp.ones((10,))
    )

def rollout_mpc_newton():
    # An MPC policy
    rollout = policy.rollout(
        model=env.step,
        state0=env.reset(PRNGKey(0)),
        policy=MPC(
            # Sample action
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost, 
            model_fn=env.step,
            horizon_length=20,
            solver=NewtonSolver()
        ),
        length=50
    )
    logger.info('MPC Rollout with Newton solver results')
    logger.info('states: {}', rollout.states)
    logger.info('actions: {}', rollout.actions)

def rollout_mpc_optax():
    optimizer = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                1000, alpha=0.01)),
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-0.01)
    )
    rollout = policy.rollout(
        model=env.step,
        state0=env.reset(PRNGKey(0)),
        policy=MPC(
            # Sample action
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost, 
            model_fn=env.step,
            horizon_length=20,
            solver=OptaxSolver(optimizer=optimizer)
        ),
        length=50
    )
    logger.info('MPC Rollout with Optax solver results')
    logger.info('states: {}', rollout.states)
    logger.info('actions: {}', rollout.actions)


def rollout_barrier():
    # Barrier-MPC based rollout
    rollout = policy.rollout(
        model=env.step,
        state0=env.reset(PRNGKey(0)),
        policy=BarrierMPC(
            # Sample action
            action_sample=env.sample_action(PRNGKey(0)),
            cost_fn=env.cost, 
            model_fn=env.step,
            horizon_length=20,
        ),
        length=100
    )

    logger.info('MPC Rollout with barrier results')
    logger.info('states: {}', rollout.states)
    logger.info('actions: {}', rollout.actions)
    logger.info('cost: {}', env.cost(rollout.states, rollout.actions))


def rollout_gradient():
    def roll_cost(actions):
        rollout = policy.rollout_inputs(
            model=env.step,
            state0=env.reset(PRNGKey(0)),
            actions=actions
        )
        return env.cost(rollout.states, rollout.actions)
    grad = jax.grad(roll_cost)(jnp.ones((20,)))
    logger.info("grad: {}", grad)

rollout_mpc_optax()
#rollout_gradient()