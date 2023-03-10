import jax
import jax.numpy as jnp
import stanza.envs as envs
import stanza.policies as policies
import optax

from jax.random import PRNGKey
from stanza.policies import Actions
from stanza.policies.mpc import MPC, BarrierMPC
from stanza.util.logging import logger

from stanza.solver.newton import NewtonSolver
from stanza.solver.optax import OptaxSolver
from stanza.solver.ilqr import iLQRSolver


# create an environment
logger.info("Creating environment")
env = envs.create("linear/di")

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

def rollout_mpc_newton():
    # An MPC policy
    rollout = policies.rollout(
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
    rollout = policies.rollout(
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
    p = BarrierMPC(
            # Sample action
            action_sample=env.sample_action(PRNGKey(0)),
            barrier_sdf=env.constraints,
            cost_fn=env.cost, 
            model_fn=env.step,
            horizon_length=10,
        )
    rollout = policies.rollout(
        model=env.step,
        state0=env.reset(PRNGKey(0)),
        policy=p,
        length=20
    )
    logger.info('MPC Rollout with barrier results')
    logger.info('states: {}', rollout.states)
    logger.info('actions: {}', rollout.actions)
    logger.info('barrier: {}', jnp.max(env.constraints(rollout.states, rollout.actions)))
    logger.info('cost: {}', env.cost(rollout.states, rollout.actions))


def rollout_gradient():
    def roll_cost(actions):
        rollout = policies.rollout_inputs(
            model=env.step,
            state0=env.reset(PRNGKey(0)),
            actions=actions
        )
        return env.cost(rollout.states, rollout.actions)
    grad = jax.grad(roll_cost)(jnp.ones((10,)))
    logger.info("grad: {}", grad)

# rollout_mpc_optax()
rollout_barrier()
# rollout_gradient()