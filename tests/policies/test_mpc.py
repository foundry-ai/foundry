from stanza.policies import PolicyInput
from stanza.policies.mpc import MPC
from stanza.solver.ilqr import iLQRSolver
from stanza.envs.linear import LinearSystem

from jax.random import PRNGKey

import jax
import jax.numpy as jnp

from chex import assert_trees_all_close

def test_mpc():
    env = LinearSystem(
        A=jnp.array([[1., 1.], [0., 1.]]),
        B=jnp.array([[0.], [1.]]),
        Q=jnp.eye(2),
        R=jnp.eye(1),
    )
    K = -jnp.linalg.inv(env.R + env.B.T @ env.P @ env.B) @ \
        (env.B.T @ env.P @ env.A)
    mpc = MPC(
        action_sample=env.sample_action(PRNGKey(0)),
        cost_fn=env.cost,
        model_fn=env.step,
        horizon_length=10,
    )
    @jax.jit
    def eval(x):
        return mpc(PolicyInput(x)).action
    eval_jac = jax.jacobian(eval)

    test_states = [
        jnp.array([0., 0.]),
        jnp.array([1., 0.]),
        jnp.array([0., 1.]),
        jnp.array([-1., -1.]),
    ]
    for test_state in test_states:
        expert_action = K @ test_state
        test_action = eval(test_state)
        test_jac = eval_jac(test_state)
        assert_trees_all_close(test_action, expert_action, atol=1e-5)
        assert_trees_all_close(test_jac, K, atol=1e-5)