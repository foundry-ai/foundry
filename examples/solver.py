from stanza.solver.newton import NewtonSolver
from stanza.solver import Minimize, IneqConstraint, EqConstraint
from functools import partial

import jax.numpy as jnp
from jax.tree_util import Partial
import jax
import sys
import stanza

def cost(x, s):
    y = x - jnp.array([3*s[0], 2*s[1]])
    v = jnp.array([[2., 0.], [0., 2.]]) @ y
    return jnp.dot(y, v)

def solve(s):
    solver = NewtonSolver()
    result = solver.run(Minimize(
        fun=partial(cost, s=s),
        initial_params=jnp.array([1.,1.])
    ))
    return result.solution.params

print(solve(jnp.array([0.5, 1.5])))
print(jax.jacrev(solve)(jnp.array([1., 1.5])))

# solver with eq constraint
solver = NewtonSolver()
result = solver.run(Minimize(
    fun=partial(cost, s=jnp.array([0.5, 1.5])),
    # constrain x < -1
    constraints=(EqConstraint(lambda x: x[0] + 1),),
    initial_params=jnp.array([1.,1.])
))
print(result.solution.params)


# solve an MPC problem
from stanza.solver.ilqr import iLQRSolver
from stanza.policies.mpc import MinimizeMPC
from stanza.envs.linear import LinearSystem

env = LinearSystem(
    A=jnp.array([[1., 1.], [0., 1.]]),
    B=jnp.array([[0.], [1.]]),
    Q=jnp.eye(2),
    R=jnp.eye(1),
)

@jax.jit
def solve(x):
    solver = iLQRSolver()
    res = solver.run(MinimizeMPC(
        initial_actions=jnp.zeros((9, 1)),
        state0=x,
        cost_fn=env.cost,
        model_fn=env.step
    ))
    return res.solution.actions[0]

# print(solve(jnp.array([0.,0.])))
print(jax.jacobian(solve)(jnp.array([0.,0.])))