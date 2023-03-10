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
    v = jnp.array([[1., 0.], [0., 1.]]) @ y
    return jnp.dot(y, v)

# solve with constraints
solver = NewtonSolver()
result = solver.run(Minimize(
    fun=partial(cost, s=jnp.array([0.5, 1.5])),
    # constrain x[0] = -1, x[1] < 10
    constraints=(EqConstraint(lambda x: x[0] + 2),),
    # constraints=(EqConstraint(lambda x: x[0] + 1),
    #              IneqConstraint(lambda x: x[1] - 10)),
    initial_params=jnp.array([1.,1.])
))
print(result.params)

def solve(s):
    solver = NewtonSolver()
    result = solver.run(Minimize(
        fun=partial(cost, s=s),
        initial_params=jnp.array([1.,1.])
    ))
    return result.params

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
print(result.params)