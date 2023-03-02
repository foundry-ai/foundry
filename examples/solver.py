from stanza.solver.newton import NewtonSolver
from stanza.solver import Minimize
from functools import partial

import jax.numpy as jnp
import jax

# How to use the newton solver


def cost(x, dsoln):
    y = x - 2*dsoln
    v = jnp.array([[1., 0.], [0., 1.]]) @ y
    return jnp.dot(y, v)

def solve(dsoln):
    solver = NewtonSolver()
    result = solver.run(Minimize(
        fun=partial(cost, dsoln=dsoln),
        init_params=jnp.array([1.,1.])
    ))
    return result.params

print(solve(jnp.array([1., 1.5])))
print(jax.jacrev(solve)(jnp.array([1., 1.5])))