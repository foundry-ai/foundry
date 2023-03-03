from stanza.solver.newton import NewtonSolver
from stanza.solver import Minimize
from functools import partial

import jax.numpy as jnp
from jax.tree_util import Partial
import jax
import stanza


def cost(x, s):
    y = x - 2*s
    v = jnp.array([[1., 0.], [0., 1.]]) @ y
    return jnp.dot(y, v)

def solve(s):
    solver = NewtonSolver()
    result = solver.run(Minimize(
        fun=partial(cost, s=s),
        init_params=jnp.array([1.,1.])
    ))
    return result.params

print(solve(jnp.array([1., 1.5])))
print(jax.jacrev(solve)(jnp.array([1., 1.5])))