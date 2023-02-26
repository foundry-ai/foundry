import jax.numpy as jnp
import optax
import pytest

from stanza.solver.optax import OptaxSolver
from functools import partial

def _quad_cost(x, soln):
    return jnp.dot(x - soln, x - soln)

def test_scalar_quadratic():
    cost = partial(_quad_cost, soln=1.5)
    solver = OptaxSolver(cost, optax.adam(0.01), 1000)
    res = solver.run(init_params=-1.0)
    assert pytest.approx(1.5, 0.01) == res.params