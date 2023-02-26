from stanza.solver.newton import NewtonSolver
from functools import partial

import jax.numpy as jnp
import pytest

def _quad_cost(x, soln):
    return jnp.dot(x - soln, x - soln)

def test_scalar_quadatic():
    fun = partial(_quad_cost, soln=5.)
    solver = NewtonSolver(fun)
    res = solver.run(init_params=10.)
    assert pytest.approx(5., 0.01) == res.params

def test_matrix_quadratic():
    fun = partial(_quad_cost, soln=jnp.array([1,2,3]))

    solver = NewtonSolver(fun)
    res = solver.run(init_params=jnp.zeros((3,)))
    assert pytest.approx(jnp.array([1,2,3]), 0.01) == res.params

def _cost_with_log_barrier(x, t, soln, max, min):
    cost = jnp.dot(x - soln, x - soln)
    barrier_cost = -jnp.log(-(x - max)) -jnp.log(-(min - x))
    return t*cost + barrier_cost

def test_log_barrier_quadratic():
    fun = partial(_cost_with_log_barrier, soln=5, max=1, min=-10, t=100)

    solver = NewtonSolver(fun)
    res = solver.run(init_params=-8.)
    assert pytest.approx(1., 0.01) == res.params