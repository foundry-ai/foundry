from jinx.solver import NewtonSolver, RelaxingSolver
from functools import partial

import jax.numpy as jnp
import pytest

def _quad_cost(x, soln):
    return jnp.dot(x - soln, x - soln)

def test_scalar_quadatic():
    fun = partial(_quad_cost, soln=5.)
    solver = NewtonSolver(fun)
    res = solver.run(10.)
    params = res.final_params
    assert pytest.approx(5., 0.01) == params

def test_matrix_quadratic():
    fun = partial(_quad_cost, soln=jnp.array([1,2,3]))

    solver = NewtonSolver(fun)
    res = solver.run(jnp.zeros((3,)))
    params = res.final_params
    assert pytest.approx(jnp.array([1,2,3]), 0.01) == params

def _cost_with_log_barrier(x, t, soln, max, min):
    cost = jnp.dot(x - soln, x - soln)
    barrier_cost = -jnp.log(-(x - max)) -jnp.log(-(min - x))
    return t*cost + barrier_cost

def test_log_barrier_quadratic():
    fun = partial(_cost_with_log_barrier, max=1, min=-10)

    sub_solver = NewtonSolver(fun)
    solver = RelaxingSolver(sub_solver)
    res = solver.run(-8., soln=5.)
    params = res.final_params
    assert pytest.approx(1., 0.01) == params