from stanza.solver import NewtonSolver

import jax.numpy as jnp

# How to use the newton solver

def cost(x):
    y = x - jnp.array([1., 2.])
    v = jnp.array([[1., 0.], [0., 1.]]) @ y
    return jnp.dot(y, v)

solver = NewtonSolver(cost)
result = solver.run(jnp.array([1., 1.]))
print('Solved', result.solved)
print('Final Params', result.final_params)
print('Final Solver State', result.final_state)