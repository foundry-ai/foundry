from stanza.solver.newton import NewtonSolver

import jax.numpy as jnp

# How to use the newton solver

def cost(x):
    y = x - jnp.array([1., 2.])
    v = jnp.array([[1., 0.], [0., 1.]]) @ y
    return jnp.dot(y, v)

solver = NewtonSolver(cost)
result = solver.run(init_params=jnp.array([1., 1.]))
print('Solved', result.solved)
print('Final Params', result.params)
print('Final State', result.state)
print('Final Solver State', result.solver_state)

# NewtonSolver with linear constraint
solver = NewtonSolver(cost, lambda x: x[0] - 2)
result = solver.run(init_params=jnp.array([1., 1.]))
print('Solved', result.solved)
print('Final Params', result.params)