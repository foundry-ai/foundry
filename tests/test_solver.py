import jax
import jax.numpy as jnp
import optax

from stanza import partial
from stanza.solver import Minimize
from stanza.solver.optax import OptaxSolver

def cost(x, s):
    y = x - jnp.array([3*s[0], 2*s[1]])
    v = jnp.array([[2., 0.], [0., 2.]]) @ y
    return jnp.dot(y, v)

def solve(s, solver=None):
    result = solver.run(Minimize(
        fun=partial(cost, s=s),
        initial_params=jnp.array([1.,1.])
    ))
    return result.solution.params

def test_optax_solver():
    iterations = 1000
    solver = OptaxSolver(
        max_iterations=iterations,
        optimizer=optax.adam(optax.cosine_decay_schedule(1, iterations))
    )
    solve_fn = partial(solve, solver=solver)

    solution = solve_fn(jnp.array([1., 2.]))
    jac = jax.jacobian(solve_fn)(jnp.array([1., 2.]))