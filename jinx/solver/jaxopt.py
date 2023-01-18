from jinx.solver import SolverResults
import jax.numpy as jnp

from jaxopt import LBFGS, BFGS, GradientDescent

class JaxOptSolver:
    def __init__(self, fun, jaxopt_solver_builder, *args, **kwargs):
        self.solver = jaxopt_solver_builder(fun, *args, **kwargs)
    
    def run(self, init_params, *args, history=False, **kwargs):
        if history:
            raise ValueError("Unable to use history on jaxopt solver")
        params, state = self.solver.run(init_params, *args, **kwargs)
        return SolverResults(
            solved=True,
            final_params=params,
            final_state=state,
            history=None
        )