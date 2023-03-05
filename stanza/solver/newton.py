from stanza.util.dataclasses import dataclass
from stanza.solver import IterativeSolver, UnsupportedObectiveError, \
        SolverState, Minimize

import jax
import jax.numpy as jnp

# A newton solver with backtracking support
@dataclass(jax=True, kw_only=True)
class NewtonSolver(IterativeSolver):
    tol: float = 1e-2
    beta: float = 0.5 # backtracking beta

    def init_state(self, objective):
        return SolverState(
            iteration=0,
            solved=False,
            obj_state=objective.init_state,
            obj_params=objective.init_params,
            obj_aux=None
        )
    
    # gradient of the objective at params == 0
    def optimality(self, objective, obj_state, obj_params):
        grad = jax.grad(lambda p: objective.eval(obj_state, p)[1])(obj_params)
        return grad

    def update(self, objective, state):
        if not isinstance(objective, Minimize) or objective.constraints:
            raise UnsupportedObectiveError("Can only handle unconstrained minimization objectives")
        if state is None:
            state = self.init_state(objective)

        new_state, _, aux = objective.eval(state.obj_state, state.obj_params)

        # unravel argument structure into param_v
        param_v, p_fmt = jax.flatten_util.ravel_pytree(state.obj_params)
        vec_cost = lambda v: objective.eval(state.obj_state, p_fmt(v))[1]

        grad = jax.grad(vec_cost)(param_v)
        hess = jax.hessian(vec_cost)(param_v)
        direction = -jnp.linalg.solve(hess,grad)

        reduction = 0.5*grad.T @ direction
        current_cost = vec_cost(param_v)
        def backtrack_cond(t):
            new_cost = vec_cost(param_v + t*direction)
            exp_cost = current_cost + t*reduction
            return new_cost > exp_cost

        t = jax.lax.while_loop(backtrack_cond,
                            lambda t: self.beta*t, 1)
        new_param_v = param_v + t*direction
        eps = jnp.linalg.norm(param_v - new_param_v)

        # Find the new state
        new_params = p_fmt(new_param_v)
        solved = eps < self.tol
        return SolverState(
            iteration=state.iteration + 1,
            solved=solved,
            obj_params=new_params,
            obj_state=new_state,
            obj_aux=aux
        )