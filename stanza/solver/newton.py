from stanza.util.dataclasses import dataclass, field
from stanza.solver import IterativeSolver, sanitize_cost
from typing import Any, Callable

import jax
import jax.numpy as jnp

@dataclass(jax=True)
class NewtonState:
    ineq_dual: jnp.array = None
    eq_dual: jnp.array = None

# A newton solver with backtracking support
@dataclass(jax=True, kw_only=True)
class NewtonSolver(IterativeSolver):
    fun: Callable = None

    # If specified, performs infeasible-newton update
    # to satisfy constraint = 0
    eq_constraint: Callable = None

    terminate: Callable = None
    has_aux: bool = field(default=False, jax_static=True)
    tol: float = 1e-2
    max_iterations: int = 500
    # backtracking beta
    beta: float = 0.5

    def update(self, fun_state, fun_params, solver_state):
        if self.eq_constraint is not None and solver_state is None:
            v = jnp.atleast_1d(self.eq_constraint(fun_params))
            dual_state = jnp.zeros_like(v)
        else:
            dual_state = solver_state

        # unravel argument structure
        param_v, p_fmt = jax.flatten_util.ravel_pytree(fun_params)

        # standardize the cost function
        cost_fun = sanitize_cost(self.fun, self.has_aux)

        vec_fun = lambda v: cost_fun(fun_state, p_fmt(v))[1]
        grad = jax.grad(vec_fun)(param_v)
        hess = jax.hessian(vec_fun)(param_v)

        if self.eq_constraint is not None:
            A = jax.jacrev(lambda v: jnp.atleast_1d(self.eq_constraint(p_fmt(v))))(param_v)
            sat = jnp.atleast_1d(self.constraint(fun_params))
            comb_hess = jnp.block([
                [hess, A.T],
                [A, jnp.zeros((A.shape[0], A.shape[0]))]
            ])
            comb_grad = jnp.block([grad + A.T @ dual_state, sat])
            comb_direction = -jnp.linalg.solve(comb_hess,comb_grad)
            direction = comb_direction[:-A.shape[0]]
            dual_direction = comb_direction[-A.shape[0]:]
            
            reduction = 0.5*comb_grad @ comb_direction
            current_cost = vec_fun(param_v)
            def backtrack_cond(t):
                new_cost = vec_fun(param_v + t*direction)
                exp_cost = current_cost + t*reduction
                return new_cost > exp_cost

            t = jax.lax.while_loop(backtrack_cond,
                                lambda t: self.beta*t, 1)
            new_param_v = param_v + t*direction
            dual_state = dual_state + t*dual_direction
            eps = jnp.linalg.norm(param_v - new_param_v)
        else:
            direction = -jnp.linalg.solve(hess,grad)
            # reduction amount for the backtracking

            reduction = 0.5*grad.T @ direction
            current_cost = vec_fun(param_v)
            def backtrack_cond(t):
                new_cost = vec_fun(param_v + t*direction)
                exp_cost = current_cost + t*reduction
                return new_cost > exp_cost

            t = jax.lax.while_loop(backtrack_cond,
                                lambda t: self.beta*t, 1)
            new_param_v = param_v + t*direction
            eps = jnp.linalg.norm(param_v - new_param_v)

        # Find the new state
        new_state, old_cost, _ = cost_fun(fun_state, fun_params)
        new_param = p_fmt(new_param_v)
        _, cost, aux = cost_fun(new_state, new_param)
        #jax.debug.print("old_cost {} new_cost {}", old_cost, cost)

        if self.terminate is not None:
            solved = self.terminate(cost, aux)
        else:
            solved = eps < self.tol
        return solved, new_state, new_param, dual_state