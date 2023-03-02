from typing import Any, Callable

from stanza.solver import IterativeSolver, sanitize_cost
from stanza.util.dataclasses import dataclass, field

import optax
import jax
import jax.numpy as jnp

@dataclass(jax=True, kw_only=True)
class OptaxSolver(IterativeSolver):
    optimizer: Any = None
    terminate: Callable = None
    has_aux: bool = field(default=False, jax_static=True)
    tol: float = 1e-3

    def update(self, fun_state, fun_params, solver_state):
        if solver_state is None:
            solver_state = self.optimizer.init(fun_params)

        cost_fun = sanitize_cost(self.fun, self.has_aux)
        fun = lambda v: cost_fun(fun_state, v)[1]
        grad = jax.grad(fun)(fun_params)
        updates, opt_state = self.optimizer.update(grad, solver_state)

        new_params = optax.apply_updates(fun_params, updates)
        new_state, _, _ = cost_fun(fun_state, fun_params)
        _, cost, aux = cost_fun(new_state, new_params)

        # check the distance for solution
        if self.terminate is not None:
            solved = self.terminate(cost, aux)
        else:
            solved = False
        return solved, new_state, new_params, opt_state