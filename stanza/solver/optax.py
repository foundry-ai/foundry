from stanza.util.dataclasses import dataclass
from stanza.solver import IterativeSolver, UnsupportedObectiveError, \
        SolverState, Minimize

from typing import Any
import jax
import jax.numpy as jnp

import optax
import jax
import jax.numpy as jnp


@dataclass(jax=True)
class OptaxState(SolverState):
    optimizer_state : Any

@dataclass(jax=True, kw_only=True)
class OptaxSolver(IterativeSolver):
    tol: float = 1e-3
    optimizer: Any = None

    def init_state(self, objective):
        return OptaxState(
            iteration=0,
            solved=False,
            obj_state=objective.init_state,
            obj_params=objective.init_params,
            obj_aux=None,
            optimizer_state = self.optimizer.init(objective.init_params)
        )

    def optimality(self, objective, obj_state, obj_params):
        grad = jax.grad(lambda p: objective.eval(obj_state, p)[1])(obj_params)
        return grad

    def update(self, objective, state):
        if not isinstance(objective, Minimize) or objective.constraints:
            raise UnsupportedObectiveError("Can only handle unconstrained minimization objectives")
        if state is None:
            state = self.init_state(objective)
        def f(p):
            obj_state, cost, obj_aux = objective.eval(state.obj_state, p)
            return cost, (obj_state, obj_aux)
        grad, (obj_state, obj_aux) = jax.grad(f, has_aux=True)(state.obj_params)
        updates, new_opt_state = self.optimizer.update(grad, state.optimizer_state, state.obj_params)
        obj_params = optax.apply_updates(state.obj_params, updates)
        return OptaxState(iteration=state.iteration + 1,
                    solved=False,
                    obj_state=obj_state,
                    obj_params=obj_params,
                    obj_aux=obj_aux,
                    optimizer_state=new_opt_state
                )