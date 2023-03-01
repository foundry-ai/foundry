import jax
import jax.flatten_util
import jax.numpy as jnp

from typing import NamedTuple, Callable, Any
from functools import partial
from stanza.util.dataclasses import dataclass, replace

@dataclass(jax=True)
class SolverResults:
    solved: bool
    state: Any
    params: Any
    solver_state: Any
    history: Any = None

class Solver:
    def run(self, *, init_params=None, init_state=None, **kwargs) -> SolverResults:
        raise NotImplementedError("Solver must implement run()")

# The loop
class LoopState(NamedTuple):
    fun_state: Any
    fun_params: Any
    solver_state: Any
    i: int

class IterativeSolver(Solver):
    # Must have a max_iterations attribute!

    # solver_state=None is passed for the first iteration
    # should return new_fun_state, new_fun_params, new_solver_state
    def update(self, fun_state, fun_params, solver_state, **kwargs):
        raise NotImplementedError("update() must be implemented")

    def _do_step(self, kwargs, sloop_state):
        (solved, loop_state) = sloop_state
        solved, new_state, new_params, solver_state = self.update(loop_state.fun_state, loop_state.fun_params,
                                                          loop_state.solver_state, **kwargs)
        return solved, LoopState(new_state, new_params, solver_state, loop_state.i + 1)

    def _loop(self, kwargs, init_params, init_state):
        do_step = partial(self._do_step, kwargs)
        loop_state = LoopState(fun_state=init_state, fun_params=init_params,
                               solver_state=None, i=0)
        # do the first step
        sloop_state = do_step((False, loop_state))
        return jax.lax.while_loop(
                lambda s: jnp.logical_and(jnp.logical_not(s[0]),
                                          s[1].i < self.max_iterations),
                do_step, sloop_state)

    def run(self, *, init_params=None, init_state=None, **kwargs):
        loop = partial(self._loop, kwargs)
        solved, final_state = loop(init_params, init_state)
        return SolverResults(
            solved=solved,
            state=final_state.fun_state,
            params=final_state.fun_params,
            solver_state=final_state.solver_state,
            history=None
        )

def sanitize_cost(cost, has_aux):
    def _cost(state, params):
        new_state, res = (
                (state, cost(params))
                    if state is None else 
                cost(state, params)
            )
        cost_val, aux = res if has_aux else (res, None)
        cost_val = jnp.nan_to_num(cost_val, nan=float('inf'))
        return new_state, cost_val, aux
    return _cost