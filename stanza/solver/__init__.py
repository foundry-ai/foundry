import jax
import stanza
import jax.flatten_util
import jax.numpy as jnp

from typing import Callable, Any
from functools import partial
from stanza.util.dataclasses import dataclass, field, replace

@dataclass(jax=True)
class SolverResults:
    solved: bool
    state: Any
    params: Any
    solver_state: Any
    history: Any = None

class Objective:
    pass

# Minimize the passed-in function
@dataclass(jax=True, kw_only=True)
class Minimize(Objective):
    fun: Callable
    has_state: bool = field(default=False, jax_static=True)
    has_aux: bool = field(default=False, jax_static=True)
    init_state: Any = None # Note that has_state needs to be true in order for this
                           # to be passed into the function!
    init_params: Any = None

    # Tuple of parameter constraints
    constraints: tuple = ()

    # Always of the form (state, params) --> (new_state, cost, aux),
    # and handles the has_state, has_aux cases
    def eval(self, state, params):
        r = self.fun(state, params) if self.has_state else self.fun(params)
        state, r = (r[0], r[1:]) if self.has_state else (None, r)
        cost, aux = (r[0], r[1]) if self.has_aux else (r, None)
        return state, cost, aux

class UnsupportedObectiveError(RuntimeError):
    pass

class Solver:
    # Can raise an UnsupportedObjectiveError
    # if the objective is not compatible with this solver
    def run(self, objective, **kwargs) -> SolverResults:
        raise NotImplementedError("Solver must implement run()")

@dataclass(jax=True)
class SolverState:
    iteration: int
    solved: bool
    obj_state: Any
    obj_params: Any
    obj_aux: Any # auxiliary output of the objective

@dataclass(jax=True, kw_only=True)
class IterativeSolver(Solver):
    # This is static so we can output the optimization history
    # if requested
    max_iterations: int = field(default=1000, jax_static=True)

    # A custom terminate function,
    # which will prematurely exit when a given
    # condition is met. Solvers should *not* populate this
    # by default and allow users to override.
    terminate: Callable = None

    # step = None is passed in for the first step
    def update(self, objective, state, **kwargs):
        raise NotImplementedError("update() must be implemented")

    # implicit differentiation happens by
    # setting optimality(args) = 0
    # by default uses update(args) - args = 0
    def optimality(self, objective, state, **kwargs):
        raise NotImplementedError("optimality() must be implemented for implicit diff")

    # -------------------- Class Internals --------------------

    @jax.jit
    def _do_step(self, kwargs, objective, loop_state):
        state = self.update(objective, loop_state, **kwargs)
        if self.terminate is not None:
            solved = self.terminate(objective, state)
            state = replace(state, solved=solved)
        return state
    
    def _scan_fn(self, kwargs, objective, loop_state, _):
        state = jax.lax.cond(loop_state.solved,
            lambda _0, _1, s: s, kwargs, objective, loop_state,
            self._do_step)
        return state, loop_state
    
    def _solve_scan(self, objective, kwargs, state):
        scan_fn = partial(self._scan_fn, kwargs, objective)
        return jax.lax.scan(scan_fn, state, None,
                    length=self.max_iterations)

    def _solve_loop(self, objective, kwargs, state):
        step_fn = partial(self._do_step, kwargs, objective)
        return jax.lax.while_loop(
            lambda s: jnp.logical_and(jnp.logical_not(s.solved),
                            s.iteration < self.max_iterations),
            step_fn, state), None

    def _optimality(self, objective, kwargs, state):
        return self.optimality(objective, state, **kwargs)

    @partial(jax.jit,
        static_argnames=['history', 'unroll', 'implicit_diff'])
    def run(self, objective, *,
            history=False,
            implicit_diff=True, **kwargs):
        state = self._do_step(kwargs, objective, None)
        optimality = partial(self._optimality, objective, kwargs)
        solve = self._solve_scan if history else self._solve_loop
        solve = partial(solve, objective, kwargs)

        solve = jax.lax.custom_root(optimality, state, solve, has_aux=True)

        final_state, history = solve(objective, state, kwargs)
        return SolverResults(
            solved=final_state.solved,
            state=final_state.obj_state,
            params=final_state.obj_params,
            solver_state=final_state,
            history=history
        )