import jax
import stanza
import jax.flatten_util
import jax.numpy as jnp

from typing import Callable, Any
from functools import partial
from stanza.util.dataclasses import dataclass, field, replace

import jax.experimental.host_callback

class Objective:
    pass

# Minimize the passed-in function
@dataclass(jax=True, kw_only=True)
class Minimize(Objective):
    fun: Callable
    has_state: bool = field(default=False, jax_static=True)
    has_aux: bool = field(default=False, jax_static=True)
    initial_state: Any = None # Note that has_state needs to be true in order for this
                           # to be passed into the function!
    initial_params: Any = None

    # Tuple of parameter constraints
    constraints: tuple = ()

    # Always of the form (state, params) --> (new_state, cost, aux),
    # and handles the has_state, has_aux cases
    def eval(self, state, params):
        r = self.fun(state, params) if self.has_state else self.fun(params)
        state, r = (r[0], r[1:]) if self.has_state else (None, r)
        cost, aux = (r[0], r[1]) if self.has_aux else (r, None)
        return state, cost, aux

# Fun <= 0
@dataclass(jax=True)
class IneqConstraint:
    fun: Callable

# Fun == 0
@dataclass(jax=True)
class EqConstraint:
    fun: Callable

class UnsupportedObectiveError(RuntimeError):
    pass

class Solver:
    # Can raise an UnsupportedObjectiveError
    # if the objective is not compatible with this solver
    def run(self, objective, **kwargs) -> Any:
        raise NotImplementedError("Solver must implement run()")

# All solver states must have iteration and solved
# parameters
@dataclass(jax=True)
class SolverState:
    iteration: int
    solved: bool

# A solver state for Minimize() objectives
@dataclass(jax=True)
class MinimizeState(SolverState):
    # The function state
    state: Any
    params: Any
    aux: Any # auxiliary output of the objective

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

    def _optimality(self, objective, kwargs, unflatten, flat_state):
        state = unflatten(flat_state)
        o = self.optimality(objective, state, **kwargs)
        of, _ = jax.flatten_util.ravel_pytree(o)
        extra_d = flat_state.shape[0] - of.shape[0]
        combined = jnp.concatenate((of, jnp.zeros((extra_d,))))
        return combined
    
    def _custom_optimality(self, objective, kwargs, state, params):
        if hasattr(state, 'params'):
            state = replace(state, params=params)
        elif hasattr(state, 'actions'):
            state = replace(state, actions=params)
        grad = self.optimality(objective, state, **kwargs)
        return grad

    def _tangent_solve(self, sol_lin, y):
        jac = jax.jacobian(sol_lin)(y)
        s = jnp.linalg.solve(jac, y)
        return s

    def run(self, objective, *,
            history=False,
            implicit_diff=True, **kwargs):
        state = self._do_step(kwargs, objective, None)
        solve = self._solve_scan if history else self._solve_loop
        final_state, state_history = solve(objective, kwargs, state)

        if implicit_diff:
            # Prevent gradient backprop through final_state so we don't unroll
            # the loop.
            final_state = jax.lax.stop_gradient(final_state)

            optimality = partial(self._custom_optimality, objective, kwargs, final_state)
            if hasattr(final_state, 'params'):
                params = jax.lax.custom_root(optimality, final_state.params,
                                        lambda _, x: x, self._tangent_solve)
                final_state = replace(final_state, params=params)
            elif hasattr(final_state, 'actions'):
                actions = jax.lax.custom_root(optimality, final_state.params,
                                        lambda _, x: x, self._tangent_solve)
                final_state = replace(final_state, actions=actions)
        return (final_state, state_history) if history else final_state