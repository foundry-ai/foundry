from stanza.solver import Solver, Objective, SolverState, SolverResult
from stanza.dataclasses import dataclass, field, replace
from stanza.solver.util import implicit_diff_solve

from functools import partial

import stanza.util

import jax
import jax.numpy as jnp

@dataclass(kw_only=True)
class IterativeSolver(Solver):
    # This is static so we can output the optimization history
    # if requested
    max_iterations: int = field(default=1000, pytree_node=False)
    
    def init(self, objective, **kwargs):
        raise NotImplementedError("init() must be implemented")

    # step = None is passed in for the first step
    def update(self, objective, state, **kwargs):
        raise NotImplementedError("update() must be implemented")

    # -------------------- Class Internals --------------------

    def _do_step(self, kwargs, objective : Objective,
                               solver_state : SolverState):
        with jax.checking_leaks():
            state = self.update(objective, solver_state, **kwargs)
        # if the objective has an early_terminate condition
        if objective.terminate_condition is not None:
            solved = jnp.logical_or(
                state.solved,
                objective.terminate_condition(state)
            )
            state = replace(state, solved=solved)
        if objective.step_callback is not None:
            state = objective.step_callback(state)
        #stanza.util.assert_trees_all_equal_shapes_and_dtypes(solver_state, state)
        return state
    
    def _scan_fn(self, kwargs, objective, loop_state, _):
        state = jax.lax.cond(loop_state.solved,
            lambda _0, _1, s: s,
            self._do_step, kwargs, objective, loop_state)
        return state, loop_state
    
    def _solve_scan(self, kwargs, objective, state):
        scan_fn = partial(self._scan_fn, kwargs, objective)
        return SolverResult(jax.lax.scan(scan_fn, state, None,
                    length=self.max_iterations))

    def _solve_loop(self, kwargs, objective, state):
        step_fn = partial(self._do_step, kwargs, objective)
        return SolverResult(jax.lax.while_loop(
            lambda s: jnp.logical_and(jnp.logical_not(s.solved),
                            s.iteration < self.max_iterations),
            step_fn, state), None)

    def run(self, objective, *, implicit_diff=True,
            history=False, **kwargs) -> SolverResult:
        with jax.checking_leaks():
            state = self.init(objective, **kwargs)
        solve = self._solve_scan if history else self._solve_loop
        solve = partial(solve, kwargs)
        if implicit_diff:
            solve = implicit_diff_solve(solve)
        return solve(objective, state)