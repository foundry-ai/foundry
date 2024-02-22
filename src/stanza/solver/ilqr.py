from typing import Any

from stanza.struct import dataclass
from stanza.solver import (
    Solver, SolverResult, UnsupportedObectiveError
)
from stanza.solver.util import implicit_diff_solve
from stanza.solver.trajax.optimizer import ilqr

import stanza
import functools

import jax
import jax.numpy as jnp

@dataclass
class iLQRSolver(Solver):
    @staticmethod
    @functools.partial(stanza.jit, static_argnums=(0,))
    def _solve(history, objective, solver_state):
        from stanza.policy.mpc import MinimizeMPC, MinimizeMPCState
        if not isinstance(objective, MinimizeMPC):
            raise UnsupportedObectiveError("iLQR only supports MinimizeMPC objectives")

        state0_flat, state_uf = jax.flatten_util.ravel_pytree(objective.state0)
        a0 = jax.tree_map(lambda x: x[0], solver_state.actions)
        _, action_uf = jax.flatten_util.ravel_pytree(a0)

        # flatten everything
        def flat_model(state, action):
            state = state_uf(state)
            action = action_uf(action)
            state = objective.model_fn(state, action, None)
            state, _ = jax.flatten_util.ravel_pytree(state)
            return state

        def flat_cost(states, actions, cost_state):
            actions = actions[:-1]
            states = jax.vmap(state_uf)(states)
            actions = jax.vmap(action_uf)(actions)
            s, c, _ = objective.cost(cost_state, states, actions)
            return c, s

        initial_actions_flat = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0]
        )(solver_state.actions)
        _, actions_flat, cost_state, cost, _, _, _, it = ilqr(flat_cost, flat_model,
                objective.initial_cost_state,
                state0_flat, initial_actions_flat, make_psd=True, psd_delta=0.0,
                grad_norm_threshold=1e-5)
        actions = jax.vmap(action_uf)(actions_flat)
        res = MinimizeMPCState(it, True, actions, cost, cost_state)
        return SolverResult(res, None)

    @functools.partial(stanza.jit, static_argnames=("implicit_diff", "history"))
    def run(self, objective, *, implicit_diff=True, history=False) -> SolverResult:
        from stanza.policy.mpc import MinimizeMPCState
        init_state = MinimizeMPCState(0, False, 
                objective.initial_actions, 0., objective.initial_cost_state)
        solve = stanza.partial(self._solve, history)
        if implicit_diff:
            solve = implicit_diff_solve(solve)
        return solve(objective, init_state)