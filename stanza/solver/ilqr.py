from typing import Any

from stanza.dataclasses import dataclass
from stanza.solver import (
    Solver, SolverResult, UnsupportedObectiveError
)
from stanza.solver.util import implicit_diff_solve
from stanza.util.trajax.optimizer import ilqr

import stanza
import functools

import jax
import jax.numpy as jnp



def linearize(step_fn):
    def grad_fn(state, action, rng):
        state_flat, state_uf = jax.flatten_util.ravel_pytree(state)
        action_flat, action_uf = jax.flatten_util.ravel_pytree(action)
        def flat_step(state, action, rng):
            state = state_uf(state)
            action = action_uf(action)
            state = step_fn(state, action, rng)
            state, _ = jax.flatten_util.ravel_pytree(state)
            return state
        A, B = jax.jacobian(flat_step, argnums=(0, 1))(state_flat, action_flat, rng)
        return A, B
    return grad_fn

def multiply_gain(K, x, u):
    x_flat, _ = jax.tree_util.tree_flatten(x)
    bias_flat = K @ x_flat
    u_flat, u_uf = jax.flatten_util.ravel_pytree(u)
    return u_uf(u_flat + bias_flat)

def tvlqr(As, Bs, Qs, Rs):
    if Qs.ndim == 2:
        Qs = jnp.repeat(Qs[None, :, :], As.shape[0] + 1, axis=0)
    if Rs.ndim == 2:
        Rs = jnp.repeat(Rs[None, :, :], As.shape[0], axis=0)

    def lqr_step(P, input):
        A, B, Q, R = input
        M = A.T @ P @ B
        F = jnp.linalg.inv(R + B.T @ P @ B) @ M.T
        P_prev = A @ P @ A.T - M @ F + Q
        return P_prev, -F
    _, gains = jax.lax.scan(lqr_step, Qs[-1],
        (As, Bs, Qs[:-1], Rs),
        reverse=True
    )
    return gains


@dataclass(jax=True)
class iLQRSolver(Solver):
    @staticmethod
    @functools.partial(jax.jit, static_argnums=(0,))
    def _solve(history, objective, solver_state):
        from stanza.policies.mpc import MinimizeMPC, MinimizeMPCState
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

        def flat_cost(states, actions):
            actions = actions[:-1]
            states = jax.vmap(state_uf)(states)
            actions = jax.vmap(action_uf)(actions)
            return objective.cost_fn(states, actions)

        initial_actions_flat = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0]
        )(solver_state.actions)
        _, actions_flat, cost, _, _, _, it = ilqr(flat_cost, flat_model,
                state0_flat, initial_actions_flat)
        actions = jax.vmap(action_uf)(actions_flat)
        res = MinimizeMPCState(it, True, actions, cost)
        return SolverResult(res, None)

    @functools.partial(jax.jit, static_argnames=("implicit_diff", "history"))
    def run(self, objective, *, implicit_diff=True, history=False) -> SolverResult:
        from stanza.policies.mpc import MinimizeMPCState
        init_state = MinimizeMPCState(0, False, 
                objective.initial_actions, 0.)
        solve = stanza.partial(self._solve, history)
        if implicit_diff:
            solve = implicit_diff_solve(solve)
        return solve(objective, init_state)