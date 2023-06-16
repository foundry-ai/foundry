from typing import Any
from jax.random import PRNGKey

from stanza.policies import Policy, PolicyOutput
from stanza.dataclasses import dataclass, field
from stanza.solver import IterativeSolver, SolverState, UnsupportedObectiveError
from stanza.policies.mpc import MinimizeMPC

import warnings
import jax
import jax.numpy as jnp

from stanza.solver import Solver
from stanza.policies.mpc import MPC

from stanza.util.trajax.optimizer import ilqr


@dataclass(jax=True)
class iLQRResult:
    actions: Any
    cost: float
    iteration: int

@dataclass(jax=True)
class iLQRSolver(Solver):
    def run(self, objective, **kwargs) -> Any:
        if not isinstance(objective, MinimizeMPC):
            raise UnsupportedObectiveError("iLQR only supports MinimizeMPC objectives")

        state0_flat, state_uf = jax.flatten_util.ravel_pytree(objective.state0)
        a0 = jax.tree_map(lambda x: x[0], objective.initial_actions)
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
        )(objective.initial_actions)
        _, actions_flat, cost, _, _, _, it = ilqr(flat_cost, flat_model,
                state0_flat, initial_actions_flat)
        actions = jax.vmap(action_uf)(actions_flat)
        return iLQRResult(actions, cost, it)