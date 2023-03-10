from typing import Any
from jax.random import PRNGKey

from stanza.policies import Policy, PolicyOutput
from stanza.util.dataclasses import dataclass, field
from stanza.solver import IterativeSolver, SolverState, UnsupportedObectiveError
from stanza.policies.mpc import MinimizeMPC

import warnings
import jax
import jax.numpy as jnp


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from trajax.tvlqr import tvlqr
    from trajax.optimizers import quadratize, linearize

class iLQRState(SolverState):
    actions: Any
    rng_key: PRNGKey

@dataclass(jax=True)
class FeedbackPolicy(Policy):
    policy: Policy
    K: jnp.array
    k: jnp.array

    def __call__(self, state, policy_state=None, **kwargs):
        sps, i = (None, 0) if policy_state is None else policy_state
        output = self.policy(state, sps)

        K_t = jax.tree_util.tree_map(lambda x: x[i], self.K)
        k_t = jax.tree_util.tree_map(lambda x: x[i], self.k)
        action = output.action + K_t @ state + k_t
        new_policy_state = (output.policy_state, i + 1)
        return PolicyOutput(action, new_policy_state, output.aux)

@dataclass(jax=True)
class iLQRSolver(IterativeSolver):
    def _line_search(self, objective):
        pass

    def init_state(self, objective):
        return SolverState(
            iteration=0,
            solved=False,
            obj_state=None,
            obj_params=objective.init_params,
            obj_aux=None
        )

    def update(self, objective, state):
        if not isinstance(objective, MinimizeMPC):
            raise UnsupportedObectiveError("iLQR only supports MinimizeMPC objectives")
        if state is None:
            state = self.init_state(objective)
        return SolverState(
            iteration=state.iteration+1,
            solved=False,
            obj_state=state.obj_state,
            obj_params=state.obj_params,
            obj_aux=None
        )