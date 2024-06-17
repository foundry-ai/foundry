import jax
import jax.numpy as jnp

import stanza.policy

from typing import Any, Callable

from stanza import partial

from stanza.struct import dataclass, field, replace
from jax.random import PRNGKey

from stanza.solver import (
    Solver, SolverState, Minimize, 
    UnsupportedObectiveError, Objective, EqConstraint
)
from stanza.solver.newton import NewtonSolver
from stanza.solver.ilqr import iLQRSolver
from stanza.policy import Actions, PolicyOutput

import jax.experimental.host_callback

# A special MinimizeMPC objective
# which solvers can take
@dataclass
class MinimizeMPCState(SolverState):
    params: Any
    actions: Any
    cost: float
    cost_state: Any = None

@dataclass(kw_only=True)
class MinimizeMPC(Objective):
    # The actions. 
    initial_actions: Any
    state0: Any

    cost_fn: Callable
    model_fn: Callable

    initial_params: Any = None
    initial_cost_state : Any = None
    has_params: bool = field(default=False, pytree_node=False)
    has_state: bool = field(default=False, pytree_node=False)
    has_aux: bool = field(default=False, pytree_node=False)

    def cost(self, cost_state, params, states, actions):
        args = tuple()
        if self.has_state: args += (cost_state,)
        if self.has_params: args += (params,)
        args += (states,actions)
        r = self.cost_fn(*args)
        r = (r,) if not (self.has_state or self.has_aux) else r
        state, r = (r[0], r[1:]) if self.has_state else (None, r)
        cost, aux = (r[0], r[1]) if self.has_aux else (r[0], None)
        return state, cost, aux

    def eval(self, cost_state, params, actions):
        r = stanza.policy.rollout(
            self.model_fn, self.state0, policy=Actions(actions)
        )
        return self.cost(cost_state, params, r.observations, r.actions)
    
    def optimality(self, solver_state : MinimizeMPCState):
        return jax.grad(lambda s, p, a: self.eval(s, p, a)[1], argnums=(1,2))(
            solver_state.cost_state, solver_state.params, solver_state.actions
        )

    def extract_params(self, solver_state):
        return solver_state.params, solver_state.actions
    
    def replace_params(self, solver_state, params):
        params, actions = params
        return replace(solver_state, actions=actions, params=params)

    def as_minimize(self):
        return Minimize(
            fun=self.eval,
            initial_params=self.initial_actions,
        )

@dataclass(kw_only=True)
class MPCState:
    actions: Any
    actions_t: int # the slice of the actions

# A vanilla MPC controller
@dataclass(kw_only=True)
class MPC:
    action_sample: Any
    cost_fn : Any
    model_fn : Callable = None
    horizon_length : int = field(pytree_node=False)

    solver : Solver = iLQRSolver()

    initializer : Callable[[Any,Any], Any] = None 

    # If the cost horizon should receed or stay static
    # if receed=False, you can only rollout horizon_length
    # length trajectories with this MPC
    receed : bool = field(default=True, pytree_node=False)
    history : bool = field(default=False, pytree_node=False)
    # todo: implement replanning without receeding
    replan : bool = field(default=True, pytree_node=False)
    
    def _solve(self, state0, init_actions):
        # Try to provide a MinimizeMPC
        # problem to the solver first
        if self.initializer:
            init_actions = self.initializer(state0, init_actions)
        objective = MinimizeMPC(
            initial_actions=init_actions,
            state0=state0,
            cost_fn=self.cost_fn,
            model_fn=self.model_fn
        )
        try:
            res = self.solver.run(objective, history=self.history)
            return res.solution.actions, res.solution.cost, res.history
        except UnsupportedObectiveError:
            pass
        objective = objective.as_minimize()
        res = self.solver.run(objective, history=self.history)
        return res.solution.params, res.solution.cost, res.history

    def __call__(self, input):
        if input.policy_state is None:
            actions = jax.tree_util.tree_map(lambda x: jnp.zeros((self.horizon_length - 1,) + x.shape), self.action_sample)
            actions, _, _ = self._solve(input.observation, actions)
            policy_state = MPCState(
                actions=actions, actions_t=0
            )
        else:
            policy_state = input.policy_state
            if self.receed:
                # shift actions and re-solve
                actions = jax.tree_util.tree_map(lambda x: x.at[:-1].set(x[1:]), policy_state.actions)
                actions, _, _ = self._solve(input.observation, actions)
                policy_state = replace(
                    policy_state,
                    actions=actions,
                    actions_t=0
                )
        action = jax.tree_util.tree_map(
            lambda x: x[policy_state.actions_t],
            policy_state.actions
        )
        policy_state = replace(
            policy_state,
            actions_t=policy_state.actions_t + 1
        )
        return PolicyOutput(
            action=action,
            policy_state=policy_state
        )