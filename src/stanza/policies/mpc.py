import jax
import jax.numpy as jnp

import stanza.policies

from typing import Any, Callable

from stanza import Partial

from stanza.struct import dataclass, field, replace
from jax.random import PRNGKey

from stanza.solver import (
    Solver, SolverState, Minimize, 
    UnsupportedObectiveError, Objective, EqConstraint
)
from stanza.solver.newton import NewtonSolver
from stanza.solver.ilqr import iLQRSolver
from stanza.policies import Actions, PolicyOutput

import jax.experimental.host_callback

# A special MinimizeMPC objective
# which solvers can take
@dataclass(jax=True, kw_only=True)
class MinimizeMPC(Objective):
    # The actions. 
    initial_actions: Any
    state0: Any

    cost_fn: Callable
    model_fn: Callable

    def eval(self, actions):
        r = stanza.policies.rollout(
            self.model_fn, self.state0, policy=Actions(actions)
        )
        return self.cost_fn(r.observations, r.actions)
    
    def optimality(self, solver_state):
        return jax.grad(self.eval, argnums=0)(
            solver_state.actions
        )

    def extract_params(self, solver_state):
        return solver_state.actions
    
    def replace_params(self, solver_state, actions):
        return replace(solver_state, actions=actions)

    def as_minimize(self):
        return Minimize(
            fun=self.eval,
            initial_params=self.initial_actions,
        )

@dataclass(jax=True)
class MinimizeMPCState(SolverState):
    actions: Any
    cost: float

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
    horizon_length : int = field(static=True)

    solver : Solver = iLQRSolver()

    initializer : Callable[[Any,Any], Any] = None 

    # If the cost horizon should receed or stay static
    # if receed=False, you can only rollout horizon_length
    # length trajectories with this MPC
    receed : bool = field(default=True, static=True)
    history : bool = field(default=False, static=True)
    replan : bool = field(default=True, static=True)
    
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

def log_barrier(barrier_sdf, states, actions):
    sdf = barrier_sdf(states, actions)
    return -jnp.sum(jnp.log(-sdf))

def centered_barrier(barrier_sdf, center_state, center_action,
                     states, actions):
    s_flat, s_unflat = jax.flatten_util.ravel_pytree(states)
    a_flat, a_unflat = jax.flatten_util.ravel_pytree(actions)
    s_center = jnp.zeros_like(s_flat)
    a_center = jnp.zeros_like(a_flat)

    b = lambda s, a: log_barrier(barrier_sdf, s_unflat(s), a_unflat(a))
    v = b(s_center, a_center)
    center_s_grad, center_a_grad = jax.grad(b, argnums=(0,1))(s_center, a_center)
    return log_barrier(barrier_sdf, states, actions) + \
            jnp.dot(center_s_grad, s_flat - s_center) + \
            jnp.dot(center_a_grad, a_flat - a_center) - v

# A barrier-based MPC
@dataclass(jax=True)
class BarrierMPC(MPC):
    # Takes states, actions as inputs
    # outputs
    barrier_sdf: Callable = None

    # State at which to center the barrier around
    # if None, uses uncentered
    center_state: Any = None
    center_action: Any = None

    eta: float = 0.01

    feasibility_solver : Solver = NewtonSolver()

    # The BarrierMPC loss function has s, params
    def _loss_fn(self, state0, rollout_state, actions):
        states = stanza.policies.rollout(
            self.model_fn, state0, policy=Actions(actions)
        ).states
        cost = self.cost_fn(states, actions)
        if not self.barrier_sdf:
            return cost
        if self.center_state is not None and self.center_action is not None:
            b = centered_barrier(self.barrier_sdf,
                self.center_state, self.center_action, states, actions)
        else:
            b = log_barrier(self.barrier_sdf, states, actions)
        total_cost = cost + self.eta*b
        total_cost = jnp.nan_to_num(total_cost, nan=jnp.inf)
        return rollout_state, total_cost
    
    # The log-based loss
    # for finding a feasible point

    def _feas_loss(self, state0, params):
        s, actions = params
        states = stanza.policies.rollout(
            self.model_fn, state0, policy=Actions(actions)
        ).states
        # get all of the constraints
        constr = self.barrier_sdf(states, actions)
        # constr < s should always hold true
        val = jnp.mean(-jnp.log(s - constr))
        val = jnp.nan_to_num(val, nan=jnp.inf)
        return val
    
    # early termination handler
    def _feas_terminate(self, state0, solver_state):
        _, actions = solver_state.params
        states = stanza.policies.rollout(
            self.model_fn, state0, policy=Actions(actions)
        ).states
        constr = self.barrier_sdf(states, actions)
        # if all of the constraints are satisfied, early terminate
        sat = jnp.all(constr < -5e-3)
        # jax.debug.print("term_s: {}", states)
        # jax.debug.print("term: {} {}", sat, constr)
        return sat
    
    def _solve_feasible(self, state0, init_actions):
        init_states = stanza.policies.rollout(
            self.model_fn, state0, policy=Actions(init_actions)
        ).states
        constr = self.barrier_sdf(init_states, init_actions)
        # make the feasibility loss always feasible
        s = jnp.max(constr) + 10
        res = self.feasibility_solver.run(Minimize(
            fun=Partial(self._feas_loss, state0),
            initial_params=(s, init_actions),
            constraints=(EqConstraint(lambda p: p[0]),),
            early_terminate=Partial(self._feas_terminate, state0)
        ), implicit_diff=False)
        s, actions = res.params
        return actions
    
    def _solve(self, rollout_state, state0, init_actions):
        # phase I:
        if self.barrier_sdf:
            # jax.debug.print("---------- PHASE I-----------")
            init_actions = self._solve_feasible(state0, init_actions)
        
        # states = stanza.policies.rollout(
        #     self.model_fn, state0, policy=Actions(init_actions)
        # ).states
        # constr = self.barrier_sdf(states, init_actions)
        # phase II:
        # now that we have guaranteed feasibility, solve
        # the full loss
        # jax.debug.print("---------- PHASE II-----------")
        return super()._solve(rollout_state, state0, init_actions)