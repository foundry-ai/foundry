import jax
import jax.numpy as jnp

import stanza.policies

from typing import Any, Callable

from stanza import Partial

from stanza.util.logging import logger
from stanza.util.attrdict import Attrs
from stanza.util.dataclasses import dataclass, field, replace
from jax.random import PRNGKey

from stanza.solver import Solver, Minimize, UnsupportedObectiveError, Objective, EqConstraint
from stanza.solver.newton import NewtonSolver
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
    # Either model_fn or rollout_fn must be specified
    model_fn: Callable = None
    rollout_fn: Callable = None
    # rollout_fn has a state
    rollout_has_state: bool = field(default=False, jax_static=True)

# A vanilla MPC controller
@dataclass(jax=True, kw_only=True)
class MPC:
    action_sample: Any
    cost_fn : Any

    # either model_fn (state, u) --> state 
    # or rollout_fn (state0, inputs) --> states
    # must be specified.
    model_fn : Callable = None
    rollout_fn : Callable = None

    # if rollout_fn takes a state as the first parameter
    rollout_has_state: bool = field(default=False, jax_static=True)

    # Horizon is part of the static jax type
    horizon_length : int = field(jax_static=True)

    # Solver must be a dataclass with either (1) a "fun" argument
    # or (2) a dynamics_fn, cost_fn argument
    solver : Solver = NewtonSolver()

    # If the cost horizon should receed or stay static
    # if receed=False, you can only rollout horizon_length
    # length trajectories with this MPC
    receed : bool = field(default=True, jax_static=True)

    replan : bool = field(default=True, jax_static=True)

    # the offset, base_states, base_actions are for
    # when receed=False but replan=True
    def _loss_fn(self, state0, rollout_state, actions):
        if self.rollout_fn:
            if self.rollout_has_state:
                rollout_state, r = self.rollout_fn(rollout_state, state0, actions)
            else:
                r = self.rollout_fn(state0, actions)
        else:
            r = stanza.policies.rollout(
                    self.model_fn, state0, policy=Actions(actions)
                )
        return rollout_state, self.cost_fn(r.states, r.actions)
    
    # if the rollout_fn has an update_actions (roll_state, state0, actions) --> actions
    # to be called after each iteration, it will be called here
    def _post_step_cb(self, state0, solver_state):
        actions = solver_state.params
        roll_state = solver_state.state
        if hasattr(self.rollout_fn, 'update_actions'):
            actions = self.rollout_fn.update_actions(state0, actions)
            if self.rollout_has_state:
                roll_state, actions = self.rollout_fn.update_actions(roll_state, state0, actions)
            else:
                actions = self.rollout_fn.update_actions(state0, actions)
        return replace(solver_state, params=actions, state=roll_state)
    
    def _solve(self, rollout_state, state0, init_actions):
        # Try to provide a MinimizeMPC
        # problem to the solver first
        try:
            res = self.solver.run(MinimizeMPC(
                initial_actions=init_actions,
                state0=state0,
                cost_fn=self.cost_fn,
                model_fn=self.model_fn,
                rollout_fn=self.rollout_fn,
                rollout_has_state=self.rollout_has_state
            ))
            return res.actions
        except UnsupportedObectiveError:
            pass
        res = self.solver.run(Minimize(
            fun=Partial(self._loss_fn, state0),
            has_state=True,
            initial_params=init_actions,
            initial_state=rollout_state,
            post_step_callback=Partial(self._post_step_cb, state0)
        ))
        _, cost = self._loss_fn(state0, res.state, res.params)
        return res.state, res.params, cost

    def __call__(self, state, policy_state=None):
        if policy_state is None:
            actions = jax.tree_util.tree_map(lambda x: jnp.zeros((self.horizon_length,) + x.shape), self.action_sample)
            rollout_state = None
            t = 0
            rollout_state, actions, cost = self._solve(rollout_state, state, actions)
        else:
            actions = policy_state[0]
            rollout_state = policy_state[1]
            t = policy_state[2]
            cost = policy_state[3]

        if self.receed and policy_state is not None:
            actions = jax.tree_util.tree_map(lambda x: x.at[:-1].set(x[1:]), actions)
            rollout_state, actions, cost = self._solve(rollout_state, state, actions)
            action = jax.tree_util.tree_map(lambda x: x[0], actions)
        else:
            action = jax.tree_util.tree_map(lambda x: x[t], actions)
        return PolicyOutput(action, policy_state=(actions, rollout_state, t+1, cost),
                            extra=Attrs(cost=cost))

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