import jax
import jax.numpy as jnp

import stanza.policy

from typing import Any, Callable

from stanza import Partial

from stanza.util.logging import logger
from stanza.util.dataclasses import dataclass, field
from jax.random import PRNGKey

from stanza.solver import Solver, Minimize, UnsupportedObectiveError, Objective, EqConstraint
from stanza.solver.newton import NewtonSolver
from stanza.policy import Actions, PolicyOutput

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

    # rollout_fn or model_fn accepts rng as the first parameter
    stochastic_dynamics : bool = field(default=False, jax_static=True)

# A vanilla MPC controller
@dataclass(jax=True, kw_only=True)
class MPC:
    action_sample: Any
    cost_fn : Any

    # either model_fn (state, u) --> state 
    # or rollout_fn (state0, us) --> states
    # must be specified.
    model_fn : Callable = None
    rollout_fn : Callable = None

    stochastic_dynamics : bool = field(default=False, jax_static=True)

    # Horizon is part of the static jax type
    horizon_length : int = field(default=20, jax_static=True)

    # Solver must be a dataclass with either (1) a "fun" argument
    # or (2) a dynamics_fn, cost_fn argument
    solver : Solver = NewtonSolver()

    # If the cost horizon should receed or stay static
    # if receed=False, you can only rollout horizon_length
    # length trajectories with this MPC
    receed : bool = field(default=True, jax_static=True)

    # If the controller should replan at every iteration
    # you can have receed=False, replan=True
    # or receed=False, replan=False (plan once, blindly executed)
    # but not receed=True, replan=False
    replan : bool = field(default=True, jax_static=True)

    def _loss_fn(self, state0, actions):
        if self.rollout_fn:
            r = self.rollout_fn(state0, actions)
        else:
            r = stanza.policy.rollout(
                    self.model_fn, state0, policy=Actions(actions)
                )
        return self.cost_fn(r.states, r.actions)
    
    def _solve(self, state0, init_actions):
        # Try to provide a MinimizeMPC
        # problem to the
        try:
            res = self.solver.run(MinimizeMPC(
                initial_actions=init_actions,
                state0=state0,
                cost_fn=self.cost_fn,
                model_fn=self.model_fn,
                rollout_fn=self.rollout_fn,
                stochastic_dynamics=self.stochastic_dynamics
            ))
            return res.actions
        except UnsupportedObectiveError:
            pass
        res = self.solver.run(Minimize(
            fun=Partial(self._loss_fn, state0),
            initial_params=init_actions
        ))
        return res.params

    def __call__(self, state, policy_state=None):
        if policy_state is None:
            actions = jax.tree_util.tree_map(lambda x: jnp.zeros((self.horizon_length,) + x.shape), self.action_sample)
        else:
            actions = jax.tree_util.tree_map(lambda x: x.at[:-1].set(x[1:]), policy_state)
        actions = self._solve(state, actions)
        action0 = jax.tree_util.tree_map(lambda x: x[0], actions)
        return PolicyOutput(action0, policy_state=actions)

# A barrier-based MPC
@dataclass(jax=True)
class BarrierMPC(MPC):
    # Takes states, actions as inputs
    # outputs
    barrier_sdf: Callable = None
    eta: float = 0.001

    feasibility_solver : Solver = NewtonSolver()

    # The BarrierMPC loss function has s, params
    def _loss_fn(self, state0, actions):
        states = stanza.policy.rollout(
            self.model_fn, state0, policy=Actions(actions)
        ).states
        cost = self.cost_fn(states, actions)
        if not self.barrier_sdf:
            return cost
        b = self.barrier_sdf(states, actions)
        cost + self.eta*jnp.sum(b)
    
    # The log-based loss
    # for finding a feasible point

    def _feas_loss(self, state0, params):
        s, actions = params
        states = stanza.policy.rollout(
            self.model_fn, state0, policy=Actions(actions)
        ).states
        # get all of the constraints
        constr = self.barrier_sdf(states, actions)
        # constr < s should always hold true
        val = jnp.sum(-jnp.log(s - constr))
        val = jnp.nan_to_num(val, nan=jnp.inf)
        return jnp.sum(val)
    
    def _solve_feasible(self, state0, init_actions):
        init_states = stanza.policy.rollout(
            self.model_fn, state0, policy=Actions(init_actions)
        ).states
        constr = self.barrier_sdf(init_states, init_actions)
        # make the loss feasible
        s = 2*jnp.max(constr) + 0.0001
        res = self.feasibility_solver.run(Minimize(
            fun=Partial(self._feas_loss, state0),
            initial_params=(s, init_actions),
            constraints=(EqConstraint(lambda p: p[0]),)
        ))
        s, actions = res.params
        jax.debug.print("final s: {}", s)
        return actions


    def _solve(self, state0, init_actions):
        # phase I:
        if self.barrier_sdf:
            init_actions = self._solve_feasible(state0, init_actions)
        # now that we have guaranteed feasibility, solve
        # the full loss
        return super()._solve(state0, init_actions)