import jax
import jax.numpy as jnp

import stanza.policy

from typing import Any, Callable

from stanza import Partial

from stanza.util.logging import logger
from stanza.util.dataclasses import dataclass, replace, field
from jax.random import PRNGKey

from stanza.solver import Solver, Minimize, UnsupportedObectiveError, Objective
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
            actions = jax.tree_util.tree_map(lambda x: jnp.zeros((1,) + x.shape), self.action_sample)
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

    # The BarrierMPC loss function has s, params
    def _loss_fn(self, state0, params):
        s, actions = params

        states = stanza.policy.rollout(
            self.model_fn, state0, policy=Actions(actions)
        ).states

        cost = self.cost_fn(states, actions)
        if not self.barrier_sdf:
            return cost
        b = self.barrier_sdf(states, actions)
        cost + self.eta*jnp.sum(b)

    def _solve(self, state0, actions):
        pass