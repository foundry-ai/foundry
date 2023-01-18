import optax

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import jinx.envs
import jinx.util

from functools import partial
from typing import NamedTuple, Any
from jinx.logging import logger
from jinx.solver import NewtonSolver, RelaxingSolver, OptaxSolver
from jinx.solver.jaxopt import JaxOptSolver, BFGS, LBFGS, GradientDescent

def _centered_log(sdf, *args):
    u_x, fmt = jax.flatten_util.ravel_pytree(args)
    # calculate gradient at zero
    grad = jax.jacrev(lambda v_x: -jnp.log(-sdf(*fmt(v_x))))(jnp.zeros_like(u_x))
    grad_term = grad @ u_x
    barrier = -jnp.log(-sdf(*args)) - grad_term
    return barrier - grad_term

# A RHC barrier-based MPC controller
class BarrierMPC:
    def __init__(self, u_sample,
                cost_fn, model_fn,
                horizon_length,
                barrier_sdf=None,
                barrier_eta=0.01, barrier_zero=0):
        self.u_sample = u_sample

        # turn the per-timestep cost function
        # into a trajectory-based cost function
        self.cost_fn = partial(jinx.envs.trajectory_cost, cost_fn)
        self.model_fn = model_fn
        self.horizon_length = horizon_length

        self.barrier_sdf = barrier_sdf
        self.barrier_eta = barrier_eta

        # using a damping of 1 prevents jumping out
        # of the feasible set
        sub_solver = NewtonSolver(self._loss_fn)
        # # using a damping of 1 prevents jumping of out of feasible
        # # set for all t
        self.solver = RelaxingSolver(sub_solver, init_t=1, max_t=1/barrier_eta)
        # self.solver = NewtonSolver(self._loss_fn)
        # self.solver = JaxOptSolver(self._loss_fn, LBFGS,
        #                             stop_if_linesearch_fails=True)

        # if we don't have a barrier, do just 1 outer loop
        if not self.barrier_sdf:
            self.solver.max_t = 1.
            self.solver.init_t = 1.

    def _loss_fn(self, us, x0, t=1):
        xs = jinx.envs.rollout_input(
                self.model_fn, x0, us
            )
        cost = self.cost_fn(xs, us)

        if self.barrier_sdf is not None:
            barrier_costs = _centered_log(self.barrier_sdf, xs, us)
            loss = cost + jnp.sum(barrier_costs)/t
        else:
            loss = cost
        loss = jnp.nan_to_num(loss, nan=float('inf'))
        return loss

    def init_state(self, x0):
        u_flat, _ = jax.flatten_util.ravel_pytree(self.u_sample)
        u_flat = jnp.zeros((self.horizon_length - 1, u_flat.shape[0]))
        return u_flat

    def __call__(self, state, policy_state=None):
        us = self.init_state(state) if policy_state is None else policy_state
        us = us.at[:-1].set(us[1:])

        def solve_us(us):
            res = self.solver.run(us, x0=state)
            us = res.final_params
            us = jax.lax.cond(res.solved, lambda: us, lambda: float("nan")*us)
            return us

        if self.barrier_sdf:
            # for now we have no state constraints
            # so u = 0 is always feasible
            us = jnp.zeros_like(us)

            xs = jinx.envs.rollout_input(self.model_fn, state, us)
            feasible = jnp.max(self.barrier_sdf(xs, us))

            us = jax.lax.cond(feasible < 0, solve_us,
                    lambda x: x, us)
        else:
            us = solve_us(us)

        return us[0] if policy_state is None else (us[0], us)