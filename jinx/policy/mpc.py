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

def _centered_log(sdf, *args):
    u_x, fmt = jax.flatten_util.ravel_pytree(args)
    # calculate gradient at zero
    grad = jax.jacrev(lambda v_x: -jnp.log(-sdf(*fmt(v_x))))(jnp.zeros_like(u_x))
    grad_term = grad @ u_x
    #jax.debug.print("zero_grad: {}", grad)
    barrier = -jnp.log(-sdf(*args)) #- grad_term
    return barrier #- grad_term

# A RHC barrier-based MPC controller
class BarrierMPC:
    def __init__(self, u_sample,
                cost_fn, model_fn,
                horizon_length,
                barrier_sdf=None,
                barrier_eta=0.01,
                barrier_feasible=None):
        self.u_sample = u_sample

        self.cost_fn = cost_fn
        self.model_fn = model_fn
        self.horizon_length = horizon_length

        self.barrier_sdf = barrier_sdf

        # for now we need to be given a feasible point
        self.feasibility_solver = barrier_feasible

        # using a damping of 1 prevents jumping out
        # of the feasible set
        sub_solver = NewtonSolver(self._loss_fn, damping=1)
        # using a damping of 1 prevents jumping of out of feasible
        # set for all t
        self.solver = RelaxingSolver(sub_solver, max_t=1/barrier_eta)

        # if we don't have a barrier, do just 1 outer loop
        if not self.barrier_sdf:
            self.solver.max_t = 1.
            self.solver.init_t = 1.

    def _loss_fn(self, us, t, x0):
        xs = jinx.envs.rollout_input(
                self.model_fn, x0, us
            )
        cost = self.cost_fn(xs, us)

        if self.barrier_sdf is not None:
            barrier_costs = _centered_log(self.barrier_sdf, xs, us)
            loss = t*cost + jnp.sum(barrier_costs)
        else:
            loss = cost
        return loss

    def init_state(self, x0):
        u_flat, unflatten = jax.flatten_util.ravel_pytree(self.u_sample)
        u_flat = jnp.repeat(u_flat[jnp.newaxis, :],
                        self.horizon_length - 1, axis=0)
        us = jax.vmap(unflatten)(u_flat)
        return us

    def __call__(self, state, policy_state=None):
        if policy_state is None:
            us = self.init_state(state)
        else:
            us = policy_state
        us = us.at[:-1].set(us[1:])

        def solve_us(us):
            res = self.solver.run(us, x0=state)
            us = res.final_params
            return us

        if self.barrier_sdf:
            # first solve to make sure feasible
            # xs = jinx.envs.rollout_input(self.model_fn, state, us)
            # feasible = jnp.max(self.barrier_sdf(xs, us)) < 0
            us = self.feasibility_solver(state, us)

            xs = jinx.envs.rollout_input(self.model_fn, state, us)
            feasible = jnp.max(self.barrier_sdf(xs, us))

            us = jax.lax.cond(feasible < 0, solve_us,
                    lambda x: x, us)
        else:
            us = solve_us(us)

        if policy_state is None:
            return us[0]
        else:
            return us[0], us
