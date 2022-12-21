import jaxopt
import jax
import jax.numpy as jnp

from functools import partial

# A simple MPC which internally uses JaxOPT
class MPC:
    def __init__(self, u_dim, cost_fn, model_fn, horizon_length):
        self.cost_fn = cost_fn
        self.model_fn = model_fn
        self.horizon_length = horizon_length
        self.u_dim = u_dim

        def scan_fn(state, u):
            new_state = model_fn(state, u)
            return new_state, new_state

        def loss_fun(us, init_state):
            _, xs = jax.lax.scan(scan_fn, init_state, us)
            return cost_fn(xs, us)
        #self.solver = jaxopt.LBFGS(fun=loss_fun, maxiter=50000)
        self.solver = jaxopt.GradientDescent(fun=loss_fun, maxiter=500000)
    
    def __call__(self, state, policy_state=None):
        us = jnp.zeros((self.horizon_length, self.u_dim))

        # if we have previously solved before
        # use the previous solutions
        # as an initial starting point
        if policy_state is not None:
            us = us.at[:-1].set(policy_state)

        solved_us, _ = self.solver.run(us, init_state=state)

        # return the remainder as the solved_us
        # as the policy state, so we don't need
        # to re-solve everything for the next iteration
        return solved_us[0], solved_us[1:]
