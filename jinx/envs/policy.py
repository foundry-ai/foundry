import jaxopt
import jax
import jax.numpy as jnp

from functools import partial



# A simple MPC which internally uses JaxOPT
class MPC:
    def __init__(self, u_dim, cost_fn, model_fn,
                horizon_length, grad_estimator=None,
                receed=True, stat_reporter=None):
        self.cost_fn = cost_fn
        self.model_fn = model_fn
        self.horizon_length = horizon_length
        self.u_dim = u_dim
        self.grad_estimator = grad_estimator
        self.receed = receed

        self.stat_reporter = stat_reporter

        def scan_fn(state, u):
            new_state = model_fn(state, u)
            return new_state, new_state

        def loss_fun(us, init_state):
            _, xs = jax.lax.scan(scan_fn, init_state, us)
            if self.grad_estimator is not None:
                grad = self.grad_estimator(xs, us)
                # inject_gradient will override
                # the gradient
                xs = __inject_gradient(xs, us, grad)
            return cost_fn(xs, us)

        self.solver = jaxopt.LBFGS(fun=loss_fun, maxiter=50000)
        #self.solver = jaxopt.GradientDescent(fun=loss_fun, maxiter=500000)
    
    def _solve(self, init_us, init_state):
        solved_us, state = self.solver.run(init_us, init_state=init_state)
        if self.stat_reporter:
            solved_us = self.stat_reporter.send(
                {'iters': state.iter_num }
            ).attach(solved_us)
        return solved_us
    
    def __call__(self, state, policy_state=None):
        if self.receed:
            # If in receeding mode,
            # resolve over the same horizon length
            # every time
            us = jnp.zeros((self.horizon_length, self.u_dim))
            # if we have previously solved before
            # use the previous solutions
            # as an initial starting point
            if policy_state is not None:
                us = us.at[:-1].set(policy_state)
            solved_us = self._solve(us, state)
            # return the remainder as the solved_us
            # as the policy state, so we don't need
            # to re-solve everything for the next iteration
            return solved_us[0], solved_us[1:]
        else:
            # If in non-receeding mode, solve once
            # and use this input for the rest
            if policy_state is None:
                us = jnp.zeros((self.horizon_length, self.u_dim))
                solved_us = self._solve(us, state)
                T = 0
            else:
                solved_us, T = policy_state
            return solved_us[T], (solved_us, T+1)

@jax.custom_vjp
def __inject_gradient(xs, us, grad):
    return xs

def __inject_gradient_fwd(xs, us, grad):
    return xs, grad

def __inject_gradient_bkw(res, g):
    return res @ g
    
__inject_gradient.defvjp(__inject_gradient_fwd, __inject_gradient_bkw)