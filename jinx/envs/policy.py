import jaxopt
from jaxopt import loop

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

from functools import partial

# A simple MPC which internally uses JaxOPT
class MPC:
    def __init__(self, u_dim, cost_fn, model_fn,
                horizon_length,
                receed=True,
                grad_estimator=None,
                stat_reporter=None):
        self.u_dim = u_dim

        self.cost_fn = cost_fn
        self.model_fn = model_fn

        self.receed = receed
        self.horizon_length = horizon_length

        self.grad_estimator = grad_estimator
        self.stat_reporter = stat_reporter

        def scan_fn(state, u):
            new_state = model_fn(state, u)
            return new_state, new_state

        # the gradient injector
        @jax.custom_vjp
        def _inject_gradient(xs, us, fwd):
            return xs
        def _inject_gradient_fwd(xs, us, fwd):
            return xs, fwd
        def _inject_gradient_bkw(res, g):
            #return (g, None, None)
            return (None, self.grad_estimator.bkw(res, g), None)

        _inject_gradient.defvjp(_inject_gradient_fwd, _inject_gradient_bkw)

        def loss_fun(us, init_state, est_state=None):
            _, xs = jax.lax.scan(scan_fn, init_state, us)
            if self.grad_estimator is not None:
                est_state, fwd = self.grad_estimator.fwd(est_state, xs, us)
                # inject_gradient will override the gradient
                xs = _inject_gradient(xs, us, fwd)
            return cost_fn(xs, us), est_state
        #self.solver = jaxopt.LBFGS(fun=loss_fun, maxiter=1000, has_aux=True)
        self.solver = jaxopt.GradientDescent(fun=loss_fun, maxiter=1000, has_aux=True)
    
    # body_fun for the solver interation
    def _body_fun(self, inputs):
        (params, state), (args, kwargs) = inputs

        # get the ge_state from the aux
        est_state = state.aux
        return (
            self.solver.update(params, state, *args, **kwargs, est_state=est_state),
            (args, kwargs)
        )
    
    # A modified version of the JaxOPT base IterativeSolver
    # which propagates the estimator state
    def _solve(self, init_params, est_state=None, *args, **kwargs):
        state = self.solver.init_state(init_params, *args, **kwargs)
        zero_step = self.solver._make_zero_step(init_params, state)
        opt_step = self.solver.update(init_params, state, *args,
                                    **kwargs, est_state=est_state)

        init_val = (opt_step, (args, kwargs))
        jit, unroll = self.solver._get_loop_options()

        many_step = loop.while_loop(
            cond_fun=self.solver._cond_fun, body_fun=self._body_fun,
            init_val=init_val, maxiter=self.solver.maxiter - 1, jit=jit,
            unroll=unroll)[0]

        final_step = tree_util.tree_map(
            partial(_where, self.solver.maxiter == 0),
            zero_step, many_step, is_leaf=lambda x: x is None)
            # state attributes can sometimes be None

        solved_params, state = final_step
        if self.stat_reporter:
            solved_params = self.stat_reporter.send(
                {'iters': state.iter_num }
            ).attach(solved_params)
        return solved_params, state.aux
    
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
                old_us, ge_state = policy_state
                us = us.at[:-1].set(old_us)
            else:
                ge_state = None

            solved_us, est_state = self._solve(us, init_state=state,
                                                est_state=est_state)
            # return the remainder as the solved_us
            # as the policy state, so we don't need
            # to re-solve everything for the next iteration

            new_state = solved_us[1:], ge_state
            return solved_us[0], new_state
        else:
            # If in non-receeding mode, solve once
            # and use this input for the rest
            if policy_state is None:
                us = jnp.zeros((self.horizon_length, self.u_dim))
                solved_us, _ = self._solve(us, init_state=state,
                                            est_state=None)
                T = 0
            else:
                solved_us, T = policy_state
            return solved_us[T], (solved_us, T+1)


def _where(cond, x, y):
  if x is None: return y
  if y is None: return x
  return jnp.where(cond, x, y)

class IsingEstimator:
    def __init__(self, env, rng_key, samples, sigma):
        self.env = env
        self.rng_key = rng_key
        self.samples = samples
        self.sigma = sigma
    
    def _roll_scan_fn(self, state, u):
        new_state = self.env.step(state, u)
        return new_state, new_state
    
    def _rollout(self, x_init, us):
        _, xs = jax.lax.scan(self._roll_scan_fn, x_init, us)
        return xs

    # the forwards step
    def fwd(self, rng, xs, us):
        rng = self.rng_key if rng is None else rng
        rng, subkey = jax.random.split(rng)

        # do a bunch of rollouts
        W = jax.random.choice(subkey, jnp.array([-1,1]), (self.samples,) + us.shape)
        # rollout all of the perturbed trajectories
        x_hat = jax.vmap(self._rollout, in_axes=(None,0))(xs[0], us + W)
        # subtract off x_bar
        x_diff = x_hat - xs
        return rng, (W, x_diff)
    
    # the backwards step
    def bkw(self, res, g):
        W, x_diff = res

        W = jnp.expand_dims(W, -2)
        W = jnp.tile(W, [1, 1, W.shape[1], 1])
        # W: (samples, traj_dim, traj_dim, u_dim)
        x_diff = jnp.expand_dims(x_diff, -3)
        # x_diff: (samples, 1,  traj_dim, x_dim)

        W = jnp.transpose(W, (1, 2,0,3))
        x_diff = jnp.transpose(x_diff, (1, 2,3,0))
        # W: (traj_dim, traj_dim, samples, u_dim)
        # x_diff: (1, traj_dim, x_dim, samples)
        trans = x_diff @ W/self.sigma/self.samples
        # trans: (traj_dim, traj_dim, x_dim, u_dim)
        # (u,v) entry contains the transition operator from time u to state v

        # we need to zero out at and below the diagonal
        # (there should be no correlation, but just in case)
        tri = jax.numpy.tri(trans.shape[0], dtype=bool)
        tri = jnp.expand_dims(jnp.expand_dims(tri, -1),-1)
        tri = jnp.tile(tri, [1,1,trans.shape[-2], trans.shape[-1]])
        trans = jnp.where(tri, trans, jnp.zeros_like(trans))

        trans_T = jnp.transpose(trans, (0,1,3,2))
        # (traj_dim, traj_dim, u_dim, x_dim) @ (1, traj_dim, x_dim, 1)
        grad = trans_T @ jnp.expand_dims(jnp.expand_dims(g, -1),0)
        # grad: (traj_dim, traj_dim, u_dim, 1)
        # sum over columns to combine all transitions for a given time
        grad = jnp.sum(jnp.squeeze(grad,-1), 1)
        return grad