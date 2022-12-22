import jaxopt
from jaxopt import loop

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import jinx.envs

from jinx.stats import Reporter

from functools import partial

# A simple MPC which internally uses JaxOPT
class MPC:
    def __init__(self, u_dim, cost_fn, model_fn,
                horizon_length,
                receed=True, solver='gd',
                grad_estimator=None):
        self.u_dim = u_dim

        self.cost_fn = cost_fn
        self.model_fn = model_fn

        self.receed = receed
        self.horizon_length = horizon_length

        self.grad_estimator = grad_estimator
        self.reporter = Reporter()

        # the gradient injector
        def loss_fun(us, init_state, est_state=None):
            xs = jinx.envs.rollout_input(self.model_fn, init_state, us)
            if self.grad_estimator is not None:
                est_state, xs = self.grad_estimator.inject_gradient(est_state, xs, us)
            return cost_fn(xs, us), est_state

        if solver == 'gd':
            self.solver = jaxopt.GradientDescent(fun=loss_fun, maxiter=1000, has_aux=True)
        elif solver == 'lbfgs':
            self.solver = jaxopt.LBFGS(fun=loss_fun, maxiter=1000, has_aux=True)
        else:
            raise RuntimeError("Unrecognized solver")
    
    # body_fun for the solver interation
    def _body_fun(self, inputs):
        (us, state), init_state = inputs

        xs = jinx.envs.rollout_input(self.model_fn, init_state, us)
        init_state = self.reporter.tap(init_state, 'cost', self.cost_fn(xs, us))

        # get the ge_state from the aux
        est_state = state.aux
        return (
            self.solver.update(us, state,
                    init_state=init_state, est_state=est_state),
            init_state
        )
    
    # A modified version of the JaxOPT base IterativeSolver
    # which propagates the estimator state
    def _solve(self, init_us, init_state, est_state=None):
        state = self.solver.init_state(init_us,
                    init_state=init_state, est_state=est_state)
        zero_step = self.solver._make_zero_step(init_us, state)

        # report the first cost
        xs = jinx.envs.rollout_input(self.model_fn, init_state, init_us)
        state = self.reporter.tap(state, 'cost', self.cost_fn(xs, init_us))

        opt_step = self.solver.update(init_us, state,
                    init_state=init_state, est_state=est_state)

        init_val = (opt_step, init_state)
        jit, unroll = self.solver._get_loop_options()

        many_step = loop.while_loop(
            cond_fun=self.solver._cond_fun, body_fun=self._body_fun,
            init_val=init_val, maxiter=self.solver.maxiter - 1, jit=jit,
            unroll=unroll)[0]

        final_step = tree_util.tree_map(
            partial(_where, self.solver.maxiter == 0),
            zero_step, many_step, is_leaf=lambda x: x is None)
            # state attributes can sometimes be None

        solved_us, state = final_step
        self.reporter.send('iters', state.iter_num)
        return solved_us, state.aux
    
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

        @jax.custom_vjp
        def _inject_gradient(xs, us, W, x_diff):
            return xs

        def _inject_gradient_fwd(xs, us, W, x_diff):
            gt = jax.jacrev(partial(jinx.envs.rollout_input, self.env.step, xs[0]))(us)
            gt = jnp.transpose(gt, (2, 0, 1, 3))
            return xs, (W, x_diff, gt)

        def _inject_gradient_bkw(res, g):
            W, x_diff, gt = res
            trans = self.calculate_jacobians(W, x_diff)
            return (None, self.bkw(trans, g), None, None)

        _inject_gradient.defvjp(_inject_gradient_fwd, _inject_gradient_bkw)
        
        self._inject_gradient = _inject_gradient
    
    def inject_gradient(self, rng, xs, us):
        if rng is None:
            rng = self.rng_key
        new_rng, subkey = jax.random.split(rng)
        W, x_diff = self.rollout(subkey, xs, us)
        xs = self._inject_gradient(xs, us, W, x_diff)
        return new_rng, xs
    
    # the forwards step
    def rollout(self, rng, xs, us):
        rng = self.rng_key if rng is None else rng

        # do a bunch of rollouts
        W = self.sigma*jax.random.choice(rng, jnp.array([-1,1]), (self.samples,) + us.shape)
        # rollout all of the perturbed trajectories
        x_hat = jax.vmap(partial(jinx.envs.rollout_input, self.env.step, xs[0]))(us + W)
        # subtract off x_bar
        x_diff = x_hat - xs
        return W, x_diff
    
    def calculate_jacobians(self, W, x_diff):
        W = jnp.expand_dims(W, -2)
        W = jnp.tile(W, [1, 1, W.shape[1], 1])
        # W: (samples, traj_dim, traj_dim, u_dim)
        x_diff = jnp.expand_dims(x_diff, -3)
        # x_diff: (samples, 1,  traj_dim, x_dim)

        W = jnp.expand_dims(W, -2)
        x_diff = jnp.expand_dims(x_diff, -1)
        # W: (samples, traj_dim, traj_dim, 1, u_dim)
        # x_diff: (samples, 1, traj_dim, x_dim, 1)
        jac = jnp.mean(x_diff @ W, axis=0)/(self.sigma*self.sigma)
        # jac: (traj_dim, traj_dim, x_dim, u_dim)
        # (u,v) entry contains the jacobian from time u to state v

        # we need to zero out at and below the diagonal
        # (there should be no correlation, but just in case)
        tri = jax.numpy.tri(jac.shape[0], dtype=bool)
        tri = jnp.expand_dims(jnp.expand_dims(tri, -1),-1)
        tri = jnp.tile(tri, [1,1,jac.shape[-2], jac.shape[-1]])

        # fill lower-triangle with zeros
        jac = jnp.where(tri, jnp.zeros_like(jac), jac)

        return jac
    
    # the backwards step
    def bkw(self, jac, g):
        jac_T = jnp.transpose(jac, (0,1,3,2))
        # (traj_dim, traj_dim, u_dim, x_dim) @ (1, traj_dim, x_dim, 1)
        grad = jac_T @ jnp.expand_dims(jnp.expand_dims(g, -1),0)
        # grad: (traj_dim, traj_dim, u_dim, 1)
        # sum over columns to combine all transitions for a given time
        grad = jnp.sum(jnp.squeeze(grad,-1), 1)
        return grad