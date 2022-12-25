import optax

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import jinx.envs
import jinx.util

from jinx.stats import Reporter

from functools import partial
from typing import NamedTuple, Any

# 
class MPCState(NamedTuple):
    T: jnp.array # current timestep
    us: jnp.array
    # The optimizer state history
    optim_history: Any
    est_state: Any

# Internally during optimization
class OptimStep(NamedTuple):
    us: jnp.array
    cost: jnp.array
    est_state: Any
    opt_state: Any

# A simple MPC which internally uses JaxOPT
class MPC:
    def __init__(self, u_dim, cost_fn, model_fn,
                horizon_length,
                opt_transform,
                iterations,
                receed=True,
                grad_estimator=None):
        self.u_dim = u_dim

        self.cost_fn = cost_fn
        self.model_fn = model_fn
        self.horizon_length = horizon_length

        self.opt_transform = opt_transform
        self.iterations = iterations

        self.receed = receed
        self.grad_estimator = grad_estimator
    
    def _loss_fn(self, est_state, x0, us):
        xs = jinx.envs.rollout_input(self.model_fn, x0, us)
        if self.grad_estimator:
            est_state, obs = self.grad_estimator.inject_gradient(est_state, xs, us)
        else:
            obs = xs.obs
        cost = self.cost_fn(obs, us)
        return cost, (est_state, cost)
    
    # body_fun for the solver interation
    def _opt_scan_fn(self, x0, prev_step, _):

        loss = partial(self._loss_fn, prev_step.est_state, x0)
        grad, (est_state, cost) = jax.grad(loss, has_aux=True)(prev_step.us)
        updates, opt_state = self.opt_transform.update(grad, prev_step.opt_state, prev_step.us)
        us = optax.apply_updates(prev_step.us, updates)

        new_step = OptimStep(
            us=us,
            cost=cost,
            est_state=est_state,
            opt_state=opt_state
        )
        return new_step, prev_step
    
    def init_state(self, x0):
        us = jnp.zeros((self.horizon_length, self.u_dim))
        est_state = self.grad_estimator.init() if self.grad_estimator is not None else None
        final_step, history = self._solve(est_state, x0, us)
        return MPCState(
            T=0,
            us=final_step.us,
            optim_history=history,
            est_state=final_step.est_state
        )

    # A modified version of the JaxOPT base IterativeSolver
    # which propagates the estimator state
    def _solve(self, est_state, x0, init_us):
        init_cost, _ = self._loss_fn(est_state, x0, init_us)
        init_step = OptimStep(
            us=init_us,
            cost=init_cost,
            est_state=est_state,
            opt_state=self.opt_transform.init(init_us),
        )
        scan_fn = partial(self._opt_scan_fn, x0)
        # Do a scan with the first iteration unrolled
        # so that the est_state can potentially be properly initialized
        # final_step, history = jinx.util.scan_unrolled(
        #     scan_fn, init_step, None, length=self.iterations
        # )
        final_step, history = jax.lax.scan(scan_fn, init_step, None, length=self.iterations)
        history = jinx.util.tree_append(history, final_step)
        return final_step, history

    def __call__(self, state, policy_state):
        us = policy_state.us
        est_state = policy_state.est_state

        if self.receed:
            us = us.at[:-1].set(us[1:])
            # return the remainder as the solved_us
            # as the policy state, so we don't need
            # to re-solve everything for the next iteration
            final_step, history = self._solve(est_state, state, us)
            return final_step.us[0], MPCState(
                T=policy_state.T + 1,
                us=final_step.us,
                optim_history=history,
                est_state=final_step.est_state
            )
        else:
            return policy_state.us[policy_state.T], MPCState(
                T=policy_state.T + 1,
                us=policy_state.us,
                optim_history=policy_state.optim_history,
                est_state=policy_state.est_state
            )

class EstimatorState(NamedTuple):
    rng: jax.random.PRNGKey
    total_samples: jnp.array

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
            return xs, (W, x_diff)

        def _inject_gradient_bkw(res, g):
            W, x_diff = res
            trans = self.calculate_jacobians(W, x_diff)
            return (None, self.bkw(trans, g), None, None)

        _inject_gradient.defvjp(_inject_gradient_fwd, _inject_gradient_bkw)
        
        self._inject_gradient = _inject_gradient
    
    def init(self):
        return EstimatorState(
            rng=self.rng_key,
            total_samples=jnp.array([0])
        )
    
    def inject_gradient(self, est_state, xs, us):
        new_rng, subkey = jax.random.split(est_state.rng)
        W, x_diff = self.rollout(subkey, xs, us)
        obs = self._inject_gradient(xs.obs, us, W, x_diff)
        return EstimatorState(
            rng=new_rng,
            total_samples=est_state.total_samples + W.shape[0]
        ), obs
    
    # the forwards step
    def rollout(self, rng, traj, us):
        rng = self.rng_key if rng is None else rng

        # do a bunch of rollouts
        W = self.sigma*jax.random.choice(rng, jnp.array([-1,1]), (self.samples,) + us.shape)
        # rollout all of the perturbed trajectories

        # Get the first state
        x_init = jax.tree_util.tree_map(lambda x: x[0], traj)
        trajs = jax.vmap(partial(jinx.envs.rollout_input, self.env.step, x_init))(us + W)
        # subtract off x_bar
        x_diff = trajs.obs - traj.obs
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