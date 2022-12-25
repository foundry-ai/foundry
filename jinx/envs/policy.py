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
    gains: jnp.array
    # The optimizer state history
    optim_history: Any
    est_state: Any

# Internally during optimization
class OptimStep(NamedTuple):
    us: jnp.array
    gains: jnp.array
    cost: jnp.array
    est_state: Any
    opt_state: Any

# A simple MPC which internally uses JaxOPT
class MPC:
    def __init__(self, u_dim, cost_fn, model_fn,
                horizon_length,
                opt_transform,
                iterations,
                use_gains=False,
                burn_in=10,
                receed=True,
                grad_estimator=None):
        self.u_dim = u_dim

        self.cost_fn = cost_fn
        self.model_fn = model_fn
        self.horizon_length = horizon_length

        self.opt_transform = opt_transform
        self.iterations = iterations

        self.use_gains = use_gains
        self.burn_in = burn_in

        self.receed = receed
        self.grad_estimator = grad_estimator

    def init_state(self, state_0):
        # initial gains are all zero
        x_dim = state_0.x.shape[-1]
        us = jnp.zeros((self.horizon_length - 1, self.u_dim))
        gains = jnp.zeros((self.horizon_length - 1, self.u_dim, x_dim))
        est_state = self.grad_estimator.init() if self.grad_estimator is not None else None
        final_step, history = self._solve(est_state, state_0, gains, us)
        return MPCState(
            T=0,
            us=final_step.us,
            gains=gains,
            optim_history=history,
            est_state=final_step.est_state
        )
    
    def _compute_gains(self, jac, prev_gains):
        C = jac[:self.burn_in,self.burn_in:]
        C_k = C[:,:-1]
        C_kp = C[:,1:]
        # flatten out the first dimension into the column dimension
        C_k = jnp.transpose(C_k, (1,2,0,3))
        C_k = C_k.reshape((C_k.shape[0], C_k.shape[1],-1))
        C_kp = jnp.transpose(C_kp, (1,2,0,3))
        C_kp = C_kp.reshape((C_kp.shape[0], C_kp.shape[1],-1))
        # C_k, C_kp are (traj_length - burn_in - 1, x_dim, input x burn_in)
        # the pseudoinverse should broadcast over the first dimension
        # select 1 above the jacobian diagonal for the Bs
        Bs_est = jnp.transpose(jnp.diagonal(jac, 1), (2,0,1))[self.burn_in:]
        # estimate the A matrices
        As_est = C_kp @ jnp.linalg.pinv(C_k) - Bs_est @ prev_gains[self.burn_in:]
        # synthesize new gains
        Q = jnp.eye(jac.shape[-2])
        R = jnp.eye(self.u_dim)
        def gains_recurse(P_next, AB):
            A, B = AB
            F = jnp.linalg.inv(R + B.T @ P_next @ B) @ (A @ P_next @ B).T
            P = A.T @ P_next @ A - (A.T @ P_next @ B) @ F + Q
            return P, F
        _, gains_est = jax.lax.scan(gains_recurse, Q, (As_est, Bs_est))
        # jax.experimental.host_callback.id_print(As_est[-1])
        # jax.experimental.host_callback.id_print(As_est[-1] + Bs_est[-1] @ gains_est[-1])
        # jax.experimental.host_callback.id_print(gains_est[0])
        # jax.experimental.host_callback.id_print(prev_gains[-1])
        # jax.experimental.host_callback.id_print(gains_est[-1])
        new_gains = prev_gains.at[self.burn_in:].set(gains_est)
        return new_gains

    
    def _loss_fn(self, est_state, state_0,
                ref_states, ref_gains, ref_us, us):
        rollout = partial(jinx.env.rollout_with_gains,
            self.model_fn, state_0, 
            ref_states, ref_gains, ref_us)
        
        states = rollout(us)
        if self.grad_estimator:
            est_state, jac, xs = self.grad_estimator.inject_gradient(est_state, states, gains, us)
        else:
            xs = states.x
            jac = jax.jacrev(lambda us: rollout(us).x)(us)
            jac = jnp.transpose(jac, (2, 0, 1, 3))
        cost = self.cost_fn(xs, us)
        return cost, (est_state, jac, cost)
    
    # body_fun for the solver interation
    def _opt_scan_fn(self, state_0, prev_step, _):
        gains = prev_step.gains
        states = jinx.envs.rollout_input(self.model_fn, state_0, us)

        loss = partial(self._loss_fn, prev_step.est_state,
                        state_0, states, gains, us)

        grad, (est_state, jac, cost) = jax.grad(loss, has_aux=True)(prev_step.us)
        updates, opt_state = self.opt_transform.update(grad, prev_step.opt_state, prev_step.us)
        us = optax.apply_updates(prev_step.us, updates)

        if self.use_gains:
            gains = self._compute_gains(jac, gains)
            # update for the old gains
            states_new = jinx.env.rollout_with_gains(self.step, state_0, )

        new_step = OptimStep(
            us=us,
            cost=cost,
            gains=gains,
            est_state=est_state,
            opt_state=opt_state
        )
        return new_step, prev_step
    

    # A modified version of the JaxOPT base IterativeSolver
    # which propagates the estimator state
    def _solve(self, est_state, state0, gains, init_us):
        init_cost, _ = self._loss_fn(est_state, state0, gains, init_us)
        init_step = OptimStep(
            us=init_us,
            gains=gains,
            cost=init_cost,
            est_state=est_state,
            opt_state=self.opt_transform.init(init_us),
        )
        scan_fn = partial(self._opt_scan_fn, state0)
        final_step, history = jax.lax.scan(scan_fn, init_step, None, length=self.iterations)
        history = jinx.util.tree_append(history, final_step)
        return final_step, history

    def __call__(self, state, policy_state):
        us = policy_state.us
        gains = policy_state.gains
        est_state = policy_state.est_state

        if self.receed:
            us = us.at[:-1].set(us[1:])
            gains = gains.at[:-1].set(gains[1:])
            # return the remainder as the solved_us
            # as the policy state, so we don't need
            # to re-solve everything for the next iteration
            final_step, history = self._solve(est_state, state, gains, us)
            return final_step.us[0], MPCState(
                T=policy_state.T + 1,
                us=final_step.us,
                gains=final_step.gains,
                optim_history=history,
                est_state=final_step.est_state
            )
        else:
            return policy_state.us[policy_state.T], MPCState(
                T=policy_state.T + 1,
                us=policy_state.us,
                gains=policy_state.gains,
                optim_history=policy_state.optim_history,
                est_state=policy_state.est_state
            )

class EstimatorState(NamedTuple):
    rng: jax.random.PRNGKey
    total_samples: int

class IsingEstimator:
    def __init__(self, model_fn, rng_key, samples, sigma):
        self.model_fn = model_fn
        self.rng_key = rng_key
        self.samples = samples
        self.sigma = sigma

        @jax.custom_vjp
        def _inject_gradient(xs, us, jac):
            return xs

        def _inject_gradient_fwd(xs, us, jac):
            return xs, jac

        def _inject_gradient_bkw(res, g):
            jac = res
            return (None, self.bkw(jac, g), None)

        _inject_gradient.defvjp(_inject_gradient_fwd, _inject_gradient_bkw)
        
        self._inject_gradient = _inject_gradient
    
    def init(self):
        return EstimatorState(
            rng=self.rng_key,
            total_samples=0
        )
    
    def inject_gradient(self, est_state, states, gains, us):
        new_rng, subkey = jax.random.split(est_state.rng)
        W, x_diff = self.rollout(subkey, states, gains, us)
        jac = self.calculate_jacobians(W, x_diff)
        x = self._inject_gradient(states.x, us, jac)
        return EstimatorState(
            rng=new_rng,
            total_samples=est_state.total_samples + W.shape[0]
        ), jac, x
    
    # the forwards step
    def rollout(self, rng, traj, gains, us):
        rng = self.rng_key if rng is None else rng
        state_0 = jax.tree_util.tree_map(lambda x: x[0], traj)

        # do a bunch of rollouts
        W = self.sigma*jax.random.choice(rng, jnp.array([-1,1]), (self.samples,) + us.shape)
        # rollout all of the perturbed trajectories
        def model_with_gains(state, u):
            state, T = state
            state = self.model_fn(state, u + gains[T] @ (state.x - traj.x[T]))
            return (state, T + 1)
        rollout = lambda us: jinx.envs.rollout_input(model_with_gains, (state_0, 0), us)[0]

        # Get the first state
        trajs = jax.vmap(rollout)(us + W)
        # subtract off x_bar
        x_diff = trajs.x - traj.x
        return W, x_diff
    
    def calculate_jacobians(self, W, x_diff):
        W = jnp.expand_dims(W, -2)
        W = jnp.tile(W, [1, 1, x_diff.shape[1], 1])
        # W: (samples, traj_dim-1, traj_dim, u_dim)
        x_diff = jnp.expand_dims(x_diff, -3)
        # x_diff: (samples, 1,  traj_dim, x_dim)

        W = jnp.expand_dims(W, -2)
        x_diff = jnp.expand_dims(x_diff, -1)
        # W: (samples, traj_dim - 1, traj_dim, 1, u_dim)
        # x_diff: (samples, 1, traj_dim, x_dim, 1)
        jac = jnp.mean(x_diff @ W, axis=0)/(self.sigma*self.sigma)
        # jac: (traj_dim-1, traj_dim, x_dim, u_dim)
        # (u,v) entry contains the jacobian from time u to state v

        # we need to zero out at and below the diagonal
        # (there should be no correlation, but just in case)
        tri = jax.numpy.tri(jac.shape[0], jac.shape[1], dtype=bool)
        tri = jnp.expand_dims(jnp.expand_dims(tri, -1),-1)
        tri = jnp.tile(tri, [1,1,jac.shape[2], jac.shape[3]])

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