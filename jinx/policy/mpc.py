import optax

import sys
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import jinx.envs
import jinx.util

from jinx.stats import Reporter

from functools import partial
from typing import NamedTuple, Any

class MPCState(NamedTuple):
    T: jnp.array # current timestep
    us: jnp.array
    gains: jnp.array
    # The optimizer state history
    optim_history: Any
    est_state: Any

# Internally during optimization
class OptimStep(NamedTuple):
    iteration: jnp.array
    us: jnp.array
    barrier_eta: jnp.array
    gains: jnp.array

    grad_norm: jnp.array

    cost: jnp.array
    est_state: Any
    opt_state: Any

    done: bool

# A barrier-based MPC controller
class MPC:
    def __init__(self, u_dim,
                cost_fn, model_fn,
                horizon_length,
                opt_transform,
                iterations=10000,
                # minimize to epsilon-stationary point
                eps=0.0001,

                # barrier signed distance function
                # can return multiple distance functions
                # for multiple barriers
                barrier_sdf=None,
                # relax barrier_eta until eta < epsilon
                barrier_eps=0.0001,
                barrier_eta_start=1.,

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
        self.eps = eps

        self.barrier_sdf = barrier_sdf
        self.barrier_eps = barrier_eps
        self.barrier_eta_start = barrier_eta_start

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
        _, gains_est = jax.lax.scan(gains_recurse, Q, (As_est, Bs_est), reverse=True)
        gains_est = -gains_est
        new_gains = prev_gains.at[self.burn_in:].set(gains_est)

        def print_fun(args, _):
            new_gains, As_est, Bs_est = args
            print('synth gains:', new_gains[-1])
            print('open-loop A:', As_est[-1])
            print('open-loop B:', Bs_est[-1])
            print('synth closed-loop A:', As_est[-1] + Bs_est[-1] @ new_gains[-1])
        # jax.experimental.host_callback.id_tap(print_fun, (new_gains, As_est, Bs_est))

        #new_gains = prev_gains

        return new_gains

    
    def _loss_fn(self, est_state, state_0,
                ref_states, ref_gains, 
                barrier_eta, us):
        rollout = partial(jinx.envs.rollout_input_gains,
            self.model_fn, state_0, 
            ref_states.x, ref_gains)
        states = rollout(us)

        # for use with gradient estimation
        if self.grad_estimator:
            est_state, jac, xs = self.grad_estimator.inject_gradient(est_state, states, ref_gains, us)
        else:
            xs = states.x
            jac = jax.jacrev(lambda us: rollout(us).x)(us)
            jac = jnp.transpose(jac, (2, 0, 1, 3))

        cost = self.cost_fn(xs, us)

        if self.barrier_sdf is not None:
            # if we violate the constraints, ignore the cost
            # and just take steps to find a feasible point
            sdfs = self.barrier_sdf(xs, us)
            zero_grad = jnp.grad(lambda xs, us: -jnp.log(-self.barrier_sdf(xs, us)))(jnp.zeros_like(xs), jnp.zeros_like(us))
            barrier_costs = -jnp.log(-sdfs) + zero_grad[0] @ xs + zeros_grad[1] @ us

            loss = cost + barrier_eta*jnp.mean(barrier_costs)
        else:
            loss = cost

        return loss, (est_state, jac, cost)

    def _inner_step(self, state_0, prev_step):
        gains = prev_step.gains
        ref_states = jinx.envs.rollout_input(self.model_fn, state_0, prev_step.us)

        loss = partial(self._loss_fn, prev_step.est_state, state_0,
                        ref_states, gains, prev_step.barrier_eta)
        grad, (est_state, jac, cost) = jax.grad(loss, has_aux=True)(
            prev_step.us
        )
        updates, opt_state = self.opt_transform.update(grad, prev_step.opt_state, prev_step.us)
        us = optax.apply_updates(prev_step.us, updates)

        if self.use_gains:
            # rollout new trajectory under old gains
            states_new = jinx.envs.rollout_input_gains(self.model_fn, state_0, ref_states.x, gains, us)
            # adjust the us to include the gain-adjustments
            mod = gains @ jnp.expand_dims(states_new.x[:-1] - ref_states.x[:-1], -1)
            us = us + jnp.squeeze(mod, -1)
            # compute new gains around the adjusted trajectory
            gains = self._compute_gains(jac, gains)
        
        def print_fun(args, _):
            grad, done = args
            print('gradient:', jnp.linalg.norm(grad, 'fro'), done, self.eps)
        # jax.experimental.host_callback.id_tap(print_fun, (grad, prev_step.done))

        new_step = OptimStep(
            us=us,
            barrier_eta=prev_step.barrier_eta,
            cost=cost,
            gains=gains,
            est_state=est_state,
            opt_state=opt_state,

            grad_norm=jnp.linalg.norm(grad),
            done=jnp.linalg.norm(grad) < self.eps,
            iteration=prev_step.iteration + 1
        )
        return new_step
    
    # body_fun for the solver interation
    def _opt_iteration(self, state_0, prev_step, _):
        if self.barrier_sdf:
            def do_barrier_step():
                # if we have hit an epsilon-stationary point
                # last iteration,
                # step on the central path
                barrier_eta = jax.lax.cond(prev_step.done,
                    lambda: prev_step.barrier_eta/2,
                    lambda: prev_step.barrier_eta 
                )
                barrier_eta = jnp.max(barrier_eta, self.barrier_eps)
                prev_step.barrier_eta = barrier_eta
                prev_step.done = False
                return self._inner_step(state_0, prev_step)

            new_step = jax.lax.cond(
                prev_step.done and prev_step.barrier_eta <= self.barrier_eps,
                lambda: prev_step,
                do_barrier_step
            )
        else:
            new_step = jax.lax.cond(
                prev_step.done,
                lambda: prev_step, 
                lambda: self._inner_step(state_0, prev_step))
        return new_step, prev_step
    
    def _feasible_point(self, est_state, state_0, gains, init_us):
        def loss_fn(est_state, ref_traj, gains, us):
            pass
        
        def step_fn(state, _):
            est_state, gains, us = state
            ref_states = jinx.envs.rollout_input(self.model_fn, state_0, us)
            loss = partial(loss_fn, est_state, ref_states, state_0, gains)
            return (est_state, gains, us)
    

    # A modified version of the JaxOPT base IterativeSolver
    # which propagates the estimator state
    def _solve(self, est_state, state_0, gains, init_us):
        # if we have a barrier function first find a feasible state
        if self.barrier_sdf:
            est_state, gains, init_us = self._feasible_point(est_state, state_0, gains, init_us)

        ref_states = jinx.envs.rollout_input(self.model_fn, state_0, init_us)
        _, (_, _, init_cost) = self._loss_fn(est_state, state_0, ref_states, gains, 1, init_us)

        init_step = OptimStep(
            us=init_us,
            barrier_eta=jnp.maximum(self.barrier_eta_start, self.barrier_eps),
            gains=gains,
            cost=init_cost,
            est_state=est_state,
            opt_state=self.opt_transform.init(init_us),

            grad_norm=jnp.array(0.),
            iteration=0,
            done=False
        )
        scan_fn = partial(self._opt_iteration, state_0)
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