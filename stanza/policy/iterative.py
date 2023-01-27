import jax.numpy as jnp
import optax
import jax
import sys

import stanza.envs

from typing import NamedTuple, Any
from functools import partial

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
    gains: jnp.array

    grad_norm: jnp.array

    cost: jnp.array
    est_state: Any
    opt_state: Any

    done: bool

DEBUG = False

# An MPC with feedback gains, but no barrier functions
class FeedbackMPC:
    def __init__(self,
                x_sample, u_sample,
                cost_fn, model_fn,
                horizon_length=20,
                optimizer=optax.adam(0.01),
                iterations=10000,
                use_gains=False,
                # for the gains computation
                burn_in=10,
                Q_coef=1,
                R_coef=1,

                receed=True,
                grad_estimator=None):
        self.x_sample = x_sample
        self.u_sample = u_sample

        self.model_fn = stanza.envs.flatten_model(model_fn, x_sample, u_sample)
        self.cost_fn = stanza.envs.flatten_cost(cost_fn, x_sample, u_sample)

        self.horizon_length = horizon_length

        self.optimizer = optimizer
        self.iterations = iterations

        self.use_gains = use_gains
        self.burn_in = burn_in
        self.Q_coef = Q_coef
        self.R_coef = R_coef

        self.receed = receed
        self.grad_estimator = grad_estimator

    def init_state(self, x0):
        # make everyting x_vec, u_vec based
        x_vec, _ = jax.flatten_util.ravel_pytree(x0)
        u_vec, _ = jax.flatten_util.ravel_pytree(self.u_sample)
        x_dim = x_vec.shape[-1]
        u_dim = u_vec.shape[-1]

        us = jnp.zeros((self.horizon_length - 1, u_dim))
        gains = jnp.zeros((self.horizon_length - 1, u_dim, x_dim))
        est_state = self.grad_estimator.init() if self.grad_estimator is not None else None
        final_step, history = self._solve(est_state, x_vec, gains, us)
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
        # Q = 0.001*jnp.eye(jac.shape[-2])
        # R = 100*jnp.eye(jac.shape[-1])
        Q = self.Q_coef*jnp.eye(jac.shape[-2])
        R = self.R_coef*jnp.eye(jac.shape[-1])
        def gains_recurse(P_next, AB):
            A, B = AB
            M = R + B.T @ P_next @ B

            N = A.T @ P_next @ B
            F = jnp.linalg.inv(M) @ N.T
            P = A.T @ P_next @ A - N @ F + Q

            P_scale = 1/(1 + jnp.linalg.norm(P, ord='fro')/100)
            P = P * P_scale
            return P, (F, P)

        _, (gains_est, Ps) = jax.lax.scan(gains_recurse, Q, (As_est, Bs_est), reverse=True)
        gains_est = -gains_est
        new_gains = prev_gains.at[self.burn_in:].set(gains_est)

        def print_fun(args, _):
            prev_gains, new_gains, As_cl, As_est, Bs_est, C_k, Ps, jac = args
            # if jnp.any(jnp.isnan(new_gains)) \
            #         or not jnp.all(jnp.isfinite(new_gains)):
            if True:
                s = jnp.linalg.svd(C_k, compute_uv=False)
                sv = lambda s: jnp.max(jnp.linalg.svd(s, compute_uv=False))
                eig = lambda s: jnp.max(jnp.linalg.eig(s)[0])

                As_cl_sv = jnp.linalg.svd(As_cl, compute_uv=False)
                As_sv = jnp.linalg.svd(As_est, compute_uv=False)

                max_eig = jnp.mean(jax.vmap(eig)(As_cl))
                # if max_eig > 1:
                if True:
                    print('gain_sv', jax.vmap(sv)(new_gains))
                    print('As_cl_eig', jax.vmap(eig)(As_cl))
                    print('As_eig', jax.vmap(eig)(As_est))
                    print('As_cl_sv', jax.vmap(sv)(As_cl))
                    print('As_sv', jax.vmap(sv)(As_est))
                    print('Ps_sv', jax.vmap(sv)(Ps))
                    print('s_diff_full', As_cl_sv - As_sv)
                    # sys.exit(0)
        if DEBUG:
            As_cl = As_est + Bs_est @ gains_est
            jax.experimental.host_callback.id_tap(print_fun, (prev_gains, new_gains, \
                As_cl, As_est, Bs_est, C_k, Ps, jac))

        #new_gains = prev_gains

        return new_gains

    
    def _loss_fn(self, est_state, x0,
                ref_xs, ref_gains, us):
        rollout = partial(ode.envs.rollout_input_gains, self.model_fn, x0, ref_xs, ref_gains)
        xs = rollout(us)

        def print_func(arg, _):
            x0, xs, us, gains = arg
            print('---- Rolling out Trajectories ------')
            if jnp.any(jnp.isnan(us)) or not jnp.all(jnp.isfinite(us)):
                print('gains', gains)
                print('xs', xs)
                print('us', us)
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('xs_nan', jnp.any(jnp.isnan(xs)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, xs, us, ref_gains))

        # for use with gradient estimation
        true_jac = jax.jacrev(rollout)(us)
        true_jac = jnp.transpose(true_jac, (2, 0, 1, 3))
        if self.grad_estimator:
            est_state, jac, xs = self.grad_estimator.inject_gradient(
                est_state, self.model_fn, xs, ref_gains, us, true_jac
            )
        else:
            jac = jax.jacrev(rollout)(us)
            jac = jnp.transpose(jac, (2, 0, 1, 3))

        # we need to modify the us to include the gains
        mod = ref_gains @ jnp.expand_dims(xs[:-1] - ref_xs[:-1], -1)
        us = us + jnp.squeeze(mod, -1)
        cost = stanza.envs.trajectory_cost(self.cost_fn, xs, us)

        def print_func(arg, _):
            x0, cost, xs, us, gains = arg
            print('---- Post-loss ------')
            print('cost', cost)
            if jnp.any(jnp.isnan(us)) or not jnp.all(jnp.isfinite(us)) \
                    or jnp.any(jnp.isnan(cost)) or not jnp.all(jnp.isfinite(cost)):
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('xs_nan', jnp.any(jnp.isnan(xs)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, cost, xs, us, ref_gains))
        return cost, (est_state, jac, cost)

    def _inner_step(self, x0, prev_step):
        gains = prev_step.gains
        ref_states = stanza.envs.rollout_input(self.model_fn, x0, prev_step.us)

        loss = partial(self._loss_fn, prev_step.est_state, x0,
                        ref_states, gains)

        def print_func(arg, _):
            x0, xs, us, gains = arg
            print('----  Pre-Loss ------')
            if jnp.any(jnp.isnan(xs)):
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('xs_nan', jnp.any(jnp.isnan(xs)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, ref_states, prev_step.us, gains))

        grad, (est_state, jac, cost) = jax.grad(loss, has_aux=True)(
            prev_step.us
        )
        updates, opt_state = self.optimizer.update(grad, prev_step.opt_state, prev_step.us)
        us = optax.apply_updates(prev_step.us, updates)

        def print_func(arg, _):
            x0, grad, xs, us, gains = arg
            print('---- Applied Updates ------')
            if jnp.any(jnp.isnan(us)):
                print('grad_nan', jnp.any(jnp.isnan(grad)))
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('xs_nan', jnp.any(jnp.isnan(xs)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)), jnp.all(jnp.isfinite(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, grad, ref_states, us, gains))

        # TODO: Help! I have put this in for now
        # but hopefully there is a better way?
        us = jax.lax.cond(jnp.any(jnp.isnan(us)), lambda: prev_step.us, lambda: us)

        if self.use_gains:
            # rollout new trajectory under old gains
            states_new = stanza.envs.rollout_input_gains(self.model_fn, x0, ref_states, gains, us)
            # adjust the us to include the gain-adjustments
            mod = gains @ jnp.expand_dims(states_new[:-1] - ref_states[:-1], -1)
            us = us + jnp.squeeze(mod, -1)
            # compute new gains around the adjusted trajectory
            gains = self._compute_gains(jac, gains)

        # clamp the us
        us_norm = jax.vmap(lambda u: jnp.maximum(jnp.linalg.norm(u)/50, jnp.array(1.)))(us)
        us = us / jnp.expand_dims(us_norm, -1)

        def print_func(arg, _):
            x0, xs, us, gains = arg
            print('---- Clipped ------')
            if jnp.any(jnp.isnan(us)):
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('xs_nan', jnp.any(jnp.isnan(xs)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, ref_states, us, gains))
        
        new_step = OptimStep(
            us=us,
            cost=cost,
            gains=gains,
            est_state=est_state,
            opt_state=opt_state,

            grad_norm=jnp.linalg.norm(grad),
            done=False,
            iteration=prev_step.iteration + 1
        )
        return new_step
    
    # body_fun for the solver interation
    def _opt_iteration(self, x0, prev_step, _):
        def print_func(arg, _):
            x0, us, gains = arg
            print('---- pre_opt_iteration ------')
            if jnp.any(jnp.isnan(us)):
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, prev_step.us, prev_step.gains))

        new_step = jax.lax.cond(
            prev_step.done,
            lambda: prev_step, 
            lambda: self._inner_step(x0, prev_step))

        def print_func(arg, _):
            x0, done, us, gains = arg
            print('---- post_opt_iteration ------')
            if jnp.any(jnp.isnan(us)):
                print('done', done)
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, prev_step.done, new_step.us, new_step.gains))
        return new_step, prev_step

    # A modified version of the JaxOPT base IterativeSolver
    # which propagates the estimator state
    def _solve(self, est_state, x0, gains, init_us):
        # if we have a barrier function first find a feasible state
        ref_states = stanza.envs.rollout_input(self.model_fn, x0, init_us)
        _, (_, _, init_cost) = self._loss_fn(est_state, x0, ref_states, gains, init_us)

        init_step = OptimStep(
            us=init_us,
            gains=gains,
            cost=init_cost,
            est_state=est_state,
            opt_state=self.optimizer.init(init_us),

            grad_norm=jnp.array(0.),
            iteration=0,
            done=False
        )
        scan_fn = partial(self._opt_iteration, x0)

        def print_func(arg, _):
            x0, us, gains = arg
            print('---- starting_iteration ------')
            if jnp.any(jnp.isnan(us)):
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, init_step.us, init_step.gains))

        final_step, history = jax.lax.scan(scan_fn, init_step, None, length=self.iterations)

        def print_func(arg, _):
            x0, us, gains = arg
            print('---- starting_iteration ------')
            if jnp.any(jnp.isnan(us)):
                print('x0_nan', jnp.any(jnp.isnan(x0)))
                print('us_nan', jnp.any(jnp.isnan(us)))
                print('gains_nan', jnp.any(jnp.isnan(gains)))
                sys.exit(0)
        if DEBUG:
            jax.experimental.host_callback.id_tap(print_func, (x0, final_step.us, final_step.gains))

        history = stanza.util.tree_append(history, final_step)
        return final_step, history

    @partial(jax.jit, static_argnums=(0,))
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

            x0, _ = jax.flatten_util.ravel_pytree(state)
            final_step, history = self._solve(est_state, x0, gains, us)
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