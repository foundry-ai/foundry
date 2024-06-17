from typing import Any

from stanza.dataclasses import dataclass
from stanza.solver import (
    Solver, SolverResult, UnsupportedObectiveError
)
from stanza.solver.util import implicit_diff_solve

import trajax.optimizers

import stanza
import functools

import jax
import jax.numpy as jnp

@dataclass
class iLQRSolver(Solver):
    @staticmethod
    @functools.partial(stanza.jit, static_argnums=(0,))
    def _solve(history, objective, solver_state):
        from stanza.policy.mpc import MinimizeMPC, MinimizeMPCState
        if not isinstance(objective, MinimizeMPC):
            raise UnsupportedObectiveError("iLQR only supports MinimizeMPC objectives")

        params_flat, params_uf = jax.flatten_util.ravel_pytree(objective.initial_params)
        state0_flat, state_uf = jax.flatten_util.ravel_pytree(objective.state0)
        a0 = jax.tree_map(lambda x: x[0], solver_state.actions)
        _, action_uf = jax.flatten_util.ravel_pytree(a0)

        initial_actions_flat = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0]
        )(solver_state.actions)
        initial_params_flat = jnp.zeros((initial_actions_flat.shape[0],) + params_flat.shape)
        initial_params_flat = initial_params_flat.at[0].set(params_flat)
        T = initial_actions_flat.shape[0]


        # flatten everything
        def flat_model(state_param, action_param, t):
            state, state_param = state_param[:len(state0_flat)], state_param[len(state0_flat):]
            action, action_param = action_param[:len(a0)], action_param[len(a0):]

            state, action = state_uf(state), action_uf(action)
            state = objective.model_fn(state, action, None)
            state, _ = jax.flatten_util.ravel_pytree(state)
            param = jax.lax.cond(t == 0, lambda: action_param, lambda: state_param)
            return jnp.concatenate((state, param))

        def flat_cost(state_param, action_param, t):
            state, state_param = state_param[:len(state0_flat)], state_param[len(state0_flat):]
            action, action_param = action_param[:len(a0)], action_param[len(a0):]
            # if t == 0, use the action parameters, otherwise use the state parameters
            param = jax.lax.cond(t == 0, lambda: action_param, lambda: state_param)

            states = jnp.zeros((T+1,) + state0_flat.shape)
            actions = jnp.zeros((T,) + a0.shape)
            states = states.at[t].set(state)
            actions = jax.lax.cond(t > T, lambda: actions, lambda: actions.at[t].set(action))

            param = params_uf(param)
            states = jax.vmap(state_uf)(states)
            actions = jax.vmap(action_uf)(actions)
            _, c, _ = objective.cost(None, param, states, actions)
            return c

        # actions include "parameters" but in reality only the first parameter is considered
        initial_action_params_flat = jnp.concatenate((initial_actions_flat, initial_params_flat), axis=-1)
        state0_param_flat = jnp.concatenate((state0_flat, params_flat))
        _, actions_flat, cost, _, _, _, it = trajax.optimizers.ilqr(flat_cost, flat_model,
                state0_param_flat, initial_action_params_flat,
                make_psd=False, psd_delta=0.0,
                grad_norm_threshold=1e-4
        )
        params_flat = actions_flat[0, len(a0):]
        actions_flat = actions_flat[:, :len(a0)]
        params = params_uf(params_flat)
        actions = jax.vmap(action_uf)(actions_flat)
        res = MinimizeMPCState(it, jnp.array(True), params, actions, cost, None)
        return SolverResult(res, None)

    @functools.partial(stanza.jit, static_argnames=("implicit_diff", "history"))
    def run(self, objective, *, implicit_diff=True, history=False) -> SolverResult:
        from stanza.policy.mpc import MinimizeMPCState
        init_state = MinimizeMPCState(
            jnp.zeros(()), jnp.array(False), 
            objective.initial_params,
            objective.initial_actions,
            jnp.zeros(()),
            objective.initial_cost_state
        )
        solve = stanza.partial(self._solve, history)
        if implicit_diff:
            solve = implicit_diff_solve(solve)
        return solve(objective, init_state)

# def custom_linearize(fun):
#     jacobian_x = jax.jacobian(fun, argnums=0)
#     jacobian_u = jax.jacobian(fun, argnums=1)
#     def linearizer(X, U, *args):
#         return jacobian_x(*args), jacobian_u(*args)
#     # already vectorized!
#     return linearizer

# def custom_quadratize(fun):
#     def _fun(x, u, t, X, U, args):
#         X = X.at[t].set(x)
#         U = U.at[t].set(u)
#         return fun(X, U, *args)
#     hessian_x = jax.hessian(_fun)
#     hessian_x = jax.vmap(hessian_x, in_axes=(0, 0, 0, None, None, None))
#     hessian_u = jax.hessian(_fun, argnums=1)
#     hessian_u = jax.vmap(hessian_u, in_axes=(0, 0, 0, None, None, None))
#     hessian_x_u = jax.jacobian(jax.grad(_fun), argnums=1)
#     hessian_x_u = jax.vmap(hessian_x_u, in_axes=(0, 0, 0, None, None, None))

#     def hessian(X, U, *args):
#         T = jnp.arange(X.shape[0])
#         return hessian_x(X, U, T, X, U, args), hessian_u(X, U, T, X, U, args), hessian_x_u(X, U, T, X, U, args)
#     return hessian

# def ilqr(cost, dynamics, x0,
#          U, maxiter=100,
#          grad_norm_threshold=1e-4,
#          relative_grad_norm_threshold=0.0,
#          obj_step_threshold=0.0,
#          inputs_step_threshold=0.0,
#          make_psd=False, psd_delta=0.0,
#          alpha_0=1.0, alpha_min=0.00005,
#          vjp_method=None, vjp_options=None):
#     if vjp_options is None:
#         vjp_options = {}

#     X = jnp.zeros((U.shape[0] + 1,) + x0.shape)
#     cost_fn, cost_args = jax.custom_derivatives.closure_convert(cost, X, U)
#     dynamics_fn, dynamics_args = jax.custom_derivatives.closure_convert(dynamics, x0, U[0], 0)

#     def new_cost_fn(X, U, bundled_cost_args):
#         return cost_fn(X, U, *bundled_cost_args)
#     def new_dynamics_fn(x, u, t, bundled_dynamics_args):
#         return dynamics_fn(x, u, t, *bundled_dynamics_args)
#     if vjp_method is not None:
#         raise NotImplementedError("vjp_method not implemented")
#     return custom_ilqr_template(new_cost_fn, new_dynamics_fn, x0, U, 
#                             (tuple(cost_args),), (tuple(dynamics_args),),
#                             maxiter, grad_norm_threshold,
#                             relative_grad_norm_threshold, obj_step_threshold,
#                             inputs_step_threshold, make_psd, psd_delta, alpha_0,
#                             alpha_min, vjp_options)

# def custom_ilqr_template(cost, dynamics, x0, U, cost_args, dynamics_args, maxiter,
#                    grad_norm_threshold, relative_grad_norm_threshold,
#                    obj_step_threshold, inputs_step_threshold, make_psd,
#                    psd_delta, alpha_0, alpha_min, vjp_options):
#     T, _ = U.shape
#     n = x0.shape[0]

#     roll = functools.partial(trajax.optimizers._rollout, dynamics)
#     cost_gradients = custom_linearize(cost)
#     quadratizer = custom_quadratize(cost)
#     dynamics_jacobians = trajax.optimizers.linearize(dynamics)
#     total_cost = cost
#     psd = jax.vmap(functools.partial(trajax.optimizers.project_psd_cone, delta=psd_delta))
#     pad = trajax.optimizers.pad

#     X = roll(U, x0, *dynamics_args)
#     timesteps = jnp.arange(X.shape[0])
#     obj = total_cost(X, pad(U), *cost_args)

#     def get_lqr_params(X, U):
#         Q, R, M = quadratizer(X, pad(U), *cost_args)

#         Q = jax.lax.cond(make_psd, Q, psd, Q, lambda x: x)
#         R = jax.lax.cond(make_psd, R, psd, R, lambda x: x)

#         q, r = cost_gradients(X, pad(U), timesteps, X, pad(U), *cost_args)
#         A, B = dynamics_jacobians(X, pad(U), jnp.arange(T + 1), *dynamics_args)

#         return (Q, q, R, r, M, A, B)

#     c = jnp.zeros((T, n))  # assumes trajectory is always dynamically feasible.

#     lqr = get_lqr_params(X, U)
#     _, q, _, r, _, A, B = lqr
#     gradient, adjoints, _ = trajax.optimizers.adjoint(A, B, q, r)
#     grad_norm_initial = jnp.linalg.norm(gradient)
#     grad_norm_threshold = jnp.maximum(
#         grad_norm_threshold,
#         relative_grad_norm_threshold *
#         jnp.where(jnp.isnan(grad_norm_initial), 1.0, grad_norm_initial + 1.0))

#     def body(inputs):
#         """Solves LQR subproblem and returns updated trajectory."""
#         X, U, obj, alpha, gradient, adjoints, lqr, iteration, _, _ = inputs

#         Q, q, R, r, M, A, B = lqr

#         K, k, _, _ = trajax.optimizers.tvlqr(Q, q, R, r, M, A, B, c)
#         X_new, U_new, obj_new, alpha = trajax.optimizers.line_search_ddp(cost, dynamics, X, U, K, k,
#                                                         obj, cost_args,
#                                                         dynamics_args, alpha_0,
#                                                         alpha_min)
#         gradient, adjoints, _ = trajax.optimizers.adjoint(A, B, q, r)
#         # print("Iteration=%d, Objective=%f, Alpha=%f, Grad-norm=%f\n" %
#         #      (device_get(iteration), device_get(obj), device_get(alpha),
#         #       device_get(np.linalg.norm(gradient))))

#         lqr = get_lqr_params(X_new, U_new)
#         U_step = jnp.linalg.norm(U_new - U)
#         obj_step = jnp.abs(obj_new - obj)
#         iteration = iteration + 1
#         return X_new, U_new, obj_new, alpha, gradient, adjoints, lqr, iteration, obj_step, U_step

#     def continuation_criterion(inputs):
#         _, U_new, obj_new, alpha, gradient, _, _, iteration, obj_step, U_step = inputs
#         grad_norm = jnp.linalg.norm(gradient)
#         grad_norm = jnp.where(jnp.isnan(grad_norm), jnp.inf, grad_norm)

#         still_improving_obj = obj_step > obj_step_threshold * (
#             jnp.absolute(obj_new) + 1.0)
#         still_moving_U = U_step > inputs_step_threshold * (
#             jnp.linalg.norm(U_new) + 1.0)
#         still_progressing = jnp.logical_and(still_improving_obj, still_moving_U)
#         has_potential_to_improve = jnp.logical_and(grad_norm > grad_norm_threshold,
#                                                     still_progressing)

#         return jnp.logical_and(
#             iteration < maxiter,
#             jnp.logical_and(has_potential_to_improve, alpha > alpha_min))

#     X, U, obj, _, gradient, adjoints, lqr, it, _, _ = jax.lax.while_loop(
#         continuation_criterion, body,
#         (X, U, obj, alpha_0, gradient, adjoints, lqr, 0, jnp.inf, jnp.inf))

#     return X, U, obj, gradient, adjoints, lqr, it

# @functools.partial(jax.jit, static_argnums=(0, 1))
# def line_search_ddp(total_cost,
#                     dynamics,
#                     X,
#                     U,
#                     K,
#                     k,
#                     obj,
#                     cost_args=(),
#                     dynamics_args=(),
#                     alpha_0=1.0,
#                     alpha_min=0.00005):
#     """Performs line search with respect to DDP rollouts."""

#     obj = jnp.where(jnp.isnan(obj), jnp.inf, obj)

#     def line_search(inputs):
#         """Line search to find improved control sequence."""
#         _, _, _, alpha = inputs
#         Xnew, Unew = trajax.optimizers.ddp_rollout(dynamics, X, U, K, k, alpha, *dynamics_args)
#         obj_new = total_cost(Xnew, Unew, *cost_args)
#         alpha = 0.5 * alpha
#         obj_new = jnp.where(jnp.isnan(obj_new), obj, obj_new)

#         # Only return new trajs if leads to a strict cost decrease
#         X_return = jnp.where(obj_new < obj, Xnew, X)
#         U_return = jnp.where(obj_new < obj, Unew, U)

#         return X_return, U_return, jnp.minimum(obj_new, obj), alpha

#     return jax.lax.while_loop(
#         lambda inputs: jnp.logical_and(inputs[2] >= obj, inputs[3] > alpha_min),
#         line_search, (X, U, obj, alpha_0))