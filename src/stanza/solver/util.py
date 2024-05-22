import functools
import jax
import jax.numpy as jnp

import stanza
import stanza.transform.lift as lift
from . import SolverResult
from stanza import struct

def implicit_diff_solve(solve, assume_psd=False):
    @functools.wraps(solve)
    @stanza.jit
    def diff_solve(objective, init_state):
        optimality_fun = lift.static_lower(objective.optimality)
        objective = lift.Static.static_wrap(objective)
        init_state = lift.Static.static_wrap(init_state)
        optimality_fun, optimality_args = jax.closure_convert(
            optimality_fun, init_state)
        solve_fun = lift.static_lower(solve)
        solve_fun, solve_args = jax.closure_convert(
            solve_fun, objective, init_state)

        theta_flat, theta_uf = jax.flatten_util.ravel_pytree(optimality_args)
        _, x_uf = jax.flatten_util.ravel_pytree(
            objective.extract_params(init_state)
        )

        def vec_optimality_fun(objective, solution, theta_flat, x_flat):
            optimality_args = theta_uf(theta_flat)
            state = objective.replace_params(solution, x_uf(x_flat))
            opt = optimality_fun(state, *optimality_args)
            return jax.flatten_util.ravel_pytree(opt)[0].astype(jnp.float32)

        @jax.custom_jvp
        def optimality_solve(objective, init_state, solve_args, theta_flat):
            res = solve_fun(objective, init_state, *solve_args)
            x = objective.extract_params(res.solution)
            x_flat, _ = jax.flatten_util.ravel_pytree(x)
            return res, x_flat

        def optimality_solve_jvp(primals, tangents):
            from jax._src.custom_derivatives import _zeros_like_pytree
            objective, init_state, solve_args, theta_flat = primals
            _, _, _, dtheta = tangents
            res = solve_fun(objective, init_state, *solve_args)
            x_flat, _ = jax.flatten_util.ravel_pytree(
                objective.extract_params(res.solution)
            )
            out_primals = res, x_flat

            dfdtheta, dfdx = jax.jacobian(vec_optimality_fun, argnums=(2,3))(
                objective, res.solution, theta_flat, x_flat)
            # we have f(theta, x) = 0
            # solving df/dtheta * dtheta + df/dx * dx = 0
            # we have dx/dtheta = -(df/dx)^(-1) df/dtheta
            # equivalently we are solving
            # df/dx * dx' = -df/dtheta * dtheta
            # jax.debug.print('dfdx {}', dfdx)
            # jax.debug.print('dfdtheta {}', dfdtheta)
            # A = dfdx
            # b = -(dfdtheta @ dtheta)
            if False:
                a = jnp.linalg.norm(dfdtheta, 2)
                dfdx = dfdx / a
                dfdtheta = dfdtheta / a
                dx = jax.scipy.linalg.solve(dfdx + 1e-2*jnp.eye(dfdx.shape[0]), -dfdtheta @ dtheta)
            else:
                U1, S1, V1T = jax.scipy.linalg.svd(dfdx, full_matrices=False)
                U2, S2, V2T = jax.scipy.linalg.svd(-dfdtheta, full_matrices=False)
                # take the pseudo-inverse of dfdx
                s_max = S2[0]
                S1, S2 = S1 / s_max, S2 / s_max
                # S1_inv = jnp.diag(1.0 / S1)
                # S1_inv = jnp.diag(S1 / (S1**2 + 1e-8))
                S1_inv = jnp.diag(jnp.where(S1 > 1e-5, 1.0 / S1, 0.0))
                S2_hat = (U1.T @ U2) @ jnp.diag(S2)
                # print(S1_inv.shape, S2_hat.shape, V2T.shape, V1T.shape)
                dxdtheta = V1T.T @ ((S1_inv @ S2_hat) @ V2T)
                dx = dxdtheta @ dtheta
            dres = jax.tree_util.tree_map(jnp.zeros_like, res)
            out_tangents = dres, dx
            return out_primals, out_tangents
        optimality_solve.defjvp(optimality_solve_jvp, symbolic_zeros=True)

        # make non-jaxtypes static
        res, x_flat = optimality_solve(objective, init_state, solve_args, theta_flat)
        res = struct.replace(res, solution=objective.replace_params(res.solution, x_uf(x_flat)))
        return res
    return diff_solve