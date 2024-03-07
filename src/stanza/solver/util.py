import functools
import jax
import jax.numpy as jnp

import chex
import stanza.transform
from . import SolverResult
from stanza import struct

def implicit_diff_solve(solve):
    @functools.wraps(solve)
    @stanza.jit
    def diff_solve(objective, init_state):
        optimality_fun = stanza.transform.jaxify_function(objective.optimality)
        objective = stanza.transform.jaxify_pytree(objective)
        init_state = stanza.transform.jaxify_pytree(init_state)
        optimality_fun, optimality_args = jax.closure_convert(
            optimality_fun, init_state)

        solve_fun = stanza.transform.jaxify_function(solve)
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
            dx = jax.scipy.linalg.solve(dfdx, - dfdtheta @ dtheta)
            dres = jax.tree_map(jnp.zeros_like, res)
            out_tangents = dres, dx
            return out_primals, out_tangents
        optimality_solve.defjvp(optimality_solve_jvp, symbolic_zeros=True)

        # make non-jaxtypes static
        res, x_flat = optimality_solve(objective, init_state, solve_args, theta_flat)
        res = struct.replace(res, solution=objective.replace_params(res.solution, x_uf(x_flat)))
        return res
    return diff_solve