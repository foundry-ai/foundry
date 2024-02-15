import functools
import jax
import jax.numpy as jnp

from stanza.struct import replace

def implicit_diff_solve(solve_fun):
    @functools.wraps(solve_fun)
    def solve(objective, init_state):
        optimality_fn, optimality_args = jax.closure_convert(
            objective.optimality, init_state)
        optimality_args_flat, optimality_args_uf = \
                jax.flatten_util.ravel_pytree(optimality_args)

        res = solve_fun(objective, init_state)
        res = jax.lax.stop_gradient(res)

        @jax.custom_jvp
        def optimality_solve(res, x_flat):
            theta = objective.extract_params(res.solution)
            theta_flat, _ = jax.flatten_util.ravel_pytree(theta)
            return theta_flat

        def optimality_solve_jvp(dx, primal_out, res, x_flat):
            theta = objective.extract_params(res.solution)
            theta_flat, theta_uf = jax.flatten_util.ravel_pytree(theta)

            def vec_optimality_fn(theta_flat, x_flat):
                optimality_args = optimality_args_uf(x_flat)
                theta = theta_uf(theta_flat)
                state = objective.replace_params(res.solution, theta)
                opt = optimality_fn(state, *optimality_args)
                return jax.flatten_util.ravel_pytree(opt)[0]
            dfdtheta, dfdx = jax.jacobian(vec_optimality_fn, argnums=(0,1))(
                theta_flat, x_flat)
            # we have f(theta, x) = 0
            # solving df/dtheta * dtheta + df/dx * dx = 0
            # we have dtheta/dx = -(df/dtheta)^(-1) df/dx
            # equivalently we are solving
            # df/dtheta * dtheta' = -df/dx * dx
            tangent = jnp.linalg.solve(dfdtheta, - dfdx @ dx)
            return tangent
        optimality_solve.defjvps(None, optimality_solve_jvp)

        theta_flat = optimality_solve(res, optimality_args_flat)
        _, theta_uf = jax.flatten_util.ravel_pytree(objective.extract_params(res.solution))
        theta = theta_uf(theta_flat)
        solution = objective.replace_params(res.solution, theta)
        res = replace(res, solution=solution)
        return res
    return solve