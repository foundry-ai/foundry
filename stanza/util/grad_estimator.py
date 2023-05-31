import sys
import optax
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import jax.scipy as jsp

import stanza.envs
import stanza.util.random
import stanza.policies

from stanza.util.dataclasses import dataclass, field
from stanza.util import vmap_ravel_pytree

from jax.random import PRNGKey

from functools import partial, wraps
from typing import Callable


def use_estimator(estimator, fun):
    pass

@dataclass(jax=True)
class EstState:
    rng_key: jax.random.PRNGKey
    total_samples: int

@jax.custom_vjp
def _inject_gradient(out, input, jac):
    return out

def _inject_gradient_fwd(xs, us, jac):
    return xs, jac

def _inject_gradient_bkw(res, g):
    jac = res
    jac_T = jnp.transpose(jac, (0,1,3,2))
    # (traj_dim, traj_dim, u_dim, x_dim) @ (1, traj_dim, x_dim, 1)
    grad = jac_T @ jnp.expand_dims(jnp.expand_dims(g, -1),0)
    # grad: (traj_dim, traj_dim, u_dim, 1)
    # sum over columns to combine all transitions for a given time
    grad = jnp.sum(jnp.squeeze(grad,-1), 1)
    return (None, grad, None)

_inject_gradient.defvjp(_inject_gradient_fwd, _inject_gradient_bkw)

_flatten = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])

@dataclass(jax=True)
class IsingEstimator:
    rng_key: PRNGKey
    sigma: float = 0
    num_samples: int = field(jax_static=True, default=50)

    def __call__(self, func):
        @wraps(func)
        def wrapped(est_state, us):
            xs = func(us)

            x0 = jax.tree_map(lambda x: x[0], xs)
            u0 = jax.tree_map(lambda x: x[0], us)
            _, x_uf = jax.flatten_util.ravel_pytree(x0)
            _, u_uf = jax.flatten_util.ravel_pytree(u0)

            us = _flatten(us)

            def flat_func(us_flat):
                us = jax.vmap(u_uf)(us_flat)
                xs = func(us)
                return _flatten(xs)

            xs = flat_func(us)
            if est_state is None:
                est_state = EstState(self.rng_key, 0)
            rng_key, rng = jax.random.split(est_state.rng_key)
            W = self.sigma*jax.random.choice(rng, jnp.array([-1,1]), (self.num_samples,) + us.shape)
            perturb_xs = jax.vmap(flat_func)(us + W)
            x_diff = perturb_xs - xs
            jac = self.calculate_jacobians(W, x_diff)
            xs = _inject_gradient(xs, us, jac)
            return EstState(rng_key, est_state.total_samples + self.num_samples), \
                jac, jax.vmap(x_uf)(xs)
        return wrapped

    # true_jac is for debugging
    def calculate_jacobians(self, W, x_diff, true_jac=None):
        # W: (samples, traj_dim-1, u_dim)
        # x_diff: (samples, traj_dim, x_dim)

        T = W.shape[1] + 1
        x_dim = x_diff.shape[-1]
        u_dim = W.shape[2]
        # W: (samples, traj_dim-1, u_dim)
        W = jnp.reshape(W, (W.shape[0], -1))
        # W: (samples, traj_dim-1 * u_dim)
        W = jnp.expand_dims(W, -1)
        # W: (samples, traj_dim-1 * u_dim, 1)
        W_T = jnp.transpose(W, (0, 2, 1))
        # W_T: (samples, 1, traj_dim-1 * u_dim)

        W_W = W @ W_T
        # W_W: (samples, traj_dim-1 * u_dim, traj_dim-1 * u_dim)
        W_W = jnp.sum(W_W, 0)
        # W_W: (traj_dim-1 * u_dim, traj_dim-1 * u_dim)
        x_diff = jnp.expand_dims(x_diff, -1)
        # x_diff (samples, traj_dim, x_dim, 1)

        jac = jnp.zeros((T-1, T, x_dim, u_dim))
        for t in range(1, T):
            M = W_W[:t*u_dim,:t*u_dim]
            # M (t*u_dim, t*u_dim)
            B = x_diff[:, t, ...] @ W_T[:, :, :t*u_dim]
            B = jnp.sum(B, 0).T
            # B (t*u_dim, x_dim)
            X = jsp.linalg.solve(M + 0.0000001*jnp.eye(M.shape[0]), B, assume_a='pos')
            # X (t*u_dim, x_dim)
            X = X.reshape((t, u_dim, x_dim))
            X = jnp.transpose(X, (0, 2, 1))
            jac = jac.at[:t,t,...].set(X)

            def print_func(arg, _):
                X, B, M, t = arg
                if jnp.any(jnp.isnan(X)) or jnp.any(jnp.isnan(B)) or jnp.any(jnp.isnan(M)):
                    print('t', t)
                    print('X', X)
                    print('B', B)
                    print('M', M)
                    print('s', jnp.linalg.cond(M))
                    import pdb
                    pdb.set_trace()
                    sys.exit(0)
            # jax.experimental.host_callback.id_tap(print_func, (X, B, M,t))
        def print_func(arg, _):
            jac, true_jac, W, x_diff = arg
            if jnp.any(jnp.isnan(jac)):
                print('---- Computing Jacobian ------')
                print(jac[1,9,...])
                print(true_jac[1,9,...])
                print(jnp.max(jnp.abs(jac - true_jac)))
                sys.exit(0)
        # jax.experimental.host_callback.id_tap(print_func, (jac, true_jac, W, x_diff))
        return jac