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

@dataclass(jax=True)
class EstState:
    rng_key: jax.random.PRNGKey
    ref_old_states: jnp.array
    ref_old_gains: jnp.array
    ref_new_states: jnp.array
    ref_new_gains: jnp.array
    total_samples: int

@dataclass(jax=True)
class IsingEstimator:
    rng_key: PRNGKey
    sigma: float = 0
    num_samples: int = field(jax_static=True, default=50)

    def __call__(self, func):
        @wraps(func)
        def wrapped(est_state, *args):
            args_flat, args_uf = jax.flatten_util.ravel_pytree(*args)
            def unflat_func(args_flat):
                args = args_uf(args_flat)
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
            X = jsp.linalg.solve(M + 0.00001*jnp.eye(M.shape[0]), B, assume_a='pos')
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