# Adapted from https://github.com/google/spectral-density
# Used under the Apache License 2.0

from typing import Callable
from functools import partial

import math
import jax
import jax.flatten_util
import jax.numpy as jnp

def net_hvp(loss, params, v):
    return jax.jvp(jax.grad(loss), [params], [v])[1]

@partial(jax.jit, static_argnums=(1,))
def net_sharpness_statistics(rng_key, loss, params, samples=5):
    params_flat, uf = jax.flatten_util.ravel_pytree(params)
    loss_flat = lambda p: loss(uf(p))
    def net_hvp(v):
        return jax.jvp(
            jax.grad(loss_flat), [params_flat], [v]
        )[1]
    order = min(params_flat.shape[0], 16)
    tridiags, _ = jax.vmap(lanczos_alg, in_axes=(0, None, None, None))(
        jax.random.split(rng_key, samples), net_hvp, params_flat.shape[0], order
    )
    eig_vals, all_weights = tridiag_to_eigv(tridiags)
    q = jnp.array([5, 25, 50, 75, 95])
    p = jnp.percentile(eig_vals, q, axis=0)
    return {
        f'lambda_percentile_{q[i]}': p[i] for i in range(len(q))
    }

@partial(jax.jit, static_argnums=(1,2,3))
def lanczos_density(rng_key, hvp, dim, order, samples=5, grid_len=10000):
    tridiags, _ = jax.vmap(lanczos_alg, in_axes=(0, None, None, None))(
        jax.random.split(rng_key, samples), hvp, dim, order
    )
    return tridiag_to_density(tridiags, grid_len=grid_len)

@partial(jax.jit, static_argnums=(1,2,3))
def lanczos_alg(rng_key, matrix_vector_product, dim, order):
    tridiag = jnp.zeros((order, order))
    w_prev = jnp.zeros((dim,))
    vecs = jnp.zeros((order, dim))
    init_vec = jax.random.normal(rng_key, shape=(dim,))
    init_vec = init_vec / jnp.linalg.norm(init_vec)
    vecs = vecs.at[0].set(init_vec)

    def step(i, carry):
        _, w_prev, tridiag, vecs = carry
        v = vecs[i, :]
        w = matrix_vector_product(v)
        w = w - w_prev
        alpha = jnp.dot(w, v)
        w = w - alpha * v
        def orth_step(j, w):
            tau = vecs[j, :]
            coeff = jnp.dot(w, tau)
            w = w - coeff * tau
            return w
        w = jax.lax.fori_loop(0, i, orth_step, w)
        beta = jnp.linalg.norm(w)
        tridiag = tridiag.at[i,i].set(alpha)
        tridiag = tridiag.at[i, i+1].set(beta)
        tridiag = tridiag.at[i+1,i].set(beta)
        vecs = vecs.at[i+1].set(w/beta)
        return alpha, w, tridiag, vecs
    alpha, _, tridiag, vecs = jax.lax.fori_loop(
        0, order, step, (0., w_prev, tridiag, vecs)
    )
    # set the final alpha
    tridiag = tridiag.at[order-1, order-1].set(alpha)
    return (tridiag, vecs)

def tridiag_to_density(tridiag_list, sigma_squared=1e-5, grid_len=10000):
    eig_vals, all_weights = tridiag_to_eigv(tridiag_list)
    density, grids = eigv_to_density(eig_vals, jnp.log(all_weights),
                                    grid_len=grid_len,
                                    sigma_squared=sigma_squared)
    eig_vals = eig_vals.reshape((-1,))
    return density, grids, eig_vals

def _log_kernel(x, mean, variance, log_weight=None):
    val = -(mean - x) ** 2
    val = val / (2.0 * variance)
    if log_weight is not None:
        val = val + log_weight
    return jax.lax.psum(val, axis_name="kernel_sum")

def eigv_to_density(eig_vals, log_weights=None, grids=None,
                    grid_len=10000, sigma_squared=None, grid_expand=1e-2):
    if log_weights is None:
        log_weights = jnp.zeros(eig_vals.shape)

    lambda_max = jnp.nanmean(jnp.max(eig_vals, axis=1), axis=0) + grid_expand
    lambda_min = jnp.nanmean(jnp.min(eig_vals, axis=1), axis=0) - grid_expand
    eig_vals = eig_vals.reshape((-1,))
    log_weights = log_weights.reshape((-1,))

    sigma_squared = sigma_squared if sigma_squared is not None else 1e-5
    if grids is None:
        assert grid_len is not None, 'grid_len is required if grids is None.'
        grids = jnp.linspace(lambda_min, lambda_max, num=grid_len)
    # compute the width of the grid buckets
    grids_prev = jnp.roll(grids, 1).at[0].set(grids[0] - grids[1])
    log_grid_spacing = jnp.log(grids - grids_prev)


    grid_len = grids.shape[0]
    sigma_squared = sigma_squared * jnp.maximum(1, (lambda_max - lambda_min))

    log_density = jax.vmap(jax.vmap(
            _log_kernel, 
            in_axes=(None, 0, None, 0), axis_name="kernel_sum",
            out_axes=None
        ), in_axes=(0,None,None, None)
    )(grids, eig_vals, sigma_squared, log_weights)
    # normalize the bucket probabilities to 1
    log_norm = jax.scipy.special.logsumexp(log_density)
    log_density = log_density - log_norm
    # this is the *bucket* density,
    # multiply by the grid spacing to get the cdf density
    log_density = log_density + log_grid_spacing
    return log_density, grids


@jax.jit
def tridiag_to_eigv(tridiag_list):
    """Preprocess the tridiagonal matrices for density estimation.

    Args:
        tridiag_list: Array of shape [num_draws, order, order] List of the
        tridiagonal matrices computed from running num_draws independent runs
        of lanczos. The output of this function can be fed directly into
        eigv_to_density.

    Returns:
        eig_vals: Array of shape [num_draws, order]. The eigenvalues of the
        tridiagonal matricies.
        all_weights: Array of shape [num_draws, order]. The weights associated with
        each eigenvalue. These weights are to be used in the kernel density
        estimate.
    """
    # Calculating the node / weights from Jacobi matrices.
    def process(tridiag):
        nodes, evecs = jnp.linalg.eigh(tridiag)
        index = jnp.argsort(nodes)
        nodes = nodes[index]
        evecs = evecs[:, index]
        return nodes, evecs[0]**2
    return jax.vmap(process)(tridiag_list)
