# Adapted from https://github.com/google/spectral-density
# Used under the Apache License 2.0

from typing import Callable
from functools import partial

import math
import jax
import jax.flatten_util
import foundry.numpy as jnp


# Takes in a loss
def net_batch_hvp(loss, data, batch_size, params):
    N = jax.tree_util.tree_flatten(data)[0][0].shape[0]
    # if N is smaller than batch_size, use that
    batch_size = min(N, batch_size)
    batches = N // batch_size
    # remove the remainder to make the data batch_size
    remainder = N  - batches * batch_size
    if remainder > 0:
        data = jax.tree.map(
            lambda x: x[:-remainder],
            data
        )
    data = jax.tree.map(
        lambda x: x.reshape((-1, batch_size) + x.shape[1:]), 
        data
    )
    params_flat, uf = jax.flatten_util.ravel_pytree(params)
    def hvp(v):
        def batch_loss(batch, params_flat):
            return jnp.mean(jax.vmap(loss, in_axes=(None, 0))(uf(params_flat), batch))
        def hvp_scan(total, batch):
            loss_flat = partial(batch_loss, batch)
            def sub_hvp(v):
                return jax.jvp(
                    jax.grad(loss_flat), [params_flat], [v]
                )[1]
            w = sub_hvp(v)
            total = total + w
            return total, None
        total , _ = jax.lax.scan(hvp_scan, jnp.zeros_like(params_flat), data)
        w = total / batches
        return w
    return hvp

def net_hvp(loss, params):
    params_flat, uf = jax.flatten_util.ravel_pytree(params)
    loss_flat = lambda p: loss(uf(p))
    def hvp(v):
        return jax.jvp(
            jax.grad(loss_flat), [params_flat], [v]
        )[1]
    return hvp

@partial(jax.jit, static_argnums=(1,3,4))
def net_sharpness_statistics(rng_key, hvp_at, params, order=8, samples=2):
    params_flat, _ = jax.flatten_util.ravel_pytree(params)
    order = min(params_flat.shape[0], order)
    hvp = hvp_at(params)
    tridiags, _ = lanczos_alg(rng_key, hvp, params_flat.shape[0], order, params_flat.dtype)
    tridiags = jnp.expand_dims(tridiags, axis=0)
    # alg = lambda rng_key: lanczos_alg(rng_key, net_hvp, params_flat.shape[0], order)
    # tridiags, _ = jax.lax.map(alg, jax.random.split(rng_key, samples))
    # tridiags, _ = jax.vmap(alg)(jax.random.split(rng_key, samples))

    eig_vals, _ = tridiag_to_eigv(tridiags)
    eig_vals = eig_vals.reshape((-1,))
    sigmas = jnp.abs(eig_vals)
    q = [5, 25, 50, 75, 95]
    lam_p = jnp.percentile(eig_vals, jnp.array(q), axis=0)
    sigma_p = jnp.percentile(sigmas, jnp.array(q), axis=0)
    d = {
        f'lambda_percentile_{q[i]:02}': lam_p[i] for i in range(len(q))
    }
    d.update({
        f'sigma_percentile_{q[i]:02}': sigma_p[i] for i in range(len(q))
    })
    d["lambda_trace"] = jnp.mean(eig_vals) * params_flat.shape[0]
    d["sigma_trace"] = jnp.mean(sigmas) * params_flat.shape[0]
    return d

@partial(jax.jit, static_argnums=(1,2,3,4,5,6))
def lanczos_density(rng_key, hvp, dim, order, dtype=jnp.float32, samples=5, grid_len=10000):
    tridiags, _ = jax.vmap(lanczos_alg, in_axes=(0, None, None, None, None))(
        jax.random.split(rng_key, samples), hvp, dim, order, dtype
    )
    return tridiag_to_density(tridiags, grid_len=grid_len)

@partial(jax.jit, static_argnums=(1,2,3,4))
def lanczos_alg(rng_key, matrix_vector_product, dim, order, dtype):
    tridiag = jnp.zeros((order, order), dtype)
    w_prev = jnp.zeros((dim,), dtype=dtype)
    vecs = jnp.zeros((order, dim), dtype=dtype)
    init_vec = jax.random.normal(rng_key, shape=(dim,), dtype=dtype)
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
    # Calculating the node / weights from Jacobi matrices.
    def process(tridiag):
        nodes, evecs = jnp.linalg.eigh(tridiag)
        index = jnp.argsort(nodes)
        nodes = nodes[index]
        evecs = evecs[:, index]
        return nodes, evecs[0]**2
    return jax.vmap(process)(tridiag_list)
