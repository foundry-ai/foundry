import jax
import optax
import functools

import foundry.core as F
import foundry.core.tree as tree

import foundry.util.pca as pca
import foundry.numpy as jnp
import foundry.train

from foundry.core.dataclasses import dataclass
from foundry.data import Data
from typing import TypeVar

T = TypeVar("T")

def kl_divergence(log_p, log_q, mask=None):
    C = jnp.exp(log_p) * (log_p - log_q)
    if mask is not None: C = C * mask
    return jnp.sum(C)

def euclidean_distance_sqr(x, y):
    return jnp.sum((x - y) ** 2, axis=-1)

def _log_q_probability(distances_sqr):
    log_Q = -jnp.log(1 + distances_sqr)
    normalization = jax.scipy.special.logsumexp(log_Q)
    log_Q = log_Q - normalization
    return log_Q

def cond_prob_and_entropy(N, distances_sqr, beta):
    log_Pi = -(distances_sqr * beta)
    log_norm = jax.scipy.special.logsumexp(log_Pi)
    log_Pi = log_Pi - log_norm
    H = -jnp.sum(jnp.exp(log_Pi) * log_Pi)
    return log_Pi, H

# def _log_joint_probability(distances, distances_mask, betas):
#     assert distances.ndim == 2
#     assert betas.ndim == 1
#     log_Pi = -distances * betas[:, None]
#     log_Pi = jnp.where(distances_mask, log_Pi, -jnp.inf)
#     # normalize the rows of the matrix to sum to 1
#     # and then normalize the matrix itself to sum to 1 by dividing by the number of rows
#     log_Pi = log_Pi - jax.scipy.special.logsumexp(log_Pi, axis=1)[:,None]
#     log_Pi = log_Pi - jnp.log(log_Pi.shape[0])
#     # symmetrize the matrix
#     log_Pi = jnp.logaddexp(log_Pi, log_Pi.T) - jnp.log(2)
#     # put the masked values to zero (rather than -inf since that would cause numerical issues)
#     log_Pi = jnp.where(distances_mask, log_Pi, 0.)
#     return log_Pi

def binary_search_beta(N, distances_sqr, perp_target, tol=1e-5, max_iter=200):
    # Do not include the i-th element itself in the perplexity calculation
    H_target = jnp.log(perp_target)

    beta_min = 0.0
    beta_max = jnp.inf
    beta = 1.0
    # jax.debug.print("H_target: {}", H_target)

    _, H = cond_prob_and_entropy(N, distances_sqr, beta)
    init_val = (beta, H, 0, beta_min, beta_max)

    def cond_fun(val):
        (_, H, i, _, _) = val
        return (jnp.abs(H- H_target) > tol) & (i < max_iter)

    def body_fun(val):
        (beta, H, i, beta_min, beta_max) = val
        _, H = cond_prob_and_entropy(N, distances_sqr, beta)
        # jax.debug.print(
        #     "H: {} (beta: {}, beta_min: {}, beta_max: {})", 
        #     H, beta, beta_min, beta_max
        # )
        # increasing beta will decrease the entropy
        beta_min = jnp.where(H > H_target, beta, beta_min)
        beta_max = jnp.where(H < H_target, beta, beta_max)

        # consider the new beta to be the average of the bounds
        # where we constrain the bounds to be within a factor of 4
        # of the current beta (so that we do not jump to insanely large values)
        min_bound = jnp.maximum(beta_min, beta / 4)
        max_bound = jnp.minimum(beta_max, beta * 4)
        beta = (min_bound + max_bound) / 2
        return (beta, H, i + 1, beta_min, beta_max)
    val = jax.lax.while_loop(cond_fun, body_fun, init_val)
    beta, _, _, _, _ = val
    log_Pi, _ = cond_prob_and_entropy(N, distances_sqr, beta)
    return beta, distances_sqr, log_Pi

@dataclass
class TsneModel:
    embedding: jax.Array
    loss_history: jax.Array

@F.jit
def randomized_tsne(X : T, *,
        rng_key, 
        n_components: int = 2,
        perplexity: float = 30.0,
        initializaton: str = "pca", # pca or random
        learning_rate: float | str = "auto",
        # default optimizer is sgd with momentum
        optimizer: callable = lambda lr, n_iter: optax.sgd(
            optax.cosine_decay_schedule(lr, n_iter), momentum=0.1
        ),
        n_iter: int = 1000,
        metric_sqr_fn: callable = euclidean_distance_sqr,
        early_exaggeration: float = 12.0,
        allow_subsampling: bool = True
    ):
    init_rng, neighbor_rng, train_rng = jax.random.split(rng_key, 3)
    X = F.vmap(lambda x: tree.ravel_pytree(x)[0])(X)

    # First do PCA to reduce the dimensionality
    if initializaton == "pca":
        pca_model = pca.randomized_pca(X, 
                n_components=n_components, rng_key=init_rng)
        Y = F.vmap(pca_model.project)(X)
    elif initializaton == "random":
        Y = jax.random.normal(init_rng, (X.shape[0], n_components))
    else:
        raise ValueError(f"Unknown initialization method {initializaton}")

    # if allow_subsampling is True, we don't
    # attach "springs" between all pairs of points,
    # only a subset of the most important neighbors
    if allow_subsampling and X.shape[0] > 2048:
        def sample_neighbor(input):
            i, rng_key = input
            neighbors = jax.random.choice(
                rng_key, X.shape[0], (2049,)
            )
            neighbors = jnp.arange(2049)
            neighbors = jnp.where(neighbors == i, neighbors[-1], neighbors)
            neighbors = neighbors[:2048]
            return neighbors
        neighbors = jax.lax.map(
            sample_neighbor,
            (jnp.arange(X.shape[0]), jax.random.split(neighbor_rng, X.shape[0])),
            batch_size=1024
        )
    else:
        neighbors = None

    # will do a binary search for the perplexity
    def search_beta(input):
        x, i = input
        if neighbors is None:
            distances = jax.vmap(lambda y: metric_sqr_fn(x, y))(X)
            distances = jnp.delete(distances, i, assume_unique_indices=True)
        else:
            distances = jax.vmap(lambda i: metric_sqr_fn(x,X[i]))(neighbors[i])
        return binary_search_beta(X.shape[0], distances, perplexity)

    # beta = search_beta((X[0], 0))
    # jax.debug.print("final beta: {}", beta)
    # return
    betas, _, log_P = jax.lax.map(
        search_beta, 
        (X, jnp.arange(X.shape[0])),
        batch_size=1024
    )
    log_P = log_P - jnp.log(log_P.shape[0])
    # Note that we use the (unsymmetrized) log_P!
    # TODO: Figure out a way of implicitly symmetrizing the log_P?

    # jax.debug.print("{}", betas)
    # return

    if learning_rate == "auto":
        learning_rate = X.shape[0] / 4

    optimizer = optimizer(learning_rate, n_iter)
    opt_state = optimizer.init(Y)

    @jax.value_and_grad
    def loss_fn(Y, log_P):
        # compute the distances
        if neighbors is not None:
            distances_sqr = F.vmap(
                lambda y, i: F.vmap(
                    lambda j: metric_sqr_fn(y, Y[j])\
                )(neighbors[i])
            )(Y, jnp.arange(Y.shape[0]))
        else:
            distances_sqr = F.vmap(
                F.vmap(metric_sqr_fn, in_axes=(0,None)), in_axes=(None, 0)
            )(Y,Y)
            distances_sqr = jax.vmap(
                lambda Di, i: jnp.delete(Di, i, assume_unique_indices=True)
            )(distances_sqr, jnp.arange(distances_sqr.shape[0]))

        # use a student t-distribution with one degree of freedom
        # the potential is given by (1 + dist_sqr)^(-1),
        # meaning log_potential = -log(1 + dist_sqr)
        log_Q = _log_q_probability(distances_sqr)
        D = kl_divergence(log_P, log_Q)
        return D

    n_exaggeration = n_iter // 4
    log_exaggeration_schedule = lambda i: jax.lax.cond(
        i < n_exaggeration, 
        lambda: jnp.log(early_exaggeration),
        lambda: 0.
    )

    def train_step(carry, _):
        i, y, opt_state, rng = carry
        log_exaggeration = log_exaggeration_schedule(i)
        value, grad = loss_fn(y, log_exaggeration + log_P)
        updates, opt_state = optimizer.update(grad, opt_state)
        y = optax.apply_updates(y, updates)
        return (i + 1, y, opt_state, rng), value

    # TODO: Switch scan to while_loop
    # with convergence criteria

    (_, Y, _, _), loss_history = jax.lax.scan(
        train_step,
        (0, Y, opt_state, train_rng),
        xs=None, length=n_iter
    )
    return TsneModel(
        embedding=Y,
        loss_history=loss_history
    )