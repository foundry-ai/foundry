import foundry.core as F
import foundry.core.tree as tree

import foundry.util.pca as pca
import foundry.numpy as jnp

import jax

def kl_divergence(p, q):
    return jnp.sum(p * jnp.log(p / q))


def euclidean_distance(x, y):
    return jnp.sum((x - y) ** 2, axis=-1)


def shannon_entopy(p, eps=1e-12):
    return -jnp.sum(p * jnp.log2(p + eps))


def perplexity_fun(p):
    return 2 ** shannon_entopy(p)


def _conditional_probability(distances, sigma):
    p = jnp.exp(-distances / (2 * sigma**2))
    p = p / jnp.sum(p)
    return p

def _joint_probability(distances, sigma):
    p = _conditional_probability(distances, sigma)
    p = (p + p.T) / (2 * p.shape[0])
    return p

def _binary_search_perplexity(distances, target, tol=1e-5, max_iter=200):
    sigma_min = 1e-20
    sigma_max = 1e20
    sigma = 1.0

    def cond_fun(val):
        (_, perplexity, i, _, _) = val
        return (jnp.abs(perplexity - target) > tol) & (i < max_iter)

    def body_fun(val):
        (sigma, perp, i, sigma_min, sigma_max) = val
        p = _conditional_probability(distances, sigma)
        perp = perplexity_fun(p)
        sigma = jnp.where(perp > target, (sigma + sigma_min) / 2, (sigma + sigma_max) / 2)
        sigma_min = jnp.where(perp > target, sigma_min, sigma)
        sigma_max = jnp.where(perp > target, sigma, sigma_max)
        return (sigma, perp, i + 1, sigma_min, sigma_max)

    p = _conditional_probability(distances, sigma)
    perplexity = perplexity_fun(p)
    init_val = (sigma, perplexity, 0, sigma_min, sigma_max)
    sigma = jax.lax.while_loop(cond_fun, body_fun, init_val)[0]
    return sigma

@F.jit
def randomized_tsne(x, n_components, rng,
    perplexity: float = 30.0,
    learning_rate: float = 1e-3,
    n_iter: int = 1000,
    metric_fn: callable = euclidean_distance,
    early_exageration: float = 12.0):
    x = F.vmap(lambda x: tree.ravel_pytree(x)[0])(x)
    x_uf = tree.ravel_pytree(
        tree.map(lambda x: x[0], x)
    )[1]
    n_samples, n_features = x.shape
    pca_model = pca.randomized_pca(x, n_components, rng, n_iter=n_iter)
    # initial embedding using PCA
    x_new = F.vmap(pca_model.transform)(x)

    metric_fn = jax.vmap(jax.vmap(metric_fn, in_axes=(0, None)), in_axes=(None, 0))

    # Compute the probability of neighbours on the original embedding.
    # The matrix needs to be symetrized in order to be used as joint probability.
    distances = metric_fn(x, x)
    sigma = _binary_search_perplexity(distances, perplexity)
    P = _joint_probability(distances, sigma)

    @jax.grad
    def loss_fn(x, P):
        distances = metric_fn(x, x)
        Q = jax.nn.softmax(-distances)
        return kl_divergence(P, Q)

    def train_step(x, _):
        grads = loss_fn(x, P)
        x_new = x - learning_rate * grads
        return x_new, None

    def train_step_early_exageration(x, _):
        grads = loss_fn(x, early_exageration * P)
        x_new = x - learning_rate * grads
        return x_new, None

    n_exageration = 250
    x_new, _ = jax.lax.scan(train_step_early_exageration, x_new, xs=None, length=n_exageration)
    x_new, _ = jax.lax.scan(train_step, x_new, xs=None, length=n_iter)
    return jax.vmap(x_uf)(x_new)