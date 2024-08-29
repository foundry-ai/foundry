import jax.flatten_util
from foundry import dataclasses, partial

import foundry.util
import foundry.numpy as jnp
import jax


@dataclasses.dataclass
class PCAState:
    components: jax.Array
    mean: jax.Array
    explained_variance: jax.Array

# From https://github.com/alonfnt/pcax/blob/main/pcax/pca.py
# used under the MIT license
@partial(jax.jit, static_argnames=("n_components", "n_iter"))
def randomized_pca(x, n_components, rng, n_iter=5):
    x = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(x)
    unflatten = jax.flatten_util.ravel_pytree(
        jax.tree_map(lambda x: x[0], x)
    )[1]

    n_samples, n_features = x.shape
    means = jnp.mean(x, axis=0)
    x = x - means[None,:]
    size = min(2 * n_components, n_features)
    Q = jax.random.normal(rng, shape=(n_features, size))
    def step_fn(q, _):
        q, _ = jax.scipy.linalg.lu(x @ q, permute_l=True)
        q, _ = jax.scipy.linalg.lu(x.T @ q, permute_l=True)
        return q, None
    Q, _ = jax.lax.scan(step_fn, init=Q, xs=None, length=n_iter)
    Q, _ = jax.scipy.linalg.qr(x @ Q, mode="economic")
    B = Q.T @ x
    _, S, Vt = jax.scipy.linalg.svd(B, full_matrices=False)
    explained_variance = (S[:n_components] ** 2) / (n_samples - 1)
    A = Vt[:n_components]
    A = jax.vmap(unflatten)(A)
    means = unflatten(means)

    return PCAState(components=A, mean=means, explained_variance=explained_variance)

@jax.jit
def project(x, means, components):
    x_flat = jax.flatten_util.ravel_pytree(x)[0]
    comp_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(components)
    mean_flat = jax.flatten_util.ravel_pytree(means)[0]

    x_flat = x_flat - mean_flat
    x_projected = jnp.squeeze(comp_flat @ x_flat[:, None], -1)
    return x_projected