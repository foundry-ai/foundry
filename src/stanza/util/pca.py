from stanza import struct

import jax.numpy as jnp
import jax


@struct.dataclass
class PCAState:
    components: jax.Array
    means: jax.Array
    explained_variance: jax.Array

# From https://github.com/alonfnt/pcax/blob/main/pcax/pca.py
# used under the MIT license
@jax.jit(static_argnames=("n_components", "n_iter"))
def randomized_pca(x, n_components, rng, n_iter=5):
    """Randomized PCA based on Halko et al [https://doi.org/10.48550/arXiv.1007.5510]."""
    n_samples, n_features = x.shape
    means = jnp.mean(x, axis=0, keepdims=True)
    x = x - means

    # Generate n_features normal vectors of the given size
    size = jnp.minimum(2 * n_components, n_features)
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

    return PCAState(components=A, means=means, explained_variance=explained_variance)