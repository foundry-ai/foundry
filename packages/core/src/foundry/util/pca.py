import jax.flatten_util
import foundry.core as F
import foundry.numpy as jnp
import foundry.core.tree as tree

from foundry.core import dataclasses, partial

from typing import TypeVar, Generic
import jax

T = TypeVar("T")


@dataclasses.dataclass
class PCAState(Generic[T]):
    components: jax.Array
    mean: jax.Array
    explained_variance: jax.Array

    def project(self, x : T) -> T:
        x_flat, _ = tree.ravel_pytree(x)
        pca_components = self.components @ (x_flat - self.mean)
        return pca_components

@F.jit
def randomized_svd(A, k, n_oversamples, *, rng_key, n_iter=8):
    m, n = A.shape
    # Step 1: Draw a random Gaussian matrix
    P = jax.random.normal(rng_key, (n, k + n_oversamples))
    # Step 2: Form Y = A * P
    Y = jnp.dot(A, P)
    # Step 3: Perform power iteration
    def power_iter(Y, _):
        Y = jnp.dot(A, jnp.dot(A.T, Y))
        Y, _ = jax.scipy.linalg.qr(Y, mode="economic")
        return Y, None
    Y, _ = jax.lax.scan(power_iter, Y, xs=None, length=n_iter)
    Q, _ = jax.scipy.linalg.qr(Y, mode="economic")

    # Step 4: Form B = Q^T A
    B = jnp.dot(Q.T, A)
    U_tilde, S, Vt = jnp.linalg.svd(B, full_matrices=False)
    # Step 7: Recover the left singular vectors of A
    U = jnp.dot(Q, U_tilde)
    return U[:, :k], S[:k], Vt[:k, :]

@F.jit
def randomized_pca(x : T, *, n_components, rng_key, n_iter=5) -> PCAState[T]:
    x = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(x)
    unflatten = jax.flatten_util.ravel_pytree(
        jax.tree_map(lambda x: x[0], x)
    )[1]
    n_samples, n_features = x.shape
    means = jnp.mean(x, axis=0)
    x = x - means[None,:]
    oversample = min(2 * n_components, n_features) - n_components

    # if the matrix is small enough, we can use the exact SVD
    if x.shape[0] < 256 and x.shape[1] < 256:
        _, S, Vt = jnp.linalg.svd(x, full_matrices=False)
    else:
        _, S, Vt = randomized_svd(x, n_components, oversample, rng_key=rng_key, n_iter=n_iter)
    explained_variance = (S ** 2) / (n_samples - 1)
    A = Vt[:n_components]
    A = jax.vmap(unflatten)(A)
    means = unflatten(means)
    return PCAState(
        components=A,
        mean=means,
        explained_variance=explained_variance
    )