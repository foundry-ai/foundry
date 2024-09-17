import foundry.numpy as jnp
import jax.random

import foundry.util.lanczos as lanczos

def test_lanczos():
    H = jnp.array([[2, 1], [1, 2]])
    hvp = lambda v: jnp.dot(H, v)
    tridiag, vecs = lanczos.lanczos_alg(
        jax.random.key(1), hvp, 2, 2, jnp.float32
    )

def test_lanczos_denisty():
    H = jnp.array([[2, 1], [1, 2]])
    hvp = lambda v: jnp.dot(H, v)
    log_density, grids, eig_vals = lanczos.lanczos_density(
        jax.random.key(0), hvp, 2, 2
    )