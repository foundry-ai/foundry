import jax
import jax.numpy as jnp
from stanza.dataclasses import dataclass, replace, field
from typing import List, Any

def vmap_ravel_pytree(x):
    i = jax.tree_util.tree_map(lambda x: x[0], x)
    _, uf = jax.flatten_util.ravel_pytree(i)

    def flatten(x):
        return jax.flatten_util.ravel_pytree(x)[0]
    flat = jax.vmap(flatten)(x)
    uf = jax.vmap(uf)
    return flat, uf

def extract_shifted(xs):
    earlier_xs = jax.tree_map(lambda x: x[:-1], xs)
    later_xs = jax.tree_map(lambda x: x[1:], xs)
    return earlier_xs, later_xs


def l2_norm_squared(x, y=None):
    if y is not None:
        x = jax.tree_map(lambda x, y: x - y, x, y)
    flat, _ = jax.flatten_util.ravel_pytree(x)
    return jnp.mean(jnp.square(flat))

def shape_tree(x):
    return jax.tree_util.tree_map(lambda x: jnp.array(x).shape, x)

def shape_dtypes(x):
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        x
    )