import jax.flatten_util
import jax.tree_util

def mapped_ravel_pytree(x):
    x0 = jax.tree_util.tree_map(lambda x: x[0], x)
    _, unflatten = jax.flatten_util.ravel_pytree(x0)
    x_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(x)
    xs_unflatten = jax.vmap(unflatten)
    return x_flat, xs_unflatten, unflatten