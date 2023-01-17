from contextlib import contextmanager
import time
import jax
from jax import custom_vjp
import jax.numpy as jnp

# returns a but disables backprop of gradients through the return value
@custom_vjp
def zero_grad(a):
    return a

def zero_grad_fwd(a):
    return a, None

def zero_grad_bkw(res, g):
    return None

zero_grad.defvjp(zero_grad_fwd, zero_grad_bkw)

def tree_append(a, b):
    return jax.tree_util.tree_map(
        lambda a,b: jnp.concatenate((a, jnp.expand_dims(b,0))) if a is not None and b is not None else None,
        a,b
    )

def tree_prepend(a, b):
    return jax.tree_util.tree_map(
        lambda a,b: jnp.concatenate((jnp.expand_dims(a,0), b)) if a is not None and b is not None else None,
        a,b
    )

def ravel_dist(a,b, ord=None):
    x, _ = jax.flatten_util.ravel_pytree(a)
    y, _ = jax.flatten_util.ravel_pytree(b)
    return jnp.linalg.norm(x-y, ord=ord)

def mapped_ravel_pytree(x):
    x0 = jax.tree_util.tree_map(lambda x: x[0], x)
    _, x_unflatten = jax.flatten_util.ravel_pytree(x0)

    xs = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(x)
    xs_unflatten = jax.vmap(x_unflatten)
    return xs, xs_unflatten