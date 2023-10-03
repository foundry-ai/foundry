import jax
import jax.numpy as jnp
from stanza.dataclasses import dataclass, replace, field
from typing import List, Any

from chex import assert_trees_all_equal_shapes_and_dtypes

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

def mat_jacobian(f, argnums=0):
    def jac(*args):
        flat_args = []
        args_uf = []
        for a in args:
            x_flat, x_uf = jax.flatten_util.ravel_pytree(a)
            flat_args.append(x_flat)
            args_uf.append(x_uf)
        def f_flat(*flat_args):
            args = []
            for a, uf in zip(flat_args, args_uf):
                args.append(uf(a))
            y = f(*args)
            y_flat, _ = jax.flatten_util.ravel_pytree(y)
            return y_flat
        return jax.jacobian(f_flat, argnums=argnums)(x_flat)
    return jac


def map(f, vsize=None):
    vf = jax.vmap(lambda args: f(*args))
    def _f(*args):
        N = jax.tree_flatten(args)[0][0].shape[0]
        vs = N if vsize is None else vsize
        N_padding = vs * ((N + vs - 1) // vs) - N
        # pad the args
        args = jax.tree_map(
            lambda x: jnp.concatenate(
                (x, jnp.zeros((N_padding,) + x.shape[1:], dtype=x.dtype)),
                axis=0
            ),
            args
        )
        # reshape to vsize chunks
        args = jax.tree_map(
            lambda x: jnp.reshape(x, 
                ((N + N_padding) // vsize, vsize) + x.shape[1:]),
            args
        )
        res = jax.lax.map(vf, args)
        # and get rid of the padding
        res = jax.tree_map(
            lambda x: jnp.reshape(
                x, (-1,) + x.shape[2:]
            )[:N], res
        )
        return res
    return _f

def check_nan(name, x, cb=None):
    def _cb(x, has_nans):
        if has_nans:
            print(f"Nans detected {name}:", x, has_nans)
            if cb is not None:
                cb(x)
            import sys
            sys.exit(1)
    flat, _ = jax.flatten_util.ravel_pytree(x)
    has_nans = jnp.any(jnp.isnan(flat))
    jax.debug.callback(_cb, x, has_nans, ordered=True),

def l2_norm_squared(x, y=None):
    if y is not None:
        x = jax.tree_map(lambda x, y: x - y, x, y)
    flat, _ = jax.flatten_util.ravel_pytree(x)
    return jnp.sum(jnp.square(flat))

def mse_loss(x, y=None):
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