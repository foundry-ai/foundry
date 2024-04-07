import stanza.struct as struct
import jax
import jax.numpy as jnp
import numpy as np
import functools

from typing import Any, TypeVar

T = TypeVar('T')
F = TypeVar('F')

def is_array(element: Any) -> bool:
    """Returns `True` if `element` is a JAX array or NumPy array."""
    return isinstance(element, (np.ndarray, np.generic, jax.Array))

class partial:
    """Partial function application with static_argnames
    """
    def __init__(self, func, /, *args, **keywords):
        if hasattr(func, "func"):
            args = func.args + args
            keywords = {**func.keywords, **keywords}
            func = func.func
        self.func = func
        self.args = args
        self.keywords = keywords

    def __call__(self, /, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        return self.func(*self.args, *args, **keywords)

from .lift import LoweredFunction, LiftedFunction

def jit(fun : F) -> F:
    lifted = lift.lift(fun)
    jitted = jax.jit(lifted)
    return lift.lower(jitted)

from jax._src.api_util import flatten_axes
from jax._src.api import _mapped_axis_size
from jax._src.api import batching

def pvmap(fun, in_axes=0, out_axes=0, axis_size=None, devices=None):
    zero_in_axes = jax.tree_map(lambda _: 0, in_axes)
    zero_out_axes = jax.tree_map(lambda _: 0, out_axes)
    devices = devices or jax.local_devices()

    vmapped = jax.vmap(fun, in_axes=zero_in_axes, out_axes=zero_out_axes)
    def map(*args, **kwargs):
        args_flat, in_tree  = jax.tree_util.tree_flatten((args, kwargs), is_leaf=batching.is_vmappable)
        in_axes_flat = flatten_axes("pvmap in_axes", in_tree, (in_axes, 0), kws=True)
        axis_size_ = (axis_size if axis_size is not None else
                    _mapped_axis_size(fun, in_tree, args_flat, in_axes_flat, "pvmap"))
        pmap_axis_size = min(len(devices), axis_size_)
        vmap_axis_size = (axis_size_ + pmap_axis_size - 1) // pmap_axis_size
        padded_axis_size_ = pmap_axis_size * vmap_axis_size
        # move any mapped axis to the 0th axis
        args_flat = [jnp.moveaxis(x, a, 0)
                        if a is not None else x
                        for x, a in zip(args_flat, in_axes_flat)]
        # pad up to padded_axis_size
        args_flat = [jnp.concatenate((
            x, jnp.zeros((padded_axis_size_ - axis_size_,) + x.shape[1:], dtype=x.dtype)
        )) if a is not None else x for x, a in zip(args_flat, in_axes_flat)]
        args_flat = [jnp.reshape(x, (pmap_axis_size, vmap_axis_size) + x.shape[1:])
                        if a is not None else x
                        for x, a in zip(args_flat, in_axes_flat)]
        (args, kwargs) = jax.tree_util.tree_unflatten(in_tree, args_flat)

        pmapped = jax.pmap(vmapped, in_axes=zero_in_axes, out_axes=zero_out_axes, devices=devices[:pmap_axis_size])
        out = pmapped(*args, **kwargs)
        # reshape the output
        out_flat, out_tree = jax.tree_util.tree_flatten(out)
        out_axes_flat = flatten_axes("pvmap out_axes", out_tree, out_axes)
        # make one mapped axis and remove padding
        out_flat = [jnp.reshape(x, (padded_axis_size_,) + x.shape[2:])[:axis_size_] if a is not None else x 
                    for (x, a) in zip(out_flat, out_axes_flat)]
        # move the mapped axis to the right place
        out_flat = [jnp.moveaxis(x, 0, a) if a is not None else x 
                    for (x, a) in zip(out_flat, out_axes_flat)]
        out = jax.tree_util.tree_unflatten(out_tree, out_flat)
        return out
    return map
