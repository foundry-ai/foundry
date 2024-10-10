import jax.tree_util
import jax
import jax.numpy as jnp

from typing import GenericAlias

from jax.tree_util import (
    tree_leaves as leaves,
    tree_map as map, 
    tree_flatten as flatten, 
    tree_flatten_with_path as flatten_with_path
)

import functools

def shape(tree):
    return map(jnp.shape, tree)

def structure(tree):
    return map(
        lambda x: jax.ShapeDtypeStruct(jnp.shape(x), jnp.array(x).dtype),
        tree
    )

def total_size(tree):
    return sum(x.size for x in leaves(tree))

def flatten_to_dict(pytree, *, join='.', prefix=None,
                    suffix=None, is_leaf=None):
    def make_path_str(path):
        path = join.join([_key_str(key) for key in path])
        if prefix is not None: path = prefix + path
        if suffix is not None: path = path + suffix
        return path

    leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree, is_leaf=is_leaf)
    paths = [make_path_str(path) for path, _ in leaves]
    nodes = [node for _, node in leaves]
    d = {path: node for path, node in zip(paths, nodes) if node is not None}
    uf = lambda d: jax.tree_util.tree_unflatten(treedef, [d.get(k,None) for k in paths])
    return d, uf

from jax._src.api_util import flatten_axes as _flatten_axes

def axis_size(pytree, axes_tree, /) -> int:
    args_flat, in_tree  = jax.tree_util.tree_flatten(pytree)
    in_axes_flat = _flatten_axes("axis_size in_axes", in_tree, axes_tree, kws=False)
    axis_sizes_ = [x.shape[i] for x, i in zip(args_flat, in_axes_flat)]
    assert all(x == axis_sizes_[0] for x in axis_sizes_)
    return axis_sizes_[0]


# Pytree raveling utilities

from jax.flatten_util import ravel_pytree
from jax.flatten_util import ravel_pytree as ravel

def ravel_pytree_structure(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    shapes, types = [l.shape for l in leaves], [l.dtype for l in leaves]
    type = jnp.result_type(*types) if leaves else jnp.float32
    with jax.ensure_compile_time_eval():
        elems = jnp.array([0] + 
                [jnp.prod(jnp.array(l.shape, dtype=jnp.uint32)) for l in leaves], dtype=jnp.uint32)
        total_elems = jnp.sum(elems).item()
        indices = tuple([i.item() for i in jnp.cumsum(elems)])
    def unravel_to_list(x):
        return [x[indices[i]:indices[i+1]].reshape(s).astype(t) 
                for i, (s, t) in enumerate(zip(shapes, types))]
    def unravel_to_pytree(x):
        nodes = unravel_to_list(x)
        return jax.tree_util.tree_unflatten(treedef, nodes)
    return jax.ShapeDtypeStruct((total_elems,), type), unravel_to_pytree

ravel_structure = ravel_pytree_structure

def _key_str(key):
    if isinstance(key, jax.tree_util.DictKey):
        return key.key
    elif isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    elif isinstance(key, jax.tree_util.SequenceKey):
        return str(key.idx)
    else:
        raise ValueError(f"Unknown key type: {key}")


class static_property:
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__
        self.cache = {}

    def __set_name__(self, owner, name):
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same jax_cached_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        nodes, treedef = jax.tree.flatten(instance)
        val = self.cache.get(treedef, functools._NOT_FOUND)
        if val is functools._NOT_FOUND:
            # map node values to none so they can't be used
            nodes = [None if isinstance(n, jax.Array) else n for n in nodes]
            instance = jax.tree.unflatten(treedef, nodes)

            with jax.ensure_compile_time_eval():
                val = self.func(instance)
            self.cache[treedef] = val
        return val

    __class_getitem__ = classmethod(GenericAlias)