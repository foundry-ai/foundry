import jax
import jax.tree_util
import jax.flatten_util
import jax.numpy as jnp

from typing import Any

class FrozenInstanceError(AttributeError): pass

from jax._src.api_util import flatten_axes

def axis_size(pytree, axes_tree) -> int:
    args_flat, in_tree  = jax.tree_util.tree_flatten(pytree)
    in_axes_flat = flatten_axes("axis_size in_axes", in_tree, axes_tree, kws=False)
    axis_sizes_ = [x.shape[i] for x, i in zip(args_flat, in_axes_flat)]
    assert all(x == axis_sizes_[0] for x in axis_sizes_)
    return axis_sizes_[0]

def ravel_pytree(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    if len(leaves) == 0:
        def unflatten(x):
            return jax.tree_util.tree_unflatten(treedef, x.reshape(leaves.shape))
        return leaves.reshape((-1,)), unflatten
    return jax.flatten_util.ravel_pytree(pytree)

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

def _key_str(key):
    if isinstance(key, jax.tree_util.DictKey):
        return key.key
    elif isinstance(key, jax.tree_util.GetAttrKey):
        return key.name
    elif isinstance(key, jax.tree_util.SequenceKey):
        return str(key.idx)
    else:
        raise ValueError(f"Unknown key type: {key}")

def flatten_to_dict(pytree):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(pytree)
    paths = ['.'.join([_key_str(key) for key in path]) for path, _ in leaves]
    nodes = [node for _, node in leaves]
    d = {path: node for path, node in zip(paths, nodes)}
    uf = lambda d: jax.tree_util.tree_unflatten(treedef, [d[k] for k in paths])
    return d, uf