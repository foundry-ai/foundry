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

from rich.text import Text as RichText
from rich.progress import ProgressColumn

class MofNColumn(ProgressColumn):
    def __init__(self):
        super().__init__()

    def render(self, task) -> RichText:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return RichText(
            f"{completed:{total_width}d}/{total}",
            style="progress.percentage",
        )

class AttrMap:
    def __init__(self, *args, **kwargs):
        d = dict()
        for a in args:
            if isinstance(a, AttrMap):
                a = a._dict
            a = dict(a)
            d.update(a)
        k = dict(**kwargs)
        d.update(k)
        # Remove None values from the attr structure
        self._dict = { k: v for k,v in d.items() if v is not None }

    def __getitem__(self, name: str):
        return self._dict.__getitem__(name)
    
    def __contains__(self, name: str):
        return self._dict.__contains__(name)

    def __setattr__(self, name: str, value: Any):
        if name != '_dict':
            raise RuntimeError(f"Unable to set {name}, Attrs is immutable")
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name.startswith('__'):
            raise AttributeError("Not found")
        return self._dict.get(name)
    
    def set(self, k,v):
        return AttrMap(self, **{k:v})
    
    def items(self):
        return self._dict.items()
    
    def values(self):
        return self._dict.values()

    def keys(self):
        return self._dict.keys()
    
    def __repr__(self) -> str:
        return self._dict.__repr__()

    def __str__(self) -> str:
        return self._dict.__str__()
    
    def __hash__(self) -> int:
        return self._dict.__hash__()
