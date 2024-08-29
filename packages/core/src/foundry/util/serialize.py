import foundry.numpy as jnp
import jax.tree_util
import jax.tree
import contextlib
import zarr
import numpy as np
import importlib

from typing import Any
from pathlib import Path

def _fq_import(module, name):
    module = importlib.import_module(module)
    subpaths = name.split('.')
    value = module
    for s in subpaths:
        value = getattr(value, s)
    return value

# Converts a treedef
# to a JSON-serializable structure
def pack_treedef(treedef):
    if treedef is None:
        return None
    node_data = treedef.node_data()
    if node_data is None:
        return {"type": "leaf"}
    node_type, node_aux = node_data
    if node_type is type(None):
        return None
    children = treedef.children()
    mod = node_type.__module__
    qualname = node_type.__qualname__
    type_name = f"{mod}::{qualname}"
    serialized = { "type": type_name }
    children = [pack_treedef(c) for c in children]
    if children:
        serialized['children'] = children
    # If the node has auxiliary data, serialize it!
    # Convert jax/numpy arrays to lists.
    if not (isinstance(node_aux, tuple) and node_aux == ()):
        def jsonify_aux(aux):
            if isinstance(aux, (jax.Array, np.ndarray)):
                if aux.size > 1:
                    return aux.item()
                else:
                    return {"array_type": aux.dtype.str,
                            "array_data": aux.tolist()}
            else: return aux
        node_aux = jax.tree_util.tree_map(
            jsonify_aux, node_aux,
            is_leaf=lambda x: isinstance(x, (jax.Array, np.ndarray))
        )
        serialized['meta'] = node_aux
    return serialized

from foundry.core.dataclasses import dataclass

@dataclass
class Foo:
    x : jax.Array

def unpack_treedef(treedef_packed):
    from jax._src.tree_util import pytree
    if treedef_packed is None:
        return jax.tree.structure(None)
    type = treedef_packed.get('type', None)
    if type == "leaf":
        return jax.tree.structure(1)
    # Unpack the type and import it
    module, type_name = type.split('::')
    node_type = _fq_import(module, type_name)

    aux = treedef_packed.get('meta', ())
    def dejsonify_aux(aux):
        if isinstance(aux, dict) and "array_type" in aux:
            return np.array(aux['array_data'], dtype=np.dtype(aux['array_type']))
        else:
            return aux
    aux = jax.tree_util.tree_map(dejsonify_aux, aux,
        is_leaf=lambda x: isinstance(x, dict) and "array_type" in x
    )
    node_data = (node_type, aux)
    children = [unpack_treedef(c) for c in treedef_packed.get('children', [])]
    registry = pytree.default_registry()
    treedef = pytree.PyTreeDef.make_from_node_data_and_children(
        registry, node_data, children
    )
    return treedef

def _keystr(k):
    from jax.tree_util import GetAttrKey, FlattenedIndexKey, DictKey, SequenceKey
    if isinstance(k, GetAttrKey):
        return k.name
    elif isinstance(k, DictKey):
        return k.key
    elif isinstance(k, FlattenedIndexKey):
        return str(k.key)
    elif isinstance(k, SequenceKey):
        return str(k.idx)
    else:
        return str(k)

def _zarr_keystr(path):
    return "/".join(_keystr(k) for k in path)

# Utilities for saving/loading from a zarr file
def save_zarr(zarr_path, tree, meta):
    if isinstance(zarr_path, (str, Path)):
        zarr_file = zarr.open(zarr_path, 'w')
    else:
        zarr_file = contextlib.nullcontext(zarr_path)
    with zarr_file as zf:
        nodes, treedef = jax.tree_util.tree_flatten_with_path(tree)
        paths = list([_zarr_keystr(p) for (p, _) in nodes])
        treedef_packed = pack_treedef(treedef)
        zf.attrs['tree'] = treedef_packed
        zf.attrs['paths'] = paths
        if meta is not None:
            zf.attrs['meta'] = meta
        for (path, node) in nodes:
            key = _zarr_keystr(path)
            zf[key] = np.array(node)

def load_zarr(zarr_path) -> tuple[Any, Any]:
    if isinstance(zarr_path, (str, Path)):
        zarr_file = zarr.open(zarr_path, 'r')
    else:
        zarr_file = contextlib.nullcontext(zarr)

    with zarr_file as zf:
        treedef = unpack_treedef(zf.attrs['tree'])
        paths = zf.attrs['paths']
        meta = zf.attrs.get('meta', None)
        nodes = []
        for p in paths:
            nodes.append(jnp.array(zf[p]))
        tree = jax.tree.unflatten(treedef, nodes)
        return tree, meta