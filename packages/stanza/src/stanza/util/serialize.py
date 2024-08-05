import jax.numpy as jnp
import jax.tree_util
import jax.tree
import contextlib
import zarr

from typing import Any
from pathlib import Path

_registry = {}

# Converts a treedef
# to a JSON-serializable structure
def pack_treedef(treedef):
    pass

def unpack_treedef(treedef_packed):
    pass

def _zarr_keystr(path):
    return "/".join(jax.tree_util.keystr((k,)) for k in path)

# Utilities for saving/loading from a zarr file
def save_zarr(zarr_path, tree, meta):
    if isinstance(zarr_path, (str, Path)):
        zarr_file = zarr.open(zarr_path, 'w')
    else:
        zarr_file = contextlib.nullcontext(zarr_path)
    with zarr_file as zf:
        nodes, treedef = jax.tree_util.tree_flatten_with_path(tree)
        paths = list([_zarr_keystr(p) for (p, _) in nodes])
        zf.attrs['tree'] = pack_treedef(treedef)
        zf.attrs['paths'] = paths
        zf.attrs['meta'] = meta
        for (path, node) in nodes:
            key = _zarr_keystr(path)
            print(key)
            zf[key] = node

def load_zarr(zarr_path) -> tuple[Any, Any]:
    if isinstance(zarr_path, (str, Path)):
        zarr_file = zarr.open(zarr_path, 'r')
    else:
        zarr_file = contextlib.nullcontext(zarr)

    with zarr_file as zf:
        treedef = unpack_treedef(zf.attrs['tree'])
        paths = zf.attrs['paths']
        meta = zf.attrs['meta']
        nodes = []
        for p in paths:
            nodes.append(zf[p])
        tree = jax.tree.unflatten(treedef, nodes)
        return tree, meta