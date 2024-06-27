import jax.numpy as jnp
import jax.tree_util
import jax.tree

import json

_registry = {}

# Converts a treedef
# to a JSON-serializable structure
def pack_treedef(treedef):
    pass

def unpack_treedef(treedef_packed):
    pass

def _zarr_keystr(path):
    pass

# Utilities for saving/loading from a zarr file
def save_zarr(zarr, tree):
    nodes, treedef = jax.tree_util.tree_flatten_with_path(pytree)
    paths = list([_zarr_keystr(p) for (p, _) in nodes])
    zarr.attrs['tree'] = pack_treedef(treedef)
    zarr.attrs['paths'] = paths
    for (path, node) in nodes:
        zarr[_zarr_keystr(path)] = node

def load_zarr(zarr):
    treedef = unpack_treedef(zarr.attrs['tree'])
    paths = zarr.attrs['paths']
    nodes = []
    for p in paths:
        nodes.append(zarr[p])
    return jax.tree.unflatten(treedef, nodes)