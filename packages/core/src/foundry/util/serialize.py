import foundry.numpy as jnp
import jax.tree_util
import jax.tree
import contextlib
import zarr
import warnings
import numpy as np
import importlib
import urllib.parse
import tempfile
import boto3

from typing import Any, Type
from pathlib import Path, PurePath

def _fq_import(module, name):
    if module == "builtins" and name == "NoneType":
        return type(None)
    module = importlib.import_module(module)
    subpaths = name.split('.')
    value = module
    for s in subpaths:
        value = getattr(value, s)
    return value

import dataclasses as _dcls

def _encode_json(aux):
    def aux_to_json(aux):
        if isinstance(aux, np.ndarray):
            return {"__tree_type__": "ndarray", 
                    "shape": aux.shape, "dtype": str(aux.dtype),
                    "data": aux.tolist()}
        if isinstance(aux, jax.ShapeDtypeStruct):
            return {"__tree_type__": "shape_dtype_struct",
                    "shape": aux.shape, "dtype": str(aux.dtype)}
        return aux
    return jax.tree.map(aux_to_json, aux)

def _decode_json(aux):
    def json_to_aux(aux):
        if isinstance(aux, dict) and "__tree_type__" in aux:
            tree_type = aux.get("__tree_type__")
            if tree_type == "ndarray":
                return np.array(aux["data"], dtype=aux["dtype"])
            if tree_type == "shape_dtype_struct":
                return jax.ShapeDtypeStruct(aux["shape"], aux["dtype"])
        return aux
    return jax.tree.map(json_to_aux, aux,
        is_leaf=lambda x: isinstance(x, dict) and
            aux.get("__tree_type__", None) is not None)

@_dcls.dataclass
class TreeDef:
    is_leaf: bool = False
    type: Type | None = None
    children: list["TreeDef"] | None = None
    aux: Any | None = None

    @staticmethod
    def from_jax_treedef(treedef):
        if treedef is None:
            return None
        node_data = treedef.node_data()
        if node_data is None:
            return TreeDef(is_leaf=True)
        node_type, node_aux = node_data
        return TreeDef(
            type=node_type,
            children=[TreeDef.from_jax_treedef(c) for c in treedef.children()],
            aux=node_aux,
        )

    def to_jax_treedef(self):
        from jax._src.tree_util import pytree
        if self.is_leaf:
            return jax.tree.structure(1)
        if self.type is None:
            return jax.tree.structure(None)
        registry = pytree.default_registry()
        node_data = (self.type, self.aux)
        children = [c.to_jax_treedef() for c in self.children]
        return pytree.PyTreeDef.make_from_node_data_and_children(
            registry, node_data, children
        )
    
    def to_json(self):
        if self.is_leaf:
            return {"type": "leaf"}
        full_type = f"{self.type.__module__}::{self.type.__name__}"
        json = {
            "type": full_type
        }
        if self.children:
            json["children"] = [c.to_json() for c in self.children]
        if self.aux is not None:
            json["aux"] = _encode_json(self.aux)
        return json
    
    @staticmethod
    def from_json(json):
        if json is None:
            return TreeDef(type=None)
        type_str = json["type"]
        if type_str == "leaf":
            return TreeDef(is_leaf=True)
        module, name = type_str.split("::")
        type = _fq_import(module, name)
        children = [TreeDef.from_json(c) for c in json.get("children", [])]
        return TreeDef(
            type=type,
            children=children,
            aux=_decode_json(json.get("aux", None))
        )

from foundry.core.dataclasses import dataclass

@dataclass
class Foo:
    x : jax.Array

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

NO_META = "__empty__"

# Utilities for saving/loading from a zarr file
def save_zarr(zarr_path, tree, meta=NO_META):
    if isinstance(zarr_path, (str, Path)):
        zarr_file = zarr.open(zarr_path, 'w')
    else:
        zarr_file = contextlib.nullcontext(zarr_path)
    with zarr_file as zf:
        nodes, treedef = jax.tree_util.tree_flatten_with_path(tree)
        paths = list([_zarr_keystr(p) for (p, _) in nodes])
        treedef_packed = TreeDef.from_jax_treedef(treedef).to_json()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zf.attrs['tree'] = treedef_packed
            zf.attrs['paths'] = paths
            if meta is not None:
                zf.attrs['meta'] = meta
            static_nodes = {}
            for (path, node) in nodes:
                key = _zarr_keystr(path)
                if isinstance(node, np.ndarray) or isinstance(node, jax.Array):
                    zf[key] = np.array(node)
                else:
                    static_nodes[key] = _encode_json(node)
            zf.attrs['static'] = static_nodes

def load_zarr(zarr_path) -> tuple[Any, Any] | Any:
    if isinstance(zarr_path, (str, Path)):
        zarr_file = zarr.open(zarr_path, 'r')
    else:
        zarr_file = contextlib.nullcontext(zarr)

    with zarr_file as zf:
        treedef = TreeDef.from_json(zf.attrs['tree']).to_jax_treedef()
        paths = zf.attrs['paths']
        meta = zf.attrs.get('meta', None)
        static_nodes = { k: _decode_json(v) 
            for k,v in zf.attrs.get('static', {}).items()}
        nodes = []
        for p in paths:
            if p in static_nodes:
                nodes.append(static_nodes[p])
            else:
                nodes.append(jnp.array(zf[p]))
        tree = jax.tree.unflatten(treedef, nodes)
        if meta == NO_META:
            return tree
        else:
            return tree, meta
    

def save(url, tree, meta=NO_META):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        if path.suffix == ".zarr" or path.suffix == ".zip":
            save_zarr(path, tree, meta)
        else:
            raise ValueError(f"Unknown file extension: {path.suffix}")
    elif parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        filename = PurePath(parsed.path).name
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = Path(tmpdirname) / filename
            save(f"file://{path.absolute()}", tree, meta)
            s3_client = boto3.client("s3")
            s3_client.upload_file(path, bucket, key)

def load(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        path = Path(parsed.path)
        if path.suffix == ".zarr" or path.suffix == ".zip":
            return load_zarr(path)
        else:
            raise ValueError(f"Unknown file extension: {path.suffix}")
    elif parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        filename = PurePath(parsed.path).name
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = Path(tmpdirname) / filename
            s3_client = boto3.client("s3")
            s3_client.download_file(bucket, key, path)
            return load(f"file://{path.absolute()}")
