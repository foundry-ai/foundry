import dataclasses as dcls
import jax

from jax.tree_util import GetAttrKey

from dataclasses import MISSING, replace
from functools import partial

def fields(cls):
    return dcls.fields(cls)

def _register_dataclass(nodetype, data_fields, meta_fields):
    meta_fields = tuple(meta_fields)
    data_fields = tuple(data_fields)

    def flatten_with_keys(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple((GetAttrKey(name), getattr(x, name)) for name in data_fields)
        return data, meta

    def unflatten_func(meta, data):
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return nodetype(**kwargs)

    def flatten_func(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta
    
    jax.tree_util.register_pytree_with_keys(
        nodetype, flatten_with_keys,
        unflatten_func, flatten_func
    )

    return nodetype

def dataclass(cls=None, **kwargs):
    if cls is None:
        return partial(dataclass, **kwargs)
    cls = dcls.dataclass(frozen=True, unsafe_hash=True, **kwargs)(cls)
    fields = dcls.fields(cls)
    data_fields = [f.name for f in fields if (not f.metadata) and f.metadata.get('pytree_node', True)]
    meta_fields = [f.name for f in fields if f.metadata and (not f.metadata.get('pytree_node', True))]
    # cls = jax.tree_util.register_dataclass(cls, data_fields, meta_fields)
    cls = _register_dataclass(cls, data_fields, meta_fields)
    return cls

def field(*, default=MISSING, default_factory=MISSING, pytree_node=True):
    return dcls.field(
        default=default, default_factory=default_factory,
        metadata={'pytree_node': pytree_node}
    )