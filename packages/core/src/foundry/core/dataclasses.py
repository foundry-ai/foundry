import dataclasses as dcls
import functools
import jax

from jax.tree_util import GetAttrKey

from dataclasses import MISSING, fields, field, replace

def fields(cls):
    return dcls.fields(cls)

def _register_dataclass(nodetype, fields):
    data_fields = tuple(fields)

    def flatten_with_keys(x):
        data = tuple((GetAttrKey(name), getattr(x, name)) for name in data_fields)
        return data, meta
    def unflatten_func(meta, data):
        kwargs = dict(zip(data_fields, data))
        return nodetype(**kwargs)
    def flatten_func(x):
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta
    jax.tree_util.register_pytree_with_keys(
        nodetype, flatten_with_keys,
        unflatten_func, flatten_func
    )
    return nodetype

def dataclass(cls=None, **kwargs):
    if cls is None:
        return functools.partial(dataclass, **kwargs)
    cls = dcls.dataclass(frozen=True, unsafe_hash=True, **kwargs)(cls)
    fields = dcls.fields(cls)
    cls = _register_dataclass(cls, fields)
    return cls