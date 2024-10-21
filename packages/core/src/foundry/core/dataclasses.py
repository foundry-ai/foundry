import dataclasses as dcls
import functools
import jax

import typing

from jax.tree_util import GetAttrKey

from dataclasses import MISSING, field, replace

def fields(cls):
    return dcls.fields(cls)

def _register_dataclass(nodetype, fields):
    data_fields = tuple(f.name for f in fields)

    def flatten_with_keys(x):
        data = tuple((GetAttrKey(name), getattr(x, name)) for name in data_fields)
        return data, None

    def unflatten_func(meta, data):
        kwargs = dict(zip(data_fields, data))
        return nodetype(**kwargs)

    def flatten_func(x):
        data = tuple(getattr(x, name) for name in data_fields)
        return data, None

    jax.tree_util.register_pytree_with_keys(
        nodetype, flatten_with_keys,
        unflatten_func, flatten_func
    )

@typing.dataclass_transform()
def dataclass(cls=None, frozen=True, **kwargs):
    if cls is None:
        return functools.partial(dataclass, frozen=frozen, **kwargs)
    cls = dcls.dataclass(frozen=frozen, unsafe_hash=True, **kwargs)(cls)
    fields = dcls.fields(cls)
    if frozen:
        _register_dataclass(cls, fields)
    return cls
