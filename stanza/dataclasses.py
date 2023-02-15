from dataclasses import dataclass as _dataclass
from jax.tree_util import register_pytree_node
from functools import partial

import jax

"""
    Like dataclass(), but has frozen=True by default and will
    automatically register a dataclass with jax
"""
def dataclass(cls=None, *, init=True, frozen=True):
    if cls is None:
        return partial(dataclass,
            init=init, frozen=frozen)
    dcls = _dataclass(cls, init=init, frozen=frozen)
    # register the dcls type in jax
    register_pytree_node(
        dcls,
        partial(_dataclass_flatten, dcls),
        partial(_dataclass_unflatten, dcls)
    )
    return dcls

def _dataclass_flatten(dcls, do):
    # for speeed use jax.util.unzip2
    keys, values = jax.util.unzip2(sorted(do.__dict__.items()))[::-1]
    return keys, values

def _dataclass_unflatten(dcls, keys, values):
    do = dcls.__new__(dcls)
    attrs = dict(zip(keys, values))
    # fill in the fields from the children
    for field in dcls.__dataclass_fields__.values():
        if field.name in attrs:
            object.__setattr__(do, field.name, attrs[field.name])
    return do