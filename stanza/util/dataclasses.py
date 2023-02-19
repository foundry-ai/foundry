import dataclasses
from dataclasses import dataclass as _dataclass, is_dataclass, fields, replace
from jax.tree_util import register_pytree_node
from functools import partial
from typing import Any

import jax

"""
    Like dataclass(), but has frozen=True by default and will
    automatically register a dataclass with jax. Unlike
    chex dataclasses, does not require a kwargs only constructor.
"""
def dataclass(cls=None, **kwargs):
    if cls is None:
        return partial(make_dataclass, **kwargs)
    return make_dataclass(cls, **kwargs)

def make_dataclass(cls=None, *, jax=True):
    dcls = _dataclass(cls, frozen=True)
    # register the dcls type in jax
    if jax:
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

# Takes in a dataclass type or instance
# and lets you set arguments on the builder object
# then call build() to get the changed object
class Builder:
    def __init__(self, dataclass):
        self._dataclass = dataclass
        self._fields = fields(dataclass)
        self._args = {}
    
    # Will fork the builder
    def fork(self):
        pass

    def __setattr__(self, name: str, value: Any):
        self[name] = value

    def __getattr__(self, name):
        return self[name]

    def __setitem__(self, name: str, value: Any):
        # Validate according to dataclass type information
        self._args[name] = value
    
    def __getitem__(self, name: str):
        if name in self._args:
            return self._args[name]
        field = self._fields[name]
        if is_dataclass(field.type):
            builder = Builder(field.type)
            self._args[name] = builder
            return builder

    def build(self):
        # transform the args
        args = { k: v.build() if isinstance(v, Builder) else v \
                    for (k, v) in self._args.items() }
        # If we already have an instance,
        # of the dataclass, call replace()
        if isinstance(self._dataclass, type):
            return self._dataclass(**args)
        else:
            return replace(self._dataclass, **args)