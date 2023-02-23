import dataclasses
from dataclasses import dataclass as _dataclass, is_dataclass, \
            fields, replace, field
from functools import partial
from typing import Any


"""
    Like dataclass(), but has frozen=True by default and will
    automatically register a dataclass with jax. Unlike
    chex dataclasses, does not require a kwargs only constructor.
"""
def dataclass(cls=None, **kwargs):
    if cls is None:
        return partial(make_dataclass, **kwargs)
    return make_dataclass(cls, **kwargs)

def make_dataclass(cls=None, *, frozen=False, jax=False):
    dcls = _dataclass(cls, frozen=jax or frozen)
    # register the dcls type in jax
    if jax:
        from jax.tree_util import register_pytree_node
        register_pytree_node(
            dcls,
            partial(_dataclass_flatten, dcls),
            partial(_dataclass_unflatten, dcls)
        )
    return dcls

def _dataclass_flatten(dcls, do):
    import jax.util
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

import inspect

# An immutable set of arguments
class Parameters:
    def __init__(self, constructor, arg_types, args=None):
        self.__args = args or {}
        self.__arg_types = arg_types
        self.__constructor = constructor

    # Will return a copy of the argument set,
    # can handle argument paths as well (i.e foo.bar)
    # will return the modified ParameterSet
    def set(self, name: str, value: Any):
        path = name.split('.')
        if len(path) == 1:
            args = dict(self.__args)
            args[name] = value
            return Parameters(self.__constructor, args)
        else:
            raise ValueError("Path parameters not yet supported")

    def __getattr__(self, name: str):
        return self[name]
    
    def __getitem__(self, name: str):
        return self.__args[name]

    # Use () on Parameters to construct the object
    def __call__(self):
        # construct any sub-parameters
        args = { k: (v() if isinstance(v, Parameters) else v) \
                            for (k,v) in self.__args.items() }
        return self.__constructor(**args)
    
    @staticmethod
    def for_dataclass(dataclass):
        return Parameters(dataclass, fields(dataclass))
    
    # Takes the cartesian product of two sets of Parameters
    # and returns the cartesian product of Parameters
    @staticmethod
    def cartesian_product(setA, setB):
        return setA