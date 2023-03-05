from dataclasses import dataclass as _dataclass, is_dataclass, \
            fields, replace, field as _field
from functools import partial
from typing import Any
import types
from itertools import chain

from stanza import _wrap, _unwrap

"""
    Like dataclass(), but has frozen=True by default and will
    automatically register a dataclass with jax. Unlike
    chex dataclasses, does not require a kwargs only constructor.
"""
def dataclass(cls=None, **kwargs):
    if cls is None:
        return partial(make_dataclass, **kwargs)
    return make_dataclass(cls, **kwargs)

def make_dataclass(cls=None, *, frozen=False, jax=False, **kwargs):
    dcls = _dataclass(cls, frozen=jax or frozen, **kwargs)
    # register the dcls type in jax
    if jax:
        from jax.tree_util import register_pytree_node
        register_pytree_node(
            dcls,
            partial(_dataclass_flatten, dcls),
            partial(_dataclass_unflatten, dcls)
        )
    return dcls

def field(*, jax_static=False, **kwargs):
    f = _field(**kwargs, metadata={'jax_static': jax_static})
    return f

def _dataclass_flatten(dcls, do):
    from stanza import is_jaxtype
    import jax.util
    # for speeed use jax.util.unzip2
    keys, values = jax.util.unzip2(sorted(do.__dict__.items()))
    # use _wrap to deal with functions as part of dataclasses
    values = [_wrap(v) for v in values]
    children = values
    aux = keys
    return children, aux

def _dataclass_unflatten(dcls, aux, children):
    do = dcls.__new__(dcls)
    dyn_keys = aux
    dyn_values = children
    dyn_values = [_unwrap(v) for v in dyn_values]
    # use _unwrap to deal with functions as part of dataclasses
    attrs = dict(zip(dyn_keys, dyn_values))
    # fill in the fields from the children
    for field in dcls.__dataclass_fields__.values():
        if field.name in attrs:
            object.__setattr__(do, field.name, attrs[field.name])
    return do

# An immutable set of arguments
class Parameters:
    def __init__(self, **args):
        self.__args = args

    # Will return a copy of the argument set,
    # can handle argument paths as well (i.e foo.bar)
    # will return the modified ParameterSet
    def set(self, name: str, value: Any):
        path = name.split('.')
        if len(path) == 1:
            args = dict(self.__args)
            args[name] = value
            return Parameters(**args)
        else:
            raise ValueError("Path parameters not yet supported")

    def __getattr__(self, name: str):
        return self[name]
    
    def __getitem__(self, name: str):
        return self.__args[name]

    # Use () on Parameters to construct the object
    def __call__(self, constructor, arg_types=None):
        if arg_types is None and is_dataclass(constructor):
            arg_types = { f.name: f.type for f in fields(constructor)}
        # construct any sub-parameters
        args = { k: (v() if isinstance(v, Parameters) else v) \
                            for (k,v) in self.__args.items() }
        return constructor(**args)
    
    @staticmethod
    def update(a, b):
        args = dict(a.__args)
        args.update(b.__args)
        return Parameters(**args)
    
    # Takes the cartesian product of two sets of Parameters
    # and returns the cartesian product of Parameters
    @staticmethod
    def cartesian_product(setA, setB):
        s = set()
        for a in setA:
            for b in setB:
                s.add(Parameters.update(a, b))
        return s
    
    def __hash__(self):
        return hash(frozenset(self.__args.items()))
    
    def __repr__(self):
        return self.__args.__repr__()