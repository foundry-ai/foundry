from dataclasses import dataclass as _dataclass, is_dataclass, \
            fields, replace, field as _field
from functools import partial
from typing import Any
import types
from itertools import chain

from stanza import _wrap, _unwrap

def dataclass(cls=None, frozen : bool = False, jax : bool = True, kw_only: bool = False):
    frozen = frozen or jax
    if cls is None:
        return partial(_make_dataclass, jax=jax, frozen=frozen, kw_only=kw_only)
    return _make_dataclass(cls, jax=jax, frozen=frozen, kw_only=kw_only)

def _make_dataclass(cls=None, jax=False, **kwargs):
    dcls = _dataclass(cls, **kwargs)
    if jax:
        import jax.tree_util
        jax.tree_util.register_pytree_node(
            dcls,
            partial(_dataclass_flatten, dcls),
            partial(_dataclass_unflatten, dcls)
        )
    return dcls

def field(*, default=None, default_factory=None, jax_static=False, **kwargs):
    kwargs['jax_static'] = jax_static
    args = {}
    if default is not None:
        args['default'] = default
    if default_factory is not None:
        args['default_factory'] = default_factory
    f = _field(*args,
               metadata=kwargs)
    return f

def _partition(pred, iterable):
    a, b = [], []
    for item in iterable:
        s, i = pred(item)
        l = a if s else b
        l.append(i)
    return a, b

def _dataclass_flatten(dcls, do):
    import jax.util
    def is_static(p):
        # Make functions automatically static
        if isinstance(p[1], types.FunctionType):
            return True, p
        f = dcls.__dataclass_fields__[p[0]]
        return (f.metadata.get('jax_static') if f.metadata else False), p
    static_items, dyn_items = _partition(is_static, sorted(do.__dict__.items()))
    static_keys, static_values = jax.util.unzip2(static_items)
    dyn_keys, dyn_values = jax.util.unzip2(dyn_items)
    children = dyn_values
    aux = dyn_keys, static_keys, static_values
    return children, aux

def _dataclass_unflatten(dcls, aux, children):
    do = dcls.__new__(dcls)
    dyn_keys, static_keys, static_values = aux
    dyn_values = children
    dyn_values = [_unwrap(v) for v in dyn_values]

    attrs = dict(chain(zip(dyn_keys, dyn_values), zip(static_keys, static_values)))
    # fill in the fields from the children
    for field in dcls.__dataclass_fields__.values():
        if field.name in attrs:
            object.__setattr__(do, field.name, attrs[field.name])
    return do
