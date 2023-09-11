from dataclasses import dataclass as _dataclass, is_dataclass, \
            fields, replace as _replace, field as _field, asdict
from functools import partial
from typing import Any
from stanza import _wrap_functions, _unwrap_functions
import types
from itertools import chain

def unpack(dc) -> dict:
    return { field.name : getattr(dc, field.name) for field in fields(dc) }

def replace(__obj, **changes):
    from stanza.util.attrdict import AttrMap
    if isinstance(__obj, AttrMap):
        return AttrMap(__obj, **changes)
    if isinstance(__obj, dict):
        return { **__obj, **changes }
    return _replace(__obj, **changes)

def replace_defaults(dc, **defaults):
    overrides = { k : v for k, v in defaults.items() if getattr(dc, k) is None }
    return replace(dc, **overrides)

def combine(clazz, *dcs):
    fields = set(clazz.__dataclass_fields__.keys())
    args = {}
    for dc in dcs:
        if not isinstance(dc, dict):
            dc = unpack(dc)
        dc = { k : v for k, v in dc.items() if k in fields}
        args.update(dc)
    return clazz(**args)

def dataclass(cls=None, frozen : bool = False, jax : bool = True, kw_only: bool = False):
    frozen = frozen or jax
    if cls is None:
        return partial(_make_dataclass, jax=jax, frozen=frozen, kw_only=kw_only)
    return _make_dataclass(cls, jax=jax, frozen=frozen, kw_only=kw_only)

def _make_dataclass(cls=None, jax=False, **kwargs):
    dcls = _dataclass(cls, **kwargs)
    if jax:
        import jax.tree_util
        jax.tree_util.register_pytree_with_keys(
            dcls,
            partial(_dataclass_flatten_with_keys, dcls),
            partial(_dataclass_unflatten, dcls),
            partial(_dataclass_flatten, dcls)
        )
    return dcls

NoArg = object()

def field(*, default=NoArg, default_factory=NoArg, 
            jax_static=False, init=NoArg, **kwargs):
    kwargs['jax_static'] = jax_static
    args = {}
    if init is not NoArg:
        args['init'] = init
    if default is not NoArg:
        args['default'] = default
    if default_factory is not NoArg:
        args['default_factory'] = default_factory
    f = _field(**args,
               metadata=kwargs)
    return f

def _partition(pred, iterable):
    a, b = [], []
    for item in iterable:
        s = pred(item)
        l = a if s else b
        l.append(item)
    return a, b

def _dataclass_flatten(dcls, do):
    import jax.util
    def is_static(p):
        # Make functions automatically static
        f = dcls.__dataclass_fields__[p[0]]
        return f.metadata.get('jax_static') if f.metadata else False
    static_items, dyn_items = _partition(is_static, sorted(do.__dict__.items()))
    static_keys, static_values = jax.util.unzip2(static_items)
    dyn_keys, dyn_values = jax.util.unzip2(dyn_items)
    dyn_values = _wrap_functions(dyn_values)
    children = dyn_values
    aux = dyn_keys, static_keys, static_values
    return children, aux

def _dataclass_flatten_with_keys(dcls, do):
    import jax.util
    def is_static(p):
        # Make functions automatically static
        f = dcls.__dataclass_fields__[p[0]]
        return f.metadata.get('jax_static') if f.metadata else False
    static_items, dyn_items = _partition(is_static, sorted(do.__dict__.items()))
    static_keys, static_values = jax.util.unzip2(static_items)
    dyn_keys, dyn_values = jax.util.unzip2(dyn_items)
    dyn_values = _wrap_functions(dyn_values)
    children = dyn_values
    aux = dyn_keys, static_keys, static_values
    dyn_keys = [jax.tree_util.GetAttrKey(k) for k in dyn_keys]
    return zip(dyn_keys, children), aux

def _dataclass_unflatten(dcls, aux, children):
    do = dcls.__new__(dcls)
    dyn_keys, static_keys, static_values = aux
    dyn_values = children
    dyn_values = _unwrap_functions(dyn_values)

    attrs = dict(chain(zip(dyn_keys, dyn_values), zip(static_keys, static_values)))
    # fill in the fields from the children
    for field in dcls.__dataclass_fields__.values():
        if field.name in attrs:
            object.__setattr__(do, field.name, attrs[field.name])
    return do