from typing import Any

import jax.tree_util

class AttrDict(dict):
    def __setattr__(self, name: str, value: Any):
        self[name] = value
    
    def __getattr__(self, name: str):
        return self[name]

# A version of attrdict
# which is immutable, jax-compatible
class Attrs:
    def __init__(self, *args, **kwargs):
        d = dict()
        for a in args:
            d.update(dict(a))
        k = dict(**kwargs)
        d.update(k)
        self._dict = d

    def __getitem__(self, name: str):
        return self._dict[name]

    def __setattr__(self, name: str, value:  Any):
        if name != '_dict':
            raise RuntimeError(f"Unable to set {name}, Attrs is immutable")
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        try:
            return self._dict.get(name)
        except KeyError:
            raise AttributeError("Missing attribute")

def _flatten_attrs(attrs):
    keys, values = jax.util.unzip2(attrs._dict.items())
    children = values
    aux = keys
    return children, aux

def _unflatten_attrs(aux, children):
    keys = aux
    values = children
    return Attrs(zip(keys, values))
jax.tree_util.register_pytree_node(Attrs, _flatten_attrs, _unflatten_attrs)