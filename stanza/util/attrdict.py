from typing import Any
import jax.tree_util

class AttrDict(dict):
    def __setattr__(self, name: str, value: Any):
        self[name] = value
    
    def __getattr__(self, name: str):
        return self[name]

# A version of attrdict
# which is immutable, jax-compatible
class AttrMap:
    def __init__(self, *args, **kwargs):
        d = dict()
        for a in args:
            if isinstance(a, AttrMap):
                a = a._dict
            a = dict(a)
            d.update(a)
        k = dict(**kwargs)
        d.update(k)
        self._dict = d

    def __getitem__(self, name: str):
        return self._dict.__getitem__(name)
    
    def __contains__(self, name: str):
        return self._dict.__contains__(name)

    def __setattr__(self, name: str, value:  Any):
        if name != '_dict':
            raise RuntimeError(f"Unable to set {name}, Attrs is immutable")
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name == '_dict':
            return None
        try:
            return self._dict.get(name)
        except KeyError:
            raise AttributeError("Missing attribute")
    
    def set(self, k,v):
        return AttrMap(self, **{k:v})
    
    def items(self):
        return self._dict.items()
    
    def values(self):
        return self._dict.values()

    def keys(self):
        return self._dict.keys()
    
    def __repr__(self) -> str:
        return self._dict.__repr__()

    def __str__(self) -> str:
        return self._dict.__str__()
    
    def __hash__(self) -> int:
        return self._dict.__hash__()

def attrmap(*args, **kwargs):
    return AttrMap(*args, **kwargs)

def _flatten_attrs(attrs):
    keys, values = jax.util.unzip2(attrs._dict.items())
    children = values
    aux = keys
    return children, aux

def _unflatten_attrs(aux, children):
    keys = aux
    values = children
    return AttrMap(zip(keys, values))
jax.tree_util.register_pytree_node(AttrMap, _flatten_attrs, _unflatten_attrs)