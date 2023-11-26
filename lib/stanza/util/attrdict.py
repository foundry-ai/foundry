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
        # Remove None values from the attr structure
        self._dict = { k: v for k,v in d.items() if v is not None }

    def __getitem__(self, name: str):
        return self._dict.__getitem__(name)
    
    def __contains__(self, name: str):
        return self._dict.__contains__(name)

    def __setattr__(self, name: str, value:  Any):
        if name != '_dict':
            raise RuntimeError(f"Unable to set {name}, Attrs is immutable")
        super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if name.startswith('__'):
            raise AttributeError("Not found")
        return self._dict.get(name)
    
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

def attrs(*args, **kwargs):
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