import jax
import stanza

from stanza.random import PRNGSequence
from stanza.transform.cell import FrozenCell, Cell
from stanza.transform.lift import is_array
from stanza import struct

from typing import Any, Callable, NamedTuple

class Collection:
    def __init__(self, vars=None):
        self._vars = Cell({}) if vars is None else vars
    
    def __setitem__(self, key, value):
        if self._vars.frozen:
            raise ValueError("Cannot modify frozen collection.")
        vars = dict(self._vars.get())
        vars[key] = value
        self._vars.set(vars)
    
    def __getitem__(self, key):
        vars = self._vars.get()
        return vars[key]

    def __repr__(self):
        return f"{self._vars.get()}"

jax.tree_util.register_pytree_node(
    Collection,
    lambda x: (x._vars,), None,
    lambda _aux, children: Collection(children[0])
)

class Variables:
    def __init__(self, submodules=None, collections=None):
        # the submodules for this module
        self._submodules : Cell[dict[str, Variables]] = (
            Cell({}) if submodules is None else submodules
        )
        # the variables for this module
        self._collections : Cell[dict[str, Collection]] = (
            Cell({}) if collections is None else collections
        )

    def has(self, collection, key):
        vars = self._collections.get()
        if not collection in vars:
            return False
        collection = vars[collection].get()
        return key in collection
    
    def get(self, collection, key) -> jax.Array:
        vars = self._collections.get()
        if not collection in vars:
            raise ValueError(f"Collection {collection} not found.")
        collection = vars[collection].get()
        if not key in collection:
            raise ValueError(f"Key {key} not found in collection {collection}.")
        return collection[key]
    
    def set(self, collection, key, value : jax.Array):
        vars = self._collections.get()
        if not collection in vars:
            # copy the dictionary to avoid modifying the original
            vars = dict(vars)
            vars[collection] = Cell({})
            self.variables.set(vars)
        collection = vars[collection]
        c = dict(collection.get())
        c[key] = value
        collection.set(c)
    
    # will return a dictionary containing all variables in
    # the specified collections
    def pop(self, collections, prefix=None):
        colls = self._collections.get()
        rem_cols = {}

        split_vars = {}
        for c in collections:
            if c not in colls:
                split_vars[c] = {}
                continue
            split_vars[c] = colls

        # add the prefix to the keys for each collection
        if prefix is not None:
            split_vars = {name: {f"{prefix}{k}": v for k, v in coll.items()} for name, coll in split_vars.items()}
        for k, mod in self._submodules.get().items():
            # split out the submodule
            sub_prefix = f"{k}." if prefix is None else f"{prefix}{k}."
            vars = mod.pop(collections, sub_prefix)
            for c in collections:
                split_vars[c].update(vars[c])
        
        return split_vars
        return Variables(
            self._submodules,
            FrozenCell({k: self._collections.get() for k in collections})
        )
    
    def push(self, vars):
        pass

    def __repr__(self):
        return f"{self._collections.get()}"

jax.tree_util.register_pytree_node(
    Variables,
    lambda x: (x.variables,), None,
    lambda _aux, children: Variables(children[0])
)

@struct.dataclass
class Context:
    vars : Variables = struct.field(default_factory=Variables)
    rngs: dict[str, PRNGSequence] = struct.field(default_factory=dict)

    def get(self, name, initializer, shape, dtype):
        if name not in self.vars:
            if "init" not in self.rngs:
                raise ValueError(f"Trying to initialize variable {name} without initialization RNG.")
            var = initializer(
                self.rngs["init"],
                shape, dtype
            )
            self.vars[name] = var
        else:
            var = self.variables[name]
        if var.shape != shape:
            raise ValueError(f"Parameter {name} has shape {var.shape}, expected {shape}")
        if var.dtype != dtype:
            raise ValueError(f"Parameter {name} has dtype {var.dtype}, expected {dtype}")
        return var
    
    def rng(self, name):
        return self.rngs[name] if name in self.rngs else None

    def __div__(self, key):
        return Context(
            self.variables[key],
            self.rngs
        )

class Lifted(NamedTuple):
    init: Callable
    apply: Callable

def transform(function: Callable) -> Lifted:
    @stanza.jit
    def init(rng : PRNGSequence, *args, **kwargs):
        ctx = Context(rngs={"init": rng})
        function(ctx, *args, **kwargs)
        return ctx

    @stanza.jit
    def apply(ctx, *args, **kwargs):
        function(ctx, *args, **kwargs)

    return Lifted(init, apply)