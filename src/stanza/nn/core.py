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
    
    def freeze(self):
        return Collection(self._vars.freeze())

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
    
    def freeze(self, mutable_collections=None):
        mc = set() if mutable_collections is None else mutable_collections
        return Variables(
            FrozenCell({k: v.freeze(mutable_collections) for k,v in self._submodules.items()}),
            FrozenCell({k: (v if k in mc else v.freeze()) for k,v in self._collections.items()}),
        )
    
    def split(self, collections):
        return Variables(
            self._submodules,
            FrozenCell({k: self._collections.get() for k in collections})
        )
    
    def flatten(self):
        sub_vars = {k: v.flatten() for k,v in self._submodules.get().items()}
        # get all of the collections
        collections = set(self._collections.get().keys()).union(*[set(v.keys()) for v in sub_vars.values()])
        vars = {c: {} for c in collections}
        # set the sub modules for each collection
        for k, v in self.sub_vars:
            for c, v in v.items():
                vars[c][k] = v
        for c, coll in self._collections.get().items():
            vars[c].update(coll.get())
        vars = {k: v for k,v in vars.items() if v}
        # add the variables from this module
        vars.update({k: v for k,v in self._collections.get().get(c, {}).items()})
        return vars

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