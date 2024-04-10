import contextlib

from stanza.random import PRNGSequence
from stanza.transform.cell import Cell
from stanza.transform.lift import is_array
from stanza import struct

from typing import Any

@struct.dataclass
class Variables:
    def __init__(self, variables=None):
        self.variables = Cell({}) if variables is None else variables

    def __contains__(self, key):
        return key in self.variables.get()

    def __setitem__(self, key, value):
        assert is_array(value) or isinstance(value, Variables)
        d = dict(self.variables.get())
        d[key] = value
        self.variables.set(d)

    def __getitem__(self, key):
        return self.variables.setdefault(key, Variables())
    
    def __repr__(self):
        return f"{self.variables}"

@struct.dataclass
class Context:
    variables: Variables = Variables()
    rngs: dict[str, PRNGSequence] = struct.field(default_factory=dict)

    def get(self, name, initializer, shape, dtype):
        if name not in self.variables:
            var = initializer(next(self.init_rng), shape, dtype)
            self.variables[name] = var
        else:
            var = self.variables[name]
        if var.shape != shape:
            raise ValueError(f"Parameter {name} has shape {var.shape}, expected {shape}")
        if var.dtype != dtype:
            raise ValueError(f"Parameter {name} has dtype {var.dtype}, expected {dtype}")
        return var

    @contextlib.contextmanager
    def scope(self, name):
        yield self[name]

    # Get a subcontext, with the same initialization RNG
    # but different variables
    def __getitem__(self, key):
        return Context(
            self.variables[key],
            self.init_rng
        )