import contextlib

from stanza.util.random import PRNGSequence
from stanza.transform.lift import Mutable
from stanza.transform import is_array
from stanza import struct

class MutVariables(Mutable):
    def __init__(self, variables=None):
        self.variables = {} if variables is None else variables
    
    def __contains__(self, key):
        return key in self.variables
    
    def __setitem__(self, key, value):
        assert is_array(value) or isinstance(value, MutVariables)
        self.variables[key] = value
    
    def __getitem__(self, key):
        return self.variables.setdefault(key, MutVariables())
    
    def freeze(self):
        return {k: (v.freeze() if isinstance(v, MutVariables) else v) \
                for k, v in self.variables.items()}

    def unfreeze(self, value):
        self.variables = MutVariables.from_frozen(value).variables
    
    def from_frozen(value):
        return MutVariables(
            {k: (MutVariables.from_frozen(v) if isinstance(v, dict) else v)
             for k, v in value.items()})
    
    def __repr__(self):
        return f"{self.variables}"

@struct.dataclass
class Context:
    variables: MutVariables = struct.field(default_factory=MutVariables)
    mutable: bool = False
    init_rng: PRNGSequence = None

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