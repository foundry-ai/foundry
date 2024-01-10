from stanza.lift.context import Module
from typing import Any

class StructModule(Module):
    def __init__(self, *, name=None):
        super().__init__(name=name)
    
    def __setattr__(self, __name: str, __value: Any):
        object.__setattr__(self, __name, __value)
