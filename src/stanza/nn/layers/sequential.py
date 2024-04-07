from stanza import struct
from typing import Any, Sequence, Callable
import jax

@struct.dataclass
class Sequential:
    layers: Sequence[Any]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x