import stanza
import jax
from typing import Sequence, Callable

from stanza import struct, nn

from stanza.nn.layers import Linear

@struct.dataclass
class MLP:
    ctx: nn.Context
    hidden_features: Sequence[int]
    output_size: int
    activation: Callable

    @stanza.jit
    def __call__(self, x : jax.Array) -> jax.Array:
        assert x.ndim == 1
        for i, features in enumerate(self.hidden_features):
            layer = Linear(self.ctx[f"layer_{i}"], features)
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        layer = Linear(self.ctx[f"layer_{i+1}"], self.output_size)
        x = layer(x) 
        return x