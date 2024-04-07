from stanza import struct

from ..core import Context
from ..initializers import Initializer, zeros, lecun_normal

import stanza
import jax
import jax.numpy as jnp

@struct.dataclass
class Linear:
    ctx: Context
    features: int
    dtype : jnp.dtype = jnp.float32
    use_bias: bool = True

    weight_initializer: Initializer = lecun_normal()
    bias_initializer: Initializer = zeros

    @stanza.jit
    def __call__(self, x : jax.Array) -> jax.Array:
        # Note: no implicit batch dimension allowed!
        assert x.ndim == 1
        W = self.ctx.create("W", self.weight_initializer,
            (self.features, x.shape[0]),
            self.dtype
        )
        b = self.ctx.create("b", self.bias_initializer,
            (self.features,),
            self.dtype
        ) if self.use_bias else None
        y = jnp.dot(W, x)
        if b is not None:
            y = y + b
        return y