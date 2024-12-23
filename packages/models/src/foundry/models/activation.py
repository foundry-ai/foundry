from flax.linen.activation import *

import jax
import foundry.numpy as jnp
import flax.linen as nn

def silu(x):
    return x * jax.nn.sigmoid(x)

# def mish(x):
#     return x * jnp.tanh(jax.nn.softplus(x))

class Mish(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jax.nn.mish(x)