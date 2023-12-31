from flax.linen.activation import *

import jax
import jax.numpy as jnp

def silu(x):
    return x * jax.nn.sigmoid(x)

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))