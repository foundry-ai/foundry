import flax.linen as nn

import jax.numpy as jnp

class SinusoidalPosEmbed(nn.Module):
    dim : int

    @nn.compact
    def __call__(self, x):
        x = jnp.array(x)
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[jnp.newaxis,...] * emb[...,jnp.newaxis]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=0)
        return emb.reshape((-1))