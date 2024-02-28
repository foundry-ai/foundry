import jax
import jax.flatten_util
import jax.numpy as jnp
import flax.linen as nn

from stanza.nn.embed import SinusoidalPosEmbed
from typing import Sequence

class DiffusionMLP(nn.Module):
    features: Sequence[int]
    time_embed_dim: int = 32

    @nn.compact
    def __call__(self, x, timestep):
        x_flat, x_uf = jax.flatten_util.ravel_pytree(x)
        x = x_flat
        embed = SinusoidalPosEmbed(self.time_embed_dim)(timestep)
        embed = nn.Dense(self.time_embed_dim)(embed)
        for f in self.features:
            x = nn.Dense(f)(x)
            shift_scale = nn.Dense(2*f)(embed)
            shift, scale = jnp.split(shift_scale, 2, axis=-1)
            x = x*(1 + scale) + shift
            x = nn.gelu(x)
        x = nn.Dense(x_flat.shape[0])(x)
        return x_uf(x)