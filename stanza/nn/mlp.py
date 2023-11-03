import flax.linen as nn
import flax.linen.activation as activations


from stanza.nn.embed import SinusoidalPosEmbed

import jax
import jax.numpy as jnp

from typing import Sequence, Any

class MLP(nn.Module):
    features: Sequence[int]
    output_sample: Any
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = getattr(activations, self.activation)
        x, _ = jax.flatten_util.ravel_pytree(x)
        x = jnp.reshape(x, (-1,))
        for feat in self.features:
            x = activation(nn.Dense(feat)(x))
        out_flat, out_uf = jax.flatten_util.ravel_pytree(self.output_sample)
        x = nn.Dense(out_flat.shape[-1])(x)
        return out_uf(x)

class DiffusionMLP(nn.Module):
    features: Sequence[int]
    output_sample: Any
    activation: str = "relu"
    embed_dim: int = 32

    @nn.compact
    def __call__(self, x, timestep=None, time_embed=None, train=False):
        activation = getattr(activations, self.activation)
        if timestep is not None and time_embed is None:
            time_embed = nn.Sequential([
                SinusoidalPosEmbed(self.embed_dim),
                nn.Dense(self.embed_dim),
                activation,
                nn.Dense(self.embed_dim)
            ])(timestep)

        x, _ = jax.flatten_util.ravel_pytree(x)
        x = jnp.reshape(x, (-1,))
        for feat in self.features:
            x = activation(nn.Dense(feat)(x))
            if time_embed is not None:
                print(time_embed.shape)
                film = nn.Dense(2*feat)(time_embed)
                scale, shift = jnp.split(film, 2, axis=-1)
                print(x.shape)
                x = x * scale + shift
                print(x.shape)
        out_flat, out_uf = jax.flatten_util.ravel_pytree(self.output_sample)
        x = nn.Dense(out_flat.shape[-1])(x)
        return out_uf(x)