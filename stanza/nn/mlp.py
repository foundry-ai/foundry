import flax.linen as nn
import flax.linen.activation as activations


from stanza.nn.embed import SinusoidalPosEmbed

import jax
import jax.numpy as jnp

from typing import Sequence, Any

class MLP(nn.Module):
    features: Sequence[int]
    output_sample: Any = None
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, embed=None, train=False, *, output_sample=None):
        output_sample = output_sample if output_sample is not None else self.output_sample
        activation = getattr(activations, self.activation)
        x, _ = jax.flatten_util.ravel_pytree(x)
        x = jnp.reshape(x, (-1,))
        for feat in self.features:
            x = activation(nn.Dense(feat)(x))
            if embed is not None:
                film = nn.Dense(2*feat)(embed)
                scale, shift = jnp.split(film, 2, axis=-1)
                x = x * scale + shift
        out_flat, out_uf = jax.flatten_util.ravel_pytree(output_sample)
        x = nn.Dense(out_flat.shape[-1])(x)
        return out_uf(x)

class DiffusionMLP(MLP):
    time_embed_dim: int = 32
    cond_embed_dim: int = 32

    @nn.compact
    def __call__(self, x,
                    # either timestep or time_embed must be passed
                    timestep=None, time_embed=None,
                    # for a second time dimension for a latent
                    latent_timestep=None, latent_time_embed=None,
                    cond=None, cond_embed=None,
                    train=False):
        activation = getattr(activations, self.activation)
        if timestep is not None and time_embed is None:
            time_embed = nn.Sequential([
                SinusoidalPosEmbed(self.time_embed_dim),
                nn.Dense(self.time_embed_dim),
                activation,
                nn.Dense(self.time_embed_dim)
            ])(timestep)
        if latent_timestep is not None and latent_time_embed is None:
            latent_time_embed = nn.Sequential([
                SinusoidalPosEmbed(self.time_embed_dim),
                nn.Dense(self.time_embed_dim),
                activation,
                nn.Dense(self.time_embed_dim)
            ])(latent_timestep)
        if latent_time_embed is not None:
            assert time_embed is not None
            time_embed = jnp.concatenate(
                [time_embed, latent_time_embed], 
                axis=-1
            )
        if cond is not None and cond_embed is None:
            cond_embed = nn.Dense(self.cond_embed_dim)(cond)
        embed = time_embed
        if cond_embed is not None:
            embed = jnp.concatenate([embed, cond_embed], axis=-1)
        return super().__call__(x, embed=embed, train=train, output_sample=x)
