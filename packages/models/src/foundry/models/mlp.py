import flax.linen as nn
import jax.flatten_util
import foundry.numpy as npx

from . import activation
from .embed import SinusoidalPosEmbed
from functools import partial
from foundry.util.registry import Registry

from collections.abc import Sequence

class MLPClassifier(nn.Module):
    n_classes: int
    features: list[int] = (64, 64, 64)
    activation: str = "gelu"

    @nn.compact
    def __call__(self, x):
        x, _ = jax.flatten_util.ravel_pytree(x)
        h = getattr(activation, self.activation)
        for i, f in enumerate(self.features):
            x = nn.Dense(f)(x)
            x = h(x)
        x = nn.Dense(self.n_classes)(x)
        return x

class DiffusionMLP(nn.Module):
    features: Sequence[int] = (64, 64, 64)
    activation: str = "gelu"
    time_embed_dim: int = 32

    @nn.compact
    def __call__(self, x, t, cond=None, train=False):
        h = getattr(activation, self.activation)
        # works even if we have multiple timesteps
        embed = SinusoidalPosEmbed(self.time_embed_dim)(t)
        embed = nn.Sequential([
            nn.Dense(self.time_embed_dim),
            h,
            nn.Dense(self.time_embed_dim),
        ])(embed)

        # concatenated embedding
        if cond is not None:
            cond_flat, _ = jax.flatten_util.ravel_pytree(cond)
            cond_flat = 2*(cond_flat - 0.5)
            cond_flat = - cond_flat
            cond_embed = nn.Sequential([
                nn.Dense(self.time_embed_dim),
                h,
                nn.Dense(self.time_embed_dim)
            ])(cond_flat)
            embed = npx.concatenate([embed, cond_embed], axis=-1)

        x, x_uf = jax.flatten_util.ravel_pytree(x)
        out_features = x.shape[-1]
        for feat in self.features:
            shift, scale = npx.split(nn.Dense(2*feat)(embed), 2, -1)
            x = h(nn.Dense(feat)(x))
            x = x * (1 + scale) + shift
        x = nn.Dense(out_features)(x)
        x = x_uf(x)
        return x

MLPLargeClassifier = partial(MLPClassifier, features=[512, 512, 128])
MLPMediumClassifier = partial(MLPClassifier, features=[128, 128, 64])
MLPSmallClassifier = partial(MLPClassifier, features=[64, 32, 32])

DiffusionMLPLarge = partial(DiffusionMLP, features=[256, 128, 256])
DiffusionMLPMedium = partial(DiffusionMLP, features=[128, 64, 128])
DiffusionMLPSmall = partial(DiffusionMLP, features=[64, 32, 64])

def register(registry: Registry, prefix=None):
    registry.register("classifier/mlp/large", MLPLargeClassifier, prefix=prefix)
    registry.register("classifier/mlp/medium", MLPMediumClassifier, prefix=prefix)
    registry.register("classifier/mlp/small", MLPSmallClassifier, prefix=prefix)

    registry.register("diffusion/mlp/large", DiffusionMLPLarge, prefix=prefix)
    registry.register("diffusion/mlp/medium", DiffusionMLPMedium, prefix=prefix)
    registry.register("diffusion/mlp/small", DiffusionMLPSmall, prefix=prefix)

    registry.register("diffusion/mlp/large/relu", partial(DiffusionMLPLarge, activation="relu"), prefix=prefix)
    registry.register("diffusion/mlp/medium/relu", partial(DiffusionMLPMedium, activation="relu"), prefix=prefix)
    registry.register("diffusion/mlp/small/relu", partial(DiffusionMLPSmall, activation="relu"), prefix=prefix)