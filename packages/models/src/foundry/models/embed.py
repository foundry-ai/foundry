import flax.linen as nn
import foundry.numpy as jnp

class SinusoidalPosEmbed(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = jnp.atleast_1d(x)
        assert len(x.shape) == 1
        assert self.dim % 2 == 0
        half_dim = self.dim // 2

        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        x = jnp.array(x, dtype=jnp.float32)
        emb = x[jnp.newaxis,...] * emb[...,jnp.newaxis]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=0)
        return emb.reshape((-1))

class RandomOrLearnedSinusoidalEmbed(nn.Module):
    dim: int
    random: bool = False

    @nn.compact
    def __call__(self, x):
        assert len(x.shape) == 1
        assert self.dim % 2 == 0
        half_dim = self.dim // 2
        if self.random:
            w = self.variable("weights", nn.initializers.normal, half_dim)
        else:
            w = self.param("weights", nn.initializers.normal, half_dim)

        x = jnp.expand_dims(x, -1)
        w = jnp.expand_dims(w, 0) * 2 * jnp.pi
        freqs = x * w
        fouriered = jnp.concatenate((jnp.sin(freqs), jnp.cos(freqs)), axis=-1)
        fouriered = jnp.concatenate((fouriered, x), axis=-1)
        return fouriered