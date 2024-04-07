import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import jax
import math

import functools

from typing import Sequence, Callable, Any

import stanza.nn.activations as activations

class AttentionPool2D(nn.Module):
    num_channels: int
    output_dim: int = None

    @nn.compact
    def __call__(self, x):
        h, w, c = x.shape[-3:]
        pos_embed = self.param("positional_embedding",
            lambda rng, shape: jax.random.normal(rng, shape, x.dtype)/shape[0]**0.5, 
                    (c, h*w + 1)
        )
        qkv_proj = nn.Conv(3*self.embed_dim, 1)
        c_proj = nn.Conv(self.output_dim or self.embed_dim, 1)
        num_heads = c // self.num_heads_channels
        attention = QKVAttention(num_heads)
        x_flat = x.reshape(x.shape[:-3] + (h*w, c))
        x_flat = jnp.concatenate((x_flat, jnp.mean(x_flat, axis=-2, keepdims=True)), axis=-2)
        pos_embed = jnp.broadcast_to(pos_embed, x_flat.shape)
        x_flat = x + pos_embed
        q, k, v = qkv_proj(x_flat).moveaxis(-1, 0)
        q = jnp.reshape(q, q.shape[:-1] + (num_heads, self.num_heads_channels))
        x = attention(q, k, v)
        x = c_proj(x)
        return x[..., 0, :]

class AttentionBlock(nn.Module):
    dims: int # signal dimension. 
    num_heads: int
    num_head_channels: int = None # if specified, use this instead of num_heads

    @nn.compact
    def __call__(self, x):
        channels = x.shape[-1]
        attention = QKVAttention(
            dims=self.dims,
            num_heads=self.num_heads, 
            num_head_channels=self.num_head_channels,
        )
        qkv_proj = nn.Conv(3*channels, self.dims*(1,))
        out_proj = nn.Conv(channels, self.dims*(1,))

        qkv = qkv_proj(x)
        q,k,v = jnp.split(qkv, 3, axis=-1)
        h = attention(q,k,v)
        h = out_proj(h)
        return h

class QKVAttention(nn.Module):
    num_heads: int = 1
    num_head_channels: int = None # if specified, use this instead of num_heads
                                  # gives a number of channels per head
    dims: int = 1 # number of spatial dimensions

    @nn.compact
    def __call__(self, q, k=None, v=None):
        """
        Apply QKV attention.

        :param q,k,v: [... x Spatial Dims X H*C] tensors of Qs, Ks, and Vs.
                        If k,v are none Q can contain all of them.
        :return: an [... x Spatial Dims X H*C] tensor after attention.
        """
        if k is None and v is None:
            # if k and v are not specified, q contains everything
            # 3*H*C -> H*C
            q,k,v = jnp.split(q, 3, axis=-1)
        # flatten the spatial dims
        spatial_dims = q.shape[-self.dims-1:-1]
        T = np.prod(spatial_dims)
        channels = q.shape[-1]
        # flatten the spatial dims, swap the last two axes
        q,k,v = jax.tree_map(
            lambda x: jnp.moveaxis(jnp.reshape(x,
                x.shape[:-self.dims-1] + (T, channels)
            ), -1, -2), (q,k,v)
        )
        # q,k,v are now [... x H*C x T]
        num_heads = (self.num_heads if self.num_head_channels is None else 
                     channels // self.num_head_channels)
        head_channels = channels // num_heads
        q,k,v = jax.tree_map(
            lambda x: jnp.reshape(x, x.shape[:-2] + (num_heads, head_channels,-1)),
            (q,k,v)
        )
        # q,k,v are [... x H x C x T ]
        scale = 1 / math.sqrt(math.sqrt(head_channels))
        # weight is [... x H x T x T]
        weight = jnp.einsum(
            "...ct,...cs->...ts",
            q*scale, k * scale
        )
        # use float32 for the softmax, then convert back to whatever type we had
        weight = jax.nn.softmax(weight.astype(jnp.float32), axis=-1).astype(weight.dtype)
        a = jnp.einsum("...ts,...cs->...ct", weight, v)
        # a is [... x H x C x T], reshape to [... x H*C x T]
        a = jnp.reshape(a, a.shape[:-3] + (channels,T))
        # and then swap the last two axes to [... x T x H*C]
        a = jnp.moveaxis(a, -1, -2)
        # lastly, reshape to [... x Spatial Dims x H*C]
        a = jnp.reshape(a, a.shape[:-2] + spatial_dims + (channels,))
        return a
