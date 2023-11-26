import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.linen.initializers as initializers

from typing import Sequence, Callable, Any
ModuleDef = Any

import stanza.nn.activation as activations
from stanza.nn.attention import AttentionBlock
from stanza.nn.embed import SinusoidalPosEmbed

class Normalization(nn.Module):
    spatial_dims: int
    num_groups: int = 32

    @nn.compact
    def __call__(self, x):
        gn = nn.GroupNorm(self.num_groups, name="group_norm")
        if len(x.shape) == self.spatial_dims + 1:
            x = jnp.expand_dims(x, 0)
            normed_x = gn(x)
            normed_x = jnp.squeeze(normed_x, 0)
        elif len(x.shape) > self.spatial_dims + 2:
            raise ValueError(f"Input tensor has too many dimensions: {x.shape}")
        else:
            normed_x = gn(x)
        return normed_x

# A sequential where keyword arguments
# are shared
class SharedSequential(nn.Module):
    layers: Sequence[Callable[..., Any]]

    @nn.compact
    def __call__(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

class IgnoreExtra(nn.Module):
    layer: Callable[..., Any]

    @nn.compact
    def __call__(self, x, **kwargs):
        return self.layer(x)

class ResBlock(nn.Module):
    out_channels: int = None
    dropout: float = None
    use_conv: bool = False
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    use_scale_shift_norm: bool = False
    dims: int = 2 # signal dimension

    @nn.compact
    def __call__(self, x, *, cond_embed=None, train=False,
                             spatial_shape=None):
        in_layers = nn.Sequential([
            Normalization(self.dims),
            activations.silu,
        ])
        out_channels = self.out_channels or x.shape[-1]
        in_conv = nn.Conv(
            out_channels,
            self.dims*(3,))
        out_norm = Normalization(self.dims)
        out_layers = nn.Sequential([
                activations.silu,
            ] + ([nn.Dropout(self.dropout, deterministic=not train)]
                if self.dropout is not None else [])
        )
        skip_connection = nn.Conv(out_channels, self.dims*(1,))

        h = in_layers(x)
        if self.up:
            h = Upsample(self.dims*(2,))(h)
            x = Upsample(self.dims*(2,))(x)
        if self.down:
            h = Downsample(self.dims*(2,))(h)
            x = Downsample(self.dims*(2,))(x)
        h = in_conv(h)

        if cond_embed is not None:
            film = nn.Sequential([
                nn.Dense(cond_embed.shape[-1]),
                activations.silu, 
                nn.Dense(2*out_channels 
                    if self.use_scale_shift_norm 
                    else out_channels
                )
            ])(cond_embed)
            if self.use_scale_shift_norm:
                scale, shift = jnp.split(film, 2, axis=-1)
                h = out_norm(h) * (1 + scale) + shift
            else:
                h = out_norm(h + film)
        h = out_layers(h)
        return skip_connection(x) + h

class Upsample(nn.Module):
    dims: int
    out_channels: int = None
    scale_factors: int | Sequence[int] = 2
    conv: bool = False

    @nn.compact
    def __call__(self, x, spatial_shape=None, **kwargs):
        scale_factors = (self.scale_factors,) * self.dims \
            if isinstance(self.scale_factors, int) else self.scale_factors
        out_ch = self.out_channels or x.shape[-1]

        # keep non-spatial dimensions the same
        if spatial_shape is not None:
            dest_shape = (
                x.shape[:-1-self.dims] + \
                tuple(spatial_shape) + 
                (x.shape[-1],)
            )
        else:
            dest_shape = (
                x.shape[:-1-self.dims] + \
                tuple((f * s for f, s in zip(scale_factors, x.shape[-1-self.dims:-1]))) + \
                (x.shape[-1],)
            )
        x = jax.image.resize(x, dest_shape, "nearest")
        if self.conv:
            x = nn.Conv(out_ch, self.dims*(3,))(x)
        return x

class Downsample(nn.Module):
    dims: int
    out_channels: int = None
    scale_factors: int | Sequence[int] = 2
    conv: bool = False

    @nn.compact
    def __call__(self, x, **kwargs):
        out_ch = self.out_channels or x.shape[-1]
        if self.conv:
            x = nn.Conv(out_ch, self.dims*(3,), self.scale_factors)(x)
        else:
            assert self.out_channels is None or self.out_channels == x.shape[-1]
            x = nn.avg_pool(x, self.scale_factors, self.scale_factors, "VALID")
        return x

class UNet(nn.Module):
    base_channels: int = 64 # channels of the first layer
    out_channels: int = None # outut channels, uses input channels if None
    num_res_blocks: int = 2 # number of res blocks per downsample
    attention_levels: Sequence[int] = (4,8) # downsampling resolutions to use attention
    channel_mult: Sequence[int] = (1, 2, 4, 8) # channel multiplier per downsample
    scale_factors: int | Sequence[int] = 2 # scale factors for upsampling/downsampling
    dropout: float = 0.0
    conv_resample: bool = True
    num_heads: int = 1
    num_head_channels: int = None # if specified, use this instead of num_heads
                                  # with a fixed number of channels per head
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    dims: int = 2 # signal dimension

    dtype: Any = jnp.float32
    num_classes: int = None # if cond is not one-hot encoded,
                            # can be used to specify the number of classes
    def _setup(self, x):
        ch = self.base_channels
        input_blocks = [IgnoreExtra(nn.Conv(ch, self.dims*(3,)))]
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        out_channels=ch*mult,
                        dropout=self.dropout,
                        use_conv=self.conv_resample,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        dims=self.dims,
                    )
                ]
            if level in self.attention_levels:
                layers.append(
                    IgnoreExtra(AttentionBlock(
                        num_heads=self.num_heads,
                        num_head_channels=self.num_head_channels,
                        dims=self.dims
                    ))
                )
            if level != len(self.channel_mult) - 1:
                if self.resblock_updown:
                    layers.append(
                        ResBlock(
                            out_channels=ch*mult,
                            dropout=self.dropout,
                            use_conv=self.conv_resample,
                            use_scale_shift_norm=self.use_scale_shift_norm,
                            down=True,
                            dims=self.dims,
                        )
                    )
                else:
                    layers.append(
                        Downsample(
                            dims=self.dims,
                            scale_factors=self.scale_factors,
                            conv=self.conv_resample,
                        )
                    )
                input_blocks.append(SharedSequential(layers))
        middle_block = SharedSequential([
            ResBlock(
                dims=self.dims,
                out_channels=ch*self.channel_mult[-1],
                dropout=self.dropout,
                use_scale_shift_norm=self.use_scale_shift_norm,
            ),
            IgnoreExtra(AttentionBlock(
                dims=self.dims,
                num_heads=self.num_heads,
                num_head_channels=self.num_head_channels,
            )),
            ResBlock(
                dims=self.dims,
                dropout=self.dropout,
                use_scale_shift_norm=self.use_scale_shift_norm,
            )
        ])
        output_blocks = []
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for _ in range(self.num_res_blocks + 1):
                layers = [
                    ResBlock(
                        dims=self.dims,
                        out_channels=self.base_channels*mult,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                    )
                ]
            if level in self.attention_levels:
                layers.append(
                    IgnoreExtra(AttentionBlock(
                        dims=self.dims,
                        num_heads=self.num_heads,
                        num_head_channels=self.num_head_channels,
                    ))
                )
            if level > 0:
                layers.append(
                    ResBlock(
                        out_channels=ch,
                        dropout=self.dropout,
                        dims=self.dims,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        up=True,
                    )
                    if self.resblock_updown
                    else Upsample(
                        conv=self.conv_resample, 
                        dims=self.dims,
                        scale_factors=self.scale_factors,
                        out_channels=ch
                    )
                )
            output_blocks.append(SharedSequential(layers))
        out_channels = self.out_channels or x.shape[-1]
        out = nn.Sequential([
            Normalization(self.dims),
            activations.silu,
            nn.Conv(out_channels, self.dims*(3,), 
                    kernel_init=initializers.zeros_init())
        ])
        return input_blocks, middle_block, output_blocks, out

    @nn.compact
    def __call__(self, x, *, cond=None, cond_embed=None, train=False):
        embed = cond_embed
        if cond is not None:
            assert cond_embed is None
            assert self.num_classes is not None
            assert cond.shape == x.shape[:-1-self.dims]
            cond_embed = nn.Embed(self.num_classes, self.embed_dim)(cond)
            if embed is not None:
                embed = jnp.concatenate([embed, cond_embed], axis=-1)

        input_blocks, middle_block, output_blocks, out = self._setup(x)
        h = x.astype(self.dtype)
        hs = []
        spatial_shapes = []
        for module in input_blocks:
            h = module(h, cond_embed=embed, train=train)
            spatial_shapes.append(h.shape[-1-self.dims:-1])
            hs.append(h)
        spatial_shapes.pop() # remove the last spatial shape
        h = middle_block(h, cond_embed=embed, train=train)
        for module in output_blocks:
            res = hs.pop()
            h = jnp.concatenate((h, res), axis=-1)
            spatial_shape = spatial_shapes.pop() if spatial_shapes else None
            h = module(
                h, 
                spatial_shape=spatial_shape,
                cond_embed=embed, 
                train=train
            )
        h = h.astype(x.dtype)
        # define the output block, so we can
        # use the input number of channels
        return out(h)


class DiffusionUNet(UNet):
    embed_dim: int = 64

    @nn.compact
    def __call__(self, x, *, timestep=None, time_embed=None,
                            cond=None, cond_embed=None, train=False):
        if timestep is not None and time_embed is None:
            embed = nn.Sequential([
                SinusoidalPosEmbed(self.embed_dim),
                nn.Dense(self.embed_dim),
                activations.silu,
                nn.Dense(self.embed_dim)
            ])(timestep)
        if cond_embed is not None:
            embed = jnp.concatenate([embed, cond_embed], axis=-1) \
                if embed is not None else cond_embed
        return super().__call__(x, cond=cond, cond_embed=embed, train=train)