import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen.initializers import variance_scaling
import flax.linen.activation as activations

from typing import Tuple

from stanza.util import vmap_ravel_pytree

_w_init = variance_scaling(1.0, "fan_in", "uniform")
_b_init = variance_scaling(1.0, "fan_in", "uniform")

class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim, name=None):
        super().__init__(name=name)
        self.dim = dim

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[jnp.newaxis,...] * emb[...,jnp.newaxis]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=0)
        return emb.reshape((-1))

class Downsample1D(nn.Module):
    @nn.compact
    def __call__(self, x):
        conv = nn.Conv(x.shape[-1],
                    kernel_size=3,
                    strides=2,
                    padding=[(1,1)], 
                    kernel_init=_w_init,
                    bias_init=_b_init,
                    name='conv')
        return conv(x)

class Upsample1D(nn.Module):
    @nn.compact
    def __call__(self, x):
        conv = nn.Conv1DTranspose(x.shape[-1],
                    kernel_size=4,
                    strides=2,
                    kernel_init=_w_init,
                    bias_init=_b_init,
                    padding=[(2,2)],
                    name='conv_transpose')
        return conv(x)

def mish(x): return x * jnp.tanh(jax.nn.softplus(x))

class Conv1DBlock(nn.Module):
    output_channels: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        conv = nn.Conv1D(self.output_channels,
                      kernel_shape=self.kernel_size,
                      padding=(self.kernel_size // 2, self.kernel_size // 2),
                      w_init=_w_init,
                      b_init=_b_init,
                      name="conv")
        gn = nn.GroupNorm(self.n_groups, axis=slice(0,None),name="group_norm")
        x = conv(x)
        # if we only have (dim, channels), expand to (1, dim, channels)
        # so the group norm does not interpret dim as the batch dimension
        if len(x.shape) == 2:
            x = jnp.expand_dims(x, 0)
            normed_x = gn(x)
            normed_x = normed_x.squeeze(0)
        else:
            normed_x = gn(x)
        x = mish(normed_x)
        return x

class CondResBlock1D(nn.Module):
    output_channels: int
    kernel_size: int = 3
    n_groups: int = 8
    
    @nn.compact
    def __call__(self, x, cond):
        block0 = Conv1DBlock(self.output_channels, 
                    self.kernel_size, n_groups=self.n_groups, name='block0')
        block1 = Conv1DBlock(self.output_channels,
                    self.kernel_size, n_groups=self.n_groups, name='block1')
        residual_conv = nn.Conv1D(
            self.output_channels, 1,
            kernel_init=_w_init, bias_init=_b_init,
            name='residual_conv'
        ) if x.shape[-1] != self.output_channels else (lambda x: x)
        out = block0(x)

        if cond is not None:
            cond_encoder = nn.Sequential(
                [mish, nn.Dense(self.output_channels*2,
                                kernel_init=_w_init, bias_init=_b_init,
                                name='cond_encoder')]
            )
            embed = cond_encoder(cond)
        # reshape into (dims, 2, output_channels)
        embed = embed.reshape(
            embed.shape[:-1] + (2, self.output_channels))
        scale = embed[...,0,:]
        bias = embed[...,1,:]
        # broadcast add the scale, bias along the spatial
        # dimension (i.e, per channel modulation, effectively
        # modulating the weights of the next layer)
        out = jnp.expand_dims(scale, -2)*out + jnp.expand_dims(bias, -2)
        out = block1(out)
        out = out + residual_conv(x)
        return out


class ConditionalUNet1D(nn.Module):
    step_embed_dim: int = None
    down_dims: Tuple[int] = (256,512,1024)
    kernel_size: int = 5
    n_groups: int = 8
    final_activation: str = 'tanh'

    @nn.compact
    def __call__(self, sample, timestep, global_cond=None):
        down_dims = self.down_dims
        start_dim = down_dims[0]
        kernel_size = self.kernel_size
        n_groups = self.n_groups
        mid_dim = down_dims[-1]

        # encode a timesteps array
        # condition on timesteps + global_cond

        # flatten the sample into (dim, channels)
        sample, sample_uf = vmap_ravel_pytree(sample)
        x = sample

        if self.step_embed_dim is not None:
            dsed = self.step_embed_dim
            diffusion_step_encoder = nn.Sequential([
                SinusoidalPosEmbed(dsed, name='diff_embed'),
                nn.Dense(4*dsed, kernel_init=_w_init, bias_init=_b_init,
                        name='diff_embed_linear_0'),
                mish,
                nn.Dense(dsed, w_init=_w_init, b_init=_b_init,
                        name='diff_embed_linear_1')
            ])
            global_feat = diffusion_step_encoder(jnp.atleast_1d(timestep))

        if global_cond is not None:
            global_cond, _ = jax.flatten_util.ravel_pytree(global_cond)
            global_feat = jnp.concatenate((global_feat, global_cond), -1) \
                if global_feat is not None else global_cond

        # skip connections
        hs = []
        for ind, dim_out in enumerate(down_dims):
            is_last = ind >= (len(down_dims) - 1)
            res0 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups,
                                  name=f'down{ind}_res0')
            res1 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups,
                                  name=f'down{ind}_res1')
            x = res0(x, global_feat)
            # print(f'down{ind}', x)
            x = res1(x, global_feat)
            hs.append(x)
            if not is_last:
                ds = Downsample1D(name=f'down{ind}_downsample')
                x = ds(x)

        # print('pre_mid', x)
        mid0 = CondResBlock1D(mid_dim, kernel_size=kernel_size,
                        n_groups=n_groups, name='mid0')
        mid1 = CondResBlock1D(mid_dim, kernel_size=kernel_size,
                        n_groups=n_groups, name='mid1')
        x = mid0(x, global_feat)
        x = mid1(x, global_feat)
        # print('post_mid', x)

        for ind, (dim_out, h) in enumerate(zip(
                    reversed(down_dims[:-1]), reversed(hs)
                )):
            is_last = ind >= (len(down_dims) - 1)
            res0 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups,
                                  name=f'up{ind}_res0')
            res1 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups,
                                  name=f'up{ind}_res1')
            x = jnp.concatenate((x, h), axis=-1)
            x = res0(x, global_feat)
            x = res1(x, global_feat)
            if not is_last:
                us = Upsample1D(name=f'up{ind}_upsample')
                x = us(x)

        final_conv = nn.Sequential([
            Conv1DBlock(start_dim, kernel_size=kernel_size,
                        name='final_conv_block'),
            nn.Conv1D(sample.shape[-1], 1, 
                      kernel_init=_w_init,
                      bias_init=_b_init,
                      name='final_conv')
        ])
        x = final_conv(x)
        if self.final_activation is not None:
            activation = getattr(activations, self.final_activation)
            x = activation(x)
        return sample_uf(x)