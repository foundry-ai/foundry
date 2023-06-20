import haiku as hk
import jax.numpy as jnp
import jax
from haiku.initializers import VarianceScaling

from typing import Optional

_w_init = VarianceScaling(1.0, "fan_in", "uniform")
_b_init = VarianceScaling(1.0, "fan_in", "uniform")

class SinusoidalPosEmbed(hk.Module):
    def __init__(self, dim, name=None):
        super().__init__(name=name)
        self.dim = dim

    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[jnp.newaxis,...] * emb[...,jnp.newaxis]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=0)
        return emb.reshape((-1))

class Downsample1D(hk.Module):
    def __call__(self, x):
        conv = hk.Conv1D(x.shape[-1],
                    kernel_shape=3,
                    stride=2,
                    padding=[(1,1)], 
                    w_init=_w_init,
                    b_init=_b_init,
                    name='conv')
        return conv(x)

class Upsample1D(hk.Module):
    def __call__(self, x):
        conv = hk.Conv1DTranspose(x.shape[-1],
                    kernel_shape=4,
                    stride=2,
                    w_init=_w_init,
                    b_init=_b_init,
                    padding=[(2,2)],
                    name='conv_transpose')
        return conv(x)

def mish(x): return x * jnp.tanh(jax.nn.softplus(x))

class Conv1DBlock(hk.Module):
    def __init__(self, output_channels, 
                 kernel_size, n_groups=8,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.n_groups = n_groups

    def __call__(self, x):
        block = hk.Sequential([
            hk.Conv1D(self.output_channels,
                      kernel_shape=self.kernel_size,
                      padding=(self.kernel_size // 2, self.kernel_size // 2),
                      w_init=_w_init,
                      b_init=_b_init,
                      name="conv"),
            # We have no batch axes in our input
            # so average over all axes
            hk.GroupNorm(self.n_groups, axis=slice(0,None),name="group_norm"),
            mish
        ])
        return block(x)

class CondResBlock1D(hk.Module):
    def __init__(self, output_channels,
                 kernel_size=3, n_groups=8, name: Optional[str] = None):
        super().__init__(name=name)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.n_groups = n_groups
    
    def __call__(self, x, cond):
        block0 = Conv1DBlock(self.output_channels, 
                    self.kernel_size, n_groups=self.n_groups, name='block0')
        block1 = Conv1DBlock(self.output_channels,
                    self.kernel_size, n_groups=self.n_groups, name='block1')
        residual_conv = hk.Conv1D(
            self.output_channels, 1,
            w_init=_w_init, b_init=_b_init,
            name='residual_conv') \
                if x.shape[-1] != self.output_channels else (lambda x: x)
        cond_encoder = hk.Sequential(
            [mish, hk.Linear(self.output_channels*2,
                            w_init=_w_init, b_init=_b_init,
                            name='cond_encoder')]
        )
        out = block0(x)
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


class ConditionalUnet1D(hk.Module):
    def __init__(self, diffusion_step_embed_dim=256,
                        down_dims=[256,512,1024],
                        kernel_size=5, n_groups=8,
                        name=None):
        super().__init__(name=name)
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.down_dims = down_dims
        self.kernel_size = 5
        self.n_groups = 8

    def __call__(self, sample, timestep, global_cond=None):
        down_dims = self.down_dims
        start_dim = down_dims[0]
        kernel_size = self.kernel_size
        n_groups = self.n_groups
        mid_dim = down_dims[-1]
        dsed = self.diffusion_step_embed_dim

        diffusion_step_encoder = hk.Sequential([
            SinusoidalPosEmbed(dsed, name='diff_embed'),
            hk.Linear(4*dsed, w_init=_w_init, b_init=_b_init,
                      name='diff_embed_linear_0'),
            mish,
            hk.Linear(dsed, w_init=_w_init, b_init=_b_init,
                      name='diff_embed_linear_1')
        ])
        # encode a timesteps array
        # condition on timesteps + global_cond
        x = sample
        global_feat = diffusion_step_encoder(jnp.atleast_1d(timestep))
        if global_cond is not None:
            global_feat = jnp.concatenate((global_feat, global_cond), -1)

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

        final_conv = hk.Sequential([
            Conv1DBlock(start_dim, kernel_size=kernel_size,
                        name='final_conv_block'),
            hk.Conv1D(sample.shape[-1], 1, 
                      w_init=_w_init,
                      b_init=_b_init,
                      name='final_conv')
        ])
        x = final_conv(x)
        return x