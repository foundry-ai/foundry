import haiku as hk
import jax.numpy as jnp
import jax

from typing import Optional

class SinusoidalPosEmbed(hk.Module):
    def __init__(self, dim, name=None):
        super().__init__(name=name)
        self.dim = dim

    def __call__(self, x):
        half_dim = x.shape[-1] // 2
        emb = jnp.log(1000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:,jnp.newaxis] * emb[jnp.newaxis,:]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb
    
class Downsample1D(hk.Module):
    def __call__(self, x):
        conv = hk.Conv1D(x.shape[-1],
                    kernel_shape=3,
                    stride=2,
                    padding=(1,1))
        return conv(x)

class Upsample1D(hk.Module):
    def __call__(self, x):
        conv = hk.Conv1DTranspose(x.shape[-1],
                    kernel_shape=4,
                    stride=2,
                    padding=(1,1))
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
                      padding=(self.kernel_size // 2, self.kernel_size // 2)),
            hk.GroupNorm(self.n_groups),
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
        block0 = Conv1DBlock(self.out_channels, self.kernel_size, n_groups=self.n_groups)
        block1 = Conv1DBlock(self.out_channels, self.kernel_size, n_groups=self.n_groups)
        residual_conv = hk.Conv1D(
            self.output_channels, 1
        ) if x.shape[-1] != self.output_channels else (lambda x: x)
        cond_encoder = hk.Sequential(
            [mish, hk.Linear(self.output_channels*2)]
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
        self.diffusion_step_embed_dim = 256
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
            SinusoidalPosEmbed(dsed),
            hk.Linear(4*dsed),
            mish,
            hk.Linear(4*dsed, dsed)
        ])
        # encode a timesteps array
        # condition on timesteps + global_cond
        x = sample
        global_feat = diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feat = jnp.concatenate((global_feat, global_cond), -1)

        # skip connections
        hs = []
        for ind, dim_out in enumerate(down_dims):
            is_last = ind >= (len(down_dims) - 1)
            res1 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups)
            res2 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups)
            x = res1(x, global_feat)
            x = res2(x, global_feat)
            hs.append(x)
            if not is_last:
                ds = Downsample1D()
                x = ds(x)

        mid1 = CondResBlock1D(mid_dim, kernel_size=kernel_size,
                        n_groups=n_groups),
        mid2 = CondResBlock1D(mid_dim, kernel_size=kernel_size,
                        n_groups=n_groups)
        x = mid1(x, global_cond)
        x = mid2(x, global_cond)

        for ind, (dim_out, h) in enumerate(reversed(zip(down_dims[:-1], h))):
            is_last = ind >= (len(down_dims) - 1)
            res1 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups),
            res2 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups),
            x = jnp.concatenate((x, h), axis=1)
            x = res1(x, global_feat)
            x = res2(x, global_feat)
            if not is_last:
                us = Upsample1D()
                x = us(x)

        final_conv = hk.Sequential([
            Conv1DBlock(start_dim, kernel_size=kernel_size),
            hk.Conv1D(sample.shape[-1], 1)
        ])
        x = final_conv(x)
        return x