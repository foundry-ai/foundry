import jax
import foundry.numpy as jnp
import flax.linen as nn
import flax.linen.initializers as initializers

from typing import Sequence, Callable, Any
ModuleDef = Any

from foundry.models import activation as activations
from foundry.models.attention import AttentionBlock
from foundry.models.embed import SinusoidalPosEmbed

class Upsample1d(nn.Module):
    out_channels: int = None
    stride: int = 2
    kernel_size: int = 4

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.ConvTranspose(self.out_channels, self.kernel_size, self.stride, padding=1)(x)
        return x

class Downsample1d(nn.Module):
    out_channels: int = None
    stride: int = 2
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.Conv(self.out_channels, self.kernel_size, self.stride, padding=1)(x)
        return x

class Conv1dBlock(nn.Module):
    '''
    Conv1d --> GroupNorm --> Mish
    '''
    out_channels: int
    kernel_size: int = 3
    n_groups: int = 8

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.Conv(self.out_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2)(x)
        x = nn.GroupNorm(self.n_groups)(x)
        x = activations.Mish(x)
        return x

class CondResBlock1d(nn.Module):
    out_channels: int = None
    cond_dim: int = None
    kernel_size: int = 3
    n_groups: int = 8
    cond_predict_scale: bool = False

    @nn.compact
    def __call__(self, x, *, cond=None, train=False):
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups)(x)
        if self.cond_predict_scale:
            cond_channels = self.out_channels * 2
        else:
            cond_channels = self.out_channels
        cond_encoder = nn.Sequential([
            activations.Mish(),
            nn.Dense(cond_channels)
        ])
        embed = cond_encoder(cond)[..., None]
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.n_groups)(out)
        out = out + nn.Conv1d(self.out_channels, kernel_size=1)(x)
        return out


class CondUNet1d(nn.Module):
    input_dim: int
    local_cond_dim: int = None
    global_cond_dim: int = None
    diffusion_step_embed_dim: int = 256
    down_dims: Sequence[int] = (256,512,1024)
    kernel_size: int = 3
    n_groups: int = 8
    cond_predict_scale: bool = False
    
    def setup(self):

        all_dims = [self.input_dim] + list(self.down_dims)
        start_dim = self.down_dims[0]

        dsed = self.diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential([
            SinusoidalPosEmbed(dsed),
            nn.Dense(dsed * 4),
            activations.Mish(),
            nn.Dense(dsed),
        ])
        cond_dim = dsed
        if self.global_cond_dim is not None:
            cond_dim += self.global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if self.local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = self.local_cond_dim
            local_cond_encoder = [
                # down encoder
                CondResBlock1d(
                    dim_out, cond_dim=cond_dim, 
                    kernel_size=self.kernel_size, n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale),
                # up encoder
                CondResBlock1d(
                    dim_out, cond_dim=cond_dim, 
                    kernel_size=self.kernel_size, n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale)
            ]

        mid_dim = all_dims[-1]
        self.mid_modules = [
            CondResBlock1d(
                mid_dim, cond_dim=cond_dim,
                kernel_size=self.kernel_size, n_groups=self.n_groups,
                cond_predict_scale=self.cond_predict_scale
            ),
            CondResBlock1d(
                mid_dim, cond_dim=cond_dim,
                kernel_size=self.kernel_size, n_groups=self.n_groups,
                cond_predict_scale=self.cond_predict_scale
            ),
        ]

        down_modules = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append([
                CondResBlock1d(
                    dim_out, cond_dim=cond_dim, 
                    kernel_size=self.kernel_size, n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale),
                CondResBlock1d(
                    dim_out, cond_dim=cond_dim, 
                    kernel_size=self.kernel_size, n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale),
                Downsample1d(dim_out) if not is_last else jnp.identity(dim_in)
            ])

        up_modules = []
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append([
                CondResBlock1d(
                    dim_in, cond_dim=cond_dim,
                    kernel_size=self.kernel_size, n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale),
                CondResBlock1d(
                    dim_in, cond_dim=cond_dim,
                    kernel_size=self.kernel_size, n_groups=self.n_groups,
                    cond_predict_scale=self.cond_predict_scale),
                Upsample1d(dim_in) if not is_last else jnp.identity(dim_in)
            ])
        
        final_conv = nn.Sequential([
            Conv1dBlock(start_dim, kernel_size=self.kernel_size),
            nn.Conv(self.input_dim, kernel_size=1),
        ])

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv
    
    @nn.compact
    def __call__(self, 
            sample: jnp.ndarray,
            timestep: jnp.ndarray | int = None, 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = jnp.swapaxes(sample, -1, -2)

        # 1. time
        timesteps = timestep
        if isinstance(timestep, int):
            timesteps = jnp.repeat(timestep, sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = jnp.concatenate([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = jnp.swapaxes(local_cond, -1, -2)
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = jnp.concatenate((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = jnp. swapaxes(x, -1, -2)
        return x

