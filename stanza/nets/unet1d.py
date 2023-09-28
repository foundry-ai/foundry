import flax.linen as nn
import jax.numpy as jnp
import jax
import flax.linen.initializers as initializers
import flax.linen.activation as activations

from typing import Tuple, Sequence

from stanza.util import vmap_ravel_pytree

_w_init = initializers.lecun_normal()
_b_init = initializers.zeros_init()

class SinusoidalPosEmbed(nn.Module):
    dim: int

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
                    kernel_size=(3,),
                    strides=(2,),
                    padding=[(1,1)], 
                    kernel_init=_w_init,
                    bias_init=_b_init,
                    name='conv')
        return conv(x)

class Upsample1D(nn.Module):
    @nn.compact
    def __call__(self, x):
        conv = nn.ConvTranspose(x.shape[-1],
                    kernel_size=(4,),
                    strides=(2,),
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
        conv = nn.Conv(self.output_channels,
                      kernel_size=(self.kernel_size,),
                      padding=[(self.kernel_size // 2, self.kernel_size // 2)],
                      kernel_init=_w_init,
                      bias_init=_b_init,
                      name="conv")
        x = conv(x)
        # if we only have (dim, channels), expand to (1, dim, channels)
        # so the group norm does not interpret dim as the batch dimension
        gn = nn.GroupNorm(self.n_groups, name="group_norm")
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
        residual_conv = nn.Conv(
            self.output_channels,
            kernel_size=(1,),
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
    def __call__(self, sample, timestep, cond=None):
        timestep = jnp.atleast_1d(timestep)
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

        global_feat = None
        if self.step_embed_dim is not None:
            dsed = self.step_embed_dim
            diffusion_step_encoder = nn.Sequential([
                SinusoidalPosEmbed(dsed, name='diff_embed'),
                nn.Dense(4*dsed, kernel_init=_w_init, bias_init=_b_init,
                        name='diff_embed_linear_0'),
                mish,
                nn.Dense(dsed, kernel_init=_w_init, bias_init=_b_init,
                        name='diff_embed_linear_1')
            ])
            global_feat = diffusion_step_encoder(timestep)

        if cond is not None:
            cond , _ = jax.flatten_util.ravel_pytree(cond)
            global_feat = jnp.concatenate((global_feat, cond), -1) \
                if global_feat is not None else cond

        # skip connections
        hs = []
        for ind, dim_out in enumerate(down_dims):
            is_last = ind >= (len(down_dims) - 1)
            res0 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups,
                                  name=f'down{ind}_res0')
            res1 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups,
                                  name=f'down{ind}_res1')
            x = res0(x, global_feat)
            x = res1(x, global_feat)
            hs.append(x)
            if not is_last and x.shape[-2] > 1:
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

        for ind, (dim_out, h, h_next) in enumerate(zip(
                    reversed(down_dims[:-1]), reversed(hs),
                    reversed([None] + hs[:-1])
                )):
            is_last = ind >= (len(down_dims) - 1)
            res0 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups,
                                  name=f'up{ind}_res0')
            res1 = CondResBlock1D(dim_out, kernel_size=kernel_size, n_groups=n_groups,
                                  name=f'up{ind}_res1')
            x = jnp.concatenate((x, h), axis=-1)
            x = res0(x, global_feat)
            x = res1(x, global_feat)
            if not is_last and h_next.shape[-2] > h.shape[-2]:
                us = Upsample1D(name=f'up{ind}_upsample')
                x = us(x)

        final_conv = nn.Sequential([
            Conv1DBlock(start_dim, kernel_size=kernel_size,
                        name='final_conv_block'),
            nn.Conv(sample.shape[-1], 
                    kernel_size=(1,), 
                    kernel_init=_w_init,
                    bias_init=_b_init,
                    name='final_conv')
        ])
        x = final_conv(x)
        if self.final_activation is not None:
            activation = getattr(activations, self.final_activation)
            x = activation(x)
        return sample_uf(x)

class ConditionalMLP(nn.Module):
    features: Sequence[int]
    step_embed_dim: int = None
    activation: str = "relu"
    final_activation: str = "tanh"

    @nn.compact
    def __call__(self, x, timestep=None, cond=None):
        activation = getattr(activations, self.activation)
        x, x_uf = jax.flatten_util.ravel_pytree(x)
        x_dim = x.shape[-1]

        if timestep is not None and self.step_embed_dim is not None:
            dsed = self.step_embed_dim
            diffusion_step_encoder = nn.Sequential([
                SinusoidalPosEmbed(dsed, name='diff_embed'),
                nn.Dense(4*dsed, kernel_init=_w_init, bias_init=_b_init,
                        name='diff_embed_linear_0'),
                mish,
                nn.Dense(dsed, kernel_init=_w_init, bias_init=_b_init,
                        name='diff_embed_linear_1')
            ])
            global_feat = diffusion_step_encoder(timestep)
        else:
            global_feat = None

        if cond is not None:
            cond , _ = jax.flatten_util.ravel_pytree(cond)
            # put cond both in x and in the global_feat
            x = jnp.concatenate((x, cond), -1)
            global_feat = jnp.concatenate((global_feat, cond), -1) \
                if global_feat is not None else cond

        for feat in self.features:
            x = activation(nn.Dense(feat)(x))
            x_film = nn.Sequential([
                nn.Dense(global_feat.shape[-1]//2),
                nn.Dense(2*x.shape[-1])
            ])(global_feat)
            x_scale, x_bias = x_film[:x.shape[-1]], x_film[-x.shape[-1]]
            x = x*x_scale + x_bias
        x = nn.Dense(x_dim)(x)

        if self.final_activation is not None:
            activation = getattr(activations, self.final_activation)
            x = activation(x)

        return x_uf(x)