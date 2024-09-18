# From https://github.com/jenkspt/gpt-jax
# used under MIT License

from typing import Any, Optional, Tuple
from functools import partial
import jax
import foundry.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from foundry.core.dataclasses import dataclass, field
from foundry.util.registry import Registry

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    num_embeds: int = 768
    dropout_rate: float = 0.1

    use_bias: bool = True
    dtype: Optional[str] = None

class SelfAttention(nn.Module):
    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None
    use_proj_bias: bool = True

    @nn.compact
    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape
        assert C % self.num_heads == 0
        head_dim = C // self.num_heads
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)

        qkv = nn.Dense(3 * C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_attn')(x)
        qkv = qkv.reshape(B, T, 3 * self.num_heads, head_dim)
        q, k, v = jnp.array_split(qkv, 3, axis=2)
        # calculate attention matrix
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.dtype)
        # attn weight shape is (batch..., num_heads, q_length, kv_length)
        attn = jnp.einsum('...qhd,...khd->...hqk', q, k) * scale
        attn = jnp.where(mask, attn, jnp.finfo(self.dtype).min)
        attn = jax.nn.softmax(attn).astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)

        # return weighted sum over values for each query position
        x = jnp.einsum('...hqk,...khd->...qhd', attn, v).reshape(B, T, C)
        x = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name='c_proj')(x)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = nn.Dense(4 * C, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_fc')(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(C, dtype=self.config.dtype, use_bias=self.config.use_bias, name='c_proj')(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.attn = SelfAttention(self.config.num_heads,
                                  self.config.dtype,
                                  dropout_rate=self.config.dropout_rate)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        x = x + self.attn(self.ln_1(x), mask, deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic)
        return x


class GPT(nn.Module):
    config: GPTConfig
    # If specified overrides the GPTConfig vocab_size
    vocab_size: int | None = None

    flash_attention: bool = True

    @nn.compact
    def __call__(self, idx, deterministic=None):
        vocab_size = self.vocab_size or self.config.vocab_size
        T, = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        # Add a fake batch dimension
        idx = idx[None, :]

        pos = jnp.arange(0, T)[None]
        attn_mask = nn.make_causal_mask(idx, dtype=bool)

        wte = nn.Embed(vocab_size, self.config.num_embeds, dtype=self.config.dtype, name='wte')
        wpe = nn.Embed(self.config.block_size, self.config.num_embeds, dtype=self.config.dtype, name='wpe')

        token_embed = wte(idx)      # [T, num_embeds]
        pos_embed = wpe(pos)        # [T, num_embeds]
        x = nn.Dropout(self.config.dropout_rate)(token_embed + pos_embed, deterministic)

        for i in range(self.config.num_layers):
            x = Block(self.config, name=str(i))(x, attn_mask, deterministic=deterministic)

        x = nn.LayerNorm(1e-5, dtype=self.config.dtype, use_bias=self.config.use_bias, name='ln_f')(x)
        logits = wte.attend(x)
        logits = logits.squeeze(0)
        return logits


def convert_hf_params(hf_params: FrozenDict, num_heads, num_embeds) -> FrozenDict:
    params = unfreeze(hf_params['transformer'])
    for k, v in params.pop('h', {}).items():
        params[k] = v

    params = flatten_dict(params, sep='.')
    for k in params.keys():
        #if k.endswith('attn.c_attn.bias'):
        #    params[k] = params[k].reshape(num_heads, -1)
        if k.endswith('attn.c_attn.kernel'):
            #params[k] = params[k].reshape(num_embeds, num_heads, -1) 
            params[k] = params[k].T
        elif k.endswith('attn.c_proj.kernel'):
            #params[k] = params[k].reshape(num_heads, -1, num_embeds)
            params[k] = params[k].T
        elif k.split('.')[1] == 'mlp' and k.endswith('kernel'):
            params[k] = params[k].T

    params = unflatten_dict({f'params.{k}': v for k, v in params.items()}, sep='.')
    return freeze(params)


GPT2Nano = partial(GPT, GPTConfig(num_layers=8, num_heads=4, num_embeds=64))  # tiny # params
GPT2Small = partial(GPT, GPTConfig(num_layers=12, num_heads=12, num_embeds=768))  # 124M params
GPT2Medium = partial(GPT, GPTConfig(num_layers=24, num_heads=16, num_embeds=1024)) # 350M params
GPT2Large = partial(GPT, GPTConfig(num_layers=36, num_heads=20, num_embeds=1280)) # 774M params
GPT2ExtraLarge = partial(GPT, GPTConfig(num_layers=48, num_heads=25, num_embeds=1600)) # 1558M params

def register(registry: Registry, prefix=None):
    registry.register("llm/gpt2/nano", GPT2Nano, prefix=prefix)
    registry.register("llm/gpt2/small", GPT2Small, prefix=prefix)
    registry.register("llm/gpt2/medium", GPT2Medium, prefix=prefix)
    registry.register("llm/gpt2/large", GPT2Large, prefix=prefix)
    registry.register("llm/gpt2/extra_large", GPT2ExtraLarge, prefix=prefix)