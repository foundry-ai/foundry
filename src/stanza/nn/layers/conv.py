from stanza import struct
from stanza.nn import initializers

from typing import Sequence, Callable

import stanza
import jax
import jax.numpy as jnp
import itertools

def _make_tuple(n: int) -> Callable:
    def parse(x):
        if isinstance(x, Sequence):
            if len(x) == n: return tuple(x)
            else: raise ValueError(f"Expected sequence of length {n}, got {len(x)}")
        else:
            return tuple(itertools.repeat(x, n))
    return parse

@struct.dataclass
class Conv:
    num_spatial_dims: int
    in_channels: int
    out_channels: int
    kernel_size: int | Sequence[int]
    stride: int | Sequence[int] = 1
    padding: int | Sequence[int] = 0
    dilation: int | Sequence[int] = 1
    groups: int = 1
    use_bias: bool = True

    weight_initializer: initializers.Initializer = initializers.lecun_normal()
    bias_initializer: initializers.Initializer = initializers.lecun_normal()

    @jax.jit
    def init(self, rng_key: jax.Array):
        parse = _make_tuple(self.num_spatial_dims)
        kernel_size = parse(self.kernel_size)
        stride = parse(self.stride)
        dilation = parse(self.dilation)

        if self.in_channels % self.groups != 0:
            raise ValueError(
                f"Input channels ({self.in_channels})"
                f" must be divisible by number of groups ({self.groups})"
            )
        channels_per_group = self.in_channels // self.groups
        if isinstance(self.padding, int):
            padding = tuple((self.padding, self.padding) for _ in range(self.num_spatial_dims))
        else:
            padding = tuple(self.padding)

        w_rng, b_rng = jax.random.split(rng_key, 2) if self.use_bias else (rng_key, None)

        return ConvModel(
        )

@struct.dataclass
class ConvModel:
    kernel: jax.Array
    bias: jax.Array | None

    @jax.jit
    def __call__(self, x: jax.Array):
        expected_dim = 1 + (len(self.kernel.shape) - 2)
        if x.ndim != expected_dim:
            raise ValueError(
                f"Expected input tensor of shape (..., {self.in_channels}) of rank {expected_dim}, got {x.shape}"
            )
        x = jnp.expand_dims(x, 0)
        x = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, 0)
        return x

    def apply(self, x: jax.Array):
        return self(x)