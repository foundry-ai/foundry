import foundry.numpy as jnp
from flax import linen as nn
import flax

from foundry.util.registry import Registry

from functools import partial
from typing import (
    Callable, Optional, Sequence, Tuple,
    Iterable, Tuple, Union, Any, Dict, Mapping
)

ModuleDef = Callable[..., Callable]
# InitFn = Callable[[PRNGKey, Shape, DType], Array]
InitFn = Callable[[Any, Iterable[int], Any], Any]

class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.relu
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1
    kernel_init: InitFn = nn.initializers.kaiming_normal()
    bias_init: InitFn = nn.initializers.zeros

    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)

    force_conv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = self.conv_cls(
            self.n_filters,
            self.kernel_size,
            self.strides,
            use_bias=(not self.norm_cls or self.force_conv_bias),
            padding=self.padding,
            feature_group_count=self.groups,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.norm_cls:
            scale_init = (nn.initializers.zeros
                          if self.is_last else nn.initializers.ones)
            mutable = self.is_mutable_collection('batch_stats')
            if isinstance(self.norm_cls(), nn.BatchNorm):
                x = self.norm_cls(use_running_average=not mutable, scale_init=scale_init)(x)
            else:
                x = self.norm_cls()(x)

        if not self.is_last:
            x = self.activation(x)
        return x

def rsoftmax(x, radix, cardinality):
    # (batch_size, features) -> (batch_size, features)
    batch = x.shape[0]
    if radix > 1:
        x = x.reshape((batch, cardinality, radix, -1)).swapaxes(1, 2)
        return nn.softmax(x, axis=1).reshape((batch, -1))
    else:
        return nn.sigmoid(x)


class SplAtConv2d(nn.Module):
    channels: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int] = (1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    groups: int = 1
    radix: int = 2
    reduction_factor: int = 4

    conv_block_cls: ModuleDef = ConvBlock
    cardinality: int = groups

    # Match extra bias here:
    # github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/splat.py#L39
    match_reference: bool = False

    @nn.compact
    def __call__(self, x):
        inter_channels = max(x.shape[-1] * self.radix // self.reduction_factor, 32)

        conv_block = self.conv_block_cls(self.channels * self.radix,
                                         kernel_size=self.kernel_size,
                                         strides=self.strides,
                                         groups=self.groups * self.radix,
                                         padding=self.padding)
        conv_cls = conv_block.conv_cls  # type: ignore
        x = conv_block(x)

        if self.radix > 1:
            # torch split takes split_size: int(rchannel//self.radix)
            # jnp split takes num sections: self.radix
            split = jnp.split(x, self.radix, axis=-1)
            gap = sum(split)
        else:
            gap = x

        gap = gap.mean((1, 2), keepdims=True)  # type: ignore # global average pool

        # Remove force_conv_bias after resolving
        # github.com/zhanghang1989/ResNeSt/issues/125
        gap = self.conv_block_cls(inter_channels,
                                  kernel_size=(1, 1),
                                  groups=self.cardinality,
                                  force_conv_bias=self.match_reference)(gap)

        attn = conv_cls(self.channels * self.radix,
                        kernel_size=(1, 1),
                        feature_group_count=self.cardinality)(gap)  # n x 1 x 1 x c
        attn = attn.reshape((x.shape[0], -1))
        attn = rsoftmax(attn, self.radix, self.cardinality)
        attn = attn.reshape((x.shape[0], 1, 1, -1))

        if self.radix > 1:
            attns = jnp.split(attn, self.radix, axis=-1)
            out = sum(a * s for a, s in zip(attns, split))
        else:
            out = attn * x

        return out


def slice_variables(variables: Mapping[str, Any],
                    start: int = 0,
                    end: Optional[int] = None) -> flax.core.FrozenDict:
    """Returns variables dict correspond to a sliced model.

    You can retrieve the model corresponding to the slices variables via
    `Sequential(model.layers[start:end])`.

    The variables mapping should have the same structure as a Sequential
    model's variable dict (based on Flax):

        variables = {
            'group1': ['layers_a', 'layers_b', ...]
            'group2': ['layers_a', 'layers_b', ...]
            ...,
        }

    Typically, 'group1' and 'group2' would be 'params' and 'batch_stats', but
    they don't have to be. 'a, b, ...' correspond to the integer indices of the
    layers.

    Args:
        variables: A dict (typically a flax.core.FrozenDict) containing the
            model parameters and state.
        start: integer indicating the first layer to keep.
        end: integer indicating the first layer to exclude (can be negative,
            has the same semantics as negative list indexing).

    Returns:
        A flax.core.FrozenDict with the subset of parameters/state requested.
    """
    last_ind = max(int(s.split('_')[-1]) for s in variables['params'])
    if end is None:
        end = last_ind + 1
    elif end < 0:
        end += last_ind + 1

    sliced_variables: Dict[str, Any] = {}
    for k, var_dict in variables.items():  # usually params and batch_stats
        sliced_variables[k] = {
            f'layers_{i-start}': var_dict[f'layers_{i}']
            for i in range(start, end)
            if f'layers_{i}' in var_dict
        }

    return flax.core.freeze(sliced_variables)


STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    269: [3, 30, 48, 8],
}


class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    base_filters: int = 64

    strides: Sequence[int] = (2, 2)
    kernel_size: Tuple[int, int] = (7, 7)
    padding: Sequence[Tuple[int, int]] = ((3, 3), (3, 3))

    @nn.compact
    def __call__(self, x):
        return self.conv_block_cls(self.base_filters,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   padding=self.padding)(x)


class ResNetDStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock
    stem_width: int = 32

    # If True, n_filters for first conv is (input_channels + 1) * 8
    adaptive_first_width: bool = False

    @nn.compact
    def __call__(self, x):
        cls = partial(self.conv_block_cls, kernel_size=(3, 3), padding=((1, 1), (1, 1)))
        first_width = (8 * (x.shape[-1] + 1)
                       if self.adaptive_first_width else self.stem_width)
        x = cls(first_width, strides=(2, 2))(x)
        x = cls(self.stem_width, strides=(1, 1))(x)
        x = cls(self.stem_width * 2, strides=(1, 1))(x)
        return x


class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = self.conv_block_cls(out_shape[-1],
                                    kernel_size=(1, 1),
                                    strides=self.strides,
                                    activation=lambda y: y)(x)
        return x


class ResNetDSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if self.strides != (1, 1):
            x = nn.avg_pool(x, (2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        if x.shape[-1] != out_shape[-1]:
            x = self.conv_block_cls(out_shape[-1], (1, 1), activation=lambda y: y)(x)
        return x


class ResNeStSkipConnection(ResNetDSkipConnection):
    # Inheritance to ensures our variables dict has the right names.
    pass


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        y = self.conv_block_cls(self.n_hidden,
                                padding=[(1, 1), (1, 1)],
                                strides=self.strides)(x)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)],
                                is_last=True)(y)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))


class ResNetBottleneckBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    groups: int = 1  # cardinality
    base_width: int = 64

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        # Downsampling strides in 3x3 conv instead of 1x1 conv, which improves accuracy.
        # This variant is called ResNet V1.5 (matches torchvision).
        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x)
        y = self.conv_block_cls(group_width,
                                strides=self.strides,
                                groups=self.groups,
                                padding=((1, 1), (1, 1)))(y)
        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))


class ResNetDBlock(ResNetBlock):
    skip_cls: ModuleDef = ResNetDSkipConnection


class ResNetDBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNetDSkipConnection


class ResNeStBottleneckBlock(ResNetBottleneckBlock):
    skip_cls: ModuleDef = ResNeStSkipConnection
    avg_pool_first: bool = False
    radix: int = 2

    splat_cls: ModuleDef = SplAtConv2d

    @nn.compact
    def __call__(self, x):
        assert self.radix == 2  # TODO: implement radix != 2

        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        group_width = int(self.n_hidden * (self.base_width / 64.)) * self.groups

        y = self.conv_block_cls(group_width, kernel_size=(1, 1))(x)

        if self.strides != (1, 1) and self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding=[(1, 1), (1, 1)])

        y = self.splat_cls(group_width,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           padding=[(1, 1), (1, 1)],
                           groups=self.groups,
                           radix=self.radix)(y)

        if self.strides != (1, 1) and not self.avg_pool_first:
            y = nn.avg_pool(y, (3, 3), strides=self.strides, padding=[(1, 1), (1, 1)])

        y = self.conv_block_cls(self.n_hidden * self.expansion,
                                kernel_size=(1, 1),
                                is_last=True)(y)

        return self.activation(y + skip_cls(self.strides)(x, y.shape))


class ResNet(nn.Module):

    block_cls: ModuleDef
    stage_sizes: Sequence[int]
    n_classes: int

    base_filters: int = 64
    conv_cls: ModuleDef = nn.Conv
    # norm_cls: Optional[ModuleDef] = nn.GroupNorm
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9, axis_name="batch")
    conv_block_cls: ModuleDef = ConvBlock
    stem_cls: ModuleDef = ResNetStem
    pool_fn: Callable = partial(nn.max_pool,
                                window_shape=(3, 3),
                                strides=(2, 2),
                                padding=((1, 1), (1, 1)))
    
    @nn.compact
    def __call__(self, x):
        conv_block_cls = partial(self.conv_block_cls, 
                        conv_cls=self.conv_cls, 
                        norm_cls=self.norm_cls)
        block_cls = partial(self.block_cls, conv_block_cls=conv_block_cls)
        if self.stem_cls is not None:
            stem_cls = partial(self.stem_cls, conv_block_cls=conv_block_cls, base_filters=self.base_filters)
            x = stem_cls()(x)
            x = self.pool_fn(x)
        for i, n_blocks in enumerate(self.stage_sizes):
            hsize = self.base_filters * 2**i
            for b in range(n_blocks):
                strides = (1, 1) if i == 0 or b != 0 else (2, 2)
                block = block_cls(n_hidden=hsize, strides=strides)
                x = block(x)
        x = jnp.mean(x, axis=(-3, -2))
        x = nn.Dense(self.n_classes)(x)
        return x

# Small models are for CIFAR-10 size images (rather than ImageNet size)
SmallResNet = partial(ResNet,
    stem_cls=partial(ResNetStem, # smol stem boy
        strides=(1, 1), kernel_size=(3, 3),
        padding=((1, 1), (1, 1))
    )
)

SmallResNet18 = partial(SmallResNet, stage_sizes=STAGE_SIZES[18],
                        block_cls=ResNetBlock)

SmallResNet50 = partial(SmallResNet, stage_sizes=STAGE_SIZES[50],
                        block_cls=ResNetBlock)

ResNet18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
                   stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=STAGE_SIZES[34],
                   stem_cls=ResNetStem, block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                   stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)
ResNet200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                    stem_cls=ResNetStem, block_cls=ResNetBottleneckBlock)


WideResNet18 = partial(ResNet18, base_filters=128,
                       block_cls=partial(ResNetBottleneckBlock, expansion=2))
WideResNet50 = partial(ResNet50, base_filters=128,
                       block_cls=partial(ResNetBottleneckBlock, expansion=2))
WideResNet101 = partial(ResNet101, base_filters=128,
                        block_cls=partial(ResNetBottleneckBlock, expansion=2))

ResNeXt50 = partial(ResNet50,
                    block_cls=partial(ResNetBottleneckBlock, groups=32, base_width=4))
ResNeXt101 = partial(ResNet101,
                     block_cls=partial(ResNetBottleneckBlock, groups=32, base_width=8))

ResNetD18 = partial(ResNet, stage_sizes=STAGE_SIZES[18],
                    stem_cls=ResNetDStem, block_cls=ResNetDBlock)
ResNetD34 = partial(ResNet, stage_sizes=STAGE_SIZES[34],
                    stem_cls=ResNetDStem, block_cls=ResNetDBlock)
ResNetD50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                    stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)
ResNetD200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                     stem_cls=ResNetDStem, block_cls=ResNetDBottleneckBlock)

ResNeSt50Fast = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                        stem_cls=ResNetDStem,
                        block_cls=partial(ResNeStBottleneckBlock, avg_pool_first=True))
ResNeSt50 = partial(ResNet, stage_sizes=STAGE_SIZES[50],
                    stem_cls=ResNetDStem, block_cls=ResNeStBottleneckBlock)
ResNeSt101 = partial(ResNet, stage_sizes=STAGE_SIZES[101],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt152 = partial(ResNet, stage_sizes=STAGE_SIZES[152],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt200 = partial(ResNet, stage_sizes=STAGE_SIZES[200],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)
ResNeSt269 = partial(ResNet, stage_sizes=STAGE_SIZES[269],
                     stem_cls=partial(ResNetDStem, stem_width=64),
                     block_cls=ResNeStBottleneckBlock)

models = Registry()
models.register("SmallResNet18", SmallResNet18)
models.register("SmallResNet50", SmallResNet50)
models.register("ResNet18", ResNet18)
models.register("ResNet34", ResNet34)
models.register("ResNet50", ResNet50)
models.register("ResNet101", ResNet101)
models.register("ResNet152", ResNet152)
models.register("ResNet200", ResNet200)