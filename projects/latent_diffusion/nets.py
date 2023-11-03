from stanza.util.logging import logger

from stanza.nn.unet import DiffusionUNet
from stanza.nn.mlp import DiffusionMLP

import jax

def make_network(rng_key, sample):
    if sample.ndim == 1:
        net = DiffusionMLP(
            features=(64, 32, 32),
            output_sample=sample,
        )
    else:
        net = DiffusionUNet(
            channel_mult=(1,1,1,1),
            attention_levels=(1, 3,),
            num_head_channels=8,
            base_channels=32,
            num_res_blocks=2,
            use_scale_shift_norm=True,
            # unet over the spatial dimensions
            dims=sample.ndims - 1
        )
    params = jax.jit(net.init)(rng_key, sample, timestep=0)
    n_params = jax.flatten_util.ravel_pytree(params)[0].shape[-1]
    logger.info("Input: {}", jax.tree_map(lambda x: x.shape, sample))
    logger.info("Number of parameters: {}", n_params)
    return net, params