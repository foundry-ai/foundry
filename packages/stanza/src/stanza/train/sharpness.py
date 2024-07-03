from . import LossOutput
from functools import partial

from stanza.util import axis_size, lanczos

import flax
import jax

@partial(jax.jit, static_argnums=(0,4))
def sharpness_stats(loss_fn, vars, rng_key, data, batch_size):
    # only compute the sharpenss wrt trainable params
    other_vars, params = flax.core.pop(vars, "params")
    N = axis_size(data, 0)
    rng_keys = jax.random.split(rng_key, N)
    def loss(params, sample):
        rng_key, sample = sample
        vars = {"params": params, **other_vars}
        output = loss_fn(vars, rng_key, sample)
        return output.loss
    hvp_at = partial(lanczos.net_batch_hvp, 
        loss, (rng_keys, data), batch_size
    )
    sharpness_stats = lanczos.net_sharpness_statistics(rng_key, hvp_at, params)
    return sharpness_stats