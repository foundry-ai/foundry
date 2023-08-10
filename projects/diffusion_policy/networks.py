from stanza.util.logging import logger
from stanza.nets.unet1d import ConditionalUnet1D

import stanza
import haiku as hk
import jax.numpy as jnp

def pusht_model_fn(curr_sample, timestep, cond):
    logger.trace("Tracing model", only_tracing=True)
    sample_flat, sample_uf = stanza.util.vmap_ravel_pytree(curr_sample)
    # flatten each observation
    cond_flat, _ = stanza.util.vmap_ravel_pytree(cond)
    # flatten the observations along the time axis
    cond_flat = cond_flat.reshape((-1))
    model = ConditionalUnet1D(name='net',
        down_dims=[64, 128, 256], diffusion_step_embed_dim=128)
    r = model(sample_flat, timestep, cond_flat)
    r = sample_uf(r)
    return r
pusht_net = hk.transform(pusht_model_fn)

def quadrotor_model_fn(curr_sample, timestep, cond):
    logger.trace("Tracing model", only_tracing=True)
    sample_flat, sample_uf = stanza.util.vmap_ravel_pytree(curr_sample)
    # flatten each observation
    cond_flat, _ = stanza.util.vmap_ravel_pytree(cond)
    # flatten the observations along the time axis
    cond_flat = cond_flat.reshape((-1))
    model = ConditionalUnet1D(name='net',
        down_dims=[64, 128, 256], diffusion_step_embed_dim=128)
    sample_flat = jnp.clip(sample_flat, -50, 50)
    cond_flat = jnp.clip(cond_flat, -50, 50)
    r = model(sample_flat, timestep, cond_flat)
    r = jnp.clip(r, -50, 50)
    r = sample_uf(r)
    return r
quadrotor_net = hk.transform(quadrotor_model_fn)