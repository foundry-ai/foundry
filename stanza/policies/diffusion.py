from typing import Any
from stanza import Partial

import jax
import jax.numpy as jnp

import flax.linen as nn

from stanza.policies import PolicyOutput

class DiffusionPolicy:
    model: Callable
    schedule: Diffuser

    def __init__(input):
        pass

def make_diffusion_policy(net_fn, diffuser, obs_norm, action_norm, 
                          action_sample, action_chunk_length,
                          action_horizon_offset, action_horizon_length):
    action_sample_traj = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), action_chunk_length, axis=0),
        action_sample
    )
    def policy(input):
        norm = obs_norm.normalize(input.observation)
        model_fn = Partial(net_fn, cond=norm)
        sample = diffuser.sample(input.rng_key, model_fn,
                action_sample_traj, 
                num_steps=diffuser.num_steps)
        action = action_norm.unnormalize(sample)
        start = action_horizon_offset
        end = action_horizon_offset + action_horizon_length
        action = jax.tree_util.tree_map(
            lambda x: x[start:end], action
        )
        return PolicyOutput(action)
    return policy