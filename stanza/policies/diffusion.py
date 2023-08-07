from typing import Any
from stanza import Partial
from stanza.util.attrdict import AttrMap

import jax
import jax.numpy as jnp

import flax.linen as nn

from stanza.policies import PolicyOutput

def make_diffusion_policy(net_fn, diffuser, normalizer,
                          action_chunk_length, action_horizon_offset, 
                          action_horizon_length, diffuse_gains=False, noise=0.):
    obs_norm = normalizer.map(lambda x: x.observation)
    action_norm = normalizer.map(lambda x: x.action)
    gain_norm = normalizer.map(lambda x: x.info.K) \
        if hasattr(normalizer.instance.info, 'K') is not None and diffuse_gains else None
    states_norm = normalizer.map(lambda x: x.state) \
        if normalizer.instance.state is not None and diffuse_gains else None
    action_sample_traj = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), action_chunk_length, axis=0),
        action_norm.instance
    )
    gain_sample_traj = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), action_chunk_length, axis=0),
        gain_norm.instance
    ) if gain_norm is not None else None
    states_sample_traj = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), action_chunk_length, axis=0),
        states_norm.instance
    ) if states_norm is not None else None

    def policy(input):
        smooth_rng, sample_rng = jax.random.split(input.rng_key)
        norm = obs_norm.normalize(input.observation)

        norm_flat, norm_uf = jax.flatten_util.ravel_pytree(norm)
        if noise > 0:
            norm_flat = norm_flat + noise * jax.random.normal(smooth_rng, norm_flat.shape)
        norm = norm_uf(norm_flat)

        model_fn = Partial(net_fn, cond=norm)
        sample = action_sample_traj, states_sample_traj, gain_sample_traj
        sample = diffuser.sample(sample_rng, model_fn,
                sample, 
                num_steps=diffuser.num_steps)
        actions, states, gains = sample
        actions = action_norm.unnormalize(actions)
        start = action_horizon_offset
        end = action_horizon_offset + action_horizon_length
        actions = jax.tree_util.tree_map(
            lambda x: x[start:end], actions
        )
        if diffuse_gains:
            states = jax.tree_util.tree_map(
                lambda x: x[start:end], states
            )
            gains = jax.tree_util.tree_map(
                lambda x: x[start:end], gains
            )
            actions = actions, states, gains
        return PolicyOutput(actions)
    return policy