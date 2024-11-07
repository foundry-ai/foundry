from ..common import Sample, Inputs
from typing import Callable
import foundry.util
from foundry.data import Data
from foundry.policy import PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.env.core import Environment

from foundry.core.dataclasses import dataclass

import jax
import foundry.numpy as jnp

@dataclass
class NearestConfig:
    action_horizon: int = 8

def nearest_neighbor(
            config: NearestConfig,
            wandb_run,
            train_data : Data[Sample],
            env : Environment,
            eval : Callable,
            rng: jax.Array
        ):
    train_data : Sample = train_data.as_pytree()
    obs_length, action_length = (
        foundry.util.axis_size(train_data.observations, 1),
        foundry.util.axis_size(train_data.actions, 1)
    )

    def chunk_policy(input: PolicyInput) -> PolicyOutput:
        obs_flat, _ = jax.flatten_util.ravel_pytree(input.observation)

        vf = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])
        train_obs_flat = vf(train_data.observations)

        dists = jnp.sum(jnp.square(
            obs_flat - train_obs_flat
        ), axis=-1)
        i = jnp.argmin(dists)
        action = jax.tree_map(lambda x: x[i], train_data.actions)
        action = action[:config.action_horizon]
        return PolicyOutput(action=action, info=action)
    
    policy = ChunkingTransform(
        obs_length, config.action_horizon
    ).apply(chunk_policy)
    return policy
