from ..common import Result, Inputs, DataConfig
from typing import Callable
import foundry.util
from foundry.data import Data
from foundry.policy import Policy, PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.env.core import Environment
from foundry.core import tree

from foundry.core.dataclasses import dataclass

import jax
import foundry.numpy as jnp
import logging
logger = logging.getLogger(__name__)

@dataclass
class Nearest(Result):
    data: DataConfig
    action_horizon: int = 16
    
    def create_policy(self) -> Policy:
        env, splits = self.data.load({"train"})
        # convert to a pytree...
        logger.info("Materializing all chunks...")
        train_data = splits["train"].as_pytree()

        logger.info("Chunks materialized...")
        obs_length, action_length = (
            tree.axis_size(train_data.observations, 1),
            tree.axis_size(train_data.actions, 1)
        )
        if self.action_horizon > action_length:
            raise ValueError("Action length must be at least action horizon")
        actions_structure = tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
            train_data.actions
        )
        states = train_data.state

        def chunk_policy(input: PolicyInput) -> PolicyOutput:
            obs_flat, _ = jax.flatten_util.ravel_pytree(input.observation)

            vf = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])
            train_obs_flat = vf(train_data.observations)

            dists = jnp.sum(jnp.square(
                obs_flat - train_obs_flat
            ), axis=-1)
            i = jnp.argmin(dists)
            action = jax.tree_map(lambda x: x[i], train_data.actions)
            action = action[:self.action_horizon]
            return PolicyOutput(action=action, info=action)
        
        policy = ChunkingTransform(
            obs_length, self.action_horizon
        ).apply(chunk_policy)
        return policy

@dataclass
class NearestConfig:
    action_horizon: int = 16

    def run(self, inputs: Inputs):
        return Nearest(
            data=inputs.data,
            action_horizon=min(self.action_horizon, inputs.data.action_length)
        )
