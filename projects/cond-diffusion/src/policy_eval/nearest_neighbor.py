from policy_eval import Sample
from typing import Callable

import stanza.util
from stanza.data import Data
from stanza.runtime import ConfigProvider
from stanza.policy import PolicyInput, PolicyOutput
from stanza.policy.transforms import ChunkingTransform

from stanza.env import Environment

from stanza.dataclasses import dataclass

import jax
import jax.numpy as jnp

@dataclass
class NNConfig:
    action_horizon: int = 8

    def parse(self, config: ConfigProvider) -> "NNConfig":
        default = NNConfig()
        return config.get_dataclass(default, flatten={"train"})

    def train_policy(self, wandb_run, train_data, env, eval, rng):
        return nearest_neighbor(self, wandb_run, train_data, env, eval, rng)

def nearest_neighbor(
            config: NNConfig,
            wandb_run,
            train_data : Data[Sample],
            env : Environment,
            eval : Callable,
            rng: jax.Array
        ):
    train_data : Sample = train_data.as_pytree()
    obs_length, action_length = (
        stanza.util.axis_size(train_data.observations, 1),
        stanza.util.axis_size(train_data.actions, 1)
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