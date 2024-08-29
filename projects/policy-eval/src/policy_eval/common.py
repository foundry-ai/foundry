from foundry.random import PRNGSequence

from foundry.env import Environment
from foundry.policy import Policy

from foundry.data import Data
from foundry.train.reporting import Video

from foundry.core.dataclasses import dataclass
from foundry.core.typing import Array

import foundry.core as F
import foundry.numpy as jnp

from typing import Any, Callable

@dataclass
class Sample:
    state: Any
    observations: Array
    actions: Array

@dataclass
class Inputs:
    timesteps: int
    rng: PRNGSequence
    env: Environment

    # Will rollout specifically on the validation dataset
    # returns normalized rewards, raw rewards, 
    # as well as a grid Video
    # (useful for logging i.e. during training)
    validate : Callable[[Array, Policy], Array]
    validate_render: Callable[[Array, Policy], tuple[Array, Video]]

    train_data : Data[Sample]
    test_data : Data[Sample]

class MethodConfig:
    def run(self, inputs: Inputs) -> Policy:
        raise NotImplementedError()

def process_data(config, env, data):
    def process_element(element):
        return env.full_state(element.reduced_state)
    data = data.map_elements(process_element).cache()
    logger.info("Chunking data")
    data = data.chunk(
        config.action_length + config.obs_length
    )
    def process_chunk(chunk):
        states = chunk.elements
        actions = jax.vmap(lambda s: env.observe(s, config.action_config))(states)
        actions = jax.tree.map(lambda x: x[-config.action_length:], actions)
        obs_states = jax.tree.map(lambda x: x[:config.obs_length], states)
        curr_state = jax.tree.map(lambda x: x[-1], obs_states)
        obs = jax.vmap(env.observe)(obs_states)
        return Sample(
            curr_state, obs, actions
        )
    return data.map(process_chunk)


def evaluate(config, env, x0s, T, policy, rng_key):
    N = foundry.util.axis_size(x0s, 0)

    # shard the x0s
    # sharding = PositionalSharding(
    #     mesh_utils.create_device_mesh((jax.device_count(),))
    # )
    # x0s = jax.tree.map(
    #     lambda x: jax.lax.with_sharding_constraint(x, sharding.reshape((jax.device_count(),) + (1,)*(x.ndim-1))),
    #     x0s
    # )
    rewards, videos = jax.vmap(partial(eval, config, env, policy, T))(
        x0s,
        jax.random.split(rng_key, N)
    )
    # reshape all the videos into a single video
    video = jax.vmap(
        lambda x: foundry.canvas.image_grid(x), 
        in_axes=1, out_axes=0
    )(videos)
    return {
        "mean_reward": jnp.mean(rewards),
        "std_reward": jnp.std(rewards),
        "test_demonstrations": Video(video)
    }