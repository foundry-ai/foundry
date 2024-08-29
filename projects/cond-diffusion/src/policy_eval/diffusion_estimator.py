
from policy_eval import Sample

from foundry.data import Data
from foundry.diffusion import DDPMSchedule
from foundry.runtime import ConfigProvider
from foundry.core.random import PRNGSequence
from foundry.policy import PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.env import Environment

from foundry.core.dataclasses import dataclass
from foundry.diffusion import nonparametric

from foundry.env.core import ObserveConfig
from foundry.env.mujoco.pusht import PushTAgentPos
from foundry.env.mujoco.robosuite import ManipulationTaskEEFPose

from typing import Callable

import foundry.util

import jax
import foundry.numpy as jnp
import logging
logger = logging.getLogger(__name__)

@dataclass
class DiffusionEstimatorConfig:
    estimator: str = "nw"
    kernel_bandwidth: float = 0.01
    diffusion_steps: int = 50
    relative_actions: bool = False
    action_config: ObserveConfig = None
    action_horizon: int = 16

    def parse(self, config: ConfigProvider) -> "DiffusionEstimatorConfig":
        return config.get_dataclass(self)

    def train_policy(self, wandb_run, train_data, env, eval, rng):
        return estimator_diffusion_policy(self, wandb_run, train_data, env, eval, rng)

def estimator_diffusion_policy(
            config: DiffusionEstimatorConfig,
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
    action_sample = jax.tree_map(lambda x: x[0], train_data.actions)
    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        prediction_type="sample"
    )
    def chunk_policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        if config.relative_actions:
            data_agent_pos = jax.vmap(
                lambda x: env.observe(x, config.action_config)
            )(train_data.state)
            if config.action_config == PushTAgentPos():
                actions = train_data.actions - data_agent_pos[:, None, :]
            elif config.action_config == ManipulationTaskEEFPose():
                expand_agent_pos = jnp.zeros_like(train_data.actions)
                expand_agent_pos = expand_agent_pos.at[...,0:3].set(data_agent_pos[:,None,0:3])
                actions = train_data.actions - expand_agent_pos
            else:
                raise ValueError(f"Unsupported action_config {config.action_config}")
        else:
            actions = train_data.actions
        data = train_data.observations, actions
        if config.estimator == "nw":
            kernel = nonparametric.log_gaussian_kernel
            estimator = lambda obs: nonparametric.nw_cond_diffuser(
                obs, data, schedule, kernel, config.kernel_bandwidth
            )
        diffuser = estimator(obs)
        action = schedule.sample(input.rng_key, diffuser, action_sample)
        if config.relative_actions:
            agent_pos = env.observe(input.state, config.action_config)
            if config.action_config == PushTAgentPos():
                action = action + agent_pos
            elif config.action_config == ManipulationTaskEEFPose():
                expand_agent_pos = jnp.zeros_like(action)
                expand_agent_pos = expand_agent_pos.at[...,0:3].set(agent_pos[0:3])
                action = action + agent_pos
            else:
                raise ValueError(f"Unsupported action_config {config.action_config}")
        action = action[:config.action_horizon]
        return PolicyOutput(action=action, info=action)
    policy = ChunkingTransform(
        obs_length, config.action_horizon
    ).apply(chunk_policy)
    return policy