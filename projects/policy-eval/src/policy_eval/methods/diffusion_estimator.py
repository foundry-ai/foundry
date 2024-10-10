from ..common import Result, Inputs, DataConfig

from foundry.core import tree
from foundry.diffusion import DDPMSchedule
from foundry.random import PRNGSequence
from foundry.policy import PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.env import Environment

from foundry.core.dataclasses import dataclass
from foundry.diffusion import nonparametric

from foundry.policy import Policy
from foundry.env.core import ObserveConfig
from foundry.env.mujoco.pusht import PushTAgentPos
from foundry.env.mujoco.robosuite import EEfPose

from typing import Callable

import foundry.util

import jax
import foundry.numpy as jnp
import logging
logger = logging.getLogger(__name__)

@dataclass
class Estimator(Result):
    type: str
    kernel_bandwidth: float
    action_horizon : int

    schedule: DDPMSchedule
    data: DataConfig

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
        actions_structure = tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
            train_data.actions
        )
        train_data = train_data.observations, train_data.actions
        def chunk_policy(input: PolicyInput) -> PolicyOutput:
            if self.type == "nw":
                estimator = lambda obs: nonparametric.nw_cond_diffuser(
                    obs, train_data, self.schedule, nonparametric.log_gaussian_kernel,
                    self.kernel_bandwidth
                )
            obs = input.observation
            diffuser = estimator(obs)
            actions = self.schedule.sample(input.rng_key, 
                                    diffuser, actions_structure)
            return PolicyOutput(
                action=actions,
                info=actions
            )
        policy = ChunkingTransform(
            obs_length, self.action_horizon
        ).apply(chunk_policy)
        return policy

@dataclass
class EstimatorConfig:
    type: str = "nw"
    kernel_bandwidth: float = 0.005
    diffusion_steps: int = 50
    relative_actions: bool = False
    action_horizon: int = 16

    def run(self, inputs: Inputs) -> Estimator:
        schedule = DDPMSchedule.make_squaredcos_cap_v2(
            self.diffusion_steps,
            prediction_type="sample"
        )
        return Estimator(
            type=self.type,
            kernel_bandwidth=self.kernel_bandwidth,
            action_horizon=self.action_horizon,
            schedule=schedule,
            data=inputs.data
        )