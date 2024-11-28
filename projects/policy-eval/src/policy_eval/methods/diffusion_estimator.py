from ..common import Result, Inputs, DataConfig

from foundry.core import tree
from foundry.diffusion import DDPMSchedule
from foundry.random import PRNGSequence
from foundry.policy import PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.env.core import Environment

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

    relative_actions : bool

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
        if self.action_horizon > action_length:
            raise ValueError("Action length must be at least action horizon")
        actions_structure = tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
            train_data.actions
        )
        states = train_data.state
        train_data = train_data.observations, train_data.actions
        if self.relative_actions:
            train_data = train_data[0], tree.map(
                lambda x, b: x - b[:, None, ...],
                train_data[1],
                jax.vmap(lambda s: env.observe(s, self.data.action_observation()))(states)
            )
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
            if self.relative_actions:
                base_action = env.observe(input.state,
                            self.data.action_observation())
                actions = tree.map(lambda x, y: x + y, base_action, actions)
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
    kernel_bandwidth: float = 0.03
    diffusion_steps: int = 32
    relative_actions: bool = True
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
            relative_actions=self.relative_actions,
            schedule=schedule,
            data=inputs.data
        )