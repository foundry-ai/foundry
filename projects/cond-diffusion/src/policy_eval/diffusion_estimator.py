from common import net, TrainConfig
from policy_eval import Sample

from stanza.data import Data
from stanza.diffusion import DDPMSchedule
from stanza.runtime import ConfigProvider
from stanza.random import PRNGSequence
from stanza.policy import PolicyInput, PolicyOutput
from stanza.policy.transforms import ChunkingTransform

from stanza.env import Environment
from stanza.env.mujoco.pusht import PushTObs

from stanza.dataclasses import dataclass
from stanza.diffusion import nonparametric

from typing import Callable

import stanza.util

import jax
import logging
logger = logging.getLogger(__name__)

@dataclass
class DiffusionEstimatorConfig:
    seed: int = 42
    estimator: str = "nw"
    kernel_bandwidth: float = 0.02
    diffusion_steps: int = 16
    relative_actions: bool = True

    def parse(self, config: ConfigProvider) -> "DiffusionEstimatorConfig":
        default = DiffusionEstimatorConfig()
        return config.get_dataclass(default, flatten={"train"})

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
        stanza.util.axis_size(train_data.observations, 1),
        stanza.util.axis_size(train_data.actions, 1)
    )
    action_sample = jax.tree_map(lambda x: x[0], train_data.actions)
    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        prediction_type="sample"
    )
    def policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        if config.relative_actions:
            data_agent_pos = jax.vmap(
                lambda x: env.observe(x, PushTObs()).agent_pos
            )(train_data.state)
            actions = train_data.actions - data_agent_pos[:, None, :]
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
            agent_pos = env.observe(input.state, PushTObs()).agent_pos
            action = action + agent_pos
        return PolicyOutput(action=action)
    policy = ChunkingTransform(
        obs_length, action_length
    ).apply(policy)
    return policy