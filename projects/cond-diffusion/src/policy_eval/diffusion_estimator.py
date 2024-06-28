from common import net, TrainConfig
from policy_eval import Sample

from stanza.data import Data
from stanza.diffusion import DDPMSchedule
from stanza.runtime import ConfigProvider
from stanza.random import PRNGSequence
from stanza.policy import PolicyInput, PolicyOutput
from stanza.policy.transforms import ChunkingTransform

from stanza.dataclasses import dataclass
from stanza.diffusion import nonparametric

from typing import Callable

import stanza.util

import wandb
import jax
import logging
logger = logging.getLogger(__name__)

@dataclass
class DiffusionEstimatorConfig:
    estimator: str = "nw"
    kernel_bandwidth: float = 0.02
    diffusion_steps: int = 16

    def parse(self, config: ConfigProvider) -> "DiffusionEstimatorConfig":
        default = DiffusionEstimatorConfig()
        return config.get_dataclass(default, flatten={"train"})

    def train_policy(self, wandb_run, train_data, eval, rng):
        return estimator_diffusion_policy(self, wandb_run, train_data, eval, rng)

def estimator_diffusion_policy(
            config: DiffusionEstimatorConfig,
            wandb_run,
            train_data : Data[Sample],
            eval : Callable,
            rng: jax.Array
        ):
    train_data = train_data.as_pytree()
    data = (train_data.observations, train_data.actions)
    obs_length, action_length = (
        stanza.util.axis_size(data[0], 1),
        stanza.util.axis_size(data[1], 1)
    )
    if config.estimator == "nw":
        kernel = nonparametric.log_gaussian_kernel
        eval_estimator = lambda obs: nonparametric.nw_cond_diffuser(
            obs, data, kernel, config.kernel_bandwidth
        )
    action_sample = jax.tree_map(lambda x: x[0], data[1])
    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        prediction_type="sample"
    )
    def policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        diffuser = nonparametric.nw_cond_diffuser(obs, data, schedule, kernel, config.kernel_bandwidth)
        action = schedule.sample(input.rng_key, diffuser, action_sample)
        return PolicyOutput(action=action)
    policy = ChunkingTransform(
        obs_length, action_length
    ).apply(policy)
    return policy