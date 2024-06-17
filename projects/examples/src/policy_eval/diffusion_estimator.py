from common import TrainConfig
from policy_eval import Sample

from stanza.diffusion import DDPMSchedule
from stanza.config import ConfigProvider
from stanza.random import PRNGSequence
from stanza.policy import PolicyInput, PolicyOutput

from stanza import struct
from stanza.diffusion import nonparametric

import net
import jax
import logging
logger = logging.getLogger(__name__)


@dataclass
class DiffusionEstimatorConfig:
    seed: int = 42
    model: str = "nadaraya-watson"
    kernel_bandwidth: float = 0.01
    T: int = 100

    def parse(self, config: ConfigProvider) -> "DiffusionEstimatorConfig":
        default = DiffusionEstimatorConfig()
        return config.get_struct(default, flatten={"train"})

    def train_policy(self, wandb_run, train_data, eval):
        if self.model.startswith("estimator/"):
            return estimator_diffusion_policy(self, wandb_run, train_data, eval)

def estimator_diffusion_policy(
            config: DiffusionEstimatorConfig,
            wandb_run, train_data, eval
        ):
    estimator = config.model.split("/")[1]
    if estimator == "gaussian":
        kernel = nonparametric.log_gaussian_kernel

    train_data = train_data.as_pytree()
    data = (train_data.observations, train_data.actions)
    obs_length, action_length = data[0].shape[1], data[1].shape[1]

    action_sample = jax.tree_map(lambda x: x[0], data[1])

    schedule = DDPMSchedule.make_squaredcos_cap_v2(128, prediction_type="sample")

    def policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        diffuser = nonparametric.nw_cond_diffuser(obs, data, schedule, kernel, config.kernel_bandwidth)
        action = schedule.sample(input.rng_key, diffuser, action_sample)
        return PolicyOutput(action=action)
    return policy