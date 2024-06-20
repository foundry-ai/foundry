from common import net, TrainConfig
from policy_eval import Sample

from stanza.diffusion import DDPMSchedule
from stanza.runtime import ConfigProvider
from stanza.random import PRNGSequence
from stanza.policy import PolicyInput, PolicyOutput

from stanza.dataclasses import dataclass
from stanza.diffusion import nonparametric

import jax
import logging
logger = logging.getLogger(__name__)


@dataclass
class DiffusionPolicyConfig:
    seed: int = 42
    model: str = "ResNet18"
    train: TrainConfig = TrainConfig()
    kernel_bandwidth: float = 0.01
    T: int = 100

    def parse(self, config: ConfigProvider) -> "DiffusionPolicyConfig":
        return config.get_dataclass(self, flatten={"train"})

    def train_policy(self, wandb_run, train_data, eval):
        return train_net_diffusion_policy(self, wandb_run, train_data, eval)

def train_net_diffusion_policy(
        config : DiffusionPolicyConfig, wandb_run, train_data, eval):
    rng = PRNGSequence(config.seed)
    Model = getattr(net, config.model.split("/")[1])
    model = Model()
    sample = train_data[0]
    vars = jax.jit(model.init)(next(rng), sample.observations, sample.actions)
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

    def loss_fn(vars, _, rng_key, sample: Sample, trian=True):
        obs = sample.observations
        actions = sample.actions