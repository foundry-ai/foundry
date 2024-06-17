from common import TrainConfig
from policy_eval import Sample

from stanza.diffusion import DDPMSchedule
from stanza.config import ConfigProvider
from stanza.random import PRNGSequence
from stanza.policy import PolicyInput, PolicyOutput

from stanza import dataclass
from stanza.diffusion import nonparametric

import net
import jax
import logging
logger = logging.getLogger(__name__)

@dataclass
class DiffusionPolicyConfig:
    seed: int = 42
    model: str = "estimator/gaussian"
    train: TrainConfig = TrainConfig()
    kernel_bandwidth: float = 0.01
    T: int = 100

    def parse(self, config: ConfigProvider) -> "DiffusionPolicyConfig":
        return config.get_struct(self, flatten={"train"})

    def train_policy(self, wandb_run, train_data, eval):
        if self.model.startswith("estimator/"):
            estimator_diffusion_policy(self, wandb_run, train_data, eval)
        elif self.model.startswith("net/"):
            train_net_diffusion_policy(self, wandb_run, train_data, eval)

def estimator_diffusion_policy(
            config: DiffusionPolicyConfig,
            wandb_run, train_data, eval
        ):
    estimator = config.model.split("/")[1]
    if estimator == "gaussian":
        kernel = nonparametric.log_gaussian_kernel

    train_data = train_data.as_pytree()
    data = (train_data.observations, train_data.actions)
    action_sample = jax.tree_map(lambda x: x[0], data[1])

    schedule = DDPMSchedule.make_squaredcos_cap_v2(128, prediction_type="sample")

    def policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        diffuser = nonparametric.nw_cond_diffuser(obs, data, schedule, kernel, config.kernel_bandwidth)
        action = schedule.sample(input.rng_key, diffuser, action_sample)
        return PolicyOutput(action=action)

    return policy

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