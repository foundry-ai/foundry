from stanza import struct
from stanza.config import ConfigProvider, command

from common import TrainConfig, AdamConfig, SGDConfig

from stanza.datasets import env_datasets
from stanza.random import PRNGSequence

import stanza.util

import net
import jax
import functools
import wandb
import stanza
import logging
logger = logging.getLogger(__name__)

@dataclass
class PolicyConfig:
    pass

    @staticmethod
    def parse(config: ConfigProvider) -> "PolicyConfig":
        raise NotImplementedError()
    
    def train_policy(self, config, env, train_data):
        pass

@dataclass
class Config:
    seed: int = 42
    dataset: str = "pusht/chen"
    obs_length: int = 2
    action_length: int = 8
    policy: PolicyConfig = None

    @staticmethod
    def parse(config: ConfigProvider) -> "Config":
        defaults = Config()
        res = config.get_struct(defaults)
        from . import diffusion_policy
        from . import diffusion_estimator
        policy = config.get_cases("policy", "The policy to use.", {
            "diffusion_estimator": diffusion_estimator.DiffusionEstimatorConfig(),
            "diffusion_policy": diffusion_policy.DiffusionPolicyConfig()
        }, "diffusion_estimator")
        return struct.replace(res,
            policy=policy
        )

@dataclass
class Sample:
    observations: jax.Array
    actions: jax.Array

def process_data(config, data):
    data = data.chunk(
        config.action_length + config.obs_length - 1
    )
    def process_chunk(sample):
        obs = (sample.chunk.observation \
            if sample.chunk.observation is not None else \
            sample.chunk.state
        )[:config.obs_length]
        return Sample(
            obs,
            sample.chunk.action[config.obs_length - 1:]
        )
    return data.map(process_chunk)

def evaluate(config, env, test_data, policy):
    pass

def main(config : Config):
    logger.info(f"Running {config}")

    logger.info(f"Loading dataset [blue]{config.dataset}[/blue]")
    dataset = env_datasets.create(config.dataset)
    env = dataset.create_env()
    train_data = process_data(config, dataset.splits["train"])
    test_data = process_data(config, dataset.splits["test"])
    wandb_run = wandb.init(
        project="policy_eval",
        config=stanza.util.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")
    eval = functools.partial(evaluate, config, env, test_data)
    policy = config.policy.train_policy(wandb_run, train_data, eval)
    logger.info(f"Performing final evaluation...")
    stats = eval(policy)
    wandb_run.summary.update(stanza.util.flatten_to_dict(stats)[0])
    wandb_run.finish()

@command
def run(config: ConfigProvider):
    logger.setLevel(logging.DEBUG)
    main(Config.parse(config))