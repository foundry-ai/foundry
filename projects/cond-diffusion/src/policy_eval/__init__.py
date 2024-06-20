from stanza import dataclasses
from stanza.runtime import ConfigProvider, command

from common import net, TrainConfig, AdamConfig, SGDConfig

from stanza.datasets import env_datasets
from stanza.random import PRNGSequence
from functools import partial

import stanza.policy
import stanza.util

import stanza.util
import jax
import functools
import wandb
import stanza
import logging
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class PolicyConfig:
    pass

    @staticmethod
    def parse(config: ConfigProvider) -> "PolicyConfig":
        raise NotImplementedError()
    
    def train_policy(self, config, env, train_data):
        pass

@dataclasses.dataclass
class Config:
    seed: int = 42
    dataset: str = "pusht/chen"
    obs_length: int = 2
    action_length: int = 8
    policy: PolicyConfig = None

    @staticmethod
    def parse(config: ConfigProvider) -> "Config":
        defaults = Config()
        res = config.get_dataclass(defaults)
        from . import diffusion_policy
        from . import diffusion_estimator
        policy = config.get_cases("policy", "The policy to use.", {
            "diffusion_estimator": diffusion_estimator.DiffusionEstimatorConfig(),
            "diffusion_policy": diffusion_policy.DiffusionPolicyConfig()
        }, "diffusion_estimator")
        return dataclasses.replace(res,
            policy=policy
        )

@dataclasses.dataclass
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

def eval(env, policy, ref_traj, rng_key):
    T = stanza.util.axis_size(ref_traj, 0)
    r = stanza.policy.rollout(
        env.step, x0, 
        policy, env.observe,
        policy_rng_key=rng_key,
        length=T
    )
    pre_states, post_states = (
        jax.tree_util.map(lambda x: x[:-1], r.states),
        jax.tree_util.map(lambda x: x[1:], r.states)
    )
    rewards = jax.vmap(env.reward)(pre_states, r.actions, post_states)
    return jnp.max(rewards, axis=-1)

def evaluate(env, test_data, policy, rng_key):
    print(jax.tree_map(lambda x: x.shape, test_data.as_pytree()))
    x0s = jax.tree_util.tree_map(
        lambda x: x[0], test_data.as_pytree()
    )
    N = stanza.util.axis_size(x0s, 0)
    return jax.vmap(partial(eval, env, policy))(
        x0s,
        jax.random.split(rng_key, N)
    )

def main(config : Config):
    logger.info(f"Running {config}")

    logger.info(f"Loading dataset [blue]{config.dataset}[/blue]")
    dataset = env_datasets.create(config.dataset)
    env = dataset.create_env()
    train_data = process_data(config, dataset.splits["train"])
    test_data = dataset.splits["test"].uniform_truncated(300)
    wandb_run = wandb.init(
        project="policy_eval",
        config=stanza.util.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")
    eval = functools.partial(evaluate, env, test_data)
    policy = config.policy.train_policy(wandb_run, train_data, eval)
    logger.info(f"Performing final evaluation...")
    stats = jax.jit(partial(eval,policy))(jax.random.key(42))
    wandb_run.summary.update(stanza.util.flatten_to_dict(stats)[0])
    wandb_run.finish()

@command
def run(config: ConfigProvider):
    logger.setLevel(logging.DEBUG)
    main(Config.parse(config))