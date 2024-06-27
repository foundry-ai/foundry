from stanza import dataclasses
from stanza.dataclasses import dataclass, replace
from stanza.runtime import ConfigProvider, command
from stanza.datasets.env import datasets
from stanza.random import PRNGSequence
from stanza.env import ImageRender
from stanza.train.reporting import Video

from common import net, TrainConfig, AdamConfig, SGDConfig

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from functools import partial


import stanza.policy
import stanza.util
import stanza.util
import stanza.train.reporting
import stanza.train.wandb
import jax
import jax.numpy as jnp
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
    dataset: str = "pusht/chi"
    obs_length: int = 1
    action_length: int = 8
    policy: PolicyConfig = None
    timesteps: int = 300

    @staticmethod
    def parse(config: ConfigProvider) -> "Config":
        defaults = Config()

        from . import diffusion_policy
        from . import diffusion_estimator
        # Check for a default policy override
        policy = config.get("policy", str, default=None)
        if policy == "diffusion_policy":
            defaults = replace(defaults, policy=diffusion_policy.DiffusionPolicyConfig())
        elif policy == "diffusion_estimator":
            defaults = replace(defaults, policy=diffusion_estimator.DiffusionEstimatorConfig())
        else:
            defaults = replace(defaults, policy=diffusion_estimator.DiffusionEstimatorConfig())
        return config.get_dataclass(defaults)

@dataclasses.dataclass
class Sample:
    observations: jax.Array
    actions: jax.Array

def process_data(config, env, data):
    data = data.chunk(
        config.action_length + config.obs_length
    )
    def process_chunk(sample):
        obs = jax.tree.map(lambda x: x[:config.obs_length], sample.elements.reduced_state)
        obs = jax.vmap(env.full_state)(obs)
        obs = jax.vmap(env.observe)(obs)
        return Sample(
            obs,
            sample.elements.action[-config.action_length:]
        )
    return data.map(process_chunk)

def eval(env, policy, T, x0, rng_key):
    r = stanza.policy.rollout(
        env.step, x0, 
        policy, observe=env.observe,
        policy_rng_key=rng_key,
        length=T
    )
    pre_states, post_states = (
        jax.tree.map(lambda x: x[:-1], r.states),
        jax.tree.map(lambda x: x[1:], r.states)
    )
    rewards = jax.vmap(env.reward)(pre_states, r.actions, post_states)
    video = jax.vmap(
        lambda x: env.render(x, ImageRender(64, 64))
    )(r.states)
    return jnp.max(rewards, axis=-1), (255*video).astype(jnp.uint8)

def evaluate(env, x0s, T, policy, rng_key):
    N = stanza.util.axis_size(x0s, 0)

    # shard the x0s
    # sharding = PositionalSharding(
    #     mesh_utils.create_device_mesh((8,), jax.devices()[:8])
    # ).reshape((8,1))
    # x0s = jax.lax.with_sharding_constraint(x0s, sharding)

    rewards, videos = jax.vmap(partial(eval, env, policy, T))(
        x0s,
        jax.random.split(rng_key, N)
    )
    # reshape all the videos into a single video
    video = jax.vmap(
        lambda x: stanza.canvas.image_grid(x), 
        in_axes=1, out_axes=0
    )(videos)
    return {
        "mean_reward": jnp.mean(rewards),
        "std_reward": jnp.std(rewards),
        "test_demonstrations": Video(video)
    }

def main(config : Config):
    logger.info(f"Running {config}")
    rng = PRNGSequence(jax.random.key(config.seed))

    logger.info(f"Loading dataset [blue]{config.dataset}[/blue]")
    dataset = datasets.create(config.dataset)
    env = dataset.create_env()
    train_data = process_data(config, env, dataset.splits["train"])

    test_data = dataset.splits["test"].truncate(1)
    x0s = test_data.map(
        lambda x: env.full_state(
            jax.tree.map(lambda x: x[0], x.reduced_state)
        )
    ).as_pytree()
    eval = functools.partial(evaluate, env, x0s, config.timesteps)

    wandb_run = wandb.init(
        project="policy_eval",
        config=stanza.util.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")

    policy = config.policy.train_policy(
        wandb_run, train_data, eval, rng
    )
    logger.info(f"Performing final evaluation...")

    output = jax.jit(partial(eval,policy))(jax.random.key(42))
    # get the metrics and final reportables
    # from the eval output
    metrics, reportables = stanza.train.reporting.as_log_dict(output)
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    wandb_run.summary.update(metrics)
    wandb_run.log({
        k: stanza.train.wandb.map_reportable(v)
        for (k,v) in reportables.items()
    })
    wandb_run.finish()

@command
def run(config: ConfigProvider):
    logger.setLevel(logging.DEBUG)
    main(Config.parse(config))