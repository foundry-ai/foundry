from stanza.runtime import setup
setup()

from stanza import dataclasses
from stanza.dataclasses import dataclass, replace
from stanza.runtime import ConfigProvider, command
from stanza.datasets.env import datasets
from stanza.datasets.env import datasets
from stanza.random import PRNGSequence
from stanza.env import ImageRender
from stanza.train.reporting import Video
from stanza import canvas
from stanza.policy import PolicyInput, PolicyOutput


from stanza.env.mujoco.pusht import PushTObs


from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

from functools import partial
from typing import Any

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
    action_length: int = 16
    policy: PolicyConfig = None
    timesteps: int = 200

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
    state: Any
    observations: jax.Array
    actions: jax.Array

def process_data(config, env, data):
    def process_element(element):
        return env.full_state(element.reduced_state)
    data = data.map_elements(process_element).cache()
    logger.info("Chunking data")
    data = data.chunk(
        config.action_length + config.obs_length
    )
    def process_chunk(chunk):
        states = chunk.elements
        actions = jax.vmap(lambda s: env.observe(s, PushTObs()).agent_pos)(states)
        actions = jax.tree.map(lambda x: x[-config.action_length:], actions)
        obs_states = jax.tree.map(lambda x: x[:config.obs_length], states)
        curr_state = jax.tree.map(lambda x: x[-1], obs_states)
        obs = jax.vmap(env.observe)(obs_states)
        return Sample(
            curr_state, obs, actions
        )
    return data.map(process_chunk)

def eval(env, policy, chunk_policy, T, x0, rng_key):
    rng_key, action_key = jax.random.split(rng_key)
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

    # def batch_policy(obs, state, rng_key):
    #     keys = jax.random.split(rng_key, 2)
    #     return jax.vmap(chunk_policy, in_axes=(PolicyInput(None, None, rng_key=0),))(
    #         PolicyInput(obs, state, rng_key=keys)
    #     ).action

    # def draw_action_chunk(action_chunk, weight, img_config):
    #     T = action_chunk.shape[0]
    #     colors = jnp.array((jnp.arange(T)/T, jnp.zeros(T), jnp.zeros(T), weight*jnp.ones(T))).T
    #     circles = canvas.fill(
    #         canvas.circle(action_chunk, 0.02*jnp.ones(T)),
    #         color=colors
    #     )
    #     circles = canvas.stack_batch(circles)
    #     circles = canvas.transform(circles,
    #         translation=(1,-1),
    #         scale=(img_config.width/2, -img_config.height/2)
    #     )
    #     return circles

    # def render_frame(state, rng_key, img_config):
    #     image = env.render(state, img_config)

    #     obs = env.observe(state)
    #     action_chunks = batch_policy(obs, state, rng_key)
    #     weights = (0.2*jnp.ones(action_chunks.shape[0])).at[0].set(1)
    #     circles = canvas.stack_batch(jax.vmap(draw_action_chunk, in_axes=(0,0,None))(action_chunks, weights, img_config))
    #     image = canvas.paint(image, circles)
        
    #     return image
    
    # video = jax.vmap(
    #     lambda x, rng_key: render_frame(x, rng_key, ImageRender(256, 256))        
    # )(r.states, jax.random.split(action_key, T))
    video = jax.vmap(
        lambda x: env.render(x, ImageRender(64, 64))
    )(r.states)
    return jnp.max(rewards, axis=-1), (255*video).astype(jnp.uint8)

def evaluate(env, x0s, T, policy, chunk_policy, rng_key):
    N = stanza.util.axis_size(x0s, 0)

    # shard the x0s
    # sharding = PositionalSharding(
    #     mesh_utils.create_device_mesh((jax.device_count(),))
    # )
    # x0s = jax.tree.map(
    #     lambda x: jax.lax.with_sharding_constraint(x, sharding.reshape((jax.device_count(),) + (1,)*(x.ndim-1))),
    #     x0s
    # )

    rewards, videos = jax.vmap(partial(eval, env, policy, chunk_policy, T))(
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
    train_data = dataset.splits["train"].slice(0,140)
    logger.info(f"Processing dataset.")
    train_data = process_data(config, env, train_data).cache()
    # jax.debug.print("{s}", s=train_data)
    # train_data = train_data.slice(0,5)
    # jax.debug.print("{s}", s=train_data.as_pytree())

    test_data = dataset.splits["test"].truncate(1).slice(0,3)
    test_x0s = test_data.map(
        lambda x: env.full_state(
            jax.tree.map(lambda x: x[0], x.reduced_state)
        )
    ).as_pytree()
    #test_x0s = jax.tree.map(lambda x: x[:1], test_x0s)
    eval = functools.partial(evaluate, env, test_x0s, config.timesteps)

    wandb_run = wandb.init(
        project="policy_eval",
        config=stanza.util.flatten_to_dict(config)[0]
    )
    logger.info(f"Logging to [blue]{wandb_run.url}[/blue]")

    policy, chunk_policy = config.policy.train_policy(
        wandb_run, train_data, env, eval, rng
    )
    logger.info(f"Performing final evaluation...")

    output = jax.jit(partial(eval,policy, chunk_policy))(jax.random.key(42))
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