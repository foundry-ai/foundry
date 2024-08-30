from foundry import core as F
from foundry import numpy as jnp
from foundry import graphics

import foundry.random
import foundry.train.reporting
import foundry.train.wandb
import wandb

from foundry.data.sequence import Chunk
from foundry.core.dataclasses import dataclass
from foundry.datasets.env import datasets
from foundry.core.typing import Array
from foundry.core import tree
from foundry.random import PRNGSequence
from foundry.env import Environment, ImageActionsRender
from foundry.train.reporting import Video

from .methods.behavior_cloning import BCConfig
from .methods.diffusion_estimator import EstimatorConfig
from .methods.diffusion_policy import DPConfig
from .methods.nearest_neighbor import NearestConfig

from .common import Sample, Inputs, MethodConfig

from omegaconf import MISSING
from functools import partial

import functools
import logging

logger = logging.getLogger(__name__)

@dataclass
class Config:
    seed: int = 42
    # these get mixed into the "master" seed
    train_seed: int = 42
    eval_seed: int = 42

    method : str = "diffusion_policy"
    dataset : str = "robomimic/pickplace/can/ph"

    # total trajectories to load
    train_trajectories : int | None = None
    test_trajectories : int | None = None
    validation_trajectories : int | None = None

    render_trajectories : int = 4

    obs_length: int = 1
    action_length: int = 16

    timesteps: int = 200

    render_width = 128
    render_height = 128

    bc : BCConfig = BCConfig()
    estimator : EstimatorConfig = EstimatorConfig()
    dp: DPConfig = DPConfig()
    nearest: NearestConfig = NearestConfig()

    @property
    def method_config(self) -> MethodConfig:
        match self.method:
            case "bc": return self.bc
            case "estimator": return self.estimator
            case "diffusion_policy": return self.dp
            case "nearest": return self.nearest
            case _: raise ValueError(f"Unknown method: {self.method}")

@F.jit
def policy_rollout(env, T, x0, rng_key, policy):
    r_policy, r_env = foundry.random.split(rng_key)
    rollout = foundry.policy.rollout(
        env.step, x0,
        policy, observe=env.observe,
        model_rng_key=r_env,
        policy_rng_key=r_policy,
        length=T,
        last_action=True
    )
    pre_states, actions, post_states = (
        tree.map(lambda x: x[:-1], rollout.states),
        tree.map(lambda x: x[:-1], rollout.actions),
        tree.map(lambda x: x[1:], rollout.states)
    )
    rewards = F.vmap(env.reward)(
        pre_states, actions, post_states
    )
    # max over time for the rewards
    reward = jnp.max(rewards, axis=0)
    return rollout, reward

@F.jit
def render_video(env, render_width, 
                render_height, rollout):
    states = rollout.states
    actions = rollout.info
    return F.vmap(env.render)(states, ImageActionsRender(
        render_width, render_height,
        actions=actions
    ))

@F.jit
def validate(env, T, render_width, render_height,
                num_videos, x0s, rng_key, policy) -> tuple[Array, Video] | Array:
    rollout_fn = partial(policy_rollout, env, T, policy=policy) 
    render_fn = partial(render_video, env, 
        render_width, render_height
    )
    N = tree.axis_size(x0s, 0)
    rngs = foundry.random.split(rng_key, N)

    rollouts, rewards = F.vmap(rollout_fn)(x0s, rngs)

    if num_videos is None:
        return rewards
    # render the videos
    video_rollouts = tree.map(
        lambda x: x[:num_videos], rollouts
    )
    videos = F.vmap(render_fn)(video_rollouts)
    # vmap over the frames axis
    videos = F.vmap(graphics.image_grid, in_axes=1, out_axes=0)(videos)
    return rewards, Video(videos, fps=10)

def process_data(config : Config, env : Environment, data):
    logger.info("Computing full state from data...")

    def process_element(element):
        if element.state is None: return env.full_state(element.reduced_state)
        else: return element.state
    data = data.map_elements(process_element).cache()

    logger.info("Chunking data...")
    data = data.chunk(
        config.action_length + config.obs_length
    )
    def process_chunk(chunk : Chunk):
        states = chunk.elements
        if config.dataset.startswith("robomimic"):
            from foundry.env.mujoco.robosuite import EEfPose
            action_obs = EEfPose()
        elif config.dataset.startswith("pusht"):
            from foundry.env.mujoco.pusht import PushTAgentPos
            action_obs = PushTAgentPos()
        actions = F.vmap(lambda s: env.observe(s, action_obs))(states)
        actions = tree.map(lambda x: x[-config.action_length:], actions)
        obs_states = tree.map(lambda x: x[:config.obs_length], states)
        curr_state = tree.map(lambda x: x[-1], obs_states)
        obs = F.vmap(env.observe)(obs_states)
        return Sample(
            curr_state, obs, actions
        )
    data = data.map(process_chunk).cache()
    return data

def run(config: Config):
    logger.setLevel(logging.DEBUG)
    logger.info(f"Running {config}")

    wandb_run = wandb.init(project="policy-eval", config=config)
    logger.info(f"Logging to {wandb_run.url}")
    # ---- set up the random number generators ----

    # split the master RNG into train, eval
    train_key, eval_key = foundry.random.split(foundry.random.key(config.seed))

    # fold in the seeds for the train, eval
    train_key = foundry.random.fold_in(train_key, config.train_seed)
    eval_key = foundry.random.fold_in(eval_key, config.eval_seed)

    # ---- set up the training data -----

    logger.info(f"Loading dataset [blue]{config.dataset}[/blue]")
    dataset = datasets.create(config.dataset)
    env = dataset.create_env()

    logger.info("Processing training data...")
    train_data = dataset.splits["train"]
    if config.train_trajectories is not None:
        train_data = train_data.slice(0, config.train_trajectories)
    train_data = process_data(config, env, train_data)

    logger.info("Processing test data...")
    test_data = dataset.splits["test"]
    if config.test_trajectories is not None:
        test_data = test_data.slice(0, config.test_trajectories)
    test_data = process_data(config, env, test_data)

    validation_data = dataset.splits["validation"]
    if config.validation_trajectories is not None:
        validation_data = validation_data.slice(0, config.validation_trajectories)
    # get the first state of the trajectory
    validation_data = validation_data.truncate(1).map(
        lambda x: env.full_state(tree.map(lambda y: y[0], x.reduced_state))
    ).as_pytree()
    N_validation = tree.axis_size(validation_data, 0)

    # validation trajectories
    validate_fn = functools.partial(
        validate, env, config.timesteps,
        config.render_width, config.render_height,
        None, validation_data
    )
    num_render_trajectories = (config.render_trajectories 
        if config.render_trajectories is not None else
        min(4, N_validation)
    )
    validate_render_fn = functools.partial(
        validate, env, config.timesteps,
        config.render_width, config.render_height,
        num_render_trajectories, validation_data
    )

    method : MethodConfig = config.method_config

    inputs = Inputs(
        wandb_run=wandb_run,
        timesteps=config.timesteps,
        rng=PRNGSequence(train_key),
        env=env,
        validate=validate_fn,
        validate_render=validate_render_fn,
        train_data=train_data,
        test_data=test_data
    )
    final_result = method.run(inputs)
    final_policy = final_result.create_policy()

    logger.info("Running validation for final policy...")
    rewards, video = validate_render_fn(eval_key, final_policy)
    mean_reward = jnp.mean(rewards)
    std_reward = jnp.std(rewards)
    outputs = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "final_validation_demonstrations": video
    }
    metrics, reportables = foundry.train.reporting.as_log_dict(outputs)
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")
    wandb_run.summary.update(metrics)
    wandb_run.log({
        k: foundry.train.wandb.map_reportable(v)
        for (k,v) in reportables.items()
    })
    wandb_run.finish()