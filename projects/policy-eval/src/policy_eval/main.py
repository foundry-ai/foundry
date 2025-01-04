from foundry import core as F
from foundry import numpy as jnp
from foundry import graphics

import foundry.util.serialize
import foundry.random
import foundry.train.reporting
import foundry.train.wandb
import foundry.datasets.env
import wandb

from foundry.data.sequence import Chunk
from foundry.datasets.core import DatasetRegistry
from foundry.core.dataclasses import dataclass
from foundry.core.typing import Array
from foundry.core import tree
from foundry.random import PRNGSequence
from foundry.env.core import (
    Environment, ObserveConfig,
    ImageActionsRender
)
from foundry.train.reporting import Video

from foundry.env.mujoco.robosuite import EEfPose
from foundry.env.mujoco.pusht import PushTAgentPos

from .methods.behavior_cloning import BCConfig
from .methods.diffusion_estimator import EstimatorConfig
from .methods.diffusion_policy import DPConfig
from .methods.nearest_neighbor import NearestConfig
from .methods.pretrained import PretrainedConfig

from .common import DataConfig, Inputs, MethodConfig

from functools import partial

import functools
import boto3
import logging

logger = logging.getLogger(__name__)

@dataclass
class Config:
    seed: int = 42
    # these get mixed into the "master" seed
    train_seed: int = 42
    eval_seed: int = 42

    # Dataset configuration
    dataset : str = "pusht/chi" # "robomimic/pickplace/can/ph"
    env_type : str = "positional"
    train_trajectories : int | None = None
    test_trajectories : int | None = None
    validation_trajectories : int | None = None
    obs_length: int = 2
    action_length: int = 16

    # if evluate is set, contains a url
    # to the policy to evaluate
    method : str = "diffusion_policy"
    evaluate: str | None = None

    bucket_url : str | None = "s3://wandb-data"

    render_trajectories : int | None = None

    timesteps: int = 400

    render_width : int = 128
    render_height : int = 128

    bc : BCConfig = BCConfig()
    estimator : EstimatorConfig = EstimatorConfig()
    dp: DPConfig = DPConfig()
    nearest: NearestConfig = NearestConfig()
    pretrained: PretrainedConfig = PretrainedConfig()

    @property
    def method_config(self) -> MethodConfig:
        match self.method:
            case "bc": return self.bc
            case "estimator": return self.estimator
            case "diffusion_policy": return self.dp
            case "nearest": return self.nearest
            case "pretrained": return self.pretrained
            case _: raise ValueError(f"Unknown method: {self.method}")
    
    @property
    def data_config(self) -> DataConfig:
        return DataConfig(
            dataset=self.dataset,
            env_type=self.env_type,
            train_trajectories=self.train_trajectories,
            test_trajectories=self.test_trajectories,
            validation_trajectories=self.validation_trajectories,
            obs_length=self.obs_length,
            action_length=self.action_length
        )

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
    reward = env.total_reward(rollout.states, rollout.actions)
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
    import jax.lax
    videos = jax.lax.map(render_fn, video_rollouts)
    # vmap over the frames axis
    videos = F.vmap(graphics.image_grid, in_axes=1, out_axes=0)(videos)
    return rewards, Video(videos, fps=10)

def relative_action(obs_config : ObserveConfig,
                    state, action):
    if isinstance(obs_config, EEfPose):
        pass
    elif isinstance(obs_config, PushTAgentPos):
        pass
    else:
        raise ValueError(f"Unsupported obs_config {obs_config}")

def absolute_action(obs_config : ObserveConfig,
                    state, relative_action):
    if isinstance(obs_config, EEfPose):
        pass
    elif isinstance(obs_config, PushTAgentPos):
        pass
    else:
        raise ValueError(f"Unsupported obs_config {obs_config}")



def run(config: Config):
    logging.getLogger("policy_eval").setLevel(logging.DEBUG)
    logger.info(f"Running {config}")
    # ---- set up the random number generators ----

    # split the master RNG into train, eval
    train_key, eval_key = foundry.random.split(foundry.random.key(config.seed))

    # fold in the seeds for the train, eval
    train_key = foundry.random.fold_in(train_key, config.train_seed)
    eval_key = foundry.random.fold_in(eval_key, config.eval_seed)

    # ---- set up the training data -----
    env, splits = config.data_config.load({"validation"})
    validation_data = splits["validation"].as_pytree()
    N_validation = tree.axis_size(validation_data, 0)

    # validation trajectories
    validate_fn = functools.partial(
        validate, env, config.timesteps,
        config.render_width, config.render_height,
        None, validation_data
    )
    num_render_trajectories = (
        config.render_trajectories 
        if config.render_trajectories is not None else
        min(4, N_validation)
    )
    logger.info(f"Rendering {num_render_trajectories} trajectories")
    validate_render_fn = functools.partial(
        validate, env, config.timesteps,
        config.render_width, config.render_height,
        num_render_trajectories, validation_data
    )
    method : MethodConfig = config.method_config

    if config.evaluate:
        pass
    else:
        wandb_run = wandb.init(project="policy-eval", config=config)
        logger.info(f"Logging to {wandb_run.url}")
        inputs = Inputs(
            wandb_run=wandb_run,
            timesteps=config.timesteps,
            rng=PRNGSequence(train_key),
            env=env,
            bucket_url=f"{config.bucket_url}/{wandb_run.id}",
            data=config.data_config,
            validate=validate_fn,
            validate_render=validate_render_fn,
        )
        final_result = method.run(inputs)
        final_policy = final_result.create_policy()

        logger.info("Running validation for final policy...")
        rewards, video = validate_render_fn(eval_key, final_policy)
        mean_reward = jnp.mean(rewards)
        std_reward = jnp.std(rewards)
        q = jnp.array([0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100])
        quantiles = jnp.percentile(rewards, q) 
        outputs = {
            "reward_mean": mean_reward,
            "reward_std": std_reward,
            "reward_quant": {
                f"{q:03}": v for (q, v) in zip(q, quantiles)
            },
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
        if inputs.bucket_url is not None:
            final_result_url = f"{inputs.bucket_url}/final_result.zarr.zip"
            final_result.save_s3(final_result_url)
            artifact = wandb.Artifact(f"final_result", type="policy")
            artifact.add_reference(final_result_url, "checkpoint.zarr.zip")
            wandb_run.log_artifact(artifact)

        wandb_run.finish()
