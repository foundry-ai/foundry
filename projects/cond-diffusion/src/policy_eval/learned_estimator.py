import jax
import foundry.numpy as jnp
import jax.flatten_util

from policy_eval import Sample

from foundry.data import Data
from foundry.diffusion import DDPMSchedule
from foundry.runtime import ConfigProvider
from foundry.core.random import PRNGSequence
from foundry.policy import PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.env import Environment

from foundry.core.dataclasses import dataclass
from foundry.diffusion import nonparametric

from foundry.env.core import ObserveConfig
from foundry.env.mujoco.pusht import PushTAgentPos
from foundry.env.mujoco.robosuite import ManipulationTaskEEFPose

from typing import Callable

import optax
import foundry.train
import foundry.train.wandb
import foundry.util
import pickle
import os

import logging
logger = logging.getLogger(__name__)

@dataclass
class TrainedEstimatorConfig:
    estimator: str = "nw"
    kernel_bandwidth: float = 0.01
    diffusion_steps: int = 50
    relative_actions: bool = True
    agent_pos_config: ObserveConfig = ManipulationTaskEEFPose()
    action_horizon: int = 8
    checkpoint_path = ""
    iterations: int = 1000

    def parse(self, config: ConfigProvider) -> "TrainedEstimatorConfig":
        return config.get_dataclass(self)

    def train_policy(self, wandb_run, train_data, env, eval, rng):
        return trained_diffusion_estimator_policy(self, wandb_run, train_data, env, eval, rng)

def trained_diffusion_estimator_policy(
            config: TrainedEstimatorConfig,
            wandb_run,
            train_data : Data[Sample],
            env : Environment,
            eval : Callable,
            rng: jax.Array
        ):

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        prediction_type="sample"
    )

    reference_denoiser = load_checkpoint(config.checkpoint_path)

    transformation = LinearTransform()
    train_samples = train_data.as_pytree()
    obs_length, action_length = (
        foundry.util.axis_size(train_data.observations, 1),
        foundry.util.axis_size(train_data.actions, 1)
    )

    def denoiser_model(params, obs, rng_key, actions, t):
        transformation.apply(params, (obs))

        kernel = nonparametric.log_gaussian_kernel
        estimator = lambda obs: nonparametric.nw_cond_diffuser(
            obs, train_samples, schedule, kernel, config.kernel_bandwidth
        )

    match_denoiser(
        reference_denoiser, denoiser_model, 
        None, schedule, 
        rng, train_data, None, config.iterations
    )

def match_denoiser(target_denoiser, 
                   denoiser_model, init_params,
                   schedule, rng_key, train_data, test_data, iterations):
    def loss_fn(params, rng_key, sample):
        t_rng, s_rng, m1_rng, m2_rng = jax.random.split(rng_key, 4)
        t = jax.random.randint(t_rng, (), 0, schedule.num_steps) + 1
        noised_sample, _, _ = schedule.add_noise(s_rng, sample, t)

        reference = target_denoiser(m1_rng, noised_sample, t)
        prediction = denoiser_model.apply(params, m2_rng, noised_sample, t - 1)
        ref_flat, _ = jax.flatten_util.ravel_pytree(reference)
        pred_flat, _ = jax.flatten_util.ravel_pytree(prediction)
        loss = jnp.mean((ref_flat - pred_flat) ** 2)
        return foundry.train.LossOutput(loss=loss, metrics={"loss": loss})
    loss_fn = foundry.train.batch_loss(loss_fn)

    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(init_params)
    vars = init_params
    with foundry.train.loop(train_data, rng_key=rng_key, 
                      iterations=iterations) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                opt_state, vars, metrics = foundry.train.step(
                    loss_fn, optimizer, opt_state, 
                    vars, step.rng_key, step.batch
                )
                foundry.train.wandb.log(step.iteration, metrics)

def load_checkpoint(checkpoint_path):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ckpts_dir = os.path.join(current_dir, "checkpoints")
    file_path = os.path.join(ckpts_dir, checkpoint_path)
    with open(file_path, "rb") as file:
        ckpt = pickle.load(file)

    model = ckpt["model"]
    ema_vars = ckpt["ema_state"].ema
    normalizer = ckpt["normalizer"]

    def denoiser(rng_key, obs, noised_actions, t):
        obs = normalizer.map(lambda x: x.observations).normalize(obs)
        return model.apply(ema_vars, obs, noised_actions, t - 1)
    return denoiser

import flax.linen as nn

class LinearTransform(nn.Model):
    def __call__(self, x):
        x_flat, uf = jax.flatten_util.ravel_pytree(x)
        y_flat = nn.Linear(x.shape[-1])(x_flat)
        y = uf(y_flat)
        return y