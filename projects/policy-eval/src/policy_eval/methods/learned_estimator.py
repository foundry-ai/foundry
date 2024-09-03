import functools

import foundry.core.tree as tree
import foundry.core as F
import foundry.numpy as jnp

from ..common import MethodConfig, Sample, Inputs


from foundry.data import Data
from foundry.diffusion import DDPMSchedule
from foundry.policy import PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform


from foundry.env import Environment

from foundry.core.dataclasses import dataclass, replace
from foundry.diffusion import nonparametric

from foundry.env.core import ObserveConfig
from foundry.env.mujoco.pusht import PushTAgentPos
from foundry.env.mujoco.robosuite import EEfPose

import optax
import foundry.train
import foundry.train.wandb
import foundry.util
import pickle
import os

import logging
logger = logging.getLogger(__name__)

class Transform:
    def transform(self, x):
        pass

    def inv_transform(self, x):
        pass

@dataclass
class LearnedEstimatorConfig(MethodConfig):
    estimator: str = "nw"
    bandwidth: float = 0.01
    relative_actions: bool = False

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    diffusion_steps: int = 50
    action_horizon: int = 8

    def run(self, inputs: Inputs):
        schedule = DDPMSchedule.make_squaredcos_cap_v2(
            self.diffusion_steps,
            prediction_type="sample"
        )
        train_data = inputs.train_data.as_pytree()
        # make the actions relative to the agent position
        if self.relative_actions:
            pass

        N = tree.axis_size(train_data, 0)
        # split the data into two halves
        sub_data = tree.map(lambda x: x[:N//2], train_data)

        def target_denoiser(rng_key, obs, noisy_actions, t):
            return nonparametric.nw_cond_diffuser(
                obs, train_data, schedule,
                nonparametric.log_gaussian_kernel, self.bandwidth
            )(rng_key, noisy_actions, t)
        
        def denoiser_model(transform : Transform, rng_key, 
                        obs, noisy_actions, t):
            transform.transform(x, y)
            return nonparametric.nw_cond_diffuser(
                obs, sub_data, schedule,
                nonparametric.log_gaussian_kernel, self.bandwidth
            )(rng_key, noisy_actions, t)
        
        def loss_fn(transform, rng_key, sample):
            denoiser = functools.partial(denoiser_model, transform)
            schedule.add_noise(n_key, )

        transformation = LinearTransform()
        train_samples = train_data.as_pytree()
        obs_length, action_length = (
            foundry.util.axis_size(train_data.observations, 1),
            foundry.util.axis_size(train_data.actions, 1)
        )

import flax.nnx as nnx
import jax.scipy.linalg

class LinearTransform(Transform, nnx.Model):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        W = jax.random.orthogonal(key, (dim, dim))
        P, L, U = jax.scipy.linalg.lu(W)
        S = jnp.diag(L)
        U = jnp.triu(U, 1)
        self.dim = dim
        self.P = nnx.Variable(P)
        self.L, self.U, self.S = nnx.Param(L), nnx.Param(U), nnx.Param(S)

    def transform(self, x):
        P, L, U, S = self.P, self.L, self.U, self.S
        identity = jnp.eye(self.dim)
        L = jnp.tril(L, -1) + identity
        U = jnp.triu(U, 1)
        W = P @ L @ (U + jnp.diag(S))
        output = jnp.dot(x, W)
        log_det_jacobian = jnp.log(jnp.abs(S)).sum()
        return output, log_det_jacobian

    def inv_transform(self, x):
        P, L, U, S = self.P, self.L, self.U, self.S
        identity = jnp.eye(self.dim)
        L = jnp.tril(L, -1) + identity
        U = jnp.triu(U, 1)
        W = P @ L @ (U + jnp.diag(S))
        outputs = x @ jnp.linalg.inv(W)
        log_det_jacobian = -jnp.log(jnp.abs(S)).sum()
        return outputs, log_det_jacobian