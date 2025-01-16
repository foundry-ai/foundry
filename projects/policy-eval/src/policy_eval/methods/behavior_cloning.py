from ..common import Sample, Inputs, Result, DataConfig
from typing import Callable

import foundry.core as F
from foundry.core import tree
from foundry.random import PRNGSequence
from foundry.policy import Policy, PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.core.dataclasses import dataclass

from foundry.data import Data, PyTreeData
from foundry.data.normalizer import Normalizer, LinearNormalizer, StdNormalizer
from foundry.train import Vars

from foundry import train
from foundry.env.core import Environment

import foundry.train.console

import wandb
import optax
import flax.linen as nn
import flax.linen.activation as activations

from typing import Sequence
from foundry.models.embed import SinusoidalPosEmbed
from foundry.models.unet import UNet

import jax
import foundry.numpy as jnp
import logging
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class MLPConfig:
    net_width: int = 16
    net_depth: int = 3
    activation: str = "gelu"

    def create_model(self, rng_key, observations, actions):
        model = MLP(
            features=[self.net_width]*self.net_depth,
            activation=self.activation
        )
        vars = F.jit(model.init)(rng_key, observations, actions, jnp.zeros((), dtype=jnp.uint32))
        def model_fn(vars, rng_key, observations, noised_actions):
            return model.apply(vars, observations, noised_actions)
        return model_fn, vars

@dataclass
class Checkpoint(Result):
    data: DataConfig # dataset this model was trained on
    observations_structure: tuple[int]
    actions_structure: tuple[int]
    action_horizon: int

    model_config: MLPConfig 

    obs_normalizer: Normalizer
    action_normalizer: Normalizer

    vars: Vars

    def create_denoiser(self):
        model, _ = self.model_config.create_model(foundry.random.key(42), 
            self.observations_structure,
            self.actions_structure
        )
        return lambda obs, rng_key, noised_actions, t: model(
            self.vars, rng_key, obs, noised_actions, t - 1
        )

    def create_policy(self) -> Policy:
        model, _ = self.model_config.create_model(foundry.random.key(42), 
            self.observations_structure,
            self.actions_structure
        )
        # TODO: assert that the vars are the same type/shape
        def chunk_policy(input: PolicyInput) -> PolicyOutput:
            obs = input.observation
            obs = self.obs_normalizer.normalize(obs)
            action_sample = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), self.actions_structure)
            action = model(self.vars, input.rng_key, obs, action_sample)
            action = self.action_normalizer.unnormalize(action)
            return PolicyOutput(action=action[:self.action_horizon], info=action)
        obs_horizon = tree.axis_size(self.observations_structure, 0)
        return ChunkingTransform(
            obs_horizon, self.action_horizon
        ).apply(chunk_policy)


@dataclass
class BCConfig:
    model: str = "mlp"

    mlp : MLPConfig = MLPConfig()

    epochs: int | None = None
    iterations: int | None = None
    test_interval: int = 50
    eval_interval: int = 2000
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    action_horizon: int = 16

    log_video: bool = True

    @property
    def model_config(self) -> MLPConfig:
        if self.model == "mlp":
            return self.mlp
        elif self.model == "unet":
            return self.unet
        else:
            raise ValueError(f"Unknown model type: {self.model}")
    
    def run(self, inputs: Inputs):
        _, data = inputs.data.load({"train", "test"})
        train_data = data["train"].cache()
        test_data = data["test"].cache()

        train_sample = train_data[0]
        observations_structure = tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), 
                                          train_sample.observations)
        action_horizon = min(self.action_horizon, inputs.data.action_length)
        actions_structure = tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), 
                                          train_sample.actions)
        model, vars = self.model_config.create_model(
            next(inputs.rng),
            observations_structure,
            actions_structure
        )
        total_params = sum(v.size for v in tree.leaves(vars))

        logger.info(f"Total parameters: {total_params}")

        normalizer = StdNormalizer.from_data(train_data)

        epoch_iterations = len(train_data) // self.batch_size
        if self.epochs is not None:
            total_iterations = self.epochs * epoch_iterations
        elif self.iterations is not None:
            total_iterations = self.iterations
        else:
            raise ValueError("Must specify either epochs or iterations")

        # initialize optimizer, EMA
        opt_schedule = optax.warmup_cosine_decay_schedule(
            self.learning_rate/10, self.learning_rate,
            min(int(total_iterations*0.01), 500), total_iterations
        )
        optimizer = optax.adamw(opt_schedule,
            weight_decay=self.weight_decay
        )
        opt_state = F.jit(optimizer.init)(vars["params"])
        ema = optax.ema(0.9)
        ema_state = F.jit(ema.init)(vars)
        ema_update = F.jit(ema.update)

        def loss_fn(vars, rng_key, sample: Sample):
            sample_norm = normalizer.normalize(sample)
            obs = sample_norm.observations
            action = sample_norm.actions
            pred_action = model(vars, rng_key, obs, action)
            loss = jnp.mean(jnp.square(pred_action - action))
            return train.LossOutput(
                loss=loss,
                metrics={
                    "loss": loss
                }
            )
        
        def make_checkpoint() -> Checkpoint:
            return Checkpoint(
                data=inputs.data,
                observations_structure=observations_structure,
                actions_structure=actions_structure,
                action_horizon=action_horizon,
                model_config=self.model_config,
                obs_normalizer=normalizer.map(lambda x: x.observations),
                action_normalizer=normalizer.map(lambda x: x.actions),
                vars=vars
            )

        batched_loss_fn = train.batch_loss(loss_fn)

        train_stream = train_data.stream().batch(self.batch_size)
        test_stream = test_data.stream().batch(self.batch_size)
        with train.loop(
                data=train_stream,
                rng_key=next(inputs.rng),
                iterations=total_iterations,
                progress=True
            ) as loop, test_stream.build() as test_stream:
            for epoch in loop.epochs():
                for step in epoch.steps():
                    # print(step.batch.observations)
                    # print(step.batch.actions)
                    # *note*: consumes opt_state, vars
                    train_rng, test_rng, val_rng = jax.random.split(step.rng_key, 3)
                    opt_state, vars, metrics = train.step(
                        batched_loss_fn, optimizer, opt_state, vars, 
                        train_rng, step.batch,
                    )
                    _, ema_state = ema_update(vars, ema_state)
                    train.wandb.log(step.iteration, metrics, 
                                    run=inputs.wandb_run, prefix="train/")
                    if step.iteration % self.test_interval == 0:
                        test_stream, test_metrics = train.eval_stream(
                            batched_loss_fn, vars, 
                            test_rng, test_stream
                        )
                        train.wandb.log(step.iteration, test_metrics, 
                                        run=inputs.wandb_run, prefix="test/")
                        if step.iteration % 100 == 0:
                            train.console.log(step.iteration, test_metrics, prefix="test.")
                    if step.iteration % 100 == 0:
                        train.console.log(step.iteration, metrics, 
                                          prefix="train.")
                    if step.iteration % self.eval_interval == 0:
                        logger.info("Evaluating policy...")
                        if self.log_video:
                            rewards, video = inputs.validate_render(val_rng, make_checkpoint().create_policy())
                            reward_metrics = {
                                "mean_reward": jnp.mean(rewards),
                                "std_reward": jnp.std(rewards),
                                "demonstrations": video
                            }
                        else:
                            rewards = inputs.validate(val_rng, make_checkpoint().create_policy())
                            reward_metrics = {
                                "mean_reward": jnp.mean(rewards),
                                "std_reward": jnp.std(rewards),
                            }
                        train.console.log(step.iteration, reward_metrics, prefix="eval.")
                        train.wandb.log(step.iteration, reward_metrics,
                                        run=inputs.wandb_run, prefix="eval/")
            # log the last iteration
            if step.iteration % 100 != 0:
                train.console.log(step.iteration, metrics)
                train.wandb.log(step.iteration, metrics, run=inputs.wandb_run)

        # Return the final checkpoint
        return make_checkpoint()
    

class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "gelu"

    @nn.compact
    def __call__(self, obs, action_sample, train=False):
        activation = getattr(activations, self.activation)
        obs_flat, _ = jax.flatten_util.ravel_pytree(obs)
        action_sample_flat, action_sample_uf = jax.flatten_util.ravel_pytree(action_sample)
        obs_embed_dim = max(self.features)
        obs_embed = nn.Sequential([
            nn.Dense(obs_embed_dim),
            activation,
            nn.Dense(obs_embed_dim),
        ])(obs_flat)
        x = obs_embed
        for feat in self.features:
            x = activation(nn.Dense(feat)(x))
        x = nn.Dense(action_sample_flat.shape[-1])(x)
        # x = jax.nn.tanh(x)
        x = action_sample_uf(x)
        return x