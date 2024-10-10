from ..common import Sample, Inputs, Result, DataConfig
from typing import Callable

from foundry.random import PRNGSequence
from foundry.policy import Policy, PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.core.dataclasses import dataclass

from foundry.data import Data, PyTreeData
from foundry.data.normalizer import Normalizer, LinearNormalizer, StdNormalizer
from foundry.train import Vars

from foundry import train
from foundry.env import Environment

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
    net_width: int = 64
    net_depth: int = 3

    def create_model(self, rng_key, observations, actions):
        model = DiffusionMLP(
            features=[self.net_width]*self.net_depth
        )
        vars = F.jit(model.init)(rng_key, observations, actions, jnp.zeros((), dtype=jnp.uint32))
        def model_fn(vars, rng_key, observations, noised_actions, t):
            return model.apply(vars, observations, noised_actions, t - 1)
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
            model_fn = lambda rng_key, noised_actions, t: model(
                self.vars, rng_key, obs, noised_actions, t - 1
            )
            action = self.schedule.sample(input.rng_key, model_fn, self.actions_structure) 
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

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    diffusion_steps: int = 32
    action_horizon: int = 8
    
    save_dir: str | None = None
    from_checkpoint: bool = False
    checkpoint_filename: str | None = None

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
        total_iterations = self.epochs * epoch_iterations

        # initialize optimizer, EMA
        optimizer = optax.adamw(self.learning_rate, weight_decay=self.weight_decay)
        opt_state = F.jit(optimizer.init)(vars["params"])
        ema = optax.ema(0.9)
        ema_state = F.jit(ema.init)(vars)
        ema_update = F.jit(ema.update)

        def loss_fn(vars, rng_key, sample: Sample, iteration):
            sample_norm = normalizer.normalize(sample)
            obs = sample_norm.observations
            action = sample_norm.actions
            pred_action = model.apply(vars, obs, action_sample)
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
                action_horizon=self.action_horizon,
                model_config=self.model_config,
                schedule=schedule,
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
                    if step.iteration % 50 == 0:
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
                    if step.iteration % 1000 == 0:
                        rewards, video = inputs.validate_render(val_rng, make_checkpoint().create_policy())
                        reward_metrics = {
                            "mean_reward": jnp.mean(rewards),
                            "std_reward": jnp.std(rewards),
                            "demonstrations": video
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

# def train_net_BC(
#         config : BCConfig,  wandb_run, train_data, env, eval):
    
#     train_sample = train_data[0]
#     action_sample = train_sample.actions
#     normalizer = StdNormalizer.from_data(train_data)
#     train_data_tree = train_data.as_pytree()

#     # Get chunk lengths
#     obs_length, action_length = (
#         foundry.util.axis_size(train_data_tree.observations, 1),
#         foundry.util.axis_size(train_data_tree.actions, 1)
#     )

#     rng = PRNGSequence(config.seed)
    
#     if config.model == "mlp":
#         model = MLP(
#             features=[config.net_width]*config.net_depth, 
#             has_skip=config.has_skip
#         )
#     else:
#         raise ValueError(f"Unknown model type: {config.model}")
    
#     vars = jax.jit(model.init)(next(rng), train_sample.observations, action_sample)
    
#     total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
#     logger.info(f"Total parameters: [blue]{total_params}[/blue]")

#     def loss_fn(vars, rng_key, sample: Sample, iteration):
#         sample_norm = normalizer.normalize(sample)
#         obs = sample_norm.observations
#         action = sample_norm.actions
#         pred_action = model.apply(vars, obs, action_sample)
#         loss = jnp.mean(jnp.square(pred_action - action))
        
#         return train.LossOutput(
#             loss=loss,
#             metrics={
#                 "loss": loss
#             }
#         )
#     batched_loss_fn = train.batch_loss(loss_fn)

#     opt_sched = optax.cosine_onecycle_schedule(config.iterations, 1e-4)
#     optimizer = optax.adamw(opt_sched)
#     opt_state = optimizer.init(vars["params"])

#     # Create a directory to save checkpoints
#     current_dir = os.path.dirname(os.path.realpath(__file__))
#     ckpts_dir = os.path.join(current_dir, "checkpoints")
#     if not os.path.exists(ckpts_dir):
#         os.makedirs(ckpts_dir)

#     train_data_batched = train_data.stream().batch(config.batch_size)

#     with foundry.train.loop(train_data_batched, 
#                 rng_key=next(rng),
#                 iterations=config.iterations,
#                 progress=True
#             ) as loop:
#         for epoch in loop.epochs():
#             for step in epoch.steps():
#                 # *note*: consumes opt_state, vars
#                 opt_state, vars, metrics = train.step(
#                     batched_loss_fn, optimizer, opt_state, vars, 
#                     step.rng_key, step.batch,
#                     # extra arguments for the loss function
#                     iteration=step.iteration
#                 )
#                 if step.iteration % 100 == 0:
#                     train.ipython.log(step.iteration, metrics)
#                     train.wandb.log(step.iteration, metrics, run=wandb_run)
#                 if step.iteration > 0 and step.iteration % 20000 == 0:
#                     ckpt = {
#                         "config": config,
#                         "model": model,
#                         "vars": vars,
#                         "opt_state": opt_state,
#                         "normalizer": normalizer
#                     }
#                     file_path = os.path.join(ckpts_dir, f"{wandb_run.id}_{step.iteration}.pkl")
#                     with open(file_path, 'wb') as file:
#                         pickle.dump(ckpt, file)
#                     wandb_run.log_model(path=file_path, name=f"{wandb_run.id}_{step.iteration}")
#         train.ipython.log(step.iteration, metrics)
#         train.wandb.log(step.iteration, metrics, run=wandb_run)

#     # save model
#     ckpt = {
#         "config": config,
#         "model": model,
#         "vars": vars,
#         "opt_state": opt_state,
#         "normalizer": normalizer
#     }
#     file_path = os.path.join(ckpts_dir, f"{wandb_run.id}_final.pkl")
#     with open(file_path, 'wb') as file:
#         pickle.dump(ckpt, file)
    
#     def chunk_policy(input: PolicyInput) -> PolicyOutput:
#         obs = input.observation
#         obs = normalizer.map(lambda x: x.observations).normalize(obs)
#         action = model.apply(vars, obs, action_sample)
#         action = normalizer.map(lambda x: x.actions).unnormalize(action)
#         action = action[:config.action_horizon]
#         return PolicyOutput(action=action, info=action)
    
#     policy = ChunkingTransform(
#         obs_length, config.action_horizon
#     ).apply(chunk_policy)

#     return policy
    

class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "relu"
    obs_embed_dim: int = 256

    @nn.compact
    def __call__(self, obs, action_sample, train=False):
        activation = getattr(activations, self.activation)
        obs_flat, _ = jax.flatten_util.ravel_pytree(obs)
        action_sample_flat, action_sample_uf = jax.flatten_util.ravel_pytree(action_sample)
        obs_embed = nn.Sequential([
            nn.Dense(self.obs_embed_dim),
            activation,
            nn.Dense(self.obs_embed_dim),
        ])(obs_flat)
        x = obs_embed
        for feat in self.features:
            x = activation(nn.Dense(feat)(x))
            if self.has_skip:
                x = jnp.concatenate((x, obs_flat), axis=-1)
        x = nn.Dense(action_sample_flat.shape[-1])(x)
        # x = jax.nn.tanh(x)
        x = action_sample_uf(x)
        return x