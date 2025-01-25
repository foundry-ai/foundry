from ..common import Sample, Inputs, Result, DataConfig

import foundry.core as F
import foundry.random

from foundry.core import tree
from foundry.diffusion import DDPMSchedule
from foundry.random import PRNGSequence
from foundry.policy import Policy, PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.core.dataclasses import dataclass
from foundry.data.normalizer import Normalizer, LinearNormalizer, StdNormalizer
from foundry.train import Vars

from foundry import train

import foundry.train.console
import foundry.train.wandb
import optax

import flax.linen as nn
import flax.linen.activation as activations

from typing import Sequence
from foundry.models.embed import SinusoidalPosEmbed
from foundry.models.unet import UNet

import jax
import foundry.numpy as jnp
import logging

logger = logging.getLogger(__name__)

@dataclass
class MLPConfig:
    net_width: int = 64
    net_depth: int = 3
    activation: str = "gelu"

    def create_model(self, rng_key, observations, actions):
        model = DiffusionMLP(
            features=[self.net_width]*self.net_depth,
            activation=self.activation
        )
        vars = F.jit(model.init)(rng_key, observations, actions, jnp.zeros((), dtype=jnp.uint32))
        def model_fn(vars, rng_key, observations, noised_actions, t):
            return model.apply(vars, observations, noised_actions, t - 1)
        return model_fn, vars

@dataclass
class UNetConfig:
    base_channels: int = 128
    num_downsample: int = 4

    def create_model(self, rng_key, observations_structure, actions_structure):
        model = DiffusionUNet(
            dims=1,
            base_channels=self.base_channels,
            channel_mult=tuple([2**i for i in range(self.num_downsample)]),
        )
        observations = tree.map(lambda x: jnp.zeros_like(x), observations_structure)
        actions = tree.map(lambda x: jnp.zeros_like(x), actions_structure)
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

    model_config: MLPConfig | UNetConfig
    schedule: DDPMSchedule
    replica_noise: float | None

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
            s_rng, n_rng = foundry.random.split(input.rng_key)
            obs = input.observation
            obs = self.obs_normalizer.normalize(obs)
            if self.replica_noise is not None and self.replica_noise > 0:
                obs_flat, uf = tree.ravel_pytree(obs)
                obs_flat = obs_flat + self.replica_noise * foundry.random.normal(n_rng, obs_flat.shape)
                obs = uf(obs_flat)
            model_fn = lambda rng_key, noised_actions, t: model(
                self.vars, rng_key, obs, noised_actions, t - 1
            )
            action = self.schedule.sample(s_rng, model_fn, self.actions_structure) 
            action = self.action_normalizer.unnormalize(action)
            return PolicyOutput(action=action[:self.action_horizon], info=action)
        obs_horizon = tree.axis_size(self.observations_structure, 0)
        return ChunkingTransform(
            obs_horizon, self.action_horizon
        ).apply(chunk_policy)

@dataclass
class DPConfig:
    model: str = "unet"

    mlp : MLPConfig = MLPConfig()
    unet : UNetConfig = UNetConfig()

    epochs: int | None = None
    iterations : int | None = None
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    replica_noise: float | None = None

    diffusion_steps: int = 32
    action_horizon: int = 16

    @property
    def model_config(self) -> MLPConfig | UNetConfig:
        if self.model == "mlp":
            return self.mlp
        elif self.model == "unet":
            return self.unet
        else:
            raise ValueError(f"Unknown model type: {self.model}")
    
    def run(self, inputs: Inputs):
        _, data = inputs.data.load({"train", "test"})
        logger.info("Materializing all data...")
        train_data = data["train"].cache()
        test_data = data["test"].cache()

        action_horizon = min(self.action_horizon, inputs.data.action_length)

        schedule = DDPMSchedule.make_squaredcos_cap_v2(
            self.diffusion_steps,
            prediction_type="sample"
        )
        train_sample = train_data[0]
        observations_structure = tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), 
                                          train_sample.observations)
        actions_structure = tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), 
                                          train_sample.actions)
        logger.info(f"Observation: {observations_structure}")
        logger.info(f"Action: {actions_structure}")

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
        # donate the ema_state argument
        ema_update = F.jit(ema.update, donate_argnums=(1,))

        def loss_fn(vars, rng_key, sample: Sample):
            sample_norm = normalizer.normalize(sample)
            obs = sample_norm.observations
            s_rng, n_rng = foundry.random.split(rng_key)
            if self.replica_noise is not None and self.replica_noise > 0:
                obs_flat, uf = tree.ravel_pytree(obs)
                obs_flat = obs_flat + self.replica_noise * foundry.random.normal(n_rng, obs_flat.shape)
                obs = uf(obs_flat)
            actions = sample_norm.actions
            denoiser = lambda rng_key, noised_actions, t: model(vars, s_rng, obs, noised_actions, t)
            loss = schedule.loss(rng_key, denoiser, actions)
            return train.LossOutput(
                loss=loss, metrics={"loss": loss}
            )
        
        def make_checkpoint(vars) -> Checkpoint:
            return Checkpoint(
                data=inputs.data,
                observations_structure=observations_structure,
                actions_structure=actions_structure,
                action_horizon=action_horizon,
                model_config=self.model_config,
                schedule=schedule,
                replica_noise=self.replica_noise,
                obs_normalizer=normalizer.map(lambda x: x.observations),
                action_normalizer=normalizer.map(lambda x: x.actions),
                vars=vars
            )

        batched_loss_fn = train.batch_loss(loss_fn)

        train_stream = train_data.stream().batch(self.batch_size)
        test_stream = test_data.stream().batch(self.batch_size)

        validate_render = F.jit(
            lambda rng_key, vars: inputs.validate_render(
                val_rng, make_checkpoint(ema_state.ema).create_policy()
            )
        )
        with train.loop(
                data=train_stream,
                rng_key=next(inputs.rng),
                iterations=total_iterations,
                progress=False
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
                    lr = opt_schedule(step.iteration).item()
                    train.wandb.log(step.iteration, metrics, {"lr": lr},
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
                    if step.iteration % 2000 == 0:
                        logger.info("Evaluating policy...")
                        # jax.profiler.save_device_memory_profile(f"memory_{step.iteration}.prof")
                        rewards, video = validate_render(val_rng, ema_state.ema)
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
        return make_checkpoint(ema_state.ema)

class DiffusionUNet(UNet):
    activation: str = "relu"
    embed_dim: int = 256

    @nn.compact
    def __call__(self, obs, actions, 
                 timestep=None, train=False):
        activation = getattr(activations, self.activation)

        # works even if we have multiple timesteps
        timestep_flat = jax.flatten_util.ravel_pytree(timestep)[0]
        time_embed = jax.vmap(
            SinusoidalPosEmbed(self.embed_dim)
        )(timestep_flat).reshape(-1)
        time_embed = nn.Sequential([
            nn.Dense(self.embed_dim),
            activation,
            nn.Dense(self.embed_dim),
        ])(time_embed)

        obs_flat, _ = jax.flatten_util.ravel_pytree(obs)
        
        # FiLM embedding
        obs_embed = nn.Sequential([
            nn.Dense(self.embed_dim),
            activation,
            nn.Dense(self.embed_dim),
        ])(obs_flat)
        cond_embed = time_embed + obs_embed
        return super().__call__(
            actions, 
            cond_embed=cond_embed,
            train=train
        )

class DiffusionMLP(nn.Module):
    features: Sequence[int]
    activation: str = "gelu"
    time_embed_dim: int = 256

    @nn.compact
    def __call__(self, obs, actions,
                    # either timestep or time_embed must be passed
                    timestep=None, train=False):
        activation = getattr(activations, self.activation)
        # works even if we have multiple timesteps
        timestep_flat = jax.flatten_util.ravel_pytree(timestep)[0]
        time_embed = jax.vmap(
            SinusoidalPosEmbed(self.time_embed_dim)
        )(timestep_flat).reshape(-1)
        time_embed = nn.Sequential([
            nn.Dense(self.time_embed_dim),
            activation,
            nn.Dense(self.time_embed_dim),
        ])(time_embed)
        obs_flat, _ = jax.flatten_util.ravel_pytree(obs)
        actions_flat, actions_uf = jax.flatten_util.ravel_pytree(actions)

        # concatenated embedding
        actions = jnp.concatenate((actions_flat, obs_flat), axis=-1)
        embed = time_embed

        for feat in self.features:
            shift, scale = jnp.split(nn.Dense(2*feat)(embed), 2, -1)
            actions = activation(nn.Dense(feat)(actions))
            actions = actions * (1 + scale) + shift
        actions = nn.Dense(actions_flat.shape[-1])(actions)
        # x = jax.nn.tanh(x)
        actions = actions_uf(actions)
        return actions
