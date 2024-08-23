from .datasets import Sample

from stanza.diffusion import DDPMSchedule
from stanza.runtime import ConfigProvider
from stanza.random import PRNGSequence

from stanza.dataclasses import dataclass, replace
from stanza.data import PyTreeData
from stanza.data.normalizer import LinearNormalizer, StdNormalizer
from stanza import train
import stanza.train.console
import optax
import flax.linen as nn
import flax.linen.activation as activations
from typing import Sequence
from projects.models.src.stanza.model.embed import SinusoidalPosEmbed
from projects.models.src.stanza.model.unet import UNet

import jax
import jax.numpy as jnp
import logging
import pickle
import os

logger = logging.getLogger(__name__)

@dataclass
class MLPConfig:
    net_width: int = 4096
    net_depth: int = 3

    def parse(self, config: ConfigProvider) -> "MLPConfig":
        return config.get_dataclass(self)

@dataclass
class UNetConfig:
    base_channels: int = 128
    num_downsample: int = 4

    def parse(self, config : ConfigProvider) -> "UNetConfig":
        return config.get_dataclass(self)

@dataclass
class DiffusionLearnedConfig:
    model: MLPConfig | None = None

    iterations: int = 5000
    batch_size: int = 64

    diffusion_steps: int = 50
    
    from_checkpoint: bool = False
    checkpoint_filename: str = None

    def parse(self, config: ConfigProvider) -> "DiffusionLearnedConfig":
        model = config.get("model", str, default="mlp")
        if model == "mlp":
            self = replace(self, model=MLPConfig())
        elif model == "unet":
            self = replace(self, model=UNetConfig())
        else: raise RuntimeError(f"{model}")
        return config.get_dataclass(self)

    def train_denoiser(self, wandb_run, train_data, rng):
        if self.from_checkpoint:
            return diffusion_policy_from_checkpoint(self, wandb_run, train_data, rng)
        else:
            return train_net_diffusion_policy(self, wandb_run, train_data, rng)

def diffusion_policy_from_checkpoint( 
        config : DiffusionLearnedConfig, wandb_run, train_data):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ckpts_dir = os.path.join(current_dir, "checkpoints")
    file_path = os.path.join(ckpts_dir, config.checkpoint_filename)
    with open(file_path, "rb") as file:
        ckpt = pickle.load(file)

    model = ckpt["model"]
    ema_vars = ckpt["ema_state"].ema
    normalizer = ckpt["normalizer"]

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        prediction_type="sample"
    )
    train_sample = jax.tree_map(lambda x: x[0], train_data)

    def denoiser(cond, rng_key) -> Sample:
        norm_cond = normalizer.map(lambda x: x.cond).normalize(cond)
        model_fn = lambda rng_key, noised_value, t: model.apply(
            ema_vars, norm_cond, noised_value, t - 1
        )
        norm_value = schedule.sample(rng_key, model_fn, train_sample.value) 
        value = normalizer.map(lambda x: x.value).unnormalize(norm_value)
        return Sample(cond, value)
    
    return denoiser

def train_net_diffusion_policy(
        config : DiffusionLearnedConfig,  wandb_run, train_data, rng):
    
    train_sample = jax.tree_map(lambda x: x[0], train_data)
    train_data = PyTreeData(train_data)
    normalizer = StdNormalizer.from_data(train_data)

    rng = PRNGSequence(rng)
    
    if isinstance(config.model, UNetConfig):
        model = DiffusionUNet(
            dims=1, 
            base_channels=config.model.base_channels, 
            channel_mult=tuple([2**i for i in range(config.model.num_downsample)]),
        ) # 1D temporal UNet
    elif isinstance(config.model, MLPConfig):
        model = DiffusionMLP(
            features=[config.model.net_width]*config.model.net_depth
        )
    else:
        raise ValueError(f"Unknown model type: {config.model}")
    
    vars = jax.jit(model.init)(next(rng), train_sample.cond, train_sample.value, 0)
    
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        prediction_type="sample"
    )

    def loss_fn(vars, rng_key, sample: Sample, iteration):
        sample_norm = normalizer.normalize(sample)
        cond = sample_norm.cond
        value = sample_norm.value
        model_fn = lambda rng_key, noised_value, t: model.apply(
            vars, cond, noised_value, t - 1
        )
        loss = schedule.loss(rng_key, model_fn, value)
        
        return train.LossOutput(
            loss=loss,
            metrics={
                "loss": loss
            }
        )
    batched_loss_fn = train.batch_loss(loss_fn)

    opt_sched = optax.cosine_onecycle_schedule(config.iterations, 1e-4)
    optimizer = optax.adamw(opt_sched)
    opt_state = optimizer.init(vars["params"])

    # Create a directory to save checkpoints
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ckpts_dir = os.path.join(current_dir, "checkpoints")
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)

    train_data_batched = train_data.stream().batch(config.batch_size)

    # Keep track of the exponential moving average of the model parameters
    ema = optax.ema(0.9)
    ema_state = ema.init(vars)

    with stanza.train.loop(train_data_batched, 
                rng_key=next(rng),
                iterations=config.iterations,
                progress=True
            ) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                # *note*: consumes opt_state, vars
                opt_state, vars, metrics = train.step(
                    batched_loss_fn, optimizer, opt_state, vars, 
                    step.rng_key, step.batch,
                    # extra arguments for the loss function
                    iteration=step.iteration
                )
                _, ema_state = ema.update(vars, ema_state)
                if step.iteration % 100 == 0:
                    train.console.log(step.iteration, metrics)
                    train.wandb.log(step.iteration, metrics, run=wandb_run)
                if step.iteration > 0 and step.iteration % 20000 == 0:
                    ckpt = {
                        "config": config,
                        "model": model,
                        "vars": vars,
                        "opt_state": opt_state,
                        "ema_state": ema_state,
                        "normalizer": normalizer
                    }
                    file_path = os.path.join(ckpts_dir, f"{wandb_run.id}_{step.iteration}.pkl")
                    with open(file_path, 'wb') as file:
                        pickle.dump(ckpt, file)
                    wandb_run.log_model(path=file_path, name=f"{wandb_run.id}_{step.iteration}")
                    # save_args = orbax_utils.save_args_from_target(ckpt)
                    # checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})
        train.console.log(step.iteration, metrics)
        train.wandb.log(step.iteration, metrics, run=wandb_run)

    # save model
    ckpt = {
        "config": config,
        "model": model,
        "vars": vars,
        "opt_state": opt_state,
        "ema_state": ema_state,
        "normalizer": normalizer
    }
    file_path = os.path.join(ckpts_dir, f"{wandb_run.id}_final.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(ckpt, file)
    
    # Rollout policy with EMA of network parameters
    ema_vars = ema_state.ema
    def denoiser(cond, rng_key) -> Sample:
        norm_cond = normalizer.map(lambda x: x.cond).normalize(cond)
        model_fn = lambda rng_key, noised_value, t: model.apply(
            ema_vars, norm_cond, noised_value, t - 1
        )
        norm_value = schedule.sample(rng_key, model_fn, train_sample.value) 
        value = normalizer.map(lambda x: x.value).unnormalize(norm_value)
        return Sample(cond, value)
    
    return denoiser
    

class DiffusionUNet(UNet):
    activation: str = "relu"
    embed_dim: int = 256

    @nn.compact
    def __call__(self, cond, value, 
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

        cond_flat, _ = jax.flatten_util.ravel_pytree(cond)
        
        # FiLM embedding
        # cond_embed = nn.Sequential([
        #     nn.Dense(self.embed_dim),
        #     activation,
        #     nn.Dense(self.embed_dim),
        # ])(cond_flat)
        # cond_embed = time_embed + cond_embed

        # concatenated embedding
        value_flat, value_uf = jax.flatten_util.ravel_pytree(value)
        value = jnp.concatenate((value_flat, cond_flat), axis=-1)

        value = super().__call__(value[...,None], cond_embed=time_embed, train=train)[...,0]
        value = nn.Dense(value_flat.shape[-1])(value)
        value = value_uf(value)
        return value

class DiffusionMLP(nn.Module):
    features: Sequence[int]
    activation: str = "relu"
    time_embed_dim: int = 256

    @nn.compact
    def __call__(self, cond, value,
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

        # concatenated embedding
        cond_flat, _ = jax.flatten_util.ravel_pytree(cond)
        value_flat, value_uf = jax.flatten_util.ravel_pytree(value)
        value = jnp.concatenate((value_flat, cond_flat), axis=-1)

        embed = time_embed
        for feat in self.features:
            shift, scale = jnp.split(nn.Dense(2*feat)(embed), 2, -1)
            value = activation(nn.Dense(feat)(value))
            value = value * (1 + scale) + shift
        value = nn.Dense(value_flat.shape[-1])(value)
        # x = jax.nn.tanh(x)
        value = value_uf(value)
        return value