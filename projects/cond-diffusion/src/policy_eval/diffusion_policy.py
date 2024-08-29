
from policy_eval import Sample

from foundry.diffusion import DDPMSchedule
from foundry.runtime import ConfigProvider
from foundry.core.random import PRNGSequence
from foundry.policy import PolicyInput, PolicyOutput
from foundry.policy.transforms import ChunkingTransform

from foundry.core.dataclasses import dataclass
from foundry.data.normalizer import LinearNormalizer, StdNormalizer
from foundry import train
import foundry.train.console
import optax
import flax.linen as nn
import flax.linen.activation as activations
from typing import Sequence
from projects.models.src.foundry.model.embed import SinusoidalPosEmbed
from projects.models.src.foundry.model.unet import UNet

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

    def parse(self, config: ConfigProvider) -> "MLPConfig":
        return config.get_dataclass(self)

@dataclass
class UNetConfig:
    base_channels: int = 128
    num_downsample: int = 4

    def parse(self, config : ConfigProvider) -> "UNetConfig":
        return config.get_dataclass(self)

@dataclass
class DiffusionPolicyConfig:
    model: MLPConfig | UNetConfig | None = None

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4

    diffusion_steps: int = 50
    action_horizon: int = 8
    
    save_dir: str = "/nobackup/users/chryu"
    from_checkpoint: bool = False
    checkpoint_filename: str = None

    def parse(self, config: ConfigProvider) -> "DiffusionPolicyConfig":
        model = config.get("model", str, default=None)
        if model == "mlp":
            self = replace(self, model=MLPConfig())
        elif model == "unet":
            self = replace(self, model=UNetConfig())
        else: 
            raise ValueError(f"Unknown model: {model}")
        return config.get_dataclass(self, flatten={"train"})

    def train_policy(self, wandb_run, train_data, env, eval, rng):
        if self.from_checkpoint:
            return diffusion_policy_from_checkpoint(self, wandb_run, train_data, env, eval)
        else:
            return train_net_diffusion_policy(self, wandb_run, train_data, env, eval, rng)

def diffusion_policy_from_checkpoint( 
        config : DiffusionPolicyConfig, wandb_run, train_data, env, eval):
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
    train_sample = train_data[0]

    def chunk_policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        obs = normalizer.map(lambda x: x.observations).normalize(obs)
        model_fn = lambda rng_key, noised_actions, t: model.apply(
            ema_vars, obs, noised_actions, t - 1
        )
        action = schedule.sample(input.rng_key, model_fn, train_sample.actions) 
        action = normalizer.map(lambda x: x.actions).unnormalize(action)
        action = action[:config.action_horizon]
        return PolicyOutput(action=action, info=action)
    
    obs_length = foundry.util.axis_size(train_data.as_pytree().observations, 1)
    policy = ChunkingTransform(
        obs_length, config.action_horizon
    ).apply(chunk_policy)
    return policy

def train_net_diffusion_policy(
        config : DiffusionPolicyConfig,  wandb_run, train_data, env, eval, rng):
    
    train_sample = train_data[0]
    normalizer = StdNormalizer.from_data(train_data)
    train_data_tree = train_data.as_pytree()
    # sample = jax.tree_map(lambda x: x[0], train_data_tree)
    # Get chunk lengths
    obs_length, action_length = (
        foundry.util.axis_size(train_data_tree.observations, 1),
        foundry.util.axis_size(train_data_tree.actions, 1)
    )

    rng = PRNGSequence(rng)
    #Model = getattr(net, config.model.split("/")[1])
    # model = DiffusionMLP(
    #     features=[config.net_width]*config.net_depth, 
    #     embed_type=config.embed_type, 
    #     has_skip=config.has_skip
    # )
    
    if isinstance(config.model, UNetConfig):
        model = DiffusionUNet(
            dims=1, 
            base_channels=config.model.base_channels, 
            channel_mult=tuple([2**i for i in range(config.model.num_downsample)]),
        ) # 1D temporal UNet
    elif isinstance(config.model, MLPConfig):
        model = DiffusionMLP(
            features=[config.net_width]*config.net_depth
        )
    else:
        raise ValueError(f"Unknown model type: {config.model}")
    
    vars = jax.jit(model.init)(next(rng), train_sample.observations, train_sample.actions, 0)
    

    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        prediction_type="sample"
    )

    def loss_fn(vars, rng_key, sample: Sample, iteration):
        noise_rng, t_rng = jax.random.split(rng_key)
        sample_norm = normalizer.normalize(sample)
        obs = sample_norm.observations
        actions = sample_norm.actions
        # obs = sample.observations
        # actions = sample.actions
        model_fn = lambda rng_key, noised_actions, t: model.apply(
            vars, obs, noised_actions, t - 1
        )

        # fit to estimator
        # estimator = nonparametric.nw_cond_diffuser(
        #     obs, (train_data_tree.observations, train_data_tree.actions), schedule, nonparametric.log_gaussian_kernel, 0.01
        # )
        # t = jax.random.randint(t_rng, (), 0, schedule.num_steps) + 1
        # noised_actions_norm, _, _ = schedule.add_noise(noise_rng, actions, t)
        # noised_actions = normalizer.map(lambda x: x.actions).unnormalize(noised_actions_norm)
        # estimator_pred = estimator(None, noised_actions, t)
        # model_pred_norm = model_fn(None, noised_actions_norm, t)
        # model_pred = normalizer.map(lambda x: x.actions).unnormalize(model_pred_norm)
        # loss = jnp.mean((estimator_pred - model_pred)**2)
        loss = schedule.loss(rng_key, model_fn, actions)
        
        return train.LossOutput(
            loss=loss,
            metrics={
                "loss": loss
            }
        )
    batched_loss_fn = train.batch_loss(loss_fn)

    train_data_batched = train_data.stream().batch(config.batch_size)
    epoch_iterations = len(train_data) // config.batch_size
    total_iterations = config.epochs * epoch_iterations

    opt_sched = optax.cosine_onecycle_schedule(total_iterations, config.learning_rate)
    optimizer = optax.adamw(opt_sched)
    opt_state = optimizer.init(vars["params"])

    # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=1000, max_to_keep=2, create=True)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     '/tmp/flax_ckpt/orbax/managed', orbax_checkpointer, options)

    # Create a directory to save checkpoints
    ckpts_dir = os.path.join(config.save_dir, "checkpoints")
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)

    

    # Keep track of the exponential moving average of the model parameters
    ema = optax.ema(0.9)
    ema_state = ema.init(vars)

    with foundry.train.loop(train_data_batched, 
                rng_key=next(rng),
                iterations=total_iterations,
                progress=True
            ) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                # print(step.batch.observations)
                # print(step.batch.actions)
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
    def chunk_policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        obs = normalizer.map(lambda x: x.observations).normalize(obs)
        model_fn = lambda rng_key, noised_actions, t: model.apply(
            ema_vars, obs, noised_actions, t - 1
        )
        action = schedule.sample(input.rng_key, model_fn, train_sample.actions) 
        action = normalizer.map(lambda x: x.actions).unnormalize(action)
        return PolicyOutput(action=action[:config.action_horizon], info=action)
    
    policy = ChunkingTransform(
        obs_length, config.action_horizon
    ).apply(chunk_policy)

    return policy
    

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
        return super().__call__(actions, cond_embed=cond_embed, train=train)

class DiffusionMLP(nn.Module):
    features: Sequence[int]
    activation: str = "relu"
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