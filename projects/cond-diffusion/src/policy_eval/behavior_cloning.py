
from policy_eval import Sample
from typing import Callable

from stanza.runtime import ConfigProvider
from stanza.random import PRNGSequence
from stanza.policy import PolicyInput, PolicyOutput
from stanza.policy.transforms import ChunkingTransform

from stanza.dataclasses import dataclass
import dataclasses
from stanza.data import Data, PyTreeData
from stanza.data.normalizer import LinearNormalizer, StdNormalizer
from stanza import train
from stanza.env import Environment
import stanza.train.console
import wandb
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
class BCConfig:
    model: str = "mlp"

    seed: int = 42
    iterations: int = 10000
    batch_size: int = 64

    # MLP config
    net_width: int = 4096
    net_depth: int = 3
    has_skip: bool = True

    action_horizon: int = 8
    
    from_checkpoint: bool = False
    checkpoint_filename: str = "5nupde5h_final.pkl"

    def parse(self, config: ConfigProvider) -> "BCConfig":
        return config.get_dataclass(self, flatten={"train"})

    def train_policy(self, wandb_run, train_data, env, eval, rng):
        if self.from_checkpoint:
            return BC_from_checkpoint(self, wandb_run, train_data, env, eval)
        else:
            return train_net_BC(self, wandb_run, train_data, env, eval)

def BC_from_checkpoint( 
        config: BCConfig, wandb_run, train_data, env, eval):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    ckpts_dir = os.path.join(current_dir, "checkpoints")
    file_path = os.path.join(ckpts_dir, config.checkpoint_filename)
    with open(file_path, "rb") as file:
        ckpt = pickle.load(file)

    model = ckpt["model"]
    vars = ckpt["vars"]
    normalizer = ckpt["normalizer"]

    train_sample = train_data[0]
    action_sample = train_sample.actions

    def chunk_policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        obs = normalizer.map(lambda x: x.observations).normalize(obs)
        action = model.apply(vars, obs, action_sample)
        action = normalizer.map(lambda x: x.actions).unnormalize(action)
        action = action[:config.action_horizon]
        return PolicyOutput(action=action, info=action)
    
    obs_length = stanza.util.axis_size(train_data.as_pytree().observations, 1)
    policy = ChunkingTransform(
        obs_length, config.action_horizon
    ).apply(chunk_policy)
    return policy

def train_net_BC(
        config : BCConfig,  wandb_run, train_data, env, eval):
    
    train_sample = train_data[0]
    action_sample = train_sample.actions
    normalizer = StdNormalizer.from_data(train_data)
    train_data_tree = train_data.as_pytree()

    # Get chunk lengths
    obs_length, action_length = (
        stanza.util.axis_size(train_data_tree.observations, 1),
        stanza.util.axis_size(train_data_tree.actions, 1)
    )

    rng = PRNGSequence(config.seed)
    
    if config.model == "mlp":
        model = DiffusionMLP(
            features=[config.net_width]*config.net_depth, 
            has_skip=config.has_skip
        )
    else:
        raise ValueError(f"Unknown model type: {config.model}")
    
    vars = jax.jit(model.init)(next(rng), train_sample.observations, action_sample)
    
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

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
                if step.iteration % 100 == 0:
                    train.ipython.log(step.iteration, metrics)
                    train.wandb.log(step.iteration, metrics, run=wandb_run)
                if step.iteration > 0 and step.iteration % 20000 == 0:
                    ckpt = {
                        "config": config,
                        "model": model,
                        "vars": vars,
                        "opt_state": opt_state,
                        "normalizer": normalizer
                    }
                    file_path = os.path.join(ckpts_dir, f"{wandb_run.id}_{step.iteration}.pkl")
                    with open(file_path, 'wb') as file:
                        pickle.dump(ckpt, file)
                    wandb_run.log_model(path=file_path, name=f"{wandb_run.id}_{step.iteration}")
        train.ipython.log(step.iteration, metrics)
        train.wandb.log(step.iteration, metrics, run=wandb_run)

    # save model
    ckpt = {
        "config": config,
        "model": model,
        "vars": vars,
        "opt_state": opt_state,
        "normalizer": normalizer
    }
    file_path = os.path.join(ckpts_dir, f"{wandb_run.id}_final.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(ckpt, file)
    
    def chunk_policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        obs = normalizer.map(lambda x: x.observations).normalize(obs)
        action = model.apply(vars, obs, action_sample)
        action = normalizer.map(lambda x: x.actions).unnormalize(action)
        action = action[:config.action_horizon]
        return PolicyOutput(action=action, info=action)
    
    policy = ChunkingTransform(
        obs_length, config.action_horizon
    ).apply(chunk_policy)

    return policy
    

class DiffusionMLP(nn.Module):
    features: Sequence[int]
    has_skip: bool
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