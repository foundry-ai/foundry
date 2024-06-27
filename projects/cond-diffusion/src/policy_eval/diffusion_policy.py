from common import net, TrainConfig
from policy_eval import Sample

from stanza.diffusion import DDPMSchedule
from stanza.runtime import ConfigProvider
from stanza.random import PRNGSequence
from stanza.policy import PolicyInput, PolicyOutput

from stanza.dataclasses import dataclass
from stanza.data.normalizer import Normalizer, LinearNormalizer
from stanza.diffusion import nonparametric
from stanza import train
import wandb
import optax
import flax.linen as nn
import flax.linen.activation as activations
from typing import Sequence
from common.net.embed import SinusoidalPosEmbed

import jax
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)


@dataclass
class DiffusionPolicyConfig:
    #model: str = "ResNet18"
    train: TrainConfig = TrainConfig()
    net_width: int = 4
    net_depth: int = 5
    embed_type: str = "concat"
    has_skip: bool = False
    timesteps: int = 100

    def parse(self, config: ConfigProvider) -> "DiffusionPolicyConfig":
        return config.get_dataclass(self, flatten={"train"})

    def train_policy(self, wandb_run, train_data, eval, rng):
        return train_net_diffusion_policy(self, wandb_run, train_data, eval)

def train_net_diffusion_policy(
        config : DiffusionPolicyConfig, wandb_run, train_data, eval):
    rng = PRNGSequence(config.seed)
    #Model = getattr(net, config.model.split("/")[1])
    model = DiffusionMLP(
        features=[config.net_width]*config.net_depth, 
        embed_type=config.embed_type, 
        has_skip=config.has_skip
    )
    sample = train_data[0]
    vars = jax.jit(model.init)(next(rng), sample.observations, sample.actions)
    #normalizer = LinearNormalizer.from_data(train_data)

    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.timesteps,
        clip_sample_range=1.5,
        prediction_type="sample"
    )

    def loss_fn(vars, rng_key, sample: Sample, iteration):
        obs = sample.observations
        actions = sample.actions
        model_fn = lambda rng_key, noised_actions, t: model.apply(
            vars, obs, noised_actions, t - 1
        )
        loss = schedule.loss(rng_key, model_fn, actions)
        return train.LossOutput(
            loss=loss,
            metrics={"loss": loss}
        )

    vars = model.init(jax.random.key(42), jnp.zeros_like(train_data.structure[0]))
    optimizer = optax.adamw(1e-4)
    opt_state = optimizer.init(vars["params"])

    with train.loop(train_data, 
                batch_size=16, 
                rng_key=jax.random.key(42),
                iterations=1000,
                progress=True
            ) as loop:
        for epoch in loop.epochs():
            for step in epoch.steps():
                # *note*: consumes opt_state, vars
                opt_state, vars, metrics = train.step(
                    loss_fn, optimizer, opt_state, vars, 
                    step.rng_key, step.batch,
                    # extra arguments for the loss function
                    iteration=step.iteration
                )
                if step.iteration % 100 == 0:
                    train.ipython.log(step.iteration, metrics)
                    train.wandb.log(step.iteration, metrics, run=wandb_run)
        train.ipython.log(step.iteration, metrics)
        train.wandb.log(step.iteration, metrics, run=wandb_run)

class DiffusionMLP(nn.Module):
    features: Sequence[int]
    embed_type: str 
    has_skip: bool
    activation: str = "relu"
    time_embed_dim: int = 32
    obs_embed_dim: int = 32

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
        if self.embed_type == "concat":
            actions = jnp.concatenate((actions_flat, obs_flat), axis=-1)
            embed = time_embed
        elif self.embed_type == "film":
            obs_embed = nn.Sequential([
                nn.Dense(self.obs_embed_dim),
                activation,
                nn.Dense(self.obs_embed_dim),
            ])(obs_flat)
            actions = actions_flat
            embed = time_embed + obs_embed
        else: 
            raise ValueError(f"Unknown embedding type: {self.embed_type}")
        for feat in self.features:
            shift, scale = jnp.split(nn.Dense(2*feat)(embed), 2, -1)
            actions = activation(nn.Dense(feat)(actions))
            actions = actions * (1 + scale) + shift
            if self.has_skip:
                actions = jnp.concatenate((actions, actions_flat, obs_flat), axis=-1)
        actions = nn.Dense(actions_flat.shape[-1])(actions)
        # x = jax.nn.tanh(x)
        actions = actions_uf(actions)
        return actions