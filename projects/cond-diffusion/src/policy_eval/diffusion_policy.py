
from policy_eval import Sample

from stanza.diffusion import DDPMSchedule
from stanza.runtime import ConfigProvider
from stanza.random import PRNGSequence
from stanza.policy import PolicyInput, PolicyOutput
from stanza.policy.transforms import ChunkingTransform
from stanza.env.mujoco.pusht import PushTObs

from stanza.dataclasses import dataclass
import dataclasses
from stanza.data import Data, PyTreeData
from stanza.data.normalizer import LinearNormalizer, StdNormalizer
from stanza.diffusion import nonparametric
from stanza import train
import stanza.train.ipython
import wandb
import optax
import flax.linen as nn
import flax.linen.activation as activations
from typing import Sequence
from projects.models.src.stanza.model.embed import SinusoidalPosEmbed
from . import diffusion_estimator

import jax
import jax.numpy as jnp
import logging
import pickle

logger = logging.getLogger(__name__)


@dataclass
class DiffusionPolicyConfig:
    #model: str = "ResNet18"
    seed: int = 42
    iterations: int = 100
    batch_size: int = 128
    net_width: int = 4096
    net_depth: int = 2
    embed_type: str = "film"
    has_skip: bool = True
    diffusion_steps: int = 50

    def parse(self, config: ConfigProvider) -> "DiffusionPolicyConfig":
        return config.get_dataclass(self, flatten={"train"})

    def train_policy(self, wandb_run, train_data, env, eval, rng):
        return train_net_diffusion_policy(self, wandb_run, train_data, env, eval)

def train_net_diffusion_policy(
        config : DiffusionPolicyConfig,  wandb_run, train_data, env, eval):
    
    
    # data_agent_pos = jax.vmap(
    #     #TODO: refactor pushtobs
    #     lambda x: env.observe(x, PushTObs()).agent_pos
    # )(train_data_tree.state)
    # actions = train_data_tree.actions - data_agent_pos[:, None, :]
    # train_data = PyTreeData(Sample(train_data_tree.state, train_data_tree.observations, actions))
    
    train_sample = train_data[0]
    normalizer = StdNormalizer.from_data(train_data)
    train_data_tree = train_data.as_pytree()
    # sample = jax.tree_map(lambda x: x[0], train_data_tree)
    # Get chunk lengths
    obs_length, action_length = (
        stanza.util.axis_size(train_data_tree.observations, 1),
        stanza.util.axis_size(train_data_tree.actions, 1)
    )

    rng = PRNGSequence(config.seed)
    #Model = getattr(net, config.model.split("/")[1])
    model = DiffusionMLP(
        features=[config.net_width]*config.net_depth, 
        embed_type=config.embed_type, 
        has_skip=config.has_skip
    )
    vars = jax.jit(model.init)(next(rng), train_sample.observations, train_sample.actions, 0)
    

    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y.size, vars, 0)
    logger.info(f"Total parameters: [blue]{total_params}[/blue]")

    schedule = DDPMSchedule.make_squaredcos_cap_v2(
        config.diffusion_steps,
        clip_sample_range=1.5,
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
        estimator = nonparametric.nw_cond_diffuser(
            obs, (train_data_tree.observations, train_data_tree.actions), schedule, nonparametric.log_gaussian_kernel, 0.01
        )
        t = jax.random.randint(t_rng, (), 0, schedule.num_steps) + 1
        noised_actions, _, _ = schedule.add_noise(noise_rng, actions, t)
        estimator_pred = estimator(None, noised_actions, t)
        model_pred_norm = model_fn(None, noised_actions, t)
        model_pred = normalizer.map(lambda x: x.actions).unnormalize(model_pred_norm)
        loss = jnp.mean((estimator_pred - model_pred)**2)

        # loss = schedule.loss(rng_key, model_fn, actions)
        
        return train.LossOutput(
            loss=loss,
            metrics={"loss": loss}
        )
    batched_loss_fn = train.batch_loss(loss_fn)

    opt_sched = optax.cosine_onecycle_schedule(config.iterations, 1e-4)
    optimizer = optax.adamw(opt_sched)
    opt_state = optimizer.init(vars["params"])

    train_data_batched = train_data.stream().batch(config.batch_size)
    with stanza.train.loop(train_data_batched, 
                rng_key=next(rng),
                iterations=config.iterations,
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
                if step.iteration % 100 == 0:
                    train.ipython.log(step.iteration, metrics)
                    train.wandb.log(step.iteration, metrics, run=wandb_run)
        train.ipython.log(step.iteration, metrics)
        train.wandb.log(step.iteration, metrics, run=wandb_run)
    
    def chunk_policy(input: PolicyInput) -> PolicyOutput:
        obs = input.observation
        obs = normalizer.map(lambda x: x.observations).normalize(obs)
        model_fn = lambda rng_key, noised_actions, t: model.apply(
            vars, obs, noised_actions, t - 1
        )
        action = schedule.sample(input.rng_key, model_fn, train_sample.actions) 
        action = normalizer.map(lambda x: x.actions).unnormalize(action)
        return PolicyOutput(action=action)
    
    policy = ChunkingTransform(
        obs_length, action_length
    ).apply(chunk_policy)

    # with open("model.pkl", 'wb') as file:
    #     pickle.dump(model, file)

    # with open("model_wts.pkl", 'wb') as file:
    #     pickle.dump(vars, file)

    return policy, chunk_policy

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