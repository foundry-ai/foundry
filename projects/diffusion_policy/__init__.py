from stanza.runtime import activity
from stanza.runtime.database import PyTree

from stanza.train import Trainer
from stanza.train.ema import EmaHook
from stanza.train.rich import RichReporter
from stanza.train.wandb import WandbReporter

from stanza.util.dataclasses import dataclass, field
from stanza.util.random import PRNGSequence
from stanza.util.logging import logger
from stanza.model.unet1d import ConditionalUnet1D
from stanza.model.diffusion import DDPMSchedule

from stanza.data.trajectory import chunk_trajectory
from stanza.data.normalizer import LinearNormalizer
from stanza.data import PyTreeData

from rich.progress import track

from functools import partial

import jax.numpy as jnp
import jax.random

import stanza.util
import haiku as hk

import optax
import time
import wandb

@dataclass
class Config:
    env: str = "pusht"
    rng_seed: int = 42
    epochs: int = 100
    batch_size: int = 256
    warmup_steps: int = 500
    smoothing_sigma: float = 0

def setup_problem(config):
    if config.env == "pusht":
        from stanza.envs.pusht import expert_data
        data = expert_data()
        def model(curr_sample, timestep, cond):
            sample_flat, sample_uf = stanza.util.vmap_ravel_pytree(curr_sample)
            cond_flat, _ = jax.flatten_util.ravel_pytree(cond)
            model = ConditionalUnet1D(name='net',
                down_dims=[64, 128, 256], diffusion_step_embed_dim=128)
            r = model(sample_flat, timestep, cond_flat)
            r = sample_uf(r)
            return r
        net = hk.transform(model)
        # Load the data into a PyTree
        logger.info("Calculating data normalizers")
        data_flat = PyTreeData.from_data(data.flatten(), chunk_size=4096)
        action_norm = LinearNormalizer.from_data(
            data_flat.map(lambda x: x.action)
        )
        obs_norm = LinearNormalizer.from_data(
            data_flat.map(lambda x: x.observation)
        )

        # chunk the data and flatten
        logger.info("Chunking trajectories")
        data = data.map(partial(chunk_trajectory,
            obs_chunk_size=2, action_chunk_size=16))
        # Load the data into a PyTree
        data = PyTreeData.from_data(data.flatten(), chunk_size=4096)
        logger.info("Data Loaded!")
        return data, net, obs_norm, action_norm

def loss(config, net, diffuser, obs_norm, action_norm,
            # these are passed in per training loop:
            params, rng, sample):
    t_sk, n_sk, s_sk = jax.random.split(rng, 3)
    timestep = jax.random.randint(t_sk, (), 0, diffuser.num_steps)
    # We do training in the normalized space!
    action = action_norm.normalize(sample.action)
    obs = obs_norm.normalize(sample.observation)

    if config.smoothing_sigma > 0:
        obs_flat, obs_uf = jax.flatten_util.ravel_pytree(obs)
        obs_flat = obs_flat + config.smoothing_sigma*jax.random.normal(s_sk, obs_flat.shape)
        obs = obs_uf(obs_flat)

    noisy_action, noise = diffuser.add_noise(n_sk, action, timestep)
    pred_noise = net.apply(params, rng, noisy_action, timestep, obs)

    pred_flat, _ = jax.flatten_util.ravel_pytree(pred_noise)
    noise_flat, _ = jax.flatten_util.ravel_pytree(noise)

    loss = jnp.mean(jnp.square(pred_flat - noise_flat))
    stats = {
        "loss": loss
    }
    return loss, stats

@activity(Config)
def train_policy(config, database):
    rng = PRNGSequence(config.rng_seed)
    exp = database.open("diffusion_policy")
    logger.info("Running diffusion policy trainer...")
    # policy_builder
    # takes params
    # and returns a full policy
    # that can be evaluated
    data, net, obs_norm, action_norm = setup_problem(config)
    logger.info("Dataset Size: {} chunks", data.length)
    train_steps = Trainer.total_iterations(data, config.batch_size, config.epochs)
    logger.info("Training for {} steps", train_steps)
    warmup_steps = config.warmup_steps

    lr_schedule = optax.join_schedules(
        [optax.constant_schedule(1),
         optax.cosine_decay_schedule(1, train_steps - warmup_steps)],
        [warmup_steps]
    )
    optimizer = optax.chain(
        optax.adamw(1e-4, weight_decay=1e-6),
        optax.scale_by_schedule(lr_schedule)
    )
    trainer = Trainer(
        optimizer=optimizer,
        batch_size=config.batch_size,
        epochs=config.epochs
    )
    # Initialize the network parameters
    sample = data.get(data.start)
    logger.info("Initializing network")
    t = time.time()
    init_params = net.init(next(rng), sample.action,
                           jnp.array(1), sample.observation)
    logger.info(f"Initialization took {time.time() - t}")
    t = time.time()
    init_params = net.init(next(rng), sample.action,
                           jnp.array(1), sample.observation)
    logger.info(f"Re-initialization took {time.time() - t}")

    params_flat, _ = jax.flatten_util.ravel_pytree(init_params)
    logger.info("params: {}", params_flat.shape[0])
    logger.info("Making diffusion schedule")
    diffuser = DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=1)
    loss_fn = partial(loss, config, net, diffuser, obs_norm, action_norm)

    ema_hook = EmaHook(
        decay=0.75
    )
    logger.info("Initialized, starting training...")

    wr = WandbReporter()
    rr = RichReporter(iter_interval=50, smoothing_epochs=1)
    wandb.init(project="diffusion_policy")

    with wr as wcb: 
        with rr as rcb:
            results = trainer.train(loss_fn, data, next(rng), init_params,
                        hooks=[ema_hook, rcb, wcb])
    params = results.fn_params
    # get the moving average params
    ema_params = results.hook_states[0]
    # save the final checkpoint
    # exp.add('final_checkpoint', PyTree(params))
    # exp.add('final_checkpoint_ema', PyTree(ema_params))