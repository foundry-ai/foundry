from stanza.runtime import activity
from stanza import partial, Partial

from stanza.train import Trainer, batch_loss
from stanza.train.ema import EmaHook
from stanza.train.validate import Validator

from stanza.policies.mpc import MPC

from stanza.dataclasses import dataclass, replace, replace_defaults, field

from stanza.util.random import PRNGSequence
from stanza.util.loop import LoggerHook, every_kth_epoch, every_kth_iteration
from stanza.util.logging import logger
from stanza.util.rich import ConsoleDisplay, \
    LoopProgress, StatisticsTable, EpochProgress
from stanza.diffusion.ddpm import DDPMSchedule

from stanza.data.chunk import chunk_data
from stanza.data.normalizer import LinearNormalizer
from stanza.data import PyTreeData
from stanza.nets.unet1d import ConditionalUNet1D

from typing import Tuple

import jax.numpy as jnp
import jax.random

import optax
import time

@dataclass
class Config:
    data: str = None

    rng_seed: int = 42

    epochs: int = 500
    batch_size: int = 256
    warmup_steps: int = 500

    obs_horizon: int = None
    action_horizon: int = None
    diffusion_horizon: int = None

    num_trajectories: int = None
    smoothing_sigma: float = 0

    diffuse_gains: bool = False
    regularize_gains: bool = False

    down_dims: Tuple[int] = None
    step_embed_dim: int = None

ENV_DEFAULTS = {
    "brax/ant": {
        "obs_horizon": 2,
        "action_horizon": 8,
        "diffusion_horizon": 16,
        "down_dims": (32, 64, 128),
        "step_embed_dim": 128
    }
}

def make_diffuser(config):
    return DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=1.)

def load_data(data_db, config):
    logger.info("Reading data...")
    data = data_db.get("trajectories")
    # chunk the data and flatten
    logger.info("Chunking trajectories")
    val_data = data[-10:]
    data = data[:-10]
    def chunk(traj):
        # Throw away the state, we use only
        # the observations and actions
        traj = traj.map(lambda x: replace(x, state=None))
        traj = chunk_data(traj,
            config.diffusion_horizon, 
            config.obs_horizon - 1,
            config.action_horizon - 1)
        return traj
    data = data.map(chunk)
    val_data = val_data.map(chunk)

    # Load the data into a PyTree
    data = PyTreeData.from_data(data.flatten(), chunk_size=4096)
    val_data = PyTreeData.from_data(val_data.flatten(), chunk_size=4096)
    logger.info("Data size: {}",
        sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.size*x.itemsize, data.data))))
    logger.info("Data Loaded! Computing normalizer")
    normalizer = LinearNormalizer.from_data(data)
    logger.info("Normalizer computed")
    return data, val_data, normalizer

def make_diffusion_input(config, normalized_sample):
    # the state/action chunks
    actions = normalized_sample.action
    # the conditioning chunks (truncated)
    obs = jax.tree_map(lambda x: x[:config.obs_horizon], states)
    states = normalized_sample.observation
    if config.diffuse_gains:
        input = actions, states, normalized_sample.info.K
    else:
        input = actions, None, None
    return input, obs

def loss(config, net, diffuser, normalizer,
            # these are passed in per training loop:
            state, params, rng, sample):
    logger.trace("Tracing loss function", only_tracing=True)
    rng = PRNGSequence(rng)
    timestep = jax.random.randint(next(rng), (), 0, diffuser.num_steps)
    # We do training in the normalized space!
    normalized_sample = normalizer.normalize(sample)
    input, obs = make_diffusion_input(config, normalized_sample)
    if config.smoothing_sigma > 0:
        obs_flat, obs_uf = jax.flatten_util.ravel_pytree(obs)
        obs_flat = obs_flat + config.smoothing_sigma*jax.random.normal(next(rng), obs_flat.shape)
        obs = obs_uf(obs_flat)
    noisy, noise = diffuser.add_noise(next(rng), input, timestep)
    pred_noise = net.apply(params, next(rng), noisy, timestep, obs)
    pred_flat, _ = jax.flatten_util.ravel_pytree(pred_noise)
    noise_flat, _ = jax.flatten_util.ravel_pytree(noise)
    loss = jnp.mean(jnp.square(pred_flat - noise_flat))

    stats = {
        "loss": loss
    }
    return state, loss, stats

@activity(Config)
def train_policy(config, database):
    rng = PRNGSequence(config.rng_seed)
    exp = database.open("diffusion_policy").create()
    logger.info(f"Running diffusion policy trainer [blue]{exp.name}[/blue]")

    data_db = database.open(f"expert_data/{config.data}")
    env_name = data_db.get("env_name")
    # load the per-env defaults into config
    config = replace_defaults(config, **ENV_DEFAULTS.get(env_name, {}))
    logger.info("Using {} with config: {}", env_name, config)
    with jax.default_device(jax.devices("cpu")[0]):
        data, val_data, normalizer = load_data(data_db, config)
    # move to GPU
    data, val_data, normalizer = jax.device_put((data, val_data, normalizer), jax.devices("gpu")[0])
    logger.info("Dataset size: {} chunks", data.length)
    train_steps = (data.length // config.batch_size) * config.epochs
    logger.info("Training for {} steps", train_steps)
    warmup_steps = config.warmup_steps

    lr_schedule = optax.join_schedules(
        [optax.linear_schedule(1e-4/500, 1e-4, warmup_steps),
         optax.cosine_decay_schedule(1e-4, 
                    train_steps - warmup_steps)],
        [warmup_steps]
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=1e-5)
    trainer = Trainer(
        optimizer=optimizer,
        batch_size=config.batch_size,
        max_epochs=config.epochs
    )
    # make the network
    net = ConditionalUNet1D(name="net",
        down_dims=config.down_dims,
        step_embed_dim=config.step_embed_dim)
    sample = normalizer.normalize(data.get(data.start))
    sample_input, sample_obs = make_diffusion_input(config, sample)
    logger.info("Instantiating network...")
    t = time.time()
    jit_init = jax.jit(net.init)
    init_params = jit_init(next(rng), sample_input,
                           jnp.array(1), sample_obs)
    logger.info(f"Initialization took {time.time() - t}")
    params_flat, _ = jax.flatten_util.ravel_pytree(init_params)
    logger.info("params: {}", params_flat.shape[0])
    logger.info("Making diffusion schedule")

    diffuser = make_diffuser(config)
    loss_fn = Partial(partial(loss, config, net),
                    diffuser, normalizer)
    loss_fn = jax.jit(loss_fn)

    ema_hook = EmaHook(
        decay=0.75
    )
    logger.info("Initialized, starting training...")

    from stanza.reporting.jax import JaxDBScope
    db = JaxDBScope(exp)
    print_hook = LoggerHook(every_kth_iteration(500))

    display = ConsoleDisplay()
    display.add("train", StatisticsTable(), interval=100)
    display.add("train", LoopProgress(), interval=100)
    display.add("train", EpochProgress(), interval=100)

    validator = Validator(
        condition=every_kth_epoch(1),
        rng_key=next(rng),
        dataset=val_data,
        batch_size=config.batch_size)

    with display as rcb, db as db:
        stat_logger = db.statistic_logging_hook(
            log_cond=every_kth_iteration(1), buffer=100)
        hooks = [ema_hook, validator, rcb.train, 
                    stat_logger, print_hook]
        results = trainer.train(
                    batch_loss(loss_fn), 
                    data, next(rng), init_params,
                    hooks=hooks
                )
    params = results.fn_params
    # get the moving average params
    ema_params = results.hook_states[0]
    # save the final checkpoint
    exp.add('config', config)
    exp.add('normalizer', normalizer)
    exp.add('final_checkpoint', params)
    exp.add('final_checkpoint_ema', ema_params)