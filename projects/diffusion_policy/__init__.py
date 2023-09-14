import stanza.policies as policies
import stanza.envs as envs

from stanza.policies.transforms import ChunkedPolicy
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

from jax.random import PRNGKey

import optax
import time

@dataclass
class Config:
    data: str = None

    wandb: str = "dpfrommer-projects/diffusion_policy"

    rng_seed: int = 42

    epochs: int = 20
    batch_size: int = 512
    warmup_steps: int = 500

    obs_horizon: int = None
    action_horizon: int = None
    diffusion_horizon: int = None

    num_trajectories: int = None
    smoothing_sigma: float = 0

    diffuse_gains: bool = False
    regularize_gains_lambda: float = 0.0

    down_dims: Tuple[int] = None
    step_embed_dim: int = None

ENV_DEFAULTS = {
    "brax/ant": {
        "obs_horizon": 1,
        "action_horizon": 1,
        "diffusion_horizon": 1,
        "down_dims": (32, 64, 128),
        "step_embed_dim": 128
    },
    "quadrotor": {
        "obs_horizon": 2,
        "action_horizon": 8,
        "diffusion_horizon": 16,
        "down_dims": (32, 64, 128),
        "step_embed_dim": 128
    },
    "pendulum": {
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
    val_data = PyTreeData.from_data(data[-20:], chunk_size=64)
    data = data[:-20]
    if config.num_trajectories is not None:
        data = data[:config.num_trajectories]
    data = PyTreeData.from_data(data, chunk_size=64)

    data_flat = PyTreeData.from_data(data.flatten(), chunk_size=4096)
    normalizer = LinearNormalizer.from_data(data_flat)

    val_data_pt = val_data.data.data
    val_trajs = Rollout(
        val_data_pt.state,
        val_data_pt.action,
        val_data_pt.observation,
        info=val_data_pt.info
    )
    def chunk(traj):
        # Throw away the state, we use only
        # the observations and actions
        traj = traj.map(lambda x: replace(x, state=None))
        traj = chunk_data(traj,
            chunk_size=config.diffusion_horizon, 
            start_padding=config.obs_horizon - 1,
            end_padding=config.action_horizon - 1)
        return traj
    data = data.map(chunk)
    val_data = val_data.map(chunk)

    # Load the data into a PyTree
    data = PyTreeData.from_data(data.flatten(), chunk_size=4096)
    val_data = PyTreeData.from_data(val_data.flatten(), chunk_size=4096)
    logger.info("Data size: {}",
        sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.size*x.itemsize, data.data))))
    logger.info("Data Loaded! Computing normalizer")
    logger.info("Normalizer computed")
    return data, val_data, val_trajs, normalizer

def make_diffusion_input(config, normalized_sample):
    # the state/action chunks
    actions = normalized_sample.action
    # the conditioning chunks (truncated)
    states = normalized_sample.observation
    obs = jax.tree_map(lambda x: x[:config.obs_horizon], states)
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
    pred_noise = net.apply(params, noisy, timestep, obs)
    pred_flat, _ = jax.flatten_util.ravel_pytree(pred_noise)
    noise_flat, _ = jax.flatten_util.ravel_pytree(noise)
    noise_loss = jnp.mean(jnp.square(pred_flat - noise_flat))
    loss = noise_loss
    if config.regularize_gains_lambda > 0:
        def jacobian(x):
            flat_obs, unflatten = jax.flatten_util.ravel_pytree(x)
            def f(x_flat):
                x = unflatten(x_flat)
                pred = net.apply(params, noisy, timestep, x)
                return jax.flatten_util.ravel_pytree(pred)[0]
            return jax.jacfwd(f)(flat_obs)
        jac = jacobian(obs)
        K_norm = normalizer.map(lambda x: x.info.K)
        jac_norm = K_norm.normalize(jac)
        jacobian_diff = jac_norm - normalized_sample.info.K
        jac_loss = jnp.mean(jnp.square(jacobian_diff))
        loss = loss + config.regularize_gains_lambda*jac_loss
    else:
        jac_loss = 0.

    stats = {
        "loss": loss,
        "noise_loss": noise_loss,
        "jac_loss": jac_loss,
    }
    return state, loss, stats

import stanza.envs as envs
from stanza.policies import Rollout

def eval(val_trajs, env, policy, rng_key):
    x0s = jax.tree_map(lambda x: x[:,0], val_trajs.states)
    N = jax.tree_util.tree_leaves(val_trajs.states)[0].shape[0]
    length = jax.tree_util.tree_leaves(val_trajs.states)[0].shape[1]
    def roll(x0, rng):
        model_rng, policy_rng = jax.random.split(rng)
        return policies.rollout(env.step, x0,
            observe=env.observe, policy=policy, length=length, 
            model_rng_key=model_rng, policy_rng_key=policy_rng,
            last_input=True
        )
    roll = jax.vmap(roll)
    rngs = jax.random.split(rng_key, N)
    rolls = roll(x0s, rngs)
    from stanza.util import extract_shifted

    state_early, state_late = jax.vmap(extract_shifted)(rolls.states)
    actions = jax.tree_map(lambda x: x[:,:-1], rolls.actions)
    vreward = jax.vmap(jax.vmap(env.reward))
    policy_r = vreward(state_early, actions, state_late)
    policy_r = jnp.sum(policy_r, axis=1)

    state_early, state_late = jax.vmap(extract_shifted)(val_trajs.states)
    actions = jax.tree_map(lambda x: x[:,:-1], val_trajs.actions)
    expert_r = vreward(state_early, actions, state_late)
    expert_r = jnp.sum(expert_r, axis=1)
    reward_ratio = policy_r / expert_r
    return rolls, jnp.mean(reward_ratio)

@activity(Config)
def train_policy(config, database):
    rng = PRNGSequence(config.rng_seed)
    exp = database.open("diffusion_policy").create()
    logger.info(f"Running diffusion policy trainer [blue]{exp.name}[/blue]")

    data_db = database.open(f"expert_data/{config.data}")
    env_name = data_db.get("env_name")
    env = envs.create(env_name)
    # load the per-env defaults into config
    config = replace_defaults(config, **ENV_DEFAULTS.get(env_name, {}))
    logger.info("Using {} with config: {}", env_name, config)
    with jax.default_device(jax.devices("cpu")[0]):
        data, val_data, val_trajs, normalizer = load_data(data_db, config)
    # move to GPU
    data, val_data, val_trajs, normalizer = jax.device_put(
        (data, val_data, val_trajs, normalizer), device=jax.devices("gpu")[0])
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

    if config.wandb is not None:
        from stanza.reporting.wandb import WandbDatabase
        from stanza.reporting.jax import JaxDBScope
        wexp = WandbDatabase(config.wandb).create()
        db = JaxDBScope(wexp)
    else:
        db = None
    print_hook = LoggerHook(every_kth_iteration(500))

    display = ConsoleDisplay()
    display.add("train", StatisticsTable(), interval=100)
    display.add("train", LoopProgress(), interval=100)
    display.add("train", EpochProgress(), interval=100)

    validator = Validator(
        condition=every_kth_epoch(1),
        rng_key=next(rng),
        dataset=val_data,
        batch_size=config.batch_size
    )

    with display as rcb, db as db:
        stat_logger = db.statistic_logging_hook(
            log_cond=every_kth_iteration(1), buffer=100)
        hooks = [ema_hook, validator, rcb.train, 
                    stat_logger, print_hook]
        results = trainer.train(data,
                    loss_fn=batch_loss(loss_fn), 
                    rng_key=next(rng),
                    init_params=init_params,
                    train_hooks=hooks
                )
    params = results.fn_params
    # get the moving average params
    ema_params = results.hook_states[0]
    # save the final checkpoint
    exp.add('config', config)
    exp.add('normalizer', normalizer)
    exp.add('final_checkpoint', params)
    exp.add('final_checkpoint_ema', ema_params)

    policy = make_diffusion_policy(
        Partial(net.apply, params), diffuser, normalizer,
        obs_chunk_length=config.obs_horizon,
        action_chunk_length=config.diffusion_horizon,
        action_horizon_offset=config.obs_horizon - 1,
        action_horizon_length=config.action_horizon
    )
    test_policy, reward = eval(val_trajs, env, policy, PRNGKey(43))
    exp.add("test_expert", val_trajs)
    exp.add("test_policy", test_policy)
    exp.add("test_reward", reward)
    logger.info("Normalized test reward: {}", reward)
    if config.wandb is not None:
        wexp.run.summary["test_reward"] = reward

def make_diffusion_policy(net_fn, diffuser, normalizer,
                          obs_chunk_length,
                          action_chunk_length, action_horizon_offset, 
                          action_horizon_length, diffuse_gains=False, 
                          gains_model=None, noise=0.):
    obs_norm = normalizer.map(lambda x: x.observation)
    action_norm = normalizer.map(lambda x: x.action)
    gain_norm = normalizer.map(lambda x: x.info.K) \
        if hasattr(normalizer.instance.info, 'K') is not None and diffuse_gains else None
    action_sample_traj = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), action_chunk_length, axis=0),
        action_norm.instance
    )
    gain_sample_traj = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), action_chunk_length, axis=0),
        gain_norm.instance
    ) if gain_norm is not None else None
    states_sample_traj = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.expand_dims(x, 0), action_chunk_length, axis=0),
        obs_norm.instance
    ) if obs_norm is not None else None

    def policy(input):
        smooth_rng, sample_rng = jax.random.split(input.rng_key)
        norm_obs = obs_norm.normalize(input.observation)

        norm_flat, norm_uf = jax.flatten_util.ravel_pytree(norm_obs)
        if noise > 0:
            norm_flat = norm_flat + noise*jax.random.normal(smooth_rng, norm_flat.shape)
        noised_norm_obs = norm_uf(norm_flat)
        model_fn = lambda _, sample, timestep: net_fn(
            sample, timestep, cond=noised_norm_obs
        )
        if diffuse_gains:
            sample = action_sample_traj, states_sample_traj, gain_sample_traj
        else:
            sample = action_sample_traj, None, None
        sample = diffuser.sample(sample_rng, model_fn,
                sample, 
                num_steps=diffuser.num_steps)
        actions, states, gains = sample
        actions = action_norm.unnormalize(actions)
        start = action_horizon_offset
        end = action_horizon_offset + action_horizon_length
        actions = jax.tree_util.tree_map(
            lambda x: x[start:end], actions
        )
        if diffuse_gains:
            states = jax.tree_util.tree_map(
                lambda x: x[start:end], states
            )
            gains = jax.tree_util.tree_map(
                lambda x: x[start:end], gains
            )
            gains = gain_norm.unnormalize(gains)
            states = obs_norm.unnormalize(states)
            actions = actions, states, gains
        return policies.PolicyOutput(actions)
    return ChunkedPolicy(policy,
        input_chunk_size=obs_chunk_length,
        output_chunk_size=action_horizon_length)