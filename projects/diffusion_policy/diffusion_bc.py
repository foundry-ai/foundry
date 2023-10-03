import stanza.policies as policies
import stanza.envs as envs

from stanza.policies.transforms import ChunkedPolicy
from stanza.reporting import BucketLogHook
from stanza.runtime import activity
from stanza import partial, Partial

from stanza.train import Trainer, batch_loss
from stanza.train.ema import EmaHook
from stanza.train.validate import Validator

from stanza.dataclasses import dataclass, replace

from stanza.util.random import PRNGSequence
from stanza.util.loop import LoggerHook, every_kth_epoch, every_kth_iteration
from stanza.util.logging import logger
from stanza.util.rich import ConsoleDisplay, \
    LoopProgress, StatisticsTable, EpochProgress
from stanza.diffusion.ddpm import DDPMSchedule

import stanza.envs as envs
import stanza.util

from stanza.nn.unet1d import ConditionalUNet1D, ConditionalMLP
from typing import Tuple, Any

import jax.numpy as jnp
import jax.random

from jax.random import PRNGKey

from diffusion_policy.util import load_data, knn_data, eval

import optax
import time

@dataclass
class Config:
    env: str
    rng_seed: int = 42

    learning_rate: float = 1e-4
    epochs: int = 100
    batch_size: int = 128
    warmup_steps: int = 500

    obs_horizon: int = 2
    action_horizon: int = 8
    action_padding: int = 7

    num_trajectories: int = None
    smoothing_sigma: float = 0

    diffuse_gains: bool = False

    jac_lambda: float = 0.0
    zorder_lambda: float = 0.
    zorder_knn: int = 3

    net : str = "mlp"
    features: Tuple[int] = (32, 64, 128)
    step_embed_dim: int = 128

@dataclass
class DiffusionSample:
    action: Any

def make_diffusion_input(config, normalized_sample):
    actions = normalized_sample.action
    return DiffusionSample(actions), normalized_sample.observation

def loss(config, net, diffuser, normalizer,
            # these are passed in per training loop:
            state, params, rng, sample):
    K_normalizer = normalizer.map(lambda x: x.info.K)
    obs_normalizer = normalizer.map(lambda x: x.observation)
    action_normalizer = normalizer.map(lambda x: x.action)

    logger.trace("Tracing loss function", only_tracing=True)
    rng = PRNGSequence(rng)
    timestep = jax.random.randint(next(rng), (), 0, diffuser.num_steps)
    # We do training in the normalized space!
    normalized_sample = normalizer.normalize(
        replace(sample, info=replace(sample.info, knn=None))
    )

    input, obs = make_diffusion_input(config, normalized_sample)
    if config.smoothing_sigma > 0:
        obs_flat, obs_uf = jax.flatten_util.ravel_pytree(obs)
        obs_flat = obs_flat + config.smoothing_sigma*jax.random.normal(next(rng), obs_flat.shape)
        obs = obs_uf(obs_flat)
    noisy, _, desired_output = diffuser.add_noise(next(rng), input, timestep)

    pred = net.apply(params, noisy, timestep, obs)

    pred_flat, _ = jax.flatten_util.ravel_pytree(pred)
    desired_flat , _ = jax.flatten_util.ravel_pytree(desired_output)
    noise_loss = jnp.mean(jnp.square(pred_flat - desired_flat))

    loss = noise_loss
    stats = {}
    stats["noise_loss"] = noise_loss
    if config.jac_lambda > 0:
        def f(x):
            x = obs_normalizer.normalize(x)
            action = net.apply(params, noisy, timestep, x).action
            return action_normalizer.unnormalize(action)
        jac = stanza.util.mat_jacobian(f)(obs_normalizer.unnormalize(obs))
        jac_norm = K_normalizer.normalize(jac)
        K_norm = jnp.squeeze(normalized_sample.info.K, axis=0)
        jacobian_diff = jac_norm - K_norm
        jac_loss = optax.safe_norm(jacobian_diff, 1e-2)
        stats["jac_loss"] = jac_loss
        loss = loss + config.jac_lambda*jac_loss
    if config.zorder_lambda > 0:
        def diff_loss(x):
            per_obs = obs_normalizer.normalize(x.observation)
            eps = stanza.util.l2_norm_squared(per_obs, obs)
            action = desired_output.action
            per_action = action_normalizer.normalize(x.action)
            action_diff = jax.tree_map(lambda x, y: x - y, 
                                       per_action, action)
            pred_action = pred.action
            pred_per_action = net.apply(params, noisy, timestep, per_obs).action
            pred_diff = jax.tree_map(lambda x, y: x - y,
                                pred_per_action, pred_action)
            loss = stanza.util.l2_norm_squared(action_diff, pred_diff)/(eps + 1e-3)
            return loss
        zorder_loss = jax.vmap(diff_loss)(sample.info.knn)
        zorder_loss = jnp.mean(zorder_loss)
        stats["zorder_loss"] = zorder_loss
        loss = loss + config.zorder_lambda * zorder_loss
    stats["loss"] = loss
    return state, loss, stats

@activity(Config)
def train_policy(config, repo):
    rng = PRNGSequence(config.rng_seed)
    data_db = repo.find(data_for=config.env).latest
    if data_db is None:
        logger.error("Unable to find data for {}", config.env)
        return
    exp = repo.create()
    logger.info(f"Running diffusion policy trainer [blue]{exp.url}[/blue]")
    # load the per-env defaults into config
    logger.info("Using data [blue]{}[/blue] with config: {}",
            data_db.url, config)
    with jax.default_device(jax.devices("cpu")[0]):
        data, val_data, val_trajs, normalizer = load_data(
            data_db, num_trajectories=config.num_trajectories,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            action_padding=config.action_padding)
    # move to GPU
    data, val_data, val_trajs, normalizer = jax.device_put(
        (data, val_data, val_trajs, normalizer), device=jax.devices("gpu")[0])

    sample = normalizer.normalize(data.get(data.start))
    if config.zorder_lambda > 0:
        data = knn_data(data, config.zorder_knn)
        val_data = knn_data(val_data, config.zorder_knn)
    logger.info("Dataset size: {} chunks", data.length)

    batch_size = min(config.batch_size, data.length)
    steps_per_epoch = (data.length // batch_size)
    epochs = max(config.epochs, 20000 // steps_per_epoch + 1)
    train_steps = steps_per_epoch * epochs
    logger.info("Training for {} steps ({} epochs)", train_steps, epochs)
    warmup_steps = min(config.warmup_steps, train_steps/2)
    lr_schedule = optax.join_schedules(
        [optax.linear_schedule(1e-4/500, config.learning_rate, warmup_steps),
         optax.cosine_decay_schedule(1e-4, train_steps - warmup_steps)],
        [warmup_steps]
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=1e-5)
    trainer = Trainer(
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=epochs
    )
    # make the network
    if config.net == "unet":
        net = ConditionalUNet1D(name="net",
            down_dims=config.features,
            step_embed_dim=config.step_embed_dim)
    elif config.net == "mlp":
        net = ConditionalMLP(name="net",
                features=config.features,
                step_embed_dim=config.step_embed_dim)
    else:
        raise ValueError(f"Unknown network {config.net}")
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

    diffuser = DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=1., prediction_type="sample")
    loss_fn = Partial(partial(loss, config, net),
                    diffuser, normalizer)
    loss_fn = jax.jit(loss_fn)

    ema_hook = EmaHook(
        decay=0.75
    )
    logger.info("Initialized, starting training...")

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
    with display as rcb:
        hooks = [ema_hook, validator, rcb.train, print_hook]
        hooks.append(BucketLogHook(exp))
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
        Partial(net.apply, ema_params), diffuser, normalizer,
        obs_chunk_length=config.obs_horizon,
        action_chunk_length=config.obs_horizon + \
            config.action_horizon + config.action_padding - 1,
        action_horizon_offset=config.obs_horizon - 1,
        action_horizon_length=config.action_horizon
    )
    env = envs.create(config.env)
    test_policy, reward = eval(val_trajs, env, policy, PRNGKey(43))
    exp.add("test_expert", val_trajs)
    exp.add("test_policy", test_policy)
    exp.add("test_reward", reward)
    logger.info("Normalized test reward: {}", reward)

def make_diffusion_policy(net_fn, diffuser, normalizer,
                          obs_chunk_length,
                          action_chunk_length, action_horizon_offset, 
                          action_horizon_length, diffuse_gains=False, 
                          noise=0.):
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
        sample = diffuser.sample(
            sample_rng, model_fn,
            sample, 
            num_steps=diffuser.num_steps
        )
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