import stanza.policies as policies
import stanza.envs as envs

from stanza.policies.transforms import ChunkedPolicy
from stanza.runtime import activity
from stanza import partial, Partial

from stanza.train import Trainer, batch_loss
from stanza.train.ema import EmaHook
from stanza.train.validate import Validator

from stanza.policies.mpc import MPC

from stanza.dataclasses import dataclass

from stanza.util.random import PRNGSequence
from stanza.util.loop import LoggerHook, every_kth_epoch, every_kth_iteration
from stanza.util.logging import logger
from stanza.util.rich import ConsoleDisplay, \
    LoopProgress, StatisticsTable, EpochProgress
from stanza.diffusion.ddpm import DDPMSchedule

import stanza.envs as envs

from stanza.nets.unet1d import ConditionalUNet1D, ConditionalMLP

from typing import Tuple

import jax.numpy as jnp
import jax.random

from jax.random import PRNGKey

from diffusion_policy.util import load_data, eval

import optax
import time

@dataclass
class Config:
    data: str

    wandb: str = "dpfrommer-projects/diffusion_policy"

    rng_seed: int = 42

    learning_rate: float = 1e-4
    epochs: int = 100
    batch_size: int = 128
    warmup_steps: int = 500

    obs_horizon: int = 2
    action_horizon: int = 8
    diffusion_horizon: int = 16

    num_trajectories: int = None
    smoothing_sigma: float = 0

    diffuse_gains: bool = False
    regularize_gains_lambda: float = 0.0

    net : str = "unet"
    features: Tuple[int] = (32, 64, 128)
    step_embed_dim: int = 128

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
    noisy, _, desired_output = diffuser.add_noise(next(rng), input, timestep)

    sqrt_alphas = jnp.sqrt(diffuser.alphas_cumprod[timestep])
    sqrt_one_minus_alphas = jnp.sqrt(1 - diffuser.alphas_cumprod[timestep])

    pred_noise = net.apply(params, noisy, timestep, obs)
    # pred_noise = jax.tree_map(
    #     lambda noisy, pred_out: (noisy - pred_out*sqrt_alphas)/sqrt_one_minus_alphas,
    #     noisy, pred_noise)

    pred_flat, _ = jax.flatten_util.ravel_pytree(pred_noise)
    desired_flat , _ = jax.flatten_util.ravel_pytree(desired_output)
    noise_loss = jnp.mean(jnp.square(pred_flat - desired_flat))

    loss = noise_loss

    if config.regularize_gains_lambda > 0 or True:
        K_normalizer = normalizer.map(lambda x: x.info.K)
        obs_normalizer = normalizer.map(lambda x: x.observation)
        action_normalizer = normalizer.map(lambda x: x.action)
        def jacobian(x):
            # take the jacobian of the unnormalized output
            x = obs_normalizer.unnormalize(x)
            flat_obs, unflatten = jax.flatten_util.ravel_pytree(x)
            def f(x_flat):
                x = unflatten(x_flat)
                x = obs_normalizer.normalize(x)
                pred = net.apply(params, noisy, timestep, x)
                pred = action_normalizer.unnormalize(pred[0]), None, None
                return jax.flatten_util.ravel_pytree(pred)[0]
            return jax.jacobian(f)(flat_obs)
        jac = jacobian(obs)
        #f = jnp.sqrt(diffuser.alphas_cumprod[timestep]/(1 - diffuser.alphas_cumprod[timestep]))
        jac_norm = K_normalizer.normalize(jac)
        # K = jnp.squeeze(sample.info.K, axis=0)
        K_norm = jnp.squeeze(normalized_sample.info.K, axis=0)
        jacobian_diff = jac_norm - K_norm
        jac_loss = optax.safe_norm(jacobian_diff, 1e-2)
    else:
        jac_loss = 0.

    loss = loss + config.regularize_gains_lambda*jac_loss

    stats = {
        "loss": loss,
        "noise_loss": noise_loss,
        "jac_loss": jac_loss,
        # "K_0": K[0,0],
        # "K_1": K[0,1],
        # "jac_0": jac[0,0],
        # "jac_1": jac[0,1]
    }
    return state, loss, stats

@activity(Config)
def train_policy(config, database):
    rng = PRNGSequence(config.rng_seed)
    exp = database.open("diffusion_policy").create()
    logger.info(f"Running diffusion policy trainer [blue]{exp.name}[/blue]")

    data_db = database.open(f"expert_data/{config.data}")
    env_name = data_db.get("env_name")
    env = envs.create(env_name)
    # load the per-env defaults into config
    logger.info("Using environment [blue]{}[/blue] with config: {}", env_name, config)
    with jax.default_device(jax.devices("cpu")[0]):
        data, val_data, val_trajs, normalizer = load_data(data_db, config)
    # move to GPU
    data, val_data, val_trajs, normalizer = jax.device_put(
        (data, val_data, val_trajs, normalizer), device=jax.devices("gpu")[0])
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

    diffuser = DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=1., prediction_type="sample")
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
        hooks = [ema_hook, validator, rcb.train, print_hook]
        if db is not None:
            hooks.append(db.statistic_logging_hook())
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