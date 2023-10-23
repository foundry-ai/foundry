import stanza.policies as policies
import stanza.envs as envs

from stanza.policies.transforms import ChunkedPolicy, FeedbackPolicy
from stanza.reporting import Video
from stanza.runtime import activity
from stanza import partial, Partial

import stanza.train
from stanza.train import batch_loss
from stanza.train.ema import EmaHook

from stanza.dataclasses import dataclass, replace

from stanza.util.attrdict import attrs
from stanza.util.logging import logger
from stanza.util.random import PRNGSequence
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
    action_horizon: int = 4
    action_padding: int = 4

    num_trajectories: int = None
    smoothing_sigma: float = 0

    diffuse_states: bool = False
    diffuse_gains: bool = False

    jac_lambda: float = 0.0
    zorder_lambda: float = 0.0
    zorder_knn: int = 3

    lambda_param: str = None
    lambda_val: float = None

    net : str = "mlp"
    features: Tuple[int] = (256, 128, 128, 64)
    step_embed_dim: int = 64
    render: bool = False

@dataclass
class DiffusionSample:
    actions: Any
    ref_states: Any = None
    gains: Any = None

def loss(config, net, diffuser, normalizer,
            # these are passed in per training loop:
            state, params, rng, sample):
    obs_normalizer = normalizer.map(lambda x: x.observation)
    action_normalizer = normalizer.map(lambda x: x.action)
    state_normalizer = normalizer.map(lambda x: x.state)
    gains_normalizer = normalizer.map(lambda x: x.info.K) if config.diffuse_gains else None

    logger.trace("Tracing loss function", only_tracing=True)
    rng = PRNGSequence(rng)
    timestep = jax.random.randint(next(rng), (), 0, diffuser.num_steps)

    # We do training in the normalized space!
    # Remove the k nearest neighbors, if they exist
    obs = obs_normalizer.normalize(sample.observation)
    if config.smoothing_sigma > 0:
        obs_flat, obs_uf = jax.flatten_util.ravel_pytree(obs)
        obs_flat = obs_flat + config.smoothing_sigma*jax.random.normal(next(rng), obs_flat.shape)
        obs = obs_uf(obs_flat)
    input = DiffusionSample(
        actions=action_normalizer.normalize(sample.action),
    )
    if config.diffuse_states or config.diffuse_gains:
        input = replace(input, ref_states=state_normalizer.normalize(sample.state))
    if config.diffuse_gains:
        input = replace(input, gains=gains_normalizer.normalize(sample.info.K))

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
            action = net.apply(params, noisy, timestep, x).actions
            return action_normalizer.unnormalize(action)
        jac = stanza.util.mat_jacobian(f)(obs_normalizer.unnormalize(obs))
        K = jnp.squeeze(sample.info.J, axis=0)
        jacobian_diff = K - jac
        jac_loss = optax.safe_norm(jacobian_diff, 1e-2)
        stats["jac_loss"] = jac_loss
        loss = loss + config.jac_lambda*jac_loss
    if config.zorder_lambda > 0:
        def diff_loss(x):
            per_obs = obs_normalizer.normalize(x.observation)
            eps = stanza.util.l2_norm_squared(per_obs, obs)
            action = desired_output.actions
            per_action = action_normalizer.normalize(x.actions)
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
    if config.lambda_param is not None:
        if config.lambda_param == "jac":
            config = replace(config, jac_lambda=config.lambda_val)
        elif config.lambda_param == "zorder":
            config = replace(config, zorder_lambda=config.lambda_val)
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
        data, val_data, val_trajs, normalizer = load_data(next(rng),
            data_db, num_trajectories=config.num_trajectories,
            obs_horizon=config.obs_horizon,
            action_horizon=config.action_horizon,
            action_padding=config.action_padding)
        sample = normalizer.normalize(data.get(data.start))
        if config.zorder_lambda > 0:
            data = knn_data(data, config.zorder_knn)
            val_data = knn_data(val_data, config.zorder_knn)
    # move to GPU
    data, sample, val_data, val_trajs, normalizer = jax.device_put(
        (data, sample, val_data, val_trajs, normalizer), device=jax.devices("gpu")[0])
    logger.info("Dataset size: {} chunks", data.length)
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
    sample_input = DiffusionSample(actions=sample.action)
    if config.diffuse_states or config.diffuse_gains:
        sample_input = replace(sample_input, ref_states=sample.state)
    if config.diffuse_gains:
        sample_input = replace(sample_input, gains=sample.info.K)
    logger.info("Instantiating network...")
    jit_init = jax.jit(net.init)
    init_params = jit_init(next(rng), sample_input,
                           jnp.array(1), sample.observation)
    params_flat, _ = jax.flatten_util.ravel_pytree(init_params)
    logger.info("params: {}", params_flat.shape[0])

    logger.info("Making diffusion schedule")
    diffuser = DDPMSchedule.make_squaredcos_cap_v2(
        100, clip_sample_range=1., prediction_type="sample")
    loss_fn = Partial(partial(loss, config, net),
                    diffuser, normalizer)
    loss_fn = jax.jit(loss_fn)

    logger.info("Initialized, starting training...")
    batch_size = min(config.batch_size, data.length)
    steps_per_epoch = (data.length // batch_size)
    epochs = max(config.epochs, 20_000 // steps_per_epoch + 1)
    # epochs = config.epochs
    train_steps = steps_per_epoch * epochs
    logger.info("Training for {} steps ({} epochs)", train_steps, epochs)
    warmup_steps = min(config.warmup_steps, train_steps/2)
    lr_schedule = optax.join_schedules(
        [optax.linear_schedule(1e-4/500, config.learning_rate, warmup_steps),
         optax.cosine_decay_schedule(1e-4, train_steps - warmup_steps)],
        [warmup_steps]
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=1e-5)

    ema_hook = EmaHook(
        decay=0.75
    )
    trainer = stanza.train.express(
        optimizer=optimizer,
        batch_size=batch_size,
        max_epochs=epochs,
        # hook related things
        validate_dataset=val_data,
        validate_batch_size=config.batch_size,
        validate_rng=next(rng),
        bucket=exp,
        train_hooks=[ema_hook]
    )
    results = trainer.train(data,
                loss_fn=batch_loss(loss_fn), 
                rng_key=next(rng), init_params=init_params,
            )
    params = results.fn_params.reg_params
    ema_params = results.fn_params.ema_params

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
        action_horizon_length=config.action_horizon,
        diffuse_states=config.diffuse_states, diffuse_gains=config.diffuse_gains,
        noise=config.smoothing_sigma
    )
    env = envs.create(config.env)
    test_policy, reward = eval(val_trajs, env, policy, next(rng))
    exp.add("test_expert", val_trajs)
    exp.add("test_policy", test_policy)
    exp.add("test_reward", reward)
    logger.info("Normalized test reward: {}", reward)
    if config.smoothing_sigma > 0:
        deconv_policy = make_diffusion_policy(
            Partial(net.apply, ema_params), diffuser, normalizer,
            obs_chunk_length=config.obs_horizon,
            action_chunk_length=config.obs_horizon + \
                config.action_horizon + config.action_padding - 1,
            action_horizon_offset=config.obs_horizon - 1,
            action_horizon_length=config.action_horizon,
            diffuse_states=config.diffuse_states, diffuse_gains=config.diffuse_gains,
            noise=0. # No noise injection for deconv policy
        )
        _, deconv_reward = eval(val_trajs, env, deconv_policy, next(rng))
        logger.info("Normalized test deconv reward: {}", deconv_reward)
        exp.add("test_deconv_reward", deconv_reward)


    if config.render:
        N_trajs = jax.tree_flatten(val_trajs)[0][0].shape[0]
        def video_render(traj):
            render_imgs = jax.jit(jax.vmap(lambda x, y: env.render(x, state_trajectory=y)))
            if "sample" in traj.info:
                chunks = normalizer.map(lambda x: x.state).unnormalize(
                    traj.info.sample.ref_states
                )
            else:
                chunks = None
            return render_imgs(traj.states, chunks)
        for i in range(N_trajs):
            logger.info(f"Rendering trajectory {i}")
            val_traj = jax.tree_map(lambda x: x[i], val_trajs)
            expert_video = video_render(val_traj)
            exp.log({"expert/{}".format(i): Video(expert_video, fps=10)})
            test_traj = jax.tree_map(lambda x: x[i], test_policy)
            policy_video = video_render(test_traj)
            exp.log({"learned/{}".format(i): Video(policy_video, fps=10)})

def make_diffusion_policy(net_fn, diffuser, normalizer,
                          obs_chunk_length, action_chunk_length,
                          action_horizon_offset, action_horizon_length, 
                          diffuse_states=False, diffuse_gains=False, noise=0.):
    obs_norm = normalizer.map(lambda x: x.observation)
    state_norm = normalizer.map(lambda x: x.state)
    action_norm = normalizer.map(lambda x: x.action)
    gain_norm = normalizer.map(lambda x: x.info.J) \
        if hasattr(normalizer.instance.info, 'J') is not None and diffuse_gains \
        else None

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
        state_norm.instance
    ) if state_norm is not None else None

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
        sample = DiffusionSample(actions=action_sample_traj)
        if diffuse_states or diffuse_gains:
            sample = replace(sample, ref_states=states_sample_traj)
        if diffuse_gains:
            sample = replace(sample, gains=gain_sample_traj)
        sample = diffuser.sample(
            sample_rng, model_fn,
            sample, 
            num_steps=diffuser.num_steps
        )
        actions, states, gains = sample.actions, sample.ref_states, sample.gains
        start, end = action_horizon_offset, action_horizon_offset + action_horizon_length
        actions = jax.tree_util.tree_map(
            lambda x: x[start:end], actions
        )
        states, gains = jax.tree_util.tree_map(
            lambda x: x[start:end], (states, gains)
        )
        actions = action_norm.unnormalize(actions)
        states = state_norm.unnormalize(states) \
            if states is not None else None
        gains = gain_norm.unnormalize(gains) \
            if gains is not None else None
        if diffuse_gains:
            return policies.PolicyOutput((actions, states, gains), info=attrs(sample=sample))
        return policies.PolicyOutput(actions, info=attrs(sample=sample))
    policy = ChunkedPolicy(policy,
        input_chunk_size=obs_chunk_length,
        output_chunk_size=action_horizon_length)
    if diffuse_gains:
        policy = FeedbackPolicy(policy)
    return policy