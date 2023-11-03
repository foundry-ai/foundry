from stanza.dataclasses import dataclass, replace
from stanza.runtime import activity
from stanza.util.logging import logger, LoggerHook

from stanza.train import batch_loss
from stanza.train.ema import EmaHook
from stanza.reporting import Video
from stanza.util import mat_jacobian
from stanza.nn.mlp import MLP

from stanza import partial, Partial
from stanza.util.random import PRNGSequence
from stanza.policies.transforms import ChunkedPolicy

from jax.random import PRNGKey
from diffusion_policy.util import load_data, knn_data, eval
from typing import Tuple

import stanza.envs as envs
import stanza.policies as policies
import stanza.util

import optax
import jax
import jax.numpy as jnp

@dataclass
class Config:
    env: str
    rng_seed: int = 42

    learning_rate: float = 1e-4
    epochs: int = 100
    batch_size: int = 128
    warmup_steps: int = 500

    num_trajectories: int = None
    obs_horizon: int = 1
    action_horizon: int = 1
    action_padding: int = 0

    jac_lambda: float = 0.
    zorder_lambda: float = 0.
    zorder_knn: int = 3

    lambda_param: str = None
    lambda_val: float = None

    net: str = "mlp"
    features: Tuple[int] = (128, 64, 32)

def loss(config, net, normalizer, state, params, rng, sample):
    logger.trace("Tracing loss function", only_tracing=True)
    obs_normalizer = normalizer.map(lambda x: x.observation)
    action_normalizer = normalizer.map(lambda x: x.action)

    norm_obs = obs_normalizer.normalize(sample.observation)
    norm_action = action_normalizer.normalize(sample.action)
    pred_action = net.apply(params, norm_obs)
    pred_action_flat = jax.flatten_util.ravel_pytree(pred_action)[0]
    action_flat = jax.flatten_util.ravel_pytree(norm_action)[0]
    action_loss = jnp.mean(jnp.square(pred_action_flat - action_flat))

    stats = {}
    loss = action_loss
    stats["action_loss"] = action_loss
    if config.jac_lambda > 0:
        def policy(x):
            norm_obs = obs_normalizer.normalize(x)
            norm_action = net.apply(params, norm_obs)
            return action_normalizer.unnormalize(norm_action)
        jac = mat_jacobian(policy)(sample.observation)
        J = sample.info.J
        jac_loss = jnp.mean(jnp.square(jac - J))
        stats["jac_loss"] = jac_loss
        loss = loss + config.jac_lambda * jac_loss
    if config.zorder_lambda > 0:
        def diff_loss(x):
            per_obs = obs_normalizer.normalize(x.observation)
            eps = stanza.util.l2_norm_squared(per_obs, norm_obs)

            per_action = action_normalizer.normalize(x.action)
            action_diff = jax.tree_map(lambda x, y: x - y, 
                                       per_action, norm_action)
            pred_per_action = net.apply(params, per_obs)
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
    exp = repo.create()
    env = envs.create(config.env)
    rng = PRNGSequence(config.rng_seed)
    data_db = repo.find(data_for=config.env).latest
    if data_db is None:
        logger.error("Unable to find data for {}", config.env)
        return
    logger.info(
        "Using environment [blue]{}[/blue] with config: {}",
        config.env, config
    )
    with jax.default_device(jax.devices("cpu")[0]):
        data, val_data, val_trajs, normalizer = load_data(
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

    if config.net == "mlp":
        net = MLP(name="net", features=config.features,
                  output_sample=sample.action)
    else:
        raise ValueError(f"Unknown network {config.net}")

    init_params = jax.jit(net.init)(next(rng), sample.observation)
    params_flat, _ = jax.flatten_util.ravel_pytree(init_params)
    logger.info("params: {}", params_flat.shape[0])

    # Make loss function, training schedule
    loss_fn = Partial(partial(loss, config, net), normalizer)
    batch_size = min(config.batch_size, data.length)
    steps_per_epoch = (data.length // batch_size)
    epochs = max(config.epochs, 20_000 // steps_per_epoch + 1)
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
    logger.info("Initialized, starting training...")
    results = trainer.train(data,
                loss_fn=batch_loss(loss_fn), 
                rng_key=next(rng),
                init_params=init_params)
    params = results.ema_params
    policy = make_bc_policy(Partial(net.apply, params), normalizer,
                            obs_chunk_length=config.obs_horizon,
                            action_chunk_length=config.action_horizon)
    test_policy, test_reward = eval(val_trajs, env, policy, next(rng))

    N_trajs = jax.tree_flatten(val_trajs)[0][0].shape[0]
    for i in range(N_trajs):
        logger.info(f"Rendering trajectory {i}")
        val_traj = jax.tree_map(lambda x: x[i], val_trajs)
        expert_video = jax.vmap(env.render)(val_traj.states)
        exp.log({"{}_expert".format(i): Video(expert_video, fps=10)})
        test_traj = jax.tree_map(lambda x: x[i], test_policy)
        policy_video = jax.vmap(env.render)(test_traj.states)
        exp.log({"{}_learned".format(i): Video(policy_video, fps=10)})

    logger.info("Reward: {}", test_reward)

    exp.add("test_reward", test_reward)

def make_bc_policy(net_fn, normalizer, obs_chunk_length, action_chunk_length):
    def policy(input):
        obs_norm = normalizer.map(lambda x: x.observation)
        action_norm = normalizer.map(lambda x: x.action)
        obs = obs_norm.normalize(input.observation)
        actions = net_fn(obs)
        actions = action_norm.unnormalize(actions)
        return policies.PolicyOutput(actions)
    return ChunkedPolicy(policy,
        input_chunk_size=obs_chunk_length,
        output_chunk_size=action_chunk_length)