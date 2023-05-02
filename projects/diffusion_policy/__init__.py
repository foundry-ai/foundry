from stanza.runtime import activity
from stanza.runtime.database import PyTree

from stanza.train import Trainer
from stanza.train.ema import EmaHook

from stanza.util.dataclasses import dataclass, field
from stanza.util.random import PRNGSequence
from stanza.util.logging import logger
from stanza.model.unet1d import ConditionalUnet1D
from stanza.model.diffusion import DDPMSchedule

from functools import partial

import jax.numpy as jnp
import jax.random

import stanza.envs as envs
import haiku as hk

import optax

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
            model = ConditionalUnet1D(name='net')
            r = model(curr_sample, timestep, cond)
            return r
        net = hk.transform(model)
        def policy_builder(params):
            pass
        return data, net, policy_builder

def loss(config, net, diffuser, params, rng, sample):
    t_sk, n_sk, s_sk = jax.random.split(rng, 3)
    timestep = jax.random.randint(t_sk, (), 0, diffuser.num_steps)
    action = sample.action
    obs = sample.observation

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

    # policy_builder
    # takes params
    # and returns a full policy
    # that can be evaluated
    data, net, policy_builder = setup_problem(config)

    # flatten the trajectory data
    data = data.flatten()
    logger.info("Data length: {}", data.length)

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

    # Initialize the network
    # parameters
    sample = data.get(data.start)
    init_params = net.init(next(rng), sample.action,
                           jnp.array(1), sample.observation)
                        
    diffuser = DDPMSchedule.make_squaredcos_cap_v2(100)
    loss_fn = partial(loss, config, net, diffuser)

    ema_hook = EmaHook()
    results = trainer.train(loss_fn, data, init_params,
                hooks=[ema_hook])
    params = results.fn_params
    # get the moving average params
    ema_params = results.hook_states[0]

    # save the final checkpoint
    exp.add('final_checkpoint', PyTree(params))
    exp.add('final_checkpoint_ema', PyTree(ema_params))