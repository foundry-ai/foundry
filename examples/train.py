from stanza.policies.imitation_learning import ImitationLearning
from stanza.dataset.env import EnvDataset
from stanza.dataset import PyTreeDataset

from jax.random import PRNGKey
from stanza.policies.mpc import MPC
from stanza.util.logging import logger
from functools import partial

import stanza.envs as envs
import jax.flatten_util
import jax.numpy as jnp

env = envs.create("pendulum")

policy = MPC(
    action_sample=env.sample_action(PRNGKey(0)),
    cost_fn=env.cost,
    model_fn=env.step,
    horizon_length=10
)

dataset = EnvDataset(PRNGKey(42), env, 25, policy)
# Just get the (state, action) pairs
logger.info('Trajectory: {}', dataset.get(dataset.start))
dataset = EnvDataset.as_state_actions(dataset)
x_sample, u_sample = dataset.get(dataset.start)
logger.info('State, action: {}, {}', x_sample, u_sample)

from stanza.train import Trainer
import haiku as hk

def net_fn(x, sample_action):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)
    u_s_flat, u_unflat = jax.flatten_util.ravel_pytree(sample_action)
    mlp = hk.nets.MLP([10,10, u_s_flat.shape[0]])
    u = mlp(x_flat)
    return u_unflat(u)

net = hk.transform(partial(net_fn, sample_action=u_sample))
params = net.init(PRNGKey(42), x_sample)

def loss_fn(params, rng_key, sample):
    x, u = sample
    pred_u = net.apply(params, None, x)
    loss = jnp.sum(jnp.square(u - pred_u))
    return loss, {'loss': loss}

# Train by hand using Trainer()
logger.info("Generating dataset...")
dataset = PyTreeDataset.from_dataset(dataset[:10])
logger.info("Training model...")
trainer = Trainer(loss_fn, max_iterations=1000)
trainer.train(dataset, PRNGKey(0), params)
logger.info("Training model with smaller batch size...")
trainer = Trainer(loss_fn, batch_size=5, max_iterations=1000)
trainer.train(dataset, PRNGKey(0), params)