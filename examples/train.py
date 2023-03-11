from stanza.policies.imitation_learning import ImitationLearning
from stanza.dataset.env import EnvDataset

from jax.random import PRNGKey

import stanza.envs as envs
import haiku as hk
import jax.flatten_util
import functools

def net(x, sample_action):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)

envs = envs.create("pendulum")

#dataset = EnvDataset(PRNGKey(42), env, 100, policy=iLQR)
# Train by hand using Trainer()

# A convenience wrapper for the above
# il = ImitationLearning(env, net, policy)