from stanza.policy.imitation_learning import ImitationLearning
from stanza.policy.ilqr import ILQR
from stanza.dataset.env import EnvDataset

from jax.random import PRNGKey

import stanza.envs as envs
import haiku as hk
import jax.flatten_util
import functools

def net(x, sample_action):
    x_flat, _ = jax.flatten_util.ravel_pytree(x)

env = envs.create("pendulum")

ilqr = iLQR(env.reset(PRNGKey(0)), env.sample_action(PRNGKey(0)),
            env.cost, env.step, horizon_length=20, receed=True)

dataset = EnvDataset(PRNGKey(42), env, 100, policy=iLQR)
# Train by hand using Trainer()

# A convenience wrapper for the above
# il = ImitationLearning(env, net, policy)