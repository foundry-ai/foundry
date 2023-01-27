import jax

import jax.numpy as jnp
from jax.random import PRNGKey

from ode.dataset import PyTreeDataset
from ode.policy.imitation_learning import ImitationLearning

import optax
import haiku as hk

def expert_policy(x):
    return 2*x

def net_fn(x):
    mlp = hk.nets.MLP([10, 10, x.shape[0]], activation=jax.nn.relu)
    return mlp(x)

def test_il():
    xs = 10*jax.random.uniform(PRNGKey(42), (20, 10, 3))
    us = jax.vmap(jax.vmap(expert_policy))(xs)
    dataset = PyTreeDataset((xs, us))

    net = hk.transform_with_state(net_fn)
    il = ImitationLearning(net, expert_policy, 
            batch_size=200, epochs=10000,
            optimizer=optax.adam(0.001))
    policy = il.run(PRNGKey(69), dataset)
    res = jax.vmap(jax.vmap(policy))(xs) - 2*xs
    print(res)
    assert jnp.max(jnp.abs(res)) < 0.05