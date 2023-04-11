import importlib
import inspect

import stanza
import jax
import jax.tree_util
import jax.numpy as jnp

from stanza.dataset import MappedDataset
from stanza.util.random import PRNGDataset
from stanza.util.dataclasses import dataclass
from functools import partial

from typing import Any

# Generic environment
class Environment:
    def sample_action(self, rng_key):
        raise NotImplementedError("Must impelement sample_action()")

    def reset(self, key):
        raise NotImplementedError("Must impelement reset()")

    def step(self, x, u):
        raise NotImplementedError("Must impelement step()")

    # If u is None, evaluate the terminal cost
    def cost(self, x, u=None):
        raise NotImplementedError("Must impelement cost()")
    
    def visualize(self, xs, us):
        raise NotImplementedError("Must impelement visualize()")

__ENV_BUILDERS = {}

def create(env_name, *args, **kwargs):
    env_path = env_name.split("/")
    # register buildres if empty
    builder = __ENV_BUILDERS[env_path[0]]()
    return builder(env_name, *args, **kwargs)

# Register them lazily so we don't
# import dependencies we don't actually use
# i.e the appropriate submodule will be imported
# for the first time during create()
def register_lazy(name, module_name):
    frm = inspect.stack()[1]
    pkg = inspect.getmodule(frm[0]).__name__
    def make_env_constructor():
        mod = importlib.import_module(module_name, package=pkg)
        return mod.builder
    __ENV_BUILDERS[name] = make_env_constructor

register_lazy('pusht', '.pusht')
register_lazy('pendulum', '.pendulum')
register_lazy('linear', '.linear')
register_lazy('quadrotor', '.quadrotor')

# Takes a cost function and maps it over a trajectory
# where there is one more x than u
def trajectory_cost(cost_fn, xs, us):
    final_x = jax.tree_util.tree_map(lambda x: x[-1], xs)
    seq_xs = jax.tree_util.tree_map(lambda x: x[:-1], xs)
    seq_cost = jax.vmap(cost_fn)(seq_xs, us)
    final_cost = cost_fn(final_x)
    return jnp.sum(seq_cost) + final_cost

# Given a model fn and x, u samples will
# construct a version of the dynamics that operates
# on flattened state representations
def flatten_model(model_fn, x_sample, u_sample):
    _, x_unflatten = jax.flatten_util.ravel_pytree(x_sample)
    _, u_unflatten = jax.flatten_util.ravel_pytree(u_sample)

    def flattened_model(x, u):
        x = x_unflatten(x)
        u = u_unflatten(u)
        x = model_fn(x, u)
        x_vec = jax.flatten_util.ravel_pytree(x)[0]
        return x_vec
    return flattened_model

def flatten_cost(cost_fn, x_sample, u_sample):
    _, x_unflatten = jax.flatten_util.ravel_pytree(x_sample)
    _, u_unflatten = jax.flatten_util.ravel_pytree(u_sample)
    def flattened_cost(x, u=None, t=None):
        x = x_unflatten(x)
        if u is not None:
            u = u_unflatten(u)
        return cost_fn(x, u, t)
    return flattened_cost
