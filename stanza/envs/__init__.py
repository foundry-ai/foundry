import importlib
import inspect

import jax
import jax.tree_util
import jax.numpy as jnp

from typing import Any

# Generic environment. Note that all
# environments are also adapters
class Environment:
    def sample_state(self, rng_key):
        raise NotImplementedError("Must impelement sample_state()")

    def sample_action(self, rng_key):
        raise NotImplementedError("Must impelement sample_action()")

    def reset(self, key):
        raise NotImplementedError("Must impelement reset()")

    # rng_key may be None.
    # if it is None and the environment
    # is inherently stochastic, throw an error!
    def step(self, state, action, rng_key):
        raise NotImplementedError("Must impelement step()")
    
    def reward(self, states, actions):
        pass

__ENV_BUILDERS = {}

def create(env_type, **kwargs):
    env_path = env_type.split("/")
    # register buildres if empty
    builder = __ENV_BUILDERS[env_path[0]]()
    return builder(env_type, **kwargs)

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
register_lazy('gym', '.gym')
register_lazy('robosuite', '.robosuite')

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
