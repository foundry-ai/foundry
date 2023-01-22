import importlib
import inspect

import jax
import jax.numpy as jnp

from jinx.random import PRNGDataset
from jinx.dataset import MappedDataset
from jinx.util import tree_append

from functools import partial

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


# Helper function to do rollouts with

@partial(jax.jit, static_argnums=(0,2, 3, 5, 6))
def rollout_policy(model_fn, x0, length, policy,
                    # The initial policy state. If "None" is supplied
                    # and policy has an 'init_state' function, that
                    # policy.init_state(x0) will be used instead
                    policy_state=None,
                    # Whether to return the last policy state
                    # or jacobians of the policy
                    ret_policy_state=False, jacobians=False):
    if hasattr(policy, 'init_state') and policy_state is None:
        policy_state = policy.init_state(x0)

    # This allows us to compute the jacobians
    # and output at the same time
    def policy_fun(env_state, policy_state=None):
        if policy_state is None:
            u = policy(env_state)
            return u, (u, None)
        else:
            u, new_policy_state = policy(env_state, policy_state)
            return u, (u, new_policy_state)

    def scan_fn(comb_state, _):
        env_state, policy_state = comb_state
        if jacobians:
            jac, (u, new_policy_state) = jax.jacrev(policy_fun, has_aux=True, argnums=(0,))(env_state, policy_state)
        else:
            _, (u, new_policy_state) = policy_fun(env_state, policy_state)
        new_env_state = model_fn(env_state, u)
        outputs = (env_state, u, jac) if jacobians else (env_state, u)
        return (new_env_state, new_policy_state), outputs

    # Do the first step manually to populate the policy state
    state = (x0, policy_state)

    # outputs is (xs, us, jacs) or (xs, us)
    (state_f, p_f), outputs = jax.lax.scan(scan_fn, state,
                                    None, length=length-1)
    states = tree_append(outputs[0], state_f)

    out = (states,) + outputs[1:]
    if ret_policy_state:
        out = (p_f,) + out
    return out

# Global registry
def rollout_input(model_fn, state_0, us):
    def scan_fn(state, u):
        new_state = model_fn(state, u)
        return new_state, state
    final_state, states = jax.lax.scan(scan_fn, state_0, us)
    states = tree_append(states, final_state)
    return states

def rollout_input_gains(model_fn, state_0, ref_xs, ref_gains, us):
    def scan_fn(state, i):
        ref_x, ref_gain, u = i
        new_state = model_fn(state, u + ref_gain @ (state - ref_x))
        return new_state, state
    final_state, states = jax.lax.scan(scan_fn, state_0, (ref_xs[:-1], ref_gains, us))
    states = tree_append(states, final_state)
    return states

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

__ENV_BUILDERS = {}

def create(type, *args, **kwargs):
    # register buildres if empty
    builder = __ENV_BUILDERS[type]()
    return builder(*args, **kwargs)

# Register them lazily so we don't
# import dependencies we don't actually use
# i.e the appropriate submodule will be imported
# for the first time during create()
def register_lazy(name, module_name):
    frm = inspect.stack()[1]
    pkg = inspect.getmodule(frm[0]).__name__
    def make_env_constructor():
        mod = importlib.import_module(module_name, package=pkg)
        builder = mod.builder()
        return builder
    __ENV_BUILDERS[name] = make_env_constructor

register_lazy('brax', '.brax')
register_lazy('gym', '.gym')
register_lazy('pendulum', '.pendulum')
register_lazy('linear', '.linear')