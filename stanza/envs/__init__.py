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

def create(env_type, *args, **kwargs):
    env_path = env_type.split("/")
    # register buildres if empty
    builder = __ENV_BUILDERS[env_path[0]]()
    return builder(env_type, *args, **kwargs)

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

register_lazy('brax', '.brax')
register_lazy('gym', '.gym')
register_lazy('pendulum', '.pendulum')
register_lazy('linear', '.linear')
register_lazy('quadrotor', '.quadrotor')


# ------------------- Policy Definitions -------------------------

# A policy is a function from x --> u or
# x --> PolicyOutput
# optionally (x, policy_state) --> PolicyOutput
@dataclass(frozen=True)
class PolicyOutput:
    action: Any
    # The policy state
    policy_state: Any = None
    # Aux output of the policy
    # this can be anything!
    aux: Any = None

@dataclass(frozen=True)
class Trajectory:
    states: Any
    actions: Any = None

@dataclass(frozen=True)
class Rollout(Trajectory):
    aux: Any = None
    final_policy_state: Any = None

def sanitize_policy(policy_fn, takes_policy_state=False):
    def sanitized_policy(state, policy_state):
        if takes_policy_state:
            output = policy_fn(state, policy_state)
        else:
            output = policy_fn(state)
            if not isinstance(output, PolicyOutput):
                output = PolicyOutput(u=output)
        return output
    return sanitized_policy

# stanza.jit can handle function arguments
# and intelligently makes them static and allows
# for vectorizing over functins.
@partial(stanza.jit, static_argnums=(4,))
def rollout(model, state0,
            # policy is optional. If policy is not supplied
            # it is assumed that model_fn is for an
            # autonomous system
            policy=None,
            # The initial policy state. If "None" is supplied
            # and policy has an 'init_state' function, that
            # policy.init_state(x0) will be used instead
            policy_init_state=None,
            # either length is an integer or  policy.rollout_length
            # or model.rollout_length is not None
            length=None):
    if hasattr(policy, 'init_state') and policy_init_state is None:
        policy_init_state = policy.init_state(state0)

    if hasattr(policy, 'rollout_length') and length is None:
        length = policy.rollout_length
    if hasattr(model, 'rollout_length') and length is None:
        length = model.rollout_length

    if length is None:
        raise ValueError("Rollout length must be specified")

    # policy is standardized to always output a PolicyOutput
    # and take (x, policy_state) as input
    policy_fn = sanitize_policy(policy,
                    takes_policy_state=policy_init_state is not None) \
            if policy is not None else None

    def scan_fn(comb_state, _):
        env_state, policy_state = comb_state
        if policy_fn is not None:
            policy_output = policy_fn(env_state, policy_state)
            action = policy_output.action
            aux = policy_output.aux

            new_policy_state = policy_output.policy_state
            new_env_state = model(env_state, action)
        else:
            action = None
            aux = None
            new_env_state = model(env_state)
            new_policy_state = policy_state
        return (new_env_state, new_policy_state), (env_state, action, aux)

    # Do the first step manually to populate the policy state
    state = (state0, policy_init_state)

    # outputs is (xs, us, jacs) or (xs, us)
    (state_f, policy_state_f), outputs = jax.lax.scan(scan_fn, state,
                                    None, length=length-1)
    states, us, auxs = outputs
    # append the last state
    states = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate((a, jnp.expand_dims(b, 0))),
        states, state_f)
    return Rollout(states=states, actions=us, 
        aux=auxs, final_policy_state=policy_state_f)

# An "Inputs" policy can be used to replay
# inputs from a history buffer
@dataclass
class Actions:
    actions: Any

    @property
    def rollout_length(self):
        lengths, _ = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(lambda x: x.shape[0], self.actions)
        )
        return lengths[0]

    def init_state(self, x0):
        return 0

    @jax.jit
    def __call__(self, x0, T):
        action = jax.tree_util.tree_map(lambda x: x[T], self.actions)
        return PolicyOutput(action=action, policy_state=T + 1)

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
