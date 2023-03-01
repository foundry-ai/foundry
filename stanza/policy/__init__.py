import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from typing import Callable, Any
import stanza
from stanza.util.dataclasses import dataclass
from functools import partial

# A policy is a function from x --> u or
# x --> PolicyOutput
# optionally (x, policy_state) --> PolicyOutput
@dataclass(jax=True)
class PolicyOutput:
    action: Any
    # The policy state
    policy_state: Any = None
    # Aux output of the policy
    # this can be anything!
    aux: Any = None

@dataclass(jax=True)
class Trajectory:
    states: Any
    actions: Any = None

# Rollout contains the trajectory + aux data,
# final policy state
@dataclass(jax=True)
class Rollout(Trajectory):
    aux: Any = None
    final_policy_state: Any = None

class PolicyWrapper:
    def __init__(self, fun):
        self._wrapped = fun
    
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __call__(self, state, policy_state=None, **kwargs):
        if policy_state is None:
            r = self._wrapped(state, **kwargs)
        else:
            r = self._wrapped(state, policy_state, **kwargs)
        if not isinstance(r, PolicyOutput):
            r = PolicyOutput(r)
        return r

# Will wrap a policy, guaranteeing the output is
# of type PolicyOutput and that policy_state is an optional
# input. It is recommended that functionals which take policies
# should call wrap() on the policies.
def wrap(policy):
    return PolicyWrapper(policy) if policy is not None else None

# stanza.jit can handle function arguments
# and intelligently makes them static and allows
# for vectorizing over functins.
@partial(stanza.jit, static_argnums=(4,5))
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
            length=None, last_state=True):
    # Look for a fallback to the rollout length
    # in the policy. This is useful mainly for the Actions policy
    if length is None and hasattr(policy, 'rollout_length'):
        length = policy.rollout_length
    if length is None:
        raise ValueError("Rollout length must be specified")
    if length == 0:
        raise ValueError("Rollout length must be > 0")
    # standardize the policy
    policy = wrap(policy)

    def scan_fn(comb_state, _):
        env_state, policy_state = comb_state
        if policy is not None:
            policy_output = policy(env_state, policy_state)
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
    state = (state0, None)
    new_state, first_output = scan_fn(state, None)
    # outputs is (xs, us, jacs) or (xs, us)
    (state_f, policy_state_f), outputs = jax.lax.scan(scan_fn, new_state,
                                    None, length=length-2)
    outputs = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate((jnp.expand_dims(a,0), b)),
        first_output, outputs)

    states, us, auxs = outputs
    if last_state:
        states = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate((a, jnp.expand_dims(b, 0))),
            states, state_f)
    return Rollout(states=states, actions=us, 
        aux=auxs, final_policy_state=policy_state_f)

# Shorthand alias for rollout with an actions policy
def rollout_inputs(model, state0, actions, last_state=True):
    return rollout(model, state0, policy=Actions(actions),
                   last_state=last_state)

# An "Actions" policy can be used to replay
# actions from a history buffer
@dataclass(jax=True)
class Actions:
    actions: Any

    @property
    def rollout_length(self):
        lengths, _ = jax.tree_util.tree_flatten(
            jax.tree_util.tree_map(lambda x: x.shape[0], self.actions)
        )
        return lengths[0] + 1

    @jax.jit
    def __call__(self, x0, T=None):
        T = T if T is not None else 0
        action = jax.tree_util.tree_map(lambda x: x[T], self.actions)
        return PolicyOutput(action=action, policy_state=T + 1)

@dataclass(jax=True)
class NoisyPolicy:
    rng_key: PRNGKey
    sigma: float
    base_policy: Callable
    
    def __call__(self, x, policy_state=None):
        rng_key, base_policy_state = (
            policy_state 
            if policy_state is not None else
            (self.rng_key, None)
        )

        rng_key, sk = jax.random.split(rng_key)

        output = (
            self.base_policy(x, base_policy_state) 
                if base_policy_state is not None else
            self.base_policy(x)
        )

        # flatten u, add the noise, unflatten
        u_flat, unflatten = jax.flatten_util.ravel_pytree(output.action)
        noise = self.sigma * jax.random.normal(sk, u_flat.shape)
        u_flat = u_flat + noise
        action = unflatten(u_flat)

        return PolicyOutput(
            action,
            policy_state=(rng_key, output.policy_state)
        )

@dataclass(jax=True)
class RandomPolicy:
    rng_key: PRNGKey
    sample_fn: Callable

    def __call__(self, x, policy_state=None):
        rng_key = self.rng_key if policy_state is None else policy_state
        rng_key, sk = jax.random.split(rng_key)
        u = self.sample_fn(sk)
        return u, rng_key