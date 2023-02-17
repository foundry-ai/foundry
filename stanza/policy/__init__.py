import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from typing import Callable
from stanza.util.dataclasses import dataclass

# A policy is a function from x --> u or
# x --> PolicyOutput
# optionally (x, policy_state) --> PolicyOutput
@dataclass
class PolicyOutput:
    action: Any
    # The policy state
    policy_state: Any = None
    # Aux output of the policy
    # this can be anything!
    aux: Any = None

@dataclass
class Trajectory:
    states: Any
    actions: Any = None

@dataclass
class Rollout(Trajectory):
    aux: Any = None
    final_policy_state: Any = None

# Takes a policy function (which may or may not return
# an instance of PolicyOutput and may or may not take in a
# policy_state) and returns a function of the form
# (input, policy_state) --> PolicyOutput
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

@dataclass
class NoisyPolicy:
    rng_key: PRNGKey
    sigma: float
    base_policy: Callable
    
    def init_state(self, x0):
        return (self.rng_key, self.base_policy.init_state(x0))
    
    def __call__(self, x, policy_state):
        rng_key, base_policy_state = policy_state
        rng_key, sk = jax.random.split(rng_key)

        u_base, base_policy_state = self.base_policy(x, base_policy_state)

        # flatten u, add the noise, unflatten
        u_flat, unflatten = jax.flatten_util.ravel_pytree(u_base)
        noise = self.sigma * jax.random.normal(sk, u_flat.shape)
        u_flat = u_flat + noise
        u = unflatten(u_flat)

        return u, (rng_key, base_policy_state)

@dataclass
class RandomPolicy:
    rng_key: PRNGKey
    sample_fn: Callable

    def init_state(self, x0):
        return self.rng_key
    
    def __call__(self, x, policy_state):
        rng_key = policy_state
        rng_key, sk = jax.random.split(rng_key)
        u = self.sample_fn(sk)
        return u, rng_key