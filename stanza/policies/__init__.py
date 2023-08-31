import jax
import jax.numpy as jnp
from jax.random import PRNGKey
from typing import Callable, List, Any
import stanza
from stanza.dataclasses import dataclass, field, replace
from stanza.util.attrdict import AttrMap
from functools import partial

# A policy is a function from PolicyInput --> PolicyOutput

@dataclass(jax=True)
class PolicyInput:
    observation: Any
    policy_state: Any = None
    rng_key : PRNGKey = None

@dataclass(jax=True)
class PolicyOutput:
    action: Any
    # The policy state
    policy_state: Any = None
    info: AttrMap = field(default_factory=AttrMap)

@dataclass(jax=True)
class Trajectory:
    states: Any
    actions: Any = None

# Rollout contains the trajectory + aux data,
# final policy state, final rng key
@dataclass(jax=True)
class Rollout(Trajectory):
    observations: Any = None
    info: AttrMap = field(default_factory=AttrMap)
    final_policy_state: Any = None
    final_policy_rng_key: PRNGKey = None
    final_model_rng_key: PRNGKey = None

class Policy:
    @property
    def rollout_length(self):
        return None

    def __call__(self, input):
        raise NotImplementedError("Must implement __call__()")

# stanza.jit can handle function arguments
# and intelligently makes them static and allows
# for vectorizing over functins.
@partial(stanza.jit, static_argnames=("length", "last_state"))
def rollout(model, state0,
            # policy is optional. If policy is not supplied
            # it is assumed that model is for an
            # autonomous system
            policy=None,
            *,
            # observation function
            # by default just uses the state
            observe=None,
            # The rng_key for the environment
            model_rng_key=None,
            # The initial policy state.
            policy_init_state=None,
            # The policy rng key
            policy_rng_key=None,
            # Apply a transform to the
            # policy before rolling out.
            policy_transform=None,
            # either length is an integer or policy.rollout_length is not None
            length=None, last_state=True):
    if policy_transform is not None and policy is not None:
        policy, policy_init_state = policy_transform(policy, policy_init_state)
    # Look for a fallback to the rollout length
    # in the policy. This is useful mainly for the Actions policy
    if length is None and hasattr(policy, 'rollout_length'):
        length = policy.rollout_length
    if length is None:
        raise ValueError("Rollout length must be specified")
    if length == 0:
        raise ValueError("Rollout length must be > 0")
    if observe is None:
        observe = lambda x: x

    def scan_fn(comb_state, _):
        env_state, policy_state, policy_rng, model_rng = comb_state
        new_policy_rng, p_sk = jax.random.split(policy_rng) \
            if policy_rng is not None else (None, None)
        new_model_rng, m_sk = jax.random.split(model_rng) \
            if model_rng is not None else (None, None)
        obs = observe(env_state)
        if policy is not None:
            input = PolicyInput(obs, policy_state, p_sk)
            policy_output = policy(input)
            action = policy_output.action
            info = policy_output.info
            new_policy_state = policy_output.policy_state
        else:
            action = None
            info = None
            new_policy_state = policy_state
        new_env_state = model(env_state, action, m_sk)
        return (new_env_state, new_policy_state, new_policy_rng, new_model_rng),\
                (env_state, obs, action, info)

    # Do the first step manually to populate the policy state
    state = (state0, policy_init_state, policy_rng_key, model_rng_key)
    new_state, first_output = scan_fn(state, None)
    # outputs is (xs, us, jacs) or (xs, us)
    (state_f, policy_state_f, 
     policy_rng_f, model_rng_f), outputs = jax.lax.scan(scan_fn, new_state,
                                    None, length=length-2)
    outputs = jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate((jnp.expand_dims(a,0), b)),
        first_output, outputs)

    states, observations, us, info = outputs
    if last_state:
        states = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate((a, jnp.expand_dims(b, 0))),
            states, state_f)
        observations = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate((a, jnp.expand_dims(b, 0))),
            observations, observe(state_f))
    return Rollout(states=states, actions=us, observations=observations,
        info=info, final_policy_state=policy_state_f, 
        final_policy_rng_key=policy_rng_f, final_model_rng_key=model_rng_f)

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
    def __call__(self, input):
        T = input.policy_state if input.policy_state is not None else 0
        action = jax.tree_util.tree_map(lambda x: x[T], self.actions)
        return PolicyOutput(action=action, policy_state=T + 1)