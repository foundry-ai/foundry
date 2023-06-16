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

# Transform a (policy, state) into a new (policy, state)
class PolicyTransform:
    # Override these! Not __call__!
    def transform_policy(self, policy):
        return policy

    def transform_policy_state(self, policy_state):
        return policy_state

    # Do not override unless you also keep the return behavior
    # Just override the above
    def __call__(self, policy, policy_state=None):
        tpol = self.transform_policy(policy)
        return tpol if policy_state is None else \
            (tpol, self.transform_policy_state(policy_state))
    
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

    def scan_fn(comb_state, _):
        env_state, policy_state, policy_rng, model_rng = comb_state
        new_policy_rng, p_sk = jax.random.split(policy_rng) \
            if policy_rng is not None else (None, None)
        new_model_rng, m_sk = jax.random.split(model_rng) \
            if model_rng is not None else (None, None)
        if policy is not None:
            input = PolicyInput(env_state, policy_state, p_sk)
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
                (env_state, action, info)

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

    states, us, info = outputs
    if last_state:
        states = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate((a, jnp.expand_dims(b, 0))),
            states, state_f)
    return Rollout(states=states, actions=us, 
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

@dataclass(jax=True)
class RandomPolicy(Policy):
    sample_fn: Callable

    def __call__(self, input):
        u = self.sample_fn(input.rng_key)
        return PolicyOutput(action=u)

@dataclass(jax=True)
class ChainedTransform(PolicyTransform):
    transforms: List[PolicyTransform]

    def transform_policy(self, policy):
        for t in self.transforms:
            policy = t.transform_policy(policy)
        return policy

    def transform_policy_state(self, policy_state):
        for t in self.transforms:
            policy = t.transform_policy_state(policy_state)
        return policy

def chain_transforms(*transforms):
    return ChainedTransform(transforms)

# ---- NoiseTransform ----
# Injects noise ontop of a given policy

@dataclass(jax=True)
class NoiseTransform(PolicyTransform):
    sigma: float
    def __call__(self, policy, policy_state):
        return NoisyPolicy(policy, self.sigma), policy_state

@dataclass(jax=True)
class NoisyPolicy(Policy):
    policy: Callable
    sigma: float
    
    def __call__(self, input):
        rng_key, sk = jax.random.split(input.rng_key)

        sub_input = replace(input, rng_key=rng_key)
        output = self.policy(sub_input)

        u_flat, unflatten = jax.flatten_util.ravel_pytree(output.action)
        noise = self.sigma * jax.random.normal(sk, u_flat.shape)
        u_flat = u_flat + noise
        action = unflatten(u_flat)
        output = replace(output, action=action)
        return output

## ---- ChunkTransform -----
# A chunk transform will composite inputs,
# outputs

@dataclass(jax=True)
class ChunkPolicyState:
    # The batched input
    obs_batch: Any = None
    # The last (batched) output
    # from the sub-policy
    last_batched_output: Any = None
    t: int = 0

@dataclass(jax=True)
class ChunkTransform(PolicyTransform):
    # If None, no input/output batch dimension
    input_chunk_size: int = field(default=None, jax_static=True)
    output_chunk_size: int = field(default=None, jax_static=True)

    def transform_policy(self, policy):
        return ChunkPolicy(policy, self.input_chunk_size, self.output_chunk_size)
    
    def transform_policy_state(self, policy_state):
        return ChunkPolicyState(None, PolicyOutput(None, policy_state, None),
                                self.output_chunk_size)

@dataclass(jax=True)
class ChunkPolicy(Policy):
    policy: Policy
    # If None, no input/output batch dimension
    input_chunk_size: int = field(default=None, jax_static=True)
    output_chunk_size: int = field(default=None, jax_static=True)

    @property
    def rollout_length(self):
        return (self.policy.rollout_length * self.output_chunk_size) \
            if self.output_chunk_size is not None else \
                self.policy.rollout_length
            
    def __call__(self, input):
        policy_state = input.policy_state
        if policy_state is None:
            # replicate the input batch
            obs_batch = input.observation \
                if self.input_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x[jnp.newaxis,...],
                        self.input_chunk_size, axis=0), input.observation
                )
        else:
            # create the new input chunk
            obs_batch = input.observation if self.input_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x, y: jnp.roll(x, -1, axis=0).at[-1,...].set(y), 
                    policy_state.obs_batch, input.observation
                )

        def reevaluate():
            output = self.policy(PolicyInput(
                obs_batch,
                policy_state.last_batched_output.policy_state \
                    if policy_state is not None else None,
                input.rng_key
            ))
            action = output.action \
                if self.output_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x: x[0, ...], output.action
                ) 
            return PolicyOutput(
                action=action,
                policy_state=ChunkPolicyState(obs_batch, output, 1),
                info=output.info
            )
        def index():
            action = policy_state.last_batched_output.action \
                if self.output_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x: x[policy_state.t, ...], policy_state.last_batched_output.action
                )
            return PolicyOutput(
                action,
                ChunkPolicyState(
                    obs_batch,
                    policy_state.last_batched_output,
                    policy_state.t + 1
                ),
                policy_state.last_batched_output.info
            )
        if self.output_chunk_size is None \
                or input.policy_state is None:
            return reevaluate()
        else:
            return jax.lax.cond(
                policy_state.t >= self.output_chunk_size,
                reevaluate, index
            )

## ----- SamplingTransform -----
# A FreqTransform will run a policy at
# a lower frequency, evaluating it every control_interval
# number of steps. This is useful
# if an expensive controller is combined
# with a higher-frequency low-level controller.
# This is equivalent to a ChunkPolicy where the output
# gets replicated control_interval number of times

@dataclass(jax=True)
class SampleRateTransform(PolicyTransform):
    control_interval : int = field(default=1, jax_static=True)

    def transform_policy(self, policy):
        return SampleRatePolicy(policy, self.control_interval)
    
    def transform_policy_state(self, policy_state):
        return SampleRateState(
            PolicyOutput(None, policy_state, None),
            self.control_interval
        )

@dataclass(jax=True)
class SampleRateState:
    # The last output from the low-level controller
    last_output: PolicyOutput
    # Time since last evaluation
    t: int

@dataclass(jax=True)
class SampleRatePolicy(Policy):
    policy: Policy = None
    control_interval : int = field(default=1, jax_static=True)

    @property
    def rollout_length(self):
        return self.policy.rollout_length*self.control_interval
    
    def __call__(self, input):
        ps = input.policy_state if input.policy_state is not None else \
            SampleRateState(PolicyOutput(None), None)

        # For the input to the sub-policy
        # use the policy state from the last outut we had
        sub_input = replace(input, 
            policy_state=ps.last_output.policy_state)
        if ps.t is None:
            t = 1
            sub_output = self.policy(sub_input)
        else:
            t, sub_output = jax.lax.cond(ps.t >= self.control_interval,
                        lambda: (1, self.policy(sub_input)),
                        lambda: (ps.t + 1, ps.last_output))
        # Store the last output of the high-level policy
        # in the policy state
        output = replace(sub_output,
            policy_state=SampleRateState(sub_output, t))
        return output