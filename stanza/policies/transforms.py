from stanza.policies import Policy, PolicyInput, PolicyOutput
from typing import Callable, List, Any
from stanza.dataclasses import dataclass, replace, field

import jax
import jax.numpy as jnp

# Transform a (policy, state) into a new (policy, state)
class PolicyTransform:
    # Override these! Not __call__!
    def transform_policy(self, policy):
        return policy

    def transform_policy_state(self, policy_state):
        return policy_state
    
    def as_environment_transform(self):
        raise NotImplementedError("Must implement as_environment_transform()")

    # Do not override unless you also keep the return behavior
    # Just override the above
    def __call__(self, policy, policy_state=None):
        tpol = self.transform_policy(policy)
        return tpol if policy_state is None else \
            (tpol, self.transform_policy_state(policy_state))

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
class NoiseInjector(PolicyTransform):
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
class ChunkedPolicyState:
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
        return ChunkedPolicy(policy, self.input_chunk_size, self.output_chunk_size)
    
    def transform_policy_state(self, policy_state):
        return ChunkedPolicyState(None, PolicyOutput(None, policy_state, None),
                                self.output_chunk_size)

@dataclass(jax=True)
class ChunkedPolicy(Policy):
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
                policy_state=ChunkedPolicyState(obs_batch, output, 1),
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
                ChunkedPolicyState(
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
class ActionRepeater(PolicyTransform):
    repeats : int = field(default=1, jax_static=True)

    def transform_policy(self, policy):
        return RepeatingPolicy(policy, self.repeats)
    
    def transform_policy_state(self, policy_state):
        return RepeatingState(
            PolicyOutput(None, policy_state, None),
            self.control_interval
        )

@dataclass(jax=True)
class RepeatingState:
    # The last output from the low-level controller
    last_output: PolicyOutput
    # Time since last evaluation
    t: int

@dataclass(jax=True)
class RepeatingPolicy(Policy):
    policy: Policy = None
    repeats : int = field(default=1, jax_static=True)

    @property
    def rollout_length(self):
        return self.policy.rollout_length*self.repeats
    
    def __call__(self, input):
        ps = input.policy_state if input.policy_state is not None else \
            RepeatingState(PolicyOutput(None), None)

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
            policy_state=RepeatingState(sub_output, t))
        return output

@dataclass(jax=True)
class FeedbackTransform(PolicyTransform):
    def transform_policy(self, policy):
        return FeedbackPolicy(policy)
    
    def transform_policy_state(self, policy_state):
        return policy_state

@dataclass(jax=True)
class FeedbackPolicy(Policy):
    policy: Policy = None
    
    @property
    def rollout_length(self):
        return self.policy.rollout_length

    def __call__(self, input):
        out = self.policy(input)
        action, ref_state, ref_gain = out.action
        action_flat, action_uf = jax.flatten_util.ravel_pytree(action)
        obs_flat, _ = jax.flatten_util.ravel_pytree(input.observation)
        ref_flat, _ = jax.flatten_util.ravel_pytree(ref_state)
        action_mod = 0. # ref_gain @ (obs_flat - ref_flat)
        action = action_uf(action_flat + action_mod)
        return replace(out, action=action)