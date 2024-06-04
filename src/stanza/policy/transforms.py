from stanza.policy import Policy, PolicyInput, PolicyOutput
from typing import Callable, List, Any
from stanza.struct import dataclass, replace, field

import jax
import jax.numpy as jnp

# Transform a (policy, state) into a new (policy, state)
class Transform:
    # Override these! Not __call__!
    def transform_policy(self, policy):
        raise NotImplementedError("Must implement transform_policy()")

    def transform_policy_state(self, policy_state):
        raise NotImplementedError("Must implement transform_policy_state()")
    
    def transform_environment(self):
        raise NotImplementedError("Must implement as_environment_transform()")

@dataclass
class RandomPolicy(Policy):
    sample_fn: Callable

    def __call__(self, input):
        u = self.sample_fn(input.rng_key)
        return PolicyOutput(action=u)

@dataclass
class ChainedTransform(Transform):
    transforms: List[Transform]

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

@dataclass
class NoiseInjector(Transform):
    sigma: float

    def transform_policy(self, policy):
        return NoisyPolicy(policy, self.sigma)

    def transform_policy_state(self, policy_state):
        return policy_state

@dataclass
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

@dataclass
class ChunkedPolicyState:
    # The batched (observation, state) history
    input_chunk: Any = None
    # The last (batched) output
    # from the sub-policy
    last_batched_output: Any = None
    t: int = 0

@dataclass
class ChunkTransform(Transform):
    # If None, no input/output batch dimension
    input_chunk_size: int = field(default=None, pytree_node=False)
    output_chunk_size: int = field(default=None, pytree_node=False)

    def transform_policy(self, policy):
        return ChunkedPolicy(policy, self.input_chunk_size, self.output_chunk_size)
    
    def transform_policy_state(self, policy_state):
        return ChunkedPolicyState(None, PolicyOutput(None, policy_state, None),
                                self.output_chunk_size)

@dataclass
class ChunkedPolicy(Policy):
    policy: Policy
    # If None, no input/output batch dimension
    input_chunk_size: int = field(default=None, pytree_node=False)
    output_chunk_size: int = field(default=None, pytree_node=False)

    @property
    def rollout_length(self):
        return (self.policy.rollout_length * self.output_chunk_size) \
            if self.output_chunk_size is not None else \
                self.policy.rollout_length
            
    def __call__(self, input):
        policy_state = input.policy_state
        if policy_state is None:
            # replicate the current input batch
            obs_batch, state_batch = (input.observation, input.state) \
                    if self.input_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x[jnp.newaxis,...],
                        self.input_chunk_size, axis=0), (input.observation, input.state)
                )
        else:
            # create the new input chunk
            obs_batch, state_batch = (input.observation, input.state) \
                    if self.input_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x, y: jnp.roll(x, -1, axis=0).at[-1,...].set(y), 
                    policy_state.input_chunk, (input.observation, input.state)
                )

        def reevaluate():
            output = self.policy(PolicyInput(
                obs_batch,
                state_batch,
                policy_state.last_batched_output.policy_state \
                    if policy_state is not None else None,
                input.rng_key
            ))
            action = output.action \
                if self.output_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x: x[0, ...], output.action,
                ) 
            return PolicyOutput(
                action=action,
                policy_state=ChunkedPolicyState((obs_batch, state_batch), output, 1),
                info=output.info
            )
        def index():
            action = policy_state.last_batched_output.action \
                if self.output_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x: x[policy_state.t, ...], 
                    policy_state.last_batched_output.action,
                )
            return PolicyOutput(
                action,
                ChunkedPolicyState(
                    (obs_batch, state_batch),
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

@dataclass
class ActionRepeater(Transform):
    repeats : int = field(default=1, pytree_node=False)

    def transform_policy(self, policy):
        return RepeatingPolicy(policy, self.repeats)
    
    def transform_policy_state(self, policy_state):
        return RepeatingState(
            PolicyOutput(None, policy_state, None),
            self.control_interval
        )

@dataclass
class RepeatingState:
    # The last output from the low-level controller
    last_output: PolicyOutput
    # Time since last evaluation
    t: int

@dataclass
class RepeatingPolicy(Policy):
    policy: Policy = None
    repeats : int = field(default=1, pytree_node=False)

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