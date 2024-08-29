from foundry.policy import Policy, PolicyInput, PolicyOutput
from foundry.env import (
    EnvWrapper, RenderConfig, ImageRender, 
    Render
)
from typing import Callable, List, Any
from foundry.core.dataclasses import dataclass, replace, field

import abc
import jax
import foundry.numpy as jnp

class PolicyTransform(abc.ABC):
    @abc.abstractmethod
    def apply(self, policy): ...
        

@dataclass
class ChainedTransform(PolicyTransform):
    transforms: List[PolicyTransform]

    def apply(self, policy):
        for t in self.transforms:
            policy = t.transform_policy(policy)
        return policy

# ---- NoiseTransform ----
# Injects noise ontop of a given policy

@dataclass
class NoiseInjector(PolicyTransform):
    sigma: float

    def apply(self, policy):
        return NoisyPolicy(policy, self.sigma)

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
class ChunkingPolicyState:
    # The batched observation history
    input_chunk: Any = None
    # The last (batched) output
    # from the sub-policy
    last_batched_output: Any = None
    t: int = 0

@dataclass
class ChunkingTransform(PolicyTransform):
    # If None, no input/output batch dimension
    obs_chunk_size: int = field(default=None, pytree_node=False)
    action_chunk_size: int = field(default=None, pytree_node=False)

    def apply(self, policy):
        return ChunkingPolicy(policy, self.obs_chunk_size, self.action_chunk_size)

@dataclass
class ChunkingPolicy(Policy):
    policy: Policy
    # If None, no input/output batch dimension
    obs_chunk_size: int = field(default=None, pytree_node=False)
    action_chunk_size: int = field(default=None, pytree_node=False)

    @property
    def rollout_length(self):
        return (self.policy.rollout_length * self.action_chunk_size) \
            if self.action_chunk_size is not None else \
                self.policy.rollout_length
            
    def __call__(self, input):
        policy_state = input.policy_state
        if policy_state is None:
            # replicate the current input batch
            obs_batch = input.observation \
                    if self.obs_chunk_size is None else \
                jax.tree.map(
                    lambda x: jnp.repeat(x[jnp.newaxis,...],
                        self.obs_chunk_size, axis=0), input.observation
                )
        else:
            # create the new input chunk
            obs_batch = input.observation \
                    if self.obs_chunk_size is None else \
                jax.tree.map(
                    lambda x, y: jnp.roll(x, -1, axis=0).at[-1,...].set(y), 
                    policy_state.input_chunk, input.observation
                )
        def reevaluate():
            output = self.policy(PolicyInput(
                obs_batch,
                input.state,
                policy_state.last_batched_output.policy_state \
                    if policy_state is not None else None,
                input.rng_key
            ))
            action = output.action \
                if self.action_chunk_size is None else \
                jax.tree.map(
                    lambda x: x[0, ...], output.action,
                ) 
            return PolicyOutput(
                action=action,
                policy_state=ChunkingPolicyState(obs_batch, output, 1),
                info=output.info
            )
        def index():
            output = policy_state.last_batched_output
            action = output.action \
                if self.action_chunk_size is None else \
                jax.tree.map(
                    lambda x: x[policy_state.t, ...], 
                    output.action,
                )
            return PolicyOutput(
                action,
                ChunkingPolicyState(
                    obs_batch,
                    policy_state.last_batched_output,
                    policy_state.t + 1
                ),
                output.info
            )
        if self.action_chunk_size is None \
                or input.policy_state is None:
            return reevaluate()
        else:
            return jax.lax.cond(
                policy_state.t >= self.action_chunk_size,
                reevaluate, index
            )