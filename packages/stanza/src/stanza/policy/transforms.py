from stanza.policy import Policy, PolicyInput, PolicyOutput
from stanza.env import (
    EnvWrapper, RenderConfig, ImageRender, SequenceRender,
    Render
)
from typing import Callable, List, Any
from stanza.dataclasses import dataclass, replace, field

import jax
import jax.numpy as jnp

# Transform a (policy, state) into a new (policy, state)
class Transform:
    # Override these! Not __call__!
    def transform_policy(self, policy):
        raise NotImplementedError("Must implement transform_policy()")
    
    def transform_env(self):
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

    def transform_env(self, env):
        for t in self.transforms:
            env = t.transform_env(env)
        return env

def chain_transforms(*transforms):
    return ChainedTransform(transforms)

# ---- NoiseTransform ----
# Injects noise ontop of a given policy

@dataclass
class NoiseInjector(Transform):
    sigma: float

    def transform_policy(self, policy):
        return NoisyPolicy(policy, self.sigma)

    def transform_env(self, env):
        return NoisyEnvironment(env, self.sigma)

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

@dataclass
class NoisyEnvironment(EnvWrapper):
    sigma: jax.Array

    def step(self, state, action, rng_key=None):
        u_flat, unflatten = jax.flatten_util.ravel_pytree(action)
        noise = self.sigma * jax.random.normal(rng_key, u_flat.shape)
        action = unflatten(u_flat + noise)
        return self.base.step(state, action, rng_key)

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
    obs_chunk_size: int = field(default=None, pytree_node=False)
    action_chunk_size: int = field(default=None, pytree_node=False)

    def transform_policy(self, policy):
        return ChunkedPolicy(policy, self.obs_chunk_size, self.action_chunk_size)
    
    def transform_env(self, env):
        return ChunkedEnvironment(env, self.obs_chunk_size, self.action_chunk_size)

@dataclass
class ChunkedPolicy(Policy):
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
            obs_batch, state_batch = (input.observation, input.state) \
                    if self.obs_chunk_size is None else \
                jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x[jnp.newaxis,...],
                        self.obs_chunk_size, axis=0), (input.observation, input.state)
                )
        else:
            # create the new input chunk
            obs_batch, state_batch = (input.observation, input.state) \
                    if self.obs_chunk_size is None else \
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
                if self.action_chunk_size is None else \
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
                if self.action_chunk_size is None else \
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
        if self.action_chunk_size is None \
                or input.policy_state is None:
            return reevaluate()
        else:
            return jax.lax.cond(
                policy_state.t >= self.action_chunk_size,
                reevaluate, index
            )

@dataclass
class ChunkedState:
    history: Any

@dataclass
class ChunkedEnvironment(EnvWrapper):
    obs_chunk_size: int | None = field(default=None, pytree_node=False)
    action_chunk_size: int | None = field(default=None, pytree_node=False)

    def sample_state(self, rng_key):
        history_length = max(self.obs_chunk_size or 1, self.action_chunk_size or 1)
        state = self.base.sample_state(rng_key)
        return jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[jnp.newaxis,...],
                history_length, axis=0), state
        )

    def reset(self, rng_key):
        history_length = max(self.obs_chunk_size or 1, self.action_chunk_size or 1)
        state = self.base.reset(rng_key)
        return jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[jnp.newaxis,...],
                history_length, axis=0), state
        )

    def step(self, state, action, rng_key=None):
        if self.action_chunk_size is None:
            action = jax.tree_util.tree_map(lambda x: x[None], action)
        action_size = self.action_chunk_size or 1
        obs_size = self.obs_chunk_size or 1

    def observe(self, state):
        history = jax.tree_util.tree_map(lambda x: x[-self.obs_chunk_size], state)
        if self.obs_chunk_size is None:
            return self.base.observe(state)
        else:
            return jax.tree_util.tree_map(
                lambda x: x[0, ...], self.base.observe(state)
            )
    
    def reward(self, state, action, next_state):
        raise NotImplementedError()

    def cost(self, states, actions):
        raise NotImplementedError()
    
    def is_finished(self, state: Any) -> jax.Array:
        raise NotImplementedError()
    
    def render(self, config: RenderConfig[Render], state) -> Render:
        if type(config) == ImageRender:
            return self.base.render(config, state)
        elif type(config) == SequenceRender:
            return self.base.render(config, state)
        else:
            raise NotImplementedError()

@dataclass
class ActionRepeater(Transform):
    repeats : int = field(default=1, pytree_node=False)

    def transform_policy(self, policy):
        return RepeatingPolicy(policy, self.repeats)
    
    def transform_env(self, env):
        return RepeatingEnvironment(env, self.repeats)

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

@dataclass
class RepeatedState:
    history: Any

@dataclass
class RepeatingEnvironment(EnvWrapper):
    repeats: int = field(default=1, pytree_node=False)

    def sample_state(self, rng_key):
        state = self.base.sample_state(rng_key)
        return RepeatedState(jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[jnp.newaxis,...],
                self.repeats, axis=0), state
        ))

    def reset(self, rng_key):
        state = self.base.reset(rng_key)
        return RepeatedState(jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[jnp.newaxis,...],
                self.repeats, axis=0), state
        ))

    def step(self, state, action, rng_key=None):
        state = jax.tree_util.tree_map(lambda x: x[-1], state.history)
        if rng_key is not None:
            rng_key = jax.random.split(rng_key, self.repeats)
        def scan_fn(state, rng_key):
            state = self.base.step(state, action, rng_key)
            return state, state
        _, history = jax.lax.scan(scan_fn, state, rng_key, length=self.repeats)
        return RepeatedState(history)
    
    def observe(self, state):
        state = jax.tree_util.tree_map(lambda x: x[-1], state.history)
        return self.base.observe(state)
    
    def reward(self, state, action, next_state):
        # sum up the rewards over the next_state history
        raise NotImplementedError()

    def cost(self, states, actions):
        raise NotImplementedError()
    
    def is_finished(self, state: Any) -> jax.Array:
        raise NotImplementedError()
    
    def render(self, config: RenderConfig[Render], state) -> Render:
        if type(config) == ImageRender:
            last_state = jax.tree_util.tree_map(lambda x: x[-1], state.history)
            return self.base.render(config, last_state)
        elif type(config) == SequenceRender:
            r = ImageRender(width=config.width, height=config.height)
            return jax.vmap(self.base.render, in_axes=(None, 0))(
                r, state.history
            )
        else:
            raise NotImplementedError()
