from stanza.dataclasses import dataclass, field
from stanza.util.attrdict import AttrMap
from stanza.policies import PolicyInput, PolicyOutput
from stanza.envs import Environment

from typing import Callable

import jax
import jax.numpy as jnp

@dataclass(jax=True)
class ACPolicy:
    actor_critic: Callable
    observation_normalizer: Callable = None
    action_normalizer: Callable = None

    def __call__(self, input: PolicyInput) -> PolicyOutput:
        observation = input.observation
        if self.observation_normalizer is not None:
            observation = self.observation_normalizer.normalize(observation)
        pi, value = self.actor_critic(observation)
        action = pi.sample(input.rng_key)
        log_prob = pi.log_prob(action)
        return PolicyOutput(
            action, log_prob, 
            AttrMap(log_prob=log_prob, value=value)
        )


@dataclass(jax=True)
class EpisodicEnvironment(Environment):
    env: Environment
    episode_length : jnp.array

    def sample_action(self, rng_key):
        return self.env.sample_action(rng_key)

    def sample_state(self, rng_key):
        return (0, self.env.sample_state(rng_key))

    def reset(self, rng_key):
        return (0, self.env.reset(rng_key))
    
    def step(self, state, action, rng_key):
        i, state = state
        def step(i, state):
            return (i + 1, self.env.step(state, action, rng_key))
        def reset(i, state):
            return (0, self.env.reset(rng_key))
        return jax.lax.cond(
            jnp.logical_or(i == self.episode_length,
                           self.env.done(state)),
                    reset, step, i, state)

    def reward(self, state, action, next_state):
        return self.env.reward(state[1], action, next_state[1])
    
    def done(self, state):
        return jnp.logical_or(self.env.done(state[1]),
            self.episode_length == state[0])
    
    def render(self, state, **kwargs):
        return self.env.render(state[1], **kwargs)

    def teleop_policy(self, interface):
        return self.env.teleop_policy(interface)