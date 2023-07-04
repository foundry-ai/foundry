from stanza.envs import Environment
from stanza.dataclasses import dataclass, field

from gymnax.environments import environment
from typing import Any

import jax.numpy as jnp

@dataclass(jax=True)
class GymnaxState:
    obs : Any
    state : Any
    reward : jnp.array
    done : jnp.array

@dataclass(jax=True)
class Gymnax(Environment):
    gymnax_env : environment.Environment = field(jax_static=True)
    env_params : Any

    def sample_action(self, rng_key):
        action = self.gymnax_env.action_space(self.env_params).sample(rng_key)
        return action

    def sample_state(self, rng_key):
        return self.reset(rng_key)

    def reset(self, rng_key):
        obs, state = self.gymnax_env.reset(rng_key)
        return GymnaxState(obs, state, 
            jnp.array(0.), jnp.array(False))
    
    def step(self, state, action, rng_key):
        n_obs, n_state, reward, done, _ = self.gymnax_env.step(
            rng_key, state.state, action, self.env_params
        )
        return GymnaxState(n_obs, n_state, reward, done)
    
    def reward(self, state, action, next_state):
        return next_state.reward
    
    def done(self, state):
        return state.done