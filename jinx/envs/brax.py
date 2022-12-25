from . import Environment
import brax
import brax.envs
import brax.io

import jax
import jax.numpy as jnp

class BraxEnvironment(Environment):
    def __init__(self, name):
        env = brax.envs.create(name)
        self._action_size = env.action_size
        self._env_reset_jit = jax.jit(env.reset)
        self._env_step_jit = jax.jit(env.step)
    
    @property
    def action_size(self):
        return self._action_size

    def reset(self, key):
        return self._env_reset_jit(key)
    
    def step(self, state, action):
        return self._env_step_jit(state, action)

def builder():
    return BraxEnvironment