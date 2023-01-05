from . import Environment
import brax
import brax.envs
import brax.io

import jax
import jax.numpy as jnp

from typing import Any, NamedTuple

class State(NamedTuple):
    brax: Any
    x: jnp.ndarray

class BraxEnvironment(Environment):
    def __init__(self, name):
        env = brax.envs.create(name)
        self._env_name = name
        self._env = env
        self._action_size = env.action_size
        self._env_reset_jit = jax.jit(env.reset)
        self._env_step_jit = jax.jit(env.step)
    
    @property
    def action_size(self):
        return self._action_size

    def reset(self, key):
        brax_state = self._env_reset_jit(key)
        x = brax_state.obs
        return State(brax_state, x)
    
    def step(self, state, action):
        brax_state = self._env_step_jit(state.brax, action)
        return State(brax_state, brax_state.obs)
    
    def render(self, state, width=256, height=256):
        return brax.io.image.render(self._env.sys, state.brax.qp, width=width, height=height)
    
    def cost(self, xs, us):
        return COST_FNS[self._env_name](xs, us)

def ant_cost(xs, us):
    # the x-velocity
    forward_vel = xs[:,13]
    return -jnp.sum(forward_vel) + jnp.sum(jnp.square(us))

COST_FNS = {
    'ant': ant_cost
}

def builder():
    return BraxEnvironment