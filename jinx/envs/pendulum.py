from jinx.envs import Environment

import jax
import jax.numpy as jnp
from typing import NamedTuple

class State(NamedTuple):
    x: jnp.ndarray


class PendulumEnvironment(Environment):
    def __init__(self):
        pass
    
    @property
    def action_size(self):
        return 1

    def reset(self, key):
        # pick random position between +/- radians from right
        pos = jax.random.uniform(key,shape=(1,), minval=-1,maxval=1)
        vel = jnp.zeros((1,))
        x = jnp.concatenate((pos, vel))
        return State(x)

    def step(self, state, action):
        pos = state.x[0] + 0.05*state.x[1]
        vel = state.x[1] - 0.05*jnp.sin(state.x[0]) + 0.05*action[0]
        x = jnp.stack((pos, vel))
        return State(x)
    
    def cost(self, xs, us):
        diff = xs - jnp.array([jnp.pi, 0])
        x_cost = jnp.sum(diff**2)
        u_cost = jnp.sum(us**2)
        return x_cost + u_cost

def builder():
    return PendulumEnvironment