from jinx.envs import Environment

import jax
import jax.numpy as jnp


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
        state = jnp.concatenate((pos, vel))
        return state

    def step(self, state, action):
        pos = state[0] - 0.05*state[1]
        # using g, l = 1
        vel = state[1] - 0.05*jnp.sin(state[0]) + 0.1*action[0]
        state = jnp.stack((pos, vel))
        return state

    def observe(self, state, name):
        if name == 'x':
            return state
        raise RuntimeError('No such observation')

def builder():
    return PendulumEnvironment