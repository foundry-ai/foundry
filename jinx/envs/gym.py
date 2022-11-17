from . import Environment

import gym
import jax
from jax.random import PRNGKey
import jax.numpy as jnp

class GymEnvironment(Environment):
    def __init__(self, name):
        self.name = name
        self.env = gym.make(self.name)
    
    @property
    def action_size(self):
        return self.env.action_space.shape[0]

    def reset(self, key):
        seed = jax.random.randint(key, (1,), 0, 2**31-1).item()
        self.env.seed(seed)
        obs = self.env.reset()
        istate = env_get_state(self.env)
        return (obs, 0, istate)

    def step(self, state, action):
        _, _, istate = state
        env_set_state(self.env, istate)
        obs, r, _, _ = self.env.step(action)
        istate = env_get_state(self.env)
        return (obs, r, istate)
    
    def observe(self, state, name):
        obs, r, istate = state
        if name == 'x':
            return obs
        raise RuntimeError('No such observation')


# Helper functions to serialize/deserialize
# states
def env_get_state(env):
    if hasattr(env, 'data'):
        return env.data
    elif hasattr(env, 'state'):
        return env.state
    else:
        raise RuntimeError("Unable to save gym state")

# Helper functions to serialize/deserialize
# states
def env_set_state(env, state):
    if hasattr(env, 'data'):
        env.data = state
    elif hasattr(env, 'state'):
        env.state = state

def builder():
    return GymEnvironment