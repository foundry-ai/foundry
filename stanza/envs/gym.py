from stanza.envs import Environment
from stanza.util.dataclasses import dataclass, field
from typing import Any

import gym
import jax.numpy as jnp

# Has a fancy callback-based
# wrapper for any Gym-based environment

class EnvPool:
    def __init__(self, env_factory):
        self.env_factory = env_factory
        self.envs = {}
        self.env_id = 0
    
    def create(self):
        e = self.env_factory()
        self.env_id = self.env_id + 1
        self.envs[self.env_id] = e
        e.reset()
        return self.env_id
    
    def step(self, env_id, action):
        e = self.envs[jnp.array(env_id).item()]
        e.step(action)

@dataclass(jax=True)
class GymEnv(Environment):
    # a function which produces Gym() environments
    # A gym environment is created per reset() call
    envs: EnvPool = field(jax_static=True)

    def reset(self, rng_key):
        pass

@dataclass(jax=True)
class GymState:
    t : int
    env_id : int
    obs : Any

def builder(env_type, *args, **kwargs):
    env_path = env_type.split("/")
    if len(env_path) < 2:
        raise RuntimeError("Must specify gym environment")
    en = env_path[1]
    return GymEnv(lambda: gym.make(en))