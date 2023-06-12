from stanza.envs import Environment
from stanza.policies import PolicyOutput

from stanza.util.dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np
import gymnasium
import jax

_COUNTER = 0
_GYMS = {}

# Has a fancy callback-based
# wrapper for any Gym-based environment
class GymWrapper(Environment):
    def _make_env(self, rng_key):
        pass

    def _reset_callback(self, rng_key):
        global _COUNTER
        global _GYMS
        _COUNTER += 1
        rng_key, sk = jax.random.split(rng_key)
        gym = self._make_env(sk)
        _GYMS[_COUNTER] = gym
        seed = jax.random.randint(rng_key, (), 0, 2**30)
        obs, _ = gym.reset(seed=seed.item())
        if obs.dtype == np.float64:
            obs = obs.astype(np.float32)
        img = gym.render()
        state = GymState(jnp.array(_COUNTER), obs, 
                    jnp.array(0.), jnp.array(False), img)
        return state

    def reset(self, rng_key):
        i = jax.random.randint(rng_key, (), 0, 2**30)
        # get a state sample
        state = self._reset_callback(i)
        state = jax.pure_callback(
            type(self)._reset_callback,
            state, self, i
        )
        return state

    def _step_callback(self, state, action):
        gym = _GYMS[state.env_id.item()]
        if action is None:
            action = gym.action_space.sample()
            action = np.zeros_like(action)
        obs, rew, term, _, _ = gym.step(action)
        if obs.dtype == np.float64:
            obs = obs.astype(np.float32)
        img = gym.render()
        state = GymState(state.env_id, jnp.array(obs),
                         jnp.array(rew), jnp.array(term), img)
        return state

    def step(self, state, action, rng_key):
        return jax.pure_callback(type(self)._step_callback, 
                                  state, self, state, action)
    
    def render(self, state):
        return state.render
   
    def teleop_policy(self, interface):
        def policy(input):
            return PolicyOutput(None)
        return policy

@dataclass(jax=True)
class GymEnv(GymWrapper):
    env_builder : Callable = field(jax_static=True)

    def _make_env(self):
        return self.env_builder()

@dataclass(jax=True)
class GymState:
    env_id: int
    obs : Any
    reward: float
    terminated: bool
    render: Any = None

def builder(env_type, *args, **kwargs):
    env_path = env_type.split("/")
    if len(env_path) < 2:
        raise RuntimeError("Must specify gym environment")
    en = env_path[1]
    return GymEnv(lambda: gymnasium.make(en, render_mode="rgb_array"))