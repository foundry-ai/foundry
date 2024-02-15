from stanza.envs import Environment
import brax.envs as envs

from stanza.dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

@dataclass(jax=True)
class BraxEnv(Environment):
    env : Any = field(jax_static=True)

    def sample_action(self, rng_key):
        return jax.random.uniform(rng_key,
            (self.env.action_size,), minval=-1, maxval=1)

    def sample_state(self, rng_key):
        return self.reset(rng_key)

    def reset(self, rng_key):
        state = self.env.reset(rng_key)
        return state

    def step(self, state, action, rng_key):
        next_state = self.env.step(state, action)
        return next_state
    
    def observe(self, state):
        return state.obs

    def reward(self, state, action, next_state):
        return next_state.reward
    
    def render(self, state, *, width=256, height=256, mode="image", **kwargs):
        if mode == "html":
            from brax.io import html
            if state.obs.ndim == 1:
                state = jax.tree_map(
                    lambda x: jnp.expand_dims(x,0), 
                    state
                )
            states = [jax.tree_map(lambda x: x[i], state).pipeline_state for i in range(state.obs.shape[0])]
            sys = self.env.sys.replace(dt=self.env.dt)
            return html.render(sys, states)
        else:
            raise NotImplementedError("Mode not supported")

    def done(self, state):
        return state.done

def builder(env_type, backend="positional"):
    env_path = env_type.split("/")
    if len(env_path) < 2:
        raise RuntimeError("Must specify gym environment")
    en = env_path[1]

    env = envs.get_environment(env_name=en, backend=backend)
    return BraxEnv(env)