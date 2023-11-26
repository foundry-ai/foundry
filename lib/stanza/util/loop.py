from stanza.dataclasses import dataclass, replace, field
from typing import Any, Callable


import abc
import jax
import jax.numpy as jnp

# Loop tools,
# forms the basis of the while loop
# functions
@dataclass(jax=True)
class LoopState:
    iteration: int
    max_iterations: int
    hooks : Any
    hook_states: Any
    last_stats: Any

class Hook(abc.ABC):
    def init(self, state):
        return None, state
    
    def run(self, hook_state, state):
        ...
    
    def finalize(self, hook_state, state):
        return hook_state, state

def loop(step_fn, state, jit=True):
    if jit:
        state = jax.lax.while_loop(
            lambda s: (s.iteration < s.max_iterations) \
                if s.max_iterations is not None else True,
            step_fn, state)
    elif state.max_iterations is not None:
        while state.iteration < state.max_iterations:
            state = step_fn(state)
    else:
        while True:
            state = step_fn(state)
    return state

@jax.jit
def init_hooks(state):
    new_hook_states = []
    for h in state.hooks:
        hs, state = h.init(state)
        new_hook_states.append(hs)
    state = replace(state, hook_states=new_hook_states)
    return state

@jax.jit
def run_hooks(state):
    new_hook_states = []
    for h, hs in zip(state.hooks, state.hook_states):
        hs, state = h.run(hs, state)
        new_hook_states.append(hs)
    state = replace(state, hook_states=new_hook_states)
    return state

@jax.jit
def finish_hooks(state):
    new_hook_states = []
    for h, hs in zip(state.hooks, state.hook_states):
        hs, state = h.finalize(hs, state)
        new_hook_states.append(hs)
    state = replace(state, hook_states=new_hook_states)
    return state

def every_kth_iteration(k):
    def cond(state):
        return state.iteration % k == 0
    return cond
every_iteration = lambda state: True

def every_kth_epoch(k):
    def cond(state):
        return jnp.logical_and(state.epoch % k == 0,
                state.epoch_iteration == 0)
    return cond
every_epoch = every_kth_epoch(1)
