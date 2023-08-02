from stanza.dataclasses import dataclass, replace, field
from typing import List, Any

import jax

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
        if hasattr(h, 'init'):
            hs, state = h.init(state)
            new_hook_states.append(hs)
        else:
            new_hook_states.append(None)
    state = replace(state, hook_states=new_hook_states)
    return state

@jax.jit
def run_hooks(state):
    new_hook_states = []
    if state.hook_states is None:
        state = replace(state,
            hook_states=[None] * len(state.hooks))
    for h, hs in zip(state.hooks, state.hook_states):
        hs, state = h(hs, state)
        new_hook_states.append(hs)
    state = replace(state, hook_states=new_hook_states)
    return state