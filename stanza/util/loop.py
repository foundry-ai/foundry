from stanza.dataclasses import dataclass, replace, field
from stanza.util.logging import logger
from typing import Any, Callable

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

def flat_items(d, prefix=''):
    for (k,v) in d.items():
        if isinstance(v, dict):
            yield from flat_items(v, prefix=f'{prefix}{k}.')
        else:
            yield (f'{prefix}{k}',v)

@dataclass
class LoggerHook:
    condition_fn: Any
    stat_fn: Callable = lambda state: state.last_stats

    def init(self, state):
        return state.iteration, state

    def __call__(self, hs, state):
        def log():
            stats = self.stat_fn(state)
            flat_stats = dict(flat_items(stats))
            s = [f"{k}: {{}}" for k in flat_stats.keys()]
            fmt = "\n".join(s)
            logger.info("Iteration {}:\n" + fmt, state.iteration, *flat_stats.values())
        jax.lax.cond(jnp.logical_and(
            self.condition_fn(state),
            state.iteration != hs
        ), log, lambda: None)
        return state.iteration, state