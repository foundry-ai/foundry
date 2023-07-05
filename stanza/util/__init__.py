import jax
import jax.numpy as jnp
from stanza.dataclasses import dataclass, replace
from typing import List, Any

def vmap_ravel_pytree(x):
    i = jax.tree_util.tree_map(lambda x: x[0], x)
    _, uf = jax.flatten_util.ravel_pytree(i)

    def flatten(x):
        return jax.flatten_util.ravel_pytree(x)[0]
    flat = jax.vmap(flatten)(x)
    uf = jax.vmap(uf)
    return flat, uf

def extract_shifted(xs):
    earlier_xs = jax.tree_map(lambda x: x[:-1], xs)
    later_xs = jax.tree_map(lambda x: x[1:], xs)
    return earlier_xs, later_xs

def shape_tree(x):
    return jax.tree_util.tree_map(lambda x: jnp.array(x).shape, x)

def shape_dtypes(x):
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        x
    )

# Loop tools,
# forms the basis of the while loop
# functions
@dataclass(jax=True)
class LoopState:
    iteration: int
    max_iterations: int

    hooks : List[Any]
    hook_states: List[Any]

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