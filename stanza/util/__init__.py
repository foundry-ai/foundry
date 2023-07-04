import jax
import chex
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
    return jax.tree_util.tree_map(lambda x: x.shape, x)

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
    hook_states: List[Any]
    last_stats: Any

@jax.jit
def _run_hooks(state, hooks):
    new_hook_states = []
    if state.hook_states is None:
        state = replace(state,
            hook_states=[None] * len(hooks))
    for h, hs in zip(hooks, state.hook_states):
        hs, state = h(hs, state)
        new_hook_states.append(hs)
    state = replace(state, hook_states=new_hook_states)
    return state

def loop(update_fn, loop_state, hooks=[], auto_increment=True):
    chex.assert_scalar_positive(loop_state.max_iterations)
    def loop_body(x):
        x = update_fn(x)
        if auto_increment:
            x = replace(x, iteration=x.iteration + 1)
        _run_hooks(x, hooks)
        return x
    loop_state = _run_hooks(loop_state, hooks)
    loop_state = loop_body(loop_state)
    loop_state = jax.lax.while_loop(lambda x: x.iteration < x.max_iterations,
                                loop_body, loop_state)
    return loop_state