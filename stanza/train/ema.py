import jax
import jax.numpy as jnp

from stanza.dataclasses import dataclass

@dataclass(jax=True)
class EmaHook:
    decay: float = 0.75

    def init(self, state):
        return state.fn_params, state

    def __call__(self, hs, state):
        curr_value = (1 + state.iteration)/(10 + state.iteration)
        decay = jnp.minimum(self.decay, curr_value)
        omd = 1 - decay
        hs = jax.tree_util.tree_map(lambda x, y:  omd*y + decay*x, 
                                    hs, state.fn_params)
        return hs, state