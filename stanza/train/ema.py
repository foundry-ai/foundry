import jax
import jax.numpy as jnp

from stanza.util.dataclasses import dataclass

@dataclass(jax=True)
class EmaHook:
    decay: float = 0.75

    def __call__(self, hs, state):
        if hs is None:
            hs = state.fn_params
        curr_value = (1 + state.total_iteration)/(10 + state.total_iteration)
        decay = jnp.minimum(self.decay, curr_value)
        omd = 1 - decay
        hs = jax.tree_util.tree_map(lambda x, y:  omd*y + decay*x, 
                                    hs, state.fn_params)
        return hs, state