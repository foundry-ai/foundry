import jax
import jax.numpy as jnp

from stanza.dataclasses import dataclass, replace
from stanza.util.loop import Hook

from typing import Any

@dataclass(jax=True)
class FinalParams:
    reg_params: Any
    ema_params: Any

@dataclass(jax=True)
class EmaHook(Hook):
    decay: float = 0.75

    def init(self, state):
        return state.fn_params, state

    def run(self, hs, state):
        curr_value = (1 + state.iteration)/(10 + state.iteration)
        decay = jnp.minimum(self.decay, curr_value)
        omd = 1 - decay
        hs = jax.tree_util.tree_map(lambda x, y:  omd*y + decay*x, 
                                    hs, state.fn_params)
        return hs, state
    
    def finalize(self, hook_state, state):
        params = FinalParams(ema_params=hook_state, reg_params=state.fn_params)
        state = replace(state, fn_params=params)
        return hook_state, state