import jax.numpy as jnp
import jax
import wandb

from stanza.util.dataclasses import dataclass, replace
from jax.experimental.host_callback import id_tap, barrier_wait

_REPORTERS = {}
_counter = 0

@dataclass(jax=True)
class WandbCallback:
    iter_interval: int
    reporter_id: int

    @staticmethod
    def cpu_state_callback(args, _):
        rid, state = args
        rid = rid.item()
        _REPORTERS[rid]._handle_state(state)

    def _do_state_callback(self, state):
        cpu_state = replace(state, rng_key=None,
            fn_params=None, fn_state=None, opt_state=None)
        return id_tap(self.cpu_state_callback, (self.reporter_id, cpu_state), result=state)

    def __call__(self, hs, state):
        # Don't load the parameter state to the CPU
        return hs, jax.lax.cond(
            jnp.logical_or(
                jnp.logical_and(state.total_iteration % self.iter_interval == 0,
                            state.last_stats is not None),
                # When we have reached the end, do a callback
                # so that we can finish the progress bars
                state.epoch == state.max_epoch
            ),
            self._do_state_callback,
            lambda x: x, state
        )

class WandbReporter:
    def __init__(self, iter_interval=10):
        self.iter_interval = iter_interval

        global _counter
        _counter = _counter + 1
        self.reporter_id = _counter

    def _handle_state(self, state):
        if state.last_stats is not None:
            wandb.log(state.last_stats)

    def __enter__(self):
        _REPORTERS[self.reporter_id] = self
        return WandbCallback(self.iter_interval, self.reporter_id)
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        barrier_wait()
        del _REPORTERS[self.reporter_id]