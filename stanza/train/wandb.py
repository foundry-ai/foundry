import jax.numpy as jnp
import jax
import wandb

from stanza.dataclasses import dataclass, field, replace
from jax.experimental.host_callback import id_tap, barrier_wait

_REPORTERS = {}
_counter = 0

@dataclass(jax=True)
class WandbCallback:
    iter_interval: int = field(jax_static=True)
    reporter_id: int

    @staticmethod
    def cpu_state_callback(args, _):
        rid, stats = args
        rid = rid.item()
        _REPORTERS[rid]._log_stats(stats)

    def _do_state_callback(self, state, hs):
        return id_tap(self.cpu_state_callback, (self.reporter_id, hs),
                      result=state)

    def __call__(self, hs, state):
        # Don't load the parameter state to the CPU
        if state.last_stats is None:
            return hs, state
        if hs is None:
            hs = jax.tree_util.tree_map(
                lambda x: jnp.zeros((self.iter_interval,) + x.shape),
                state.last_stats
            )
        hs = jax.tree_util.tree_map(
            lambda x, s: jnp.roll(x, 1, 0).at[0].set(s),
            hs, state.last_stats
        )
        return hs, jax.lax.cond(
            jnp.logical_and(state.total_iteration % self.iter_interval == 0,
                        state.last_stats is not None),
            self._do_state_callback,
            lambda x, _: x, state, hs
        )

class WandbReporter:
    def __init__(self, iter_interval=10):
        self.iter_interval = iter_interval

        global _counter
        _counter = _counter + 1
        self.reporter_id = _counter

    def _log_stats(self, stats):
        dim = jax.tree_util.tree_flatten(stats)[0][0].shape[0]
        for i in range(dim):
            step = jax.tree_util.tree_map(lambda x: x[i].item(), stats)
            wandb.log(step)

    def __enter__(self):
        _REPORTERS[self.reporter_id] = self
        return WandbCallback(self.iter_interval, self.reporter_id)
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        barrier_wait()
        del _REPORTERS[self.reporter_id]
