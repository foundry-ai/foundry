from stanza.dataclasses import dataclass, field, replace
from stanza.reporting import Database
from stanza import partial

from jax.experimental.host_callback import barrier_wait
from typing import Callable

import jax
import jax.numpy as jnp
import chex

_HANDLES = {}
_counter = 0

def _log_cb(args, transforms, batch=False):
    handle, data, batch_n = args
    db = _HANDLES[handle.item()]
    # if there is an limit to the batch, get the last batch_n
    # from the buffer
    if batch and batch_n is not None:
        data = jax.tree_map(lambda x: x[-batch_n:], data)
    db.log(data, batch=batch)

def log_every_kth_iteration(k):
    def cond(state):
        return state.iteration % k == 0
    return cond
log_every_iteration = lambda state: True

def log_every_kth_epoch(k):
    def cond(state):
        return jnp.logical_and(state.epoch % k == 0,
                state.epoch_iteration == 0)
    return cond
log_every_epoch = log_every_kth_epoch(1)

@dataclass(jax=True)
class JaxDBHandle(Database):
    id: int

    def log_hook(self, 
            stat_fn=lambda state: state.last_stats,
            log_cond=log_every_iteration,
            *, buffer=100):
        return LoggingHook(self, stat_fn, 
                    log_cond, buffer)

    def log(self, data, batch=False, batch_n=None):
        jax.experimental.host_callback.id_tap(
            partial(_log_cb, batch=batch), (self.id, data, batch_n))

@dataclass(jax=True)
class LoggingHook:
    handle: JaxDBHandle
    stat_fn: Callable

    condition_fn: Callable

    buffer: int = field(jax_static=True)

    def init(self, state):
        # make a buffer for the last stats
        stats = self.stat_fn(state)
        stat_buffer = jax.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x,0), self.buffer, axis=0),
            stats
        )
        return (stat_buffer, jnp.array(0), state.iteration), state

    def __call__(self, hook_state, state):
        if state.last_stats is None:
            return hook_state, state
        stat_buffer, elems, prev_iteration = hook_state

        # add the last stats to the buffer
        def update_buffer(stat_buffer, elems):
            stats = self.stat_fn(state)
            stat_buffer = jax.tree_map(
                lambda x, y: jnp.roll(x, -1, axis=0).at[-1, ...].set(y), 
                stat_buffer, stats)
            return stat_buffer, jnp.minimum(elems + 1, self.buffer)

        should_log = jnp.logical_and(self.condition_fn(state),
                        state.iteration != prev_iteration)
        stat_buffer, elems = jax.lax.cond(should_log,
            update_buffer, lambda x, y: (x, y), stat_buffer, elems)

        done = jnp.logical_and(
            should_log, 
            state.iteration == state.max_iterations
        )
        def do_log():
            self.handle.log(stat_buffer, batch=True, batch_n=elems)
            return 0
        elems = jax.lax.cond(
            jnp.logical_or(elems >= self.buffer, done),
            do_log, lambda: elems)
        new_hook_state = (stat_buffer, elems, state.iteration)
        return new_hook_state, state

class JaxDBScope:
    def __init__(self, db):
        global _counter

        self.id = _counter
        self.db = db
        _counter = _counter + 1
    
    def __enter__(self):
        _HANDLES[self.id] = self.db
        return JaxDBHandle(self.id)
    
    def __exit__(self, *args):
        barrier_wait()
        del _HANDLES[self.id]