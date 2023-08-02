from stanza.dataclasses import dataclass, field, replace
from stanza.reporting import Database
from stanza import partial

from jax.experimental.host_callback import barrier_wait
from typing import Any, Callable

import jax
import jax.numpy as jnp
import chex

_HANDLES = {}
_counter = 0

def _log_cb(args, transforms, batch=False):
    handle, data, iteration, batch_n = args
    db = _HANDLES[handle.item()]
    # if there is an limit to the batch, get the last batch_n
    # from the buffer
    if batch and batch_n is not None:
        data = jax.tree_map(lambda x: x[-batch_n:], data)
    db.log(data, step=iteration, batch=batch)

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

    def stat_logger(self, 
            stat_fn=lambda stat_state, state: (stat_state, state.last_stats),
            log_cond=log_every_iteration,
            *, buffer=100):
        return StatHook(self, stat_fn, 
                    log_cond, buffer)
    
    def logger_hook(self, log_fn,
            log_cond=log_every_iteration):
        pass

    def log(self, data, step=None, batch=False, batch_n=None):
        jax.experimental.host_callback.id_tap(
            partial(_log_cb, batch=batch), (self.id, data, step, batch_n))

# A generic log callback
@dataclass(jax=True)
class LogHook:
    handle: JaxDBHandle
    log_fn: Callable
    condition_fn: Callable

    def init(self, state):
        if hasattr(self.stat_fn, "init"):
            log_fn_state = self.stat_fn.init(state)
        else:
            log_fn_state = None
        return log_fn_state, state
    
    def __call__(self, hook_state, state):
        def do_log():
            return self.log_fn(hook_state, state, self.handle)
        hook_state = jax.lax.cond(
            self.condition_fn(hook_state, state)
        )
        pass

@dataclass(jax=True)
class StatHook:
    handle: JaxDBHandle
    stat_fn: Callable

    condition_fn: Callable

    buffer: int = field(jax_static=True)

    def init(self, state):
        # make a buffer for the last stats
        if hasattr(self.stat_fn, "init"):
            stat_fn_state = self.stat_fn.init(state)
        else:
            stat_fn_state = None
        stat_fn_state, stats = self.stat_fn(stat_fn_state, state)
        stat_buffer = jax.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x,0), self.buffer, axis=0),
            stats
        )
        return (stat_buffer, jnp.array(0), state.iteration, stat_fn_state), state

    def __call__(self, hook_state, state):
        if state.last_stats is None:
            return hook_state, state
        stat_buffer, elems, prev_iteration, stat_fn_state = hook_state

        # add the last stats to the buffer
        def update_buffer(stat_buffer, elems, stat_fn_state):
            stat_fn_state, stats = self.stat_fn(stat_fn_state, state)
            stat_buffer = jax.tree_map(
                lambda x, y: jnp.roll(x, -1, axis=0).at[-1, ...].set(y), 
                stat_buffer, stats)
            return stat_buffer, \
                jnp.minimum(elems + 1, self.buffer), stat_fn_state

        should_log = jnp.logical_and(self.condition_fn(state),
                        state.iteration != prev_iteration)
        stat_buffer, elems, stat_fn_state = jax.lax.cond(should_log,
            update_buffer, lambda x, y, z: (x, y, z),
            stat_buffer, elems, stat_fn_state)

        done = jnp.logical_and(
            should_log, 
            state.iteration == state.max_iterations
        )
        def do_log():
            self.handle.log(stat_buffer, state.iteration, batch=True, batch_n=elems)
            return 0
        elems = jax.lax.cond(
            jnp.logical_or(elems >= self.buffer, done),
            do_log, lambda: elems)
        new_hook_state = (stat_buffer, elems, state.iteration, stat_fn_state)
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