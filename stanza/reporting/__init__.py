import numpy as np
import jax.numpy as jnp

from stanza.dataclasses import dataclass, field
from stanza.util.loop import Hook

import jax
import stanza.util.loop
import stanza

from typing import Any, Callable

def _iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False

@dataclass(frozen=True)
class Media:
    pass

@dataclass(frozen=True)
class Figure(Media):
    fig: Any
    height: int = None
    width: int = None

@dataclass(frozen=True)
class Video(Media):
    data: np.array
    fps: int = 28

@dataclass(frozen=True)
class Image(Media):
    data: np.array

def flatten_media_tree(tree):
    return jax.tree_util.tree_flatten_with_path(tree,
        is_leaf=lambda x: isinstance(x, Media)
    )[0]

def map_media_tree(f, tree, *trees):
    return jax.tree_util.tree_map(f, tree, *trees,
        is_leaf=lambda x: isinstance(x, Media)
    )

class Backend:
    def create(self):
        raise NotImplementedError()

    def open(self, id):
        raise NotImplementedError()

    def find(self, **tags):
        raise NotImplementedError()

    @staticmethod
    def get(url):
        if url.startswith("local://"):
            from stanza.reporting.local import LocalBackend
            return LocalBackend(url)
        elif url.startswith("wandb://"):
            from stanza.reporting.wandb import WandbBackend
            return WandbBackend(url)
        elif url.startswith("dummy://"):
            from stanza.reporting.dummy import DummyBackend
            return DummyBackend()
        else:
            raise NotImplementedError(f"Unknown backend {url}")

class BucketBackend:
    pass

class BucketsBackend:
    pass
class Repo:
    def __init__(self, url):
        self._url = url
        self._backend = Backend.get(url)
    
    def open(self, id):
        return Bucket(impl=self._backend.open(id))
    
    def create(self):
        return Bucket(impl=self._backend.create())

    @property
    def buckets(self):
        return self._backend.buckets
    
    def find(self, id=None, **tags):
        ntags = {}
        for (k,v) in tags.items():
            v = {v} if isinstance(v, str) or not _iterable(v) else set(v)
            ntags[k] = v
        return self._backend.find(id=id, **ntags)

    @property
    def url(self):
        return self._url

class Buckets:
    def __init__(self, impl=None):
        self._impl = impl

    @property
    def latest(self):
        return self._impl.latest

    def filter(self, id=None, **tags):
        ntags = {}
        for (k,v) in tags.items():
            v = {v} if isinstance(v, str) or not _iterable(v) else set(v)
            ntags[k] = v
        return self._impl.find(id=id, **ntags)
    
    def __len__(self):
        return len(self._impl)
    
    def __iter__(self):
        return iter(self._impl)

from weakref import WeakValueDictionary

_BUCKET_COUNTER = 0
_BUCKETS = WeakValueDictionary()

class Bucket:
    def __init__(self, *, jid=None, impl=None):
        self._jid = jid
        if impl is not None and jid is None:
            global _BUCKET_COUNTER
            self._jid = _BUCKET_COUNTER
            _BUCKET_COUNTER = _BUCKET_COUNTER + 1
            _BUCKETS[self._jid] = impl
        self.__impl = impl
    
    @property
    def _impl(self):
        if self.__impl is None:
            self.__impl = _BUCKETS[self._jid]
        return self.__impl
    
    @property
    def url(self):
        return self._impl.url

    @property
    def id(self):
        return self._impl.id
    
    @property
    def creation_time(self):
        return self._impl.creation_time

    @property
    def tags(self):
        return self._impl.tags

    @property
    def keys(self):
        return self._impl.keys

    def __contains__(self, name):
        return self._impl.has_key(name)
    
    def tag(self, **tags):
        ntags = {}
        for (k,v) in tags.items():
            v = {v} if isinstance(v, str) or not _iterable(v) else set(v)
            ntags[k] = v
        self._impl.tag(**ntags)
    
    def get(self, key):
        return self._impl.get(key)

    # must have stream=True
    # to log over steps
    def add(self, name, value, *,
            append=False, step=None, batch=False, batch_lim=None):
        if batch and batch_lim is not None:
            data = jax.tree_map(lambda x: x[-batch_lim:], data)
        self._impl.add(name, value,
            append=append, step=step, batch=batch)

    @staticmethod
    def _log_cb(handle, data, iteration, batch_n, batch=False):
        impl = _BUCKETS[handle.item()]
        # if there is an limit to the batch, get the last batch_n
        # from the buffer
        if batch and batch_n is not None:
            data = jax.tree_map(lambda x: x[-batch_n:], data)
        impl.log(data, step=iteration, batch=batch)

    def log(self, data, step=None, batch=False, batch_n=None):
        import jax.experimental
        jax.experimental.io_callback(
            stanza.partial(self._log_cb, batch=batch), (),
            self._jid, data, step, batch_n, ordered=True)

jax.tree_util.register_pytree_node(
    Bucket, lambda x: ((x._jid,), None),
    lambda _, xs: Bucket(jid=xs[0])
)

@dataclass(jax=True)
class BucketLogHook(Hook):
    bucket: Bucket

    stat_fn: Callable = lambda stat_state, state: (stat_state, state.last_stats)
    condition_fn: Callable = stanza.util.loop.every_iteration
    buffer: int = field(default=100, jax_static=True)

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
        iters = jnp.zeros((self.buffer,), dtype=jnp.int32)
        return (stat_buffer, jnp.array(0), iters, state.iteration, stat_fn_state), state

    def run(self, hook_state, state):
        if state.last_stats is None:
            return hook_state, state
        stat_buffer, elems, iters, prev_iteration, stat_fn_state = hook_state

        # add the last stats to the buffer
        def update_buffer(stat_buffer, elems, iters, stat_fn_state):
            stat_fn_state, stats = self.stat_fn(stat_fn_state, state)
            stat_buffer = jax.tree_map(
                lambda x, y: jnp.roll(x, -1, axis=0).at[-1, ...].set(y), 
                stat_buffer, stats)
            iters = jnp.roll(iters, -1, axis=0).at[-1].set(state.iteration)
            return stat_buffer, \
                jnp.minimum(elems + 1, self.buffer), iters, stat_fn_state

        should_log = jnp.logical_and(self.condition_fn(state),
                        state.iteration != prev_iteration)
        stat_buffer, elems, iters, stat_fn_state = jax.lax.cond(should_log,
            update_buffer, lambda x, y, z, w: (x, y, z, w),
            stat_buffer, elems, iters, stat_fn_state)

        done = jnp.logical_and(
            state.iteration == state.max_iterations,
            state.iteration != prev_iteration)

        def do_log():
            self.bucket.log(stat_buffer, iters, batch=True, batch_n=elems)
            return 0
        elems = jax.lax.cond(
            jnp.logical_or(elems >= self.buffer, done),
            do_log, lambda: elems)
        new_hook_state = (stat_buffer, elems, iters, state.iteration, stat_fn_state)
        return new_hook_state, state
