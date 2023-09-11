from stanza.data import Data

from typing import Any, List
from stanza.dataclasses import dataclass, field, replace

def chunk_data(data, chunk_size=1,
                     start_padding=0, end_padding=0):
    # Return a dataset that turns each trajectory
    # into a ChunkedTrajectory!
    return ChunkedData(data, chunk_size, 
                            start_padding, end_padding)

import jax
import jax.numpy as jnp

@dataclass(jax=True)
class ChunkIterator:
    iterators: List[Any]
    off_end: int

@dataclass(jax=True)
class ChunkedData(Data):
    trajectory: Data
    chunk_size: int = field(default=1, jax_static=True)
    start_padding: int = field(default=0, jax_static=True)
    end_padding: int = field(default=0, jax_static=True)

    # Chunked iterators are iterators into the
    # trajectory
    @property
    def start(self):
        if self.start_padding >= self.chunk_size:
            raise ValueError("Too much start padding!")
        start = self.trajectory.start
        padded = jax.tree_util.tree_map(
            lambda s: jnp.repeat(jnp.expand_dims(s, 0), self.start_padding + 1),
            start
        )
        def scan_fn(i, _):
            n = self.trajectory.next(i)
            return n, n
        _, post = jax.lax.scan(scan_fn, start, None,
                    length=self.chunk_size - self.start_padding - 1)
        # stack padded, post
        it = jax.tree_util.tree_map(
            lambda p, s: jnp.concatenate((p, s)), padded, post
        )
        return ChunkIterator(it, 0)

    def remaining(self, iterator):
        end_iterator = jax.tree_util.tree_map(
            lambda s: s[-1], iterator.iterators)
        return self.trajectory.remaining(end_iterator) + self.end_padding

    def is_end(self, iterator):
        return iterator.off_end > self.end_padding

    def next(self, iterator):
        iterators = iterator.iterators

        last_it = jax.tree_util.tree_map(
            lambda x: x[-1], iterators)

        # propagate the last_it, but cap to the
        # end if necessary
        new_last = self.trajectory.next(last_it)

        new_last, off_end = jax.lax.cond(
            self.trajectory.is_end(new_last),
            lambda: (last_it, iterator.off_end + 1),
            lambda: (new_last, 0)
        )
        new_it = jax.tree_util.tree_map(
            lambda x, n: jnp.roll(x, -1, 0).at[-1].set(n), 
            iterators, new_last)
        return ChunkIterator(new_it, off_end)

    def get(self, iterator):
        vget = jax.vmap(self.trajectory.get)
        return vget(iterator.iterators)