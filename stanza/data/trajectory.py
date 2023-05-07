from stanza.data import Data

from stanza.util.dataclasses import dataclass, field, replace
from typing import Any, List

import jax
import jax.numpy as jnp

@dataclass(jax=True)
class Timestep:
    observation: Any
    action: Any

# A dataset of Trajectories
# has a get() method which
# returns a Dataset per-trajectory
# which can themselves be iterated
# to get each of the state entries
@dataclass(jax=True)
class TrajectoryIndices:
    start_index: int
    end_index: int

@dataclass(jax=True)
class IndexedTrajectoryData(Data):
    trajectory_indices: Data
    timesteps: Data

    @property
    def start(self):
        return self.trajectory_indices.start

    def remaining(self, iterator):
        return self.trajectory_indices.remaining(iterator)

    def is_end(self, iterator):
        return self.trajectory_indices.is_end(iterator)
    
    def next(self, iterator):
        return self.trajectory_indices.next(iterator)

    def advance(self, iterator, n):
        return self.trajectory_indices.advance(iterator, n)
    
    def get(self, iterator):
        idx = self.trajectory_indices.get(iterator)
        iter = self.timesteps.advance(
            self.timesteps.start, idx.start_index)
        return self.timesteps.slice(
            iter, idx.end_index - idx.start_index)

    # Override slice()
    # to slice just trajectory_info
    # We can't slice timesteps due to step,
    # so for now just keep all of the timesteps around
    def slice(self, start_iter, len):
        return IndexedTrajectoryData(
            self.trajectory_indices.slice(
                start_iter, len
            ),
            self.timesteps
        )
    
    def shuffle(self):
        return IndexedTrajectoryData(
            self.trajectory_indices.shuffle(),
            self.timesteps
        )

def chunk_trajectory(trajectory, obs_chunk_size=None, 
                        action_chunk_size=None):
    # Return a dataset that turns each trajectory
    # into a ChunkedTrajectory!
    return ChunkedTrajectory(trajectory,
            obs_chunk_size, action_chunk_size)

@dataclass(jax=True)
class ChunkIterator:
    iterators: List[Any]
    off_end: int

@dataclass(jax=True)
class ChunkedTrajectory(Data):
    trajectory: Data
    obs_chunk_size: int = field(default=None, jax_static=True)
    action_chunk_size: int = field(default=None, jax_static=True)

    # Chunked iterators are iterators into the
    # trajectory
    @property
    def start(self):
        i = self.obs_chunk_size \
            if self.obs_chunk_size is not None else 1
        o = self.action_chunk_size \
            if self.action_chunk_size is not None else 1

        start = self.trajectory.start
        def scan_fn(i, _):
            n = self.trajectory.next(i)
            return n, n
        _, post = jax.lax.scan(scan_fn, start, None, length=o-1)
        # stack start i times, followed by *post*
        it = jax.tree_util.tree_map(
            lambda s, p: jnp.concatenate(
                (jnp.repeat(jnp.expand_dims(s, 0), i), p)
            ), start, post
        )
        return ChunkIterator(it, 0)

    def remaining(self, iterator):
        i = self.obs_chunk_size \
            if self.obs_chunk_size is not None else 1
        mid_iterator = jax.tree_util.tree_map(
            lambda s: s[i - 1], iterator.iterators)
        return self.trajectory.remaining(mid_iterator)

    def is_end(self, iterator):
        o = self.action_chunk_size \
            if self.action_chunk_size is not None else 1
        return iterator.off_end > o - 1

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
        i = self.obs_chunk_size \
            if self.obs_chunk_size is not None else 1
        input_iterators = jax.tree_util.tree_map(
            lambda x: x[:i], iterator.iterators
        )
        output_iterators = jax.tree_util.tree_map(
            lambda x: x[i-1:], iterator.iterators
        )
        vget = jax.vmap(self.trajectory.get)
        input = vget(input_iterators).observation
        output = vget(output_iterators).action
        return Timestep(input, output)