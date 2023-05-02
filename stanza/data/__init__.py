import math
import chex
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp
import numpy as np

from jax.tree_util import register_pytree_node_class
from stanza.util.logging import logger
from stanza.util.dataclasses import dataclass, field

from typing import Callable, Any

# We use a float to represent infinite
# datasets so that we can manipulate the
# dataset size and the result is still coherent
INFINITE = jnp.inf
UNKNOWN = jnp.nan

class Data:
    # Please override the following:
    @property
    def start(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement start"
        )

    # Returns number of elements remainng
    # in the dataset *including* the provided
    # iterator. i.e. if the iterator is at the end
    # of the dataset, this should be 0.
    def remaining(self, iterator):
        return UNKNOWN

    def is_end(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement is_end"
        )

    def next(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement next"
        )

    def get(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get"
        )

    # These should NOT be overridden:

    @property
    def length(self):
        return self.remaining(self.start)

    def __iter__(self):
        it = self.start
        while not self.is_end(it):
            o = self.get(it)
            yield o
            it = self.next(it)
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.slice(i.start, i.stop, i.step)
        else:
            it = jax.lax.fori_loop(0, i, self.next, self.start)
            return self.get(it)

    # ------------------------- Transformations --------------------------------
    def slice(self, start, stop, step=1):
        assert step > 0
        if start < 0:
            start = self.length + start
        if stop < 0:
            stop = self.length + stop
        start_it = jax.lax.fori_loop(0, start, self.next, self.start)
        return DataSlice(self, start_it, stop - start, step)

    # Will return the first batch separately.
    # The first batch may be jagged if length % n != 0
    def batch(self, n):
        return BatchedData(self, n)

    # Flatten and batch are opposites
    def flatten(self):
        raise NotImplementedError("Dataset does not implement flatten()")

    def map(self, fun):
        return MappedData(self, fun)
    
    # Not all datasets implement shuffle!
    def shuffle(self, key):
        raise NotImplementedError("Dataset does not implement shuffle()")
    
    @staticmethod
    def from_pytree(data, n=None):
        nums_tree = tree_util.tree_map(lambda x: jnp.shape(x)[0], data)
        all_nums = tree_util.tree_flatten(nums_tree)[0]
        num = all_nums[0]
        return PyTreeData(data, num if n is None else n)
    
# Type alias for iterators
Iterator = Any

@dataclass(jax=True)
class PyTreeData(Data):
    data: jnp.array
    # n can be less than the data shape!
    # This allows PyTreeData to represent
    # uneven size batches
    n: int

    @property
    def start(self):
        return 0

    def remaining(self, iter):
        return self.n - iter
    
    def is_end(self, iter):
        return iter >= self.n

    def next(self, iter):
        return iter + 1

    def get(self, iter):
        return tree_util.tree_map(lambda x: x[iter], self.data)

    def batch(self, n):
        dim = tree_util.tree_flatten(self.data)[0][0].shape[0]
        # get the amount of padding we need
        pdim = (((n - dim) % n) + n) % n
        batches = (dim + pdim)//n
        def make_batches(x):
            padding = jnp.repeat(jnp.expand_dims(x[-1],0), pdim)
            padded = jnp.concatenate((x,padding), axis=0)
            return padded.reshape((-1,n) + padded.shape[1:])
        batched = jax.tree_util.tree_map(
            make_batches, self.data
        )
        batch_sizes = n*jnp.ones((batches,))
        # modify the last batch size
        batch_sizes = batch_sizes.at[-1].set(n - pdim)
        return PyTreeData(
            PyTreeData(batched, batch_sizes),
            batches
        )

    def flatten(self):
        if isinstance(self.data, PyTreeData):
            data = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[2:]),
                self.data.data)
            # sum all of the sub-sizes
            n = self.data.n.sum()
            return PyTreeData(data, n)
        raise NotImplementedError("Can only flatten() recursive PyTreeData datasets")

    # This dataset type is shuffleable
    def shuffle(self, key):
        import stanza.util.random
        dim = tree_util.tree_flatten(self.data)[0][0].shape[0]
        idx = stanza.util.random.permutation(key, dim, self.n)
        data = jax.tree_util.tree_map(
            lambda x: jnp.take(x, idx, axis=0, unique_indices=True),
        self.data)
        return PyTreeData(data, self.n)

    @staticmethod
    def from_data(data, start=None, buffer_size=None):
        if isinstance(data, PyTreeData):
            return data
        start = start or data.start
        if not math.isfinite(data.remaining(start)):
            raise ValueError("Cannot read in an infinite dataset")
        # Scan out the iterators
        def scan_fn(iter, _):
            return data.next(iter), iter
        _, iters = jax.lax.scan(scan_fn, start, None,
                    length=data.remaining(start))
        # in parallel fetch the iterators...
        data = jax.vmap(data.get)(iters)
        return PyTreeData(data)

@dataclass(jax=True)
class MappedData(Data):
    data: Data
    fun: Callable

    @property
    def start(self):
        return self.data.start
    
    def is_end(self, iterator):
        return self.data.is_end(iterator)
    
    def remaining(self, iterator):
        return self.data.remaining(iterator)
    
    @property
    def length(self):
        return self.data.length

    def next(self, iterator):
        return self.data.next(iterator)

    def get(self, iterator):
        it = self.data.get(iterator)
        return self.fun(it)
    
    # override slice, shuffle, batch transformations
    # to happen pre-map application since this is 
    # generally more efficient (i.e for PyTreeData)
    def slice(self, start, stop, step):
        return MappedData(self.data.slice(start,stop, step), self.fun)
    
    def batch(self, n):
        return MappedData(self.data.batch(n), jax.vmap(self.fun))
    
    def shuffle(self, rng_key):
        return MappedData(self.data.shuffle(rng_key), self.fun)

@dataclass(jax=True)
class DataSlice(Data):
    data: Data
    start_iter: Iterator
    n : int
    # amount to advance underlying
    # iterator per next() call
    step: int = field(default=1, jax_static=True)

    @property
    def start(self):
        return (self.start_iter, 0)

    def next(self, iterator):
        it, i = iterator
        next_it = jax.lax.fori_loop(0, self.step, self.data.next, it)
        return (i + 1, next_it)

    def remaining(self, iterator):
        _, i = iterator
        dr = self.data.remaining(i)
        r = self.n - i
        # return whichever of dr, r is less
        return jnp.minimum(dr, r)

    def get(self, iterator):
        it, _ = iterator
        return self.data.get(it)

# n should not be part
# of the jax tree
@dataclass(jax=True)
class BatchedData(Data):
    data: Data
    n : int = field(jax_static=True)

    @property
    def start(self):
        return self.data.start
    
    def is_end(self, iterator):
        return self.data.is_end(iterator)

    def remaining(self, iterator):
        return self.data.remaining(iterator) // self.n 

    def get(self, iterator):
        return DataSlice(self.data, iterator, self.n)
    
    def next(self, iterator):
        # advance the iterator by n
        return jax.lax.fori_loop(0, self.n, self.data.next, iterator)
    
    # Override flatten()
    # to just return the original dataset!
    def flatten(self):
        return self.data