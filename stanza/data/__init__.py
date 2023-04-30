import math
import chex
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp

from jax.tree_util import register_pytree_node_class
from stanza.util.logging import logger
from stanza.util.dataclasses import dataclass, field

from typing import Callable, Any

# We use a float to represent infinite
# datasets so that we can manipulate the
# dataset size and the result is still coherent
INFINITE = jnp.inf

class Data:
    # Please override the following:
    @property
    def start(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement start"
        )

    def remaining(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement remaining"
        )

    def next(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement next"
        )

    def get(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get"
        )

    # Do not need to override
    def is_end(self, iterator):
        return self.remaining(iterator) <= 0

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
        if not isinstance(i, slice):
            raise ValueError("Can only slice, not index, datasets")
        return self.slice(i.start, i.stop, i.step)

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
    def batch(self, n, ret_first=True):
        l = self.length
        n = min(l, n)
        if ret_first:
            first_dim = ((l - 1) % n) + 1
            start = self.start
            batch_first = BatchedData.batch_advance(self, start, first_dim)
            return DataSlice(self, start, first_dim), \
                   BatchedData(self, batch_first, n)
        else:
            chex.assert_equal(self.length % n, 0)
            return BatchedData(self, self.start, n)

    # Flatten and batch are opposites
    # (modulo rounding due to the first batch 
    # being a different size)
    def flatten(self):
        return FlattenedData(self)

    def map(self, fun):
        return MappedData(self, fun)
    
    # Not all datasets implement shuffle!
    def shuffle(self, key):
        raise NotImplementedError("Dataset does not implement shuffle()")
    
    @staticmethod
    def from_pytree(data):
        return PyTreeData(data)
    
# Type alias for iterators
Iterator = Any

@dataclass(jax=True)
class PyTreeData(Data):
    data: jnp.array = None

    @property
    def start(self):
        return 0
    
    def remaining(self, iter):
        nums_tree = tree_util.tree_map(lambda x: jnp.shape(x)[0], self.data)
        all_nums = tree_util.tree_flatten(nums_tree)[0]
        num = all_nums[0]
        return num - iter

    def next(self, iter):
        return iter + 1

    def get(self, iter):
        return tree_util.tree_map(lambda x: x[iter], self.data)

    # override slice/batch/flatten transformations
    # to just reshape for efficiency
    def slice(self, start, stop, step=1):
        data = jax.tree_util.tree_map(lambda x: x[start:stop:step], self.data)
        return PyTreeData(data)

    def batch(self, n, ret_first=True):
        # If length % n == 0 we can
        # return this as just a side-case of a 
        # batch dataset
        l = self.length
        n = min(l, n)
        if ret_first:
            first_dim = ((l - 1) % n) + 1
            first_batch = jax.tree_util.tree_map(
                lambda x: x[:first_dim], self.data
            )
            rest = jax.tree_util.tree_map(
                lambda x: x[first_dim:], self.data
            )
            # reshape the rest into the appropriate
            # batch shape
            rest = jax.tree_util.tree_map(
                lambda x: x.reshape((-1, n) + x.shape[1:]),
                rest
            )
            # The rest is just a PytreeData of PytreeDatas!
            return PyTreeData(first_batch), \
                PyTreeData(PyTreeData(rest))
        else:
            chex.assert_equal(self.length % n, 0)
            data = jax.tree_util.tree_map(
                lambda x: x.reshape((-1, n) + x.shape[1:]),
                self.data 
            )
            return PyTreeData(PyTreeData(data))

    def flatten(self):
        def reshape(x):
            return x.reshape((-1,) + x.shape[2:])
        data = jax.tree_util.tree_map(reshape, self.data)
        return PyTreeData(data)

    # This dataset type is shuffleable
    def shuffle(self, key):
        nums_tree = tree_util.tree_map(lambda x: jnp.shape(x)[0], self.data)
        all_nums = tree_util.tree_flatten(nums_tree)[0]
        num = all_nums[0]

        idxs = jax.random.permutation(key, jnp.arange(num))
        data = jax.tree_util.tree_map(lambda x: x[idxs,...], self.data)
        return PyTreeData(data)
    
    @staticmethod
    def from_data(data, start=None):
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

# Returns a slice into another
# data
@dataclass(jax=True)
class DataSlice(Data):
    data: Data
    start_iter: Iterator
    n : int = field(jax_static=True)

    # amount to advance underlying
    # iterator per next() call
    skip: int = field(default=1, jax_static=True)

    @property
    def start(self):
        return (self.start_iter, 0)

    def next(self, iterator):
        it, i = iterator
        next_it = jax.lax.fori_loop(0, self.skip, self.data.next, it)
        return (i + 1, next_it)

    def remaining(self, iterator):
        it, i = iterator
        return self.n - i
    
    def get(self, iterator):
        it, i = iterator
        return self.data.get(it)

# n should not be part
# of the jax tree
@dataclass(jax=True)
class BatchedData(Data):
    data: Data
    start_iter: Iterator
    n : int = field(jax_static=True)

    @property
    def start(self):
        return self.start_iter
    
    def get(self, iterator):
        return Slice(self.data, iterator, n)
    
    def next(self, iterator):
        # advance the iterator by n
        return jax.lax.fori_loop(0, self.n, self.data.next, iterator)

    def remaining(self, iterator):
        return self.data.remaining(iterator) // self.n 

@dataclass(jax=True)
class FlattenedData(Data):
    dataset: Data

    @staticmethod
    def sample_dim(sample):
        t, _ = jax.tree_util.tree_flatten(sample)
        return t[0].shape[0]

    @property
    def start(self):
        start = self.data.start
        sample = self.data.get(start)
        next = self.data.next(start)
        return sample, 0, next
    
    def remaining(self, iterator):
        sample, i, n = iterator
        w = FlatDataset.sample_dim(sample)
        return (w - i) + w*self.data.remaining(n)

    def get(self, iterator):
        sample, i, _ = iterator
        return jax.tree_util.tree_map(lambda x: x[i], sample)

    def next(self, iterator):
        sample, i, next_it = iterator
        w = FlatDataset.sample_dim(sample)
        return jax.lax.cond(i < w, lambda: (sample, i + 1, next_it),
                     lambda: (self.data.get(next_it), 0, self.data.next(next_it)))