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
INFINITE = float("inf")

# Dataset iterators should be immutable!
class Dataset:
    @property
    def start(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement start"
        )

    def remaining(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement remaining"
        )

    # Will return the next iterator.
    # If the iterator is at the end, the function does not need to return
    # the same iterator, but it must return an iterator for which is_end is also
    # true
    def next(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement next"
        )

    # This will actually do the computation
    # to get a given iterator
    def get(self, iterator):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get"
        )

    # NOT NECESSARY TO OVERRIDE

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
        return SlicedDataset(self, start, stop, step)

    # Will return the first batch separately.
    # The first batch may be jagged if length % n != 0
    def batch(self, n, ret_first=True):
        l = self.length
        n = min(l, n)
        if ret_first:
            first_dim = ((l - 1) % n) + 1
            batch, i = BatchDataset.rollout_batch(self.start, first_dim)
            batch = BatchDataset.get_batch(batch)
            return batch, BatchDataset(self, i, n)
        else:
            chex.assert_equal(self.length % n, 0)
            return BatchDataset(self, self.start, n)

    # Flatten and batch are opposites (modulo rounding due to n)
    def flatten(self):
        return FlatDataset(self)

    def map(self, fun):
        return MappedDataset(self, fun)
    
    # Not all datasets implement shuffle!
    def shuffle(self, key):
        raise NotImplementedError("Dataset does not implement shuffle()")
    
    @staticmethod
    def from_pytree(data):
        return PyTreeDataset(data)
    
# Type alias for iterators
Iterator = Any

@dataclass(jax=True)
class PyTreeDataset(Dataset):
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
        return PyTreeDataset(data)

    def batch(self, n, ret_first=True):
        # If length % n == 0 we can
        # return this as just a side-case of a 
        # batch dataset
        l = self.length
        n = min(l, n)
        if ret_first:
            first_dim = ((l - 1) % n) + 1
            first_batch = jax.tree_util.tree_map(
                lambda x: x[:first_dim].reshape((first_dim, -1) + x.shape[1:]), 
                self.data)
            batches = jax.tree_util.tree_map(
                lambda x: x[first_dim:].reshape((-1, n) + x.shape[1:]), 
                self.data)
            return first_batch, PyTreeDataset(batches)
        else:
            chex.assert_equal(self.length % n, 0)
            return BatchDataset(self, n)

    def flatten(self):
        def reshape(x):
            return x.reshape((-1,) + x.shape[2:])
        data = jax.tree_util.tree_map(reshape, self.data)
        return PyTreeDataset(data)

    # This dataset type is shuffleable
    def shuffle(self, key):
        nums_tree = tree_util.tree_map(lambda x: jnp.shape(x)[0], self.data)
        all_nums = tree_util.tree_flatten(nums_tree)[0]
        num = all_nums[0]

        idxs = jax.random.permutation(key, jnp.arange(num))
        data = jax.tree_util.tree_map(lambda x: x[idxs,...], self.data)
        return PyTreeDataset(data)
    
    @staticmethod
    def from_dataset(dataset, start=None):
        if isinstance(dataset, PyTreeDataset):
            return dataset
        start = start or dataset.start
        if not math.isfinite(dataset.remaining(start)):
            raise ValueError("Cannot read in an infinite dataset")
        # Scan out the iterators
        def scan_fn(iter, _):
            return dataset.next(iter), iter
        _, iters = jax.lax.scan(scan_fn, start, None, length=dataset.remaining(start))
        # in parallel fetch the iterators...
        data = jax.vmap(dataset.get)(iters)
        return PyTreeDataset(data)

@dataclass(jax=True)
class MappedDataset(Dataset):
    dataset: Dataset
    fun: Callable

    @property
    def start(self):
        return self.dataset.start
    
    def remaining(self, iterator):
        return self.dataset.remaining(iterator)
    
    @property
    def length(self):
        return self.dataset.length

    def next(self, iterator):
        return self.dataset.next(iterator)

    def get(self, iterator):
        it = self.dataset.get(iterator)
        return self.fun(it)
    
    # override slice, shuffle, batch transformations
    # to happen pre-map since this is more efficient
    def slice(self, start, stop, step):
        return MappedDataset(self.dataset.slice(start,stop, step), self.fun)
    
    def batch(self, n):
        return MappedDataset(self.dataset.batch(n), jax.vmap(self.fun))
    
    def shuffle(self, rng_key):
        return MappedDataset(self.dataset.shuffle(rng_key), self.fun)

# n should not be part
# of the jax tree
@dataclass(jax=True)
class BatchDataset(Dataset):
    dataset: Dataset
    start_iter: Iterator
    n : int = field(jax_static=True)

    # Static helper methods
    @staticmethod
    def rollout_batch(dataset, sub_iter, n):
        next, d = jax.lax.scan(lambda c, _: (dataset.next(c), c), sub_iter,
            None, length=n)
        return d, next

    @staticmethod
    def get_batch(dataset, sub_iters):
        return jax.vmap(dataset.get, sub_iters)

    @property
    def start(self):
        batch, n = BatchDataset.rollout_batch(self.dataset, self.start_iter, self.n)
        return batch, n
    
    def get(self, iterator):
        batch, _ = iterator
        return BatchDataset.get_batch(self.dataset, batch)
    
    def next(self, iterator):
        return BatchDataset.rollout_batch(self.dataset, iterator[1], self.n)

    def remaining(self, iterator):
        first = jax.tree_util.tree_map(lambda x: x[0], iterator)
        return self.dataset.remaining(first) // self.n 

    def tree_flatten(self):
        return (self.dataset,self.start_iter), self.n

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)

@dataclass(jax=True)
class FlatDataset(Dataset):
    dataset: Dataset

    @staticmethod
    def sample_dim(sample):
        t, _ = jax.tree_util.tree_flatten(sample)
        return t[0].shape[0]

    @property
    def start(self):
        start = self.dataset.start
        sample = self.dataset.get(start)
        next = self.dataset.next(start)
        return sample, 0, next
    
    def remaining(self, iterator):
        sample, i, n = iterator
        w = FlatDataset.sample_dim(sample)
        return (w - i) + w*self.dataset.remaining(n)

    def get(self, iterator):
        sample, i, _ = iterator
        return jax.tree_util.tree_map(lambda x: x[i], sample)

    def next(self, iterator):
        sample, i, next_it = iterator
        w = FlatDataset.sample_dim(sample)
        return jax.lax.cond(i < w, lambda: (sample, i + 1, next_it),
                     lambda: (self.dataset.get(next_it), 0, self.dataset.next(next_it)))

@dataclass(jax=True)
class ShufflingDataset(Dataset):
    dataset: Dataset
    buffer_size: int

    @property
    def start(self):
        return self.dataset.start
    
    # Override so that start() isn't
    # being called unnecessarily
    @property
    def length(self):
        return self.dataset.length


@register_pytree_node_class
class SlicedDataset(Dataset):
    def __init__(self, dataset, start, stop, step):
        self._dataset = dataset
        self._start = 0
        self._stop = stop
        self._step = 1

        if self._stop < 0:
            self._stop = dataset.length - self._stop
        if self._start < 0:
            self._start = dataset.length - self._start
    
    @property
    def start(self):
        return self._dataset.start, 0

    def next(self, iter):
        return self._dataset.next(iter[0]), iter[1] + 1

    def remaining(self, iter):
        r = self._dataset.remaining(iter[0])
        if self._stop is not None:
            return min(self._stop - iter[1], r)
        else:
            return r

    def get(self, iter):
        return self._dataset.get(iter[0])
    
    def tree_flatten(self):
        return (self._dataset,), (self._start, self._stop, self._step)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)