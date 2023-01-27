import math
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp

from jax.tree_util import register_pytree_node_class
from functools import partial as partial

from stanza.logging import logger, pbar

INFINITE = float("inf")

# Dataset iterators should be immutable!
class Dataset:
    @property
    def start(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement start")

    def remaining(self, iterator):
        raise NotImplementedError(f"{self.__class__.__name__} must implement remaining")

    # Will return the next iterator.
    # If the iterator is at the end, the function does not need to return
    # the same iterator, but it must return an iterator for which is_end is also
    # true
    def next(self, iterator):
        raise NotImplementedError(f"{self.__class__.__name__} must implement next")

    # This will actually do the computation
    # to get a given iterator
    def get(self, iterator):
        raise NotImplementedError(f"{self.__class__.__name__} must implement get")


    # NOT NECESSARY TO OVERRIDE

    # Whether an iterator from this dataset has a next
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

    def __getitem__(self, i):
        if not isinstance(i, slice):
            raise ValueError("Can only slice datasets")
        return self.slice(i.start, i.stop, i.step)

    # Transformations
    def slice(self, start, stop, step=1):
        return SlicedDataset(self, start, stop, step)

    def batch(self, n):
        return BatchDataset(self, n)

    # Flatten and batch are opposites (modulo rounding due to n)
    def flatten(self):
        return FlatDataset(self)

    def map(self, func):
        return MappedDataset(self, func)

    # Will read the whole dataset into a PyTreeDataset
    def read(self):
        logger.info("dataset", "Constructing in-memory dataset...")
        start = self.start
        if not math.isfinite(self.remaining(start)):
            raise ValueError("Cannot read in an infinite dataset")
        
        # Scan out the iterators
        def scan_fn(iter, _):
            return self.next(iter), iter
        _, iters = jax.lax.scan(scan_fn, start, None, length=self.remaining(start), unroll=10)

        def fetch_fn(iter):
            data = self.get(iter)
            return data
        # in parallel fetch the iterators...
        data = jax.vmap(fetch_fn)(iters)
        # with pbar('dataset', total=self.remaining(start)) as pb:
        #     iter = self.start
        #     data = []
        #     for i in range(self.remaining(start)):
        #         data.append(self.get(iter))
        #         iter = self.next(iter)
        #         pb.inc()
        #     data = jax.tree_util.tree_map(lambda *args: jnp.concatenate(args), *data)

            # def scan_fn(iter, _):
            #     pb.inc()
            #     return self.next(iter), self.get(iter)
            # _, data = jax.lax.scan(scan_fn, start, None, length=self.remaining(start), unroll=10)

        logger.info("dataset", f"Dataset construction complete...")
        return PyTreeDataset(data)

    # A shortcut for jax scanning through a dataset
    def fold(self, func, base, iter_limit=None):
        @partial(jax.jit, static_argnums=0)
        def wrapped_func(func, state):
            iter, accum, iter_num = state
            item = self.get(iter)
            new_state = func(accum, item)
            iter = self.next(iter)
            return iter, new_state, iter_num + 1
        state = self.start, base, 0
        if iter_limit is not None:
            _, final_state, _ = jax.lax.while_loop(lambda s: not self.is_end(s[0]) and s[3] < iter_limit,
                                            partial(wrapped_func, func), state)
        else:
            _, final_state, _ = jax.lax.while_loop(lambda s: not self.is_end(s[0]),
                                            partial(wrapped_func, func), state)
        return final_state
    
    @staticmethod
    def from_pytree(data):
        return PyTreeDataset(data)


@register_pytree_node_class
class PyTreeDataset(Dataset):
    def __init__(self, data, _num=None):
        self._data = data
        #if _num is None:
        nums_tree = tree_util.tree_map(lambda x: jnp.shape(x)[0], self._data)
        all_nums = tree_util.tree_flatten(nums_tree)[0]
        self._num = all_nums[0]
        # else:
        #     self._num = _num
    
    @property
    def start(self):
        return jnp.array(0)

    @property
    def length(self):
        return self._num
    
    def remaining(self, iter):
        return self._num - iter

    def next(self, iter):
        return iter + 1

    def get(self, iter):
        return tree_util.tree_map(lambda x: x[iter], self._data)

    # override slice/batch/flatten transformations
    # to just be reshapes for efficiency
    def slice(self, start, stop, step=1):
        data = jax.tree_util.tree_map(lambda x: x[start:stop:step])
        return PyTreeDataset(data)

    def batch(self, n):
        def reshape(x):
            # chop off remainder if we have more than 1 batch worth of data
            if x.shape[0] > n:
                x = x[:-(x.shape[0] % n)] if x.shape[0] % n > 0 else x
                bs = n
            else:
                bs = x.shape[0]
            return x.reshape((-1, bs) + x.shape[1:])
        data = jax.tree_util.tree_map(reshape, self._data)
        return PyTreeDataset(data)
    
    def flatten(self):
        def reshape(x):
            return x.reshape((-1,) + x.shape[2:])
        data = jax.tree_util.tree_map(reshape, self._data)
        return PyTreeDataset(data)

    # read just returns self
    def read(self):
        return self

    # This dataset type is shuffleable

    def shuffle(self, key):
        idxs = jax.random.permutation(key, jnp.arange(self._num))
        data = jax.tree_util.tree_map(lambda x: x[idxs,...], self._data)
        return PyTreeDataset(data)

    # jax tree methods
    def tree_flatten(self):
        return (self._data, self._num), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class MappedDataset(Dataset):
    def __init__(self, dataset, fun):
        self._dataset = dataset
        self._fun = fun

    @property
    def start(self):
        return self._dataset.start
    
    def remaining(self, iterator):
        return self._dataset.remaining(iterator)

    def next(self, iterator):
        return self._dataset.next(iterator)

    def get(self, iterator):
        it = self._dataset.get(iterator)
        return self._fun(it)

    def tree_flatten(self):
        return (self._dataset,), self._fun

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)

class BatchDataset(Dataset):
    pass

class FlatDataset(Dataset):
    pass

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