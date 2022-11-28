from loguru import logger
import jax
import jax.tree_util as tree_util
import jax.numpy as jnp
from functools import partial as partial

INFINITE = float("inf")

# Dataset iterators should be immutable!
class Dataset:
    @property
    def start(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement start")

    @property
    def remaining(self, iterator):
        raise NotImplementedError(f"{self.__class__.__name__} must implement remaining")

    # Whether an iterator from this dataset has a next iterator
    def is_end(self, iterator):
        return self.remaining(iterator) == 0

    # Will return the next iterator.
    # If the iterator is at the end, the function does not need to return
    # the same iterator, but it must return an iterator for which is_end is also
    # true
    def next(self, iterator):
        raise NotImplementedError(f"{self.__class__.__name__} must implement next")

    # This will actually do the computation
    # for a given iterator
    def get(self, iterator):
        raise NotImplementedError(f"{self.__class__.__name__} must implement get")

    # DO NOT OVERRIDE BELOW THIS LINE!
    def __iter__(self):
        it = self.start
        while not self.is_end(it):
            o = self.get(it)
            it = self.next(it)

    # Transformations
    def batch(self, n):
        return BatchedDataset(self, n)

    def map(self, func):
        return MappedDataset(self, func)

    def shuffle(self, key, buffer_size):
        return self

    # A shortcut for jax scanning through a dataset
    def fold(self, func, base):
        @partial(jax.jit, static_argnums=0)
        def wrapped_func(func, state):
            iter, accum = state
            item = self.get(iter)
            iter = self.next(iter)
            new_state = func(accum, item)
            return iter, new_state
        state = self.start, base
        _, final_state = jax.lax.while_loop(lambda s: not self.is_end(s[0]),
                            partial(wrapped_func, func), state)
        return final_state
    
    @staticmethod
    def from_pytree(data):
        return PyTreeDataset(data)


class PyTreeDataset(Dataset):
    def __init__(self, data, num=None):
        self.data = data
        if num is None:
            nums_tree = tree_util.tree_map(lambda x: jnp.shape(x)[0], self.data)
            all_nums = tree_util.tree_flatten(nums_tree)[0]
            num = all_nums[0]
            self.num = num
        else:
            self.num = num
    
    @property
    def start(self):
        return jnp.zeros((0,))

    @property
    def length(self):
        return self.num

    def is_end(self, iter):
        return iter >= self.num

    def next(self, iter):
        return iter + 1

    def get(self, iter):
        return tree_util.tree_map(lambda x: x[iter], self.data)

class MappedDataset(Dataset):
    def __init__(self, dataset, fun):
        self._dataset = dataset
        self._fun = fun

    @property
    def start(self):
        return self._dataset.start
    
    def is_end(self, iterator):
        return self._dataset.is_end(iterator)

    @property
    def length(self):
        return self._dataset.length

    def next(self, iterator):
        return self._dataset.next(iterator)

    def get(self, iterator):
        it = self._dataset.get(iterator)
        return self._fun(it)

class BatchedDataset(Dataset):
    def __init__(self, dataset, size):
        self._dataset = dataset
        self._batch_size = size

    def _roll_batch(self, sub_iter):
        next_iter, batch = jax.lax.scan(lambda a, _: (self._dataset.next(a), a), 
                                    sub_iter, None, self._batch_size)
        return sub_iter, batch, next_iter

    @property
    def start(self):
        # get a batch worth of iterators
        sub_start = self._dataset.start
        first_iter, iter_batch, sub_iter = self._roll_batch(sub_start)
        return (first_iter, iter_batch, sub_iter)

    def is_end(self, iterator):
        first_iter, _, _ = iterator
        return self._dataset.is_end(first_iter)

    def next(self, iterator):
        _, _, next_iter = iterator
        first_iter, iter_batch, next_iter = self._roll_batch(next_iter)
        return (first_iter, iter_batch, next_iter)

    def get(self, iterator):
        _, iter_batch, _ = iterator
        return jax.lax.vmap(self._dataset.get, iter_batch)

# class ShuffledDataset(Dataset):
#     def __init__(self, dataset, shuffle_key, buffer_size):
#         self._dataset = dataset
#         self._key = shuffle_key
#         self._buffer_size = buffer_size

#     def _roll_buffer(self, sub_iter):
#         next_iter, buffer = jax.lax.scan(lambda a, _: (self._dataset.next(a), a), 
#                                     sub_iter, None, self._buffer_size)
#         return sub_iter, buffer, next_iter

#     @property
#     def start(self):
#         _, buffer, next_iter = self._roll_buffer(self._dataset.start)
#         shuffle_key, next_key = jax.random.split(self._key)
        
#         # permute the initial iterator buffer
#         buffer_elems = jnp.arange(self._buffer_size)
#         buffer_elems = jax.random.shuffle(shuffle_key, buffer_elems)
#         shuffled_buffer = jax.tree_util.tree_map(lambda x: x[buffer_elems], buffer)

#         return next_key, shuffled_buffer, self._buffer_size, next_iter

#     def is_end(self, iterator):
#         next_key, shuffled_buffer, _, _ = iterator
#         iter = jax.tree_util.tree_map(lambda x: x[0], shuffled_buffer)
#         return self._dataset.is_end(iter)

#     def next(self, iterator):
#         next_key, shuffled_buffer, next_iter = iterator
#         key, next_key = jax.random.split(next_key)
#         idx = jax.random.randint(key, (1,), 0, self._buffer_size)
    
#         # do the replacement
#         next_iter = self._datasset.next(next_iter)
#         return next_key, shuffled_buffer, next_iter

#     def get(self, iterator):
#         next_key, shuffled_buffer, next_iter = iterator
#         pass