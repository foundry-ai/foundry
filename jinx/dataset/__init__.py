import jax
import jax.tree_util as tree_util
import jax.numpy as jnp

from jax.tree_util import register_pytree_node_class
from functools import partial as partial
from loguru import logger

INFINITE = float("inf")

# Dataset iterators should be immutable!
class Dataset:
    @property
    def start(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement start")

    @property
    def remaining(self, iterator):
        raise NotImplementedError(f"{self.__class__.__name__} must implement remaining")

    # Whether an iterator from this dataset has a next
    def is_end(self, iterator):
        return self.remaining(iterator) <= 0

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

    # Do not override the __iter__
    def __iter__(self):
        it = self.start
        while not self.is_end(it):
            o = self.get(it)
            yield o
            it = self.next(it)

    # Transformations
    def batch(self, n):
        return BatchedDataset(self, n)

    def map(self, func):
        return MappedDataset(self, func)

    def shuffle(self, key):
        return self

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
    def __init__(self, data):
        self._data = data
        nums_tree = tree_util.tree_map(lambda x: jnp.shape(x)[0], self._data)
        all_nums = tree_util.tree_flatten(nums_tree)[0]
        self._num = all_nums[0]
    
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

    # override transformations
    # to happen in-place
    def batch(self, n):
        def reshape(x):
            # chop off remainder
            x = x[:-(x.shape[0] % n)]
            return x.reshape((-1, n) + x.shape[1:])
        data = jax.tree_util.tree_map(reshape, self._data)
        return PyTreeDataset(data)
    
    def flatten(self):
        def reshape(x):
            # chop off remainder
            x = x[:-(x.shape[0] % n)]
            return x.reshape((-1, n) + x.shape[1:])
        data = jax.tree_util.tree_map(reshape, self._data)
        return PyTreeDataset(data)

    def map(self, func):
        new_data = jax.vmap(func)(self._data)
        return PyTreeDataset(new_data)

    def shuffle(self, key):
        idx = jax.random.permutation(key, self._num)
        data = jax.vmap(lambda x: x[idx], self._data)
        return PyTreeDataset(data)

    # jax tree methods
    def tree_flatten(self):
        return (self._data,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# @register_pytree_node_class
# class MappedDataset(Dataset):
#     def __init__(self, dataset, fun):
#         self._dataset = dataset
#         self._fun = fun

#     @property
#     def start(self):
#         return self._dataset.start
    
#     def is_end(self, iterator):
#         return self._dataset.is_end(iterator)

#     @property
#     def length(self):
#         return self._dataset.length

#     def next(self, iterator):
#         return self._dataset.next(iterator)

#     def get(self, iterator):
#         it = self._dataset.get(iterator)
#         return self._fun(it)

#     def tree_flatten(self):
#         return (self._dataset,), self._fun

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children, aux_data)