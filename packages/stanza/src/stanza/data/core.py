import abc
import itertools
import jax
import jax.numpy as jnp
import multiprocessing.pool as mp_pool

from functools import partial

from typing import (
    TypeVar, Generic, Sequence,
    Iterator, Generator, Callable, Optional
)


T = TypeVar('T')
V = TypeVar('V')

class Data(abc.ABC, Generic[T]):
    """ A dataset of elements of type T. Not necessarily a jax pytree.
    """
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, idx : int) -> T:
        ...

    def as_pytree(self) -> "T":
        return self.slice(0, len(self))

    def slice(self, off : int, length : int) -> T:
        off = off or 0
        length = length or len(self) - off
        idxs = jnp.arange(off, off+length)
        return jax.vmap(lambda i: self[i])(idxs)
    
    def map(self, fn : Callable[[T], V]) -> "Mapped[V]":
        return Mapped(self, fn)

class Mapped(Data[T]):
    def __init__(self, dataset : Data[V], fn : Callable[[V], T]):
        self.dataset = dataset
        self.fn = fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx : int) -> T:
        return self.fn(self.dataset[idx])
    
    def slice(self, off : int, length : int) -> T:
        batch = self.dataset.slice(off, length)
        return jax.vmap(self.fn)(batch)


# A pytorch dataset from a jax pytree
class PyTreeData(Data[T]):
    def __init__(self, tree: T):
        ns = jnp.array([x.shape[0] for x in jax.tree_leaves(tree)])
        n = ns[0]
        assert jnp.all(ns == n)
        self.n = n
        self.tree = tree

    def __len__(self):
        return self.n

    def __getitem__(self, idx : jax.Array) -> T:
        assert idx.ndim == 0
        return jax.tree_map(
            lambda x: x[idx],
            self.tree
        )
    
    def slice(self, off : int, length : int) -> T:
        return jax.tree_map(
            lambda x: jax.lax.dynamic_slice(x, jnp.broadcast_to(jnp.array(off), (x.ndim,)), (length,) + x.shape[1:]),
            self.tree
        )

    @staticmethod
    def from_data(data : Data[T]) -> "PyTreeData[T]":
        return PyTreeData(data.slice(0, len(data)))

@partial(jax.jit,static_argnums=(1,))
def _shuffle(rng_key, len):
    return jax.random.permutation(rng_key, jnp.arange(len))
