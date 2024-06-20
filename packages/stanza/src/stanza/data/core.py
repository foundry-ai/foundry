import abc
import itertools
import jax
import jax.numpy as jnp
import multiprocessing.pool as mp_pool

from functools import partial

from typing import (
    TypeVar, Generic, Sequence, Self,
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
    def __getitem__(self, idx : jax.typing.ArrayLike) -> T:
        ...

    def as_pytree(self) -> T:
        idxs = jnp.arange(len(self))
        return jax.vmap(lambda i: self[i])(idxs)

    def slice(self, off : int, length : int) -> "Data[T]":
        length = length or len(self) - off
        idxs = jnp.arange(length) + off
        return PyTreeData(jax.vmap(lambda i: self[i])(idxs))

    def map(self, fn : Callable[[T], V]) -> "Mapped[V]":
        return Mapped(self, fn)

    def append(self, data: "Data[T]") -> "Data[T]":
        return Concatenated(self, data)

    # def replace(self, idx: int, data: "Data[T]") -> "Data[T]":
    #     return Replaced(self, data)

    # def delete(self, start: int, 
    #         length: int | None = None) -> "Data[T]":
    #     return Deleted(self, start, length)

class Mapped(Data[T]):
    def __init__(self, dataset : Data[V], fn : Callable[[V], T]):
        self.dataset = dataset
        self.fn = fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx : jax.typing.ArrayLike) -> T:
        return self.fn(self.dataset[idx])
    
    def as_pytree(self) -> "T":
        return jax.vmap(self.fn)(self.dataset.as_pytree())
    
    def slice(self, off : int, length : int) -> T:
        return self.dataset.slice(off, length).map(self.fn)

class Concatenated:
    def __init__(self, *datas):
        self.datas = datas

# A pytorch dataset from a jax pytree
class PyTreeData(Data[T]):
    def __init__(self, tree: T | None = None):
        if tree is None:
            self.n = 0
            self.tree = tree
        else:
            ns = jnp.array([x.shape[0] for x in jax.tree_leaves(tree)])
            n = ns[0]
            assert jnp.all(ns == n)
            self.n = n
            self.tree = tree

    def __len__(self):
        return self.n

    def __getitem__(self, idx : jax.typing.ArrayLike) -> T:
        idx = jnp.array(idx)
        assert idx.ndim == 0
        return jax.tree_map(
            lambda x: x[idx],
            self.tree
        )
    
    def slice(self, off : int, length : int) -> T:
        return PyTreeData(jax.tree_map(
            lambda x: jax.lax.dynamic_slice(x, jnp.broadcast_to(jnp.array(off), (x.ndim,)), (length,) + x.shape[1:]),
            self.tree
        ))
    
    def as_pytree(self) -> T:
        return self.tree
    
    def append(self, data: Data[T]) -> "PyTreeData[T]":
        tree = data.as_pytree()
        if tree is None: return self
        if self.tree is None: return PyTreeData(tree)
        tree = jax.tree_util.tree_map(lambda x, y: jnp.concatenate((x,y), axis=0), self.tree, tree)
        return PyTreeData(tree)

@partial(jax.jit,static_argnums=(1,))
def _shuffle(rng_key, len):
    return jax.random.permutation(rng_key, jnp.arange(len))
