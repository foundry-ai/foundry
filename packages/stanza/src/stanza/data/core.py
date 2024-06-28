from stanza.dataclasses import dataclass, field, replace

import abc
import itertools
import jax
import jax.tree_util
import jax.numpy as jnp
import multiprocessing.pool as mp_pool
from contextlib import contextmanager

from functools import partial

from typing import (
    TypeVar, Generic, Sequence, Self,
    Iterator, Generator, Callable, Optional
)

T = TypeVar('T')
V = TypeVar('V')

# Represents a stream of data.
# Must be a jax type!
class DataStream(Generic[T]):
    def has_next(self):
        raise NotImplementedError()

    def next(self) -> tuple["DataStream[T]", T]:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    # Optional to implement
    def shuffle(self, rng_key: jax.Array, resample=False) -> "DataStream[T]":
        raise NotImplementedError()
    
    # Optional to override
    def map(self, fn: Callable[[T], V]) -> "DataStream[V]":
        return MappedStream(self, fn)

# The stream configuration.
@dataclass
class StreamConfig:
    # If none, don't shuffle the data
    shuffle_key: jax.Array | None
    # The (maximum) batch size to stream the data with.
    batch_size: int = field(pytree_node=False)

class Data(Generic[T]):
    """ A dataset of elements of type T. Not necessarily a jax pytree."""

    # A Data must implement these functions.
    # Non-indexable Data may choose to only implement
    # stream().

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx : jax.typing.ArrayLike) -> T:
        raise NotImplementedError()
    
    @contextmanager
    def stream(self, *, batch_size) -> DataStream:
        yield IndexedDataStream.create(
            self, batch_size
        )

    # Get the structure of one instance of the data.
    @property
    def structure(self):
        return jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self[0]
        )

    # Optional to implement. Note the functional API.
    # These may potentially choose to invalidate the original Data object,
    # depending on the implementation.

    def append(self, data: "Data[T]") -> "Data[T]":
        raise NotImplementedError()

    def replace(self, idx: int, data: "Data[T]") -> "Data[T]":
        raise NotImplementedError()

    def delete(self, start: int, 
            length: int | None = None) -> "Data[T]":
        raise NotImplementedError()

    # These methods have default implementations,
    # but may be overriden with more efficient ones
    # depending on the backing Data storage.

    def as_pytree(self) -> T:
        idxs = jnp.arange(len(self))
        return jax.vmap(lambda i: self[i])(idxs)

    def slice(self, off : int, length : int) -> "Data[T]":
        length = length or len(self) - off
        idxs = jnp.arange(length) + off
        return PyTreeData(jax.vmap(lambda i: self[i])(idxs))

    def map(self, fn : Callable[[T], V]) -> "Mapped[V]":
        return Mapped(self, fn)
    
    # "caching" data realizes any transformations,
    # by default storing the realized data in memory.
    def cache(self) -> Self:
        return PyTreeData(self.as_pytree())
    

@dataclass
class Mapped(Data[T]):
    data : Data[V]
    fn: Callable[[V], T] = field(pytree_node=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx : jax.typing.ArrayLike) -> T:
        return self.fn(self.data[idx])
    
    @contextmanager
    def stream(self, batch_size):
        with self.data.stream(batch_size=batch_size) as s:
            yield MappedStream(s, self.fn)

    # A utility which uses tracing
    # to compute the mapped structure under the given function.
    @staticmethod
    @partial(jax.jit, static_argnums=(0,1))
    def _compute_structure(fn, data_structure):
        sample = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), data_structure)
        mapped = fn(sample)
        return jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), mapped
        )
    
    @property
    def structure(self):
        return Mapped._compute_structure(self.fn, self.data.structure)

    # Cannot append, replace, or delete
    # on Mapped data!

    def as_pytree(self) -> "T":
        return jax.vmap(self.fn)(self.data.as_pytree())
    
    def slice(self, off : int, length : int) -> T:
        return self.data.slice(off, length).map(self.fn)

# A Data backed by a jax pytree
class PyTreeData(Data[T]):
    def __init__(self, tree: T | None = None):
        if tree is None:
            self.n = 0
            self.tree = tree
        else:
            with jax.ensure_compile_time_eval():
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
        return jax.tree_util.tree_map(
            lambda x: x[idx],
            self.tree
        )

    @property
    def structure(self):
        return jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
            self.tree
        )

    def slice(self, off : int, length : int) -> T:
        return PyTreeData(jax.tree_util.tree_map(
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

jax.tree_util.register_pytree_node(
    PyTreeData,
    lambda n: ((n.tree,), None),
    lambda _, c: PyTreeData(c[0])
)


@dataclass
class IndexedDataStream(DataStream[T]):
    data: Data[T]
    offset: jax.Array

    indices: jax.Array | None
    shuffle_key: jax.Array | None

    resample : bool = field(pytree_node=False)
    batch_size: int = field(pytree_node=False)
    max_offset: int = field(pytree_node=False)

    @staticmethod
    def create(data, batch_size, shuffle_key=None, resample=False):
        batch_size = min(len(data), batch_size)
        max_offset = len(data)
        max_offset = max_offset - (max_offset % batch_size)

        if shuffle_key is not None and resample:
            r, shuffle_key = jax.random.split(shuffle_key)
            indices = jax.random.permutation(r, max_offset)
        else:
            indices = None

        return IndexedDataStream(
            data=data,
            offset=jnp.zeros((), dtype=jnp.uint32),
            indices=indices,
            shuffle_key=shuffle_key,
            resample=resample,
            batch_size=batch_size,
            max_offset=max_offset
        )
    
    @jax.jit
    def has_next(self):
        return self.offset < self.max_offset
    
    @partial(jax.jit, donate_argnums=(1,2,3))
    def _next(self, offset, indices, shuffle_key):
        if self.resample:
            shuffle_key, r = jax.random.split(shuffle_key)
            idxs = jax.random.randint(r, (), minval=0, maxval=self.max_offset)
            data = jax.vmap(lambda x: self.data[x])(idxs)
        elif indices is not None:
            idxs = jax.lax.dynamic_slice(indices, offset[None], (self.batch_size,))
            data = jax.vmap(lambda i: self.data[i])(idxs)
        else:
            data = self.data.slice(offset, self.batch_size).as_pytree()
        stream = replace(self,
            offset=offset + self.batch_size,
            indices=indices,
            shuffle_key=shuffle_key,
        )
        return stream, data

    # donte the correct arguments so that
    # we don't re-use a given stream. TODO: Use jax API
    def next(self):
        return self._next(self.offset, self.indices, self.shuffle_key)
    
    @partial(jax.jit, donate_argnums=(1,2,3))
    def _reset(self, offset, indices, shuffle_key):
        if self.resample:
            shuffle_key, r = jax.random.split(shuffle_key)
            indices = jax.random.permutation(shuffle_key, self.max_offset)
        else:
            shuffle_key, indices = None, None
        return replace(self,
            offset=jnp.zeros_like(offset),
            indices=indices,
            shuffle_key=shuffle_key,
            batch_size=self.batch_size,
            max_offset=self.max_offset
        )

    def reset(self):
        return self._reset(self.offset, self.indices, self.shuffle_key)

    def shuffle(self, rng_key: jax.Array, resample : bool = False) -> "IndexedDataStream[T]":
        return IndexedDataStream.create(
            self.data, self.batch_size,
            shuffle_key=rng_key,
            resample=resample
        )

@dataclass
class MappedStream(DataStream[T]):
    stream: DataStream[V]
    fn: Callable[[V], T] = field(pytree_node=False)

    @jax.jit
    def has_next(self):
        return self.stream.has_next()

    @jax.jit
    def next(self):
        stream, batch = self.stream.next()
        stream = MappedStream(stream, self.fn)
        batch = jax.vmap(self.fn)(batch)
        return stream, batch

    @jax.jit
    def reset(self):
        return MappedStream(self.stream.reset(), self.fn)