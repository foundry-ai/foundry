from stanza.dataclasses import dataclass, field, replace
from functools import partial
from contextlib import contextmanager
from typing import (
    TypeVar, Generic, Callable, Sequence,
    Generator
)
from .stream import StreamBuilder, DataStream

import math
import jax
import jax.tree_util
import jax.numpy as jnp

T = TypeVar('T')
V = TypeVar('V')

class Data(Generic[T]):
    """ A dataset of elements of type T. Not necessarily a jax pytree."""

    # A Data must implement these functions.
    # Non-indexable Data may choose to only implement
    # stream().

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx : jax.typing.ArrayLike) -> T:
        raise NotImplementedError()
    
    def stream(self) -> StreamBuilder[T]:
        return IndexedStreamBuilder(self, len(self))

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
        idxs = jnp.arange(len(self), dtype=jnp.uint64)
        return jax.vmap(lambda i: self[i])(idxs)

    def slice(self, off : int, length : int) -> "Data[T]":
        length = length or len(self) - off
        idxs = jnp.arange(length, dtype=jnp.uint64) + off
        return PyTreeData(jax.vmap(lambda i: self[i])(idxs))

    def map(self, fn : Callable[[T], V]) -> "MappedData[V]":
        return MappedData(self, fn)
    
    # "caching" data realizes any transformations,
    # by default storing the realized data in memory.
    def cache(self) -> "PyTreeData[T]":
        return PyTreeData(self.as_pytree())
    

@dataclass
class MappedData(Data[T]):
    data : Data[V]
    fn: Callable[[V], T] = field(pytree_node=False)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx : jax.typing.ArrayLike) -> T:
        return self.fn(self.data[idx])
    
    def stream(self) -> StreamBuilder[T]:
        return self.data.stream().map(self.fn)

    # A utility which uses tracing
    # to compute the mapped structure under the given function.
    @staticmethod
    @partial(jax.jit, static_argnums=(0,1))
    def _compute_structure(fn, data_structure):
        sample = jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.type), data_structure)
        mapped = fn(sample)
        return jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), mapped
        )
    
    @property
    def structure(self):
        return MappedData._compute_structure(self.fn, self.data.structure)

    # Cannot append, replace, or delete
    # on Mapped data!

    def as_pytree(self) -> "T":
        return jax.vmap(self.fn)(self.data.as_pytree())
    
    def slice(self, off : int, length : int) -> T:
        return self.data.slice(off, length).map(self.fn)

# A Data backed by a jax pytree
class PyTreeData(Data[T]):
    def __init__(self, tree: T | None = None, n: int | None = None):
        if tree is None:
            self.n = 0
            self.tree = tree
        else:
            if n is None:
                with jax.ensure_compile_time_eval():
                    ns = jnp.array([x.shape[0] for x in jax.tree_leaves(tree)], dtype=jnp.uint64)
                    n = ns[0]
                    assert jnp.all(ns == n)
            self.n = n
            self.tree = tree

    def __len__(self):
        return self.n

    def __getitem__(self, idx : jax.typing.ArrayLike) -> T:
        idx = jnp.array(idx, dtype=jnp.uint64)
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
            lambda x: jax.lax.dynamic_slice(x,
                    jnp.broadcast_to(jnp.array(off, dtype=jnp.uint64), (x.ndim,)),
                    (length,) + x.shape[1:]),
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
    lambda d: ((d.tree,), d.n),
    lambda n, c: PyTreeData(c[0], n)
)

@dataclass
class IndexedDataStream(DataStream[T]):
    data: Data[T]
    offset: jax.Array
    max_offset: int = field(pytree_node=False)
    batch_shape: Sequence[int] = field(pytree_node=False)

    shuffle_key: jax.Array | None
    indices: jax.Array | None
    resample : bool = field(pytree_node=False)

    @staticmethod
    def create(data, max_offset, batch_shape,
               shuffle_key=None, resample=False):
        indices_per_batch = math.prod(batch_shape)
        batches = max_offset // indices_per_batch
        max_offset = batches * indices_per_batch
        if shuffle_key is not None and not resample:
            shuffle_key, r = jax.random.split(shuffle_key)
            indices = jax.random.permutation(r, max_offset)
        else: indices = None
        return IndexedDataStream(
            data=data,
            offset=jnp.zeros((), dtype=jnp.uint64),
            max_offset=max_offset,
            batch_shape=batch_shape,
            shuffle_key=shuffle_key,
            indices=indices,
            resample=resample,
        )

    @jax.jit
    def __len__(self):
        batch_size = math.prod(self.batch_shape)
        return (self.max_offset - self.offset) // batch_size

    @jax.jit
    def has_next(self):
        return self.offset < self.max_offset

    @jax.jit
    def _next(self):
        shuffle_key = self.shuffle_key
        batch_shape = self.batch_shape
        batch_size = math.prod(batch_shape)
        if self.resample:
            shuffle_key, r = jax.random.split(shuffle_key)
            idxs = jax.random.randint(r, (batch_size,), minval=0, maxval=self.max_offset)
            data = jax.vmap(lambda x: self.data[x])(idxs)
        elif self.indices is not None:
            idxs = jax.lax.dynamic_slice(self.indices, self.offset[None], self.batch_shape)
            data = jax.vmap(lambda i: self.data[i])(idxs)
        else:
            data = self.data.slice(self.offset, batch_size).as_pytree()
        data = jax.tree.map(lambda x: jnp.reshape(x, batch_shape + x.shape[1:]), data)
        return self.offset + batch_size, shuffle_key, data

    def next(self):
        offset, shuffle_key, batch = self._next()
        return replace(self, 
            offset=offset, 
            shuffle_key=shuffle_key
        ), batch
    
    @jax.jit
    def _reset(self):
        shuffle_key = self.shuffle_key
        if not self.resample and self.shuffle_key is not None:
            shuffle_key, r = jax.random.split(shuffle_key)
            indices = jax.random.permutation(r, self.max_offset)
        else:
            indices = None
        return jnp.zeros_like(self.offset), indices, shuffle_key

    def reset(self):
        offset, indices, shuffle_key = self._reset()
        return replace(
            self, offset=offset, indices=indices,
            shuffle_key=shuffle_key
        )

@dataclass
class IndexedStreamBuilder(StreamBuilder[T]):
    data: Data[T]
    max_offset: int = field(pytree_node=False)
    batch_shape: Sequence[int] | None = field(default=None, pytree_node=False)
    shuffle_key: jax.Array | None = None
    resample : bool = field(default=False, pytree_node=False)

    def batch(self, batch_size: int) -> "IndexedStreamBuilder[T]":
        return replace(self, 
            batch_shape=((batch_size,) + self.batch_shape) 
            if self.batch_shape else (batch_size,)
        )
    
    def shuffle(self, rng_key : jax.Array, resample=False) -> "IndexedStreamBuilder[T]":
        return replace(self,
            shuffle_key=rng_key, 
            resample=resample or self.resample
        )

    @contextmanager
    def build(self) -> Generator[DataStream[T], None, None]:
        yield IndexedDataStream.create(
            self.data, self.max_offset, self.batch_shape,
            self.shuffle_key, self.resample
        )