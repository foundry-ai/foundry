import foundry.core.transforms as F
import foundry.core.tree as tree
import foundry.numpy as jnp

from foundry.core.dataclasses import (
    dataclass, field, replace
)
from foundry.core.typing import ArrayLike

from functools import partial
from contextlib import contextmanager
from typing import (
    TypeVar, Generic, Callable, Sequence,
    Generator
)
from .stream import StreamBuilder, DataStream

import jax.tree_util

import math
import numpy as np

T = TypeVar('T')
V = TypeVar('V')

# Make indices 64-bit if x64 is enabled
idx_dtype = int

class Data(Generic[T]):
    """ A dataset of elements of type T. Not necessarily a jax pytree."""

    # A Data must implement these functions.
    # Non-indexable Data may choose to only implement
    # stream().

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx : ArrayLike) -> T:
        raise NotImplementedError()
    
    def stream(self) -> StreamBuilder[T]:
        return IndexedStreamBuilder(self, len(self))

    # Get the structure of one instance of the data.
    @property
    def structure(self):
        return tree.map(
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
        idxs = jnp.arange(len(self), dtype=idx_dtype)
        return jax.vmap(lambda i: self[i])(idxs)

    def slice(self, off : ArrayLike, length : ArrayLike) -> "Data[T]":
        length = np.array(length).item()
        length = length or len(self) - off
        idxs = jnp.arange(length, dtype=idx_dtype) + off
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
    fn: Callable[[V], T]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx : ArrayLike) -> T:
        return self.fn(self.data[idx])
    
    def stream(self) -> StreamBuilder[T]:
        return self.data.stream().map(self.fn)

    # A utility which uses tracing
    # to compute the mapped structure under the given function.
    @staticmethod
    @F.jit
    def _compute_structure(fn, data_structure):
        sample = tree.map(lambda x: jnp.zeros(x.shape, x.type), data_structure)
        mapped = fn(sample)
        return jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), mapped
        )
    
    @property
    def structure(self):
        return MappedData._compute_structure(self.fn, self.data.structure)

    # Cannot append, replace, or delete
    # on Mapped data!

    def as_pytree(self) -> "T":
        return jax.vmap(self.fn)(self.data.as_pytree())
    
    def slice(self, off : ArrayLike, length : ArrayLike) -> T:
        return self.data.slice(off, length).map(self.fn)

# A Data backed by a jax pytree
class PyTreeData(Data[T]):
    def __init__(self, tree: T | None = None):
        if tree is None:
            self.n = 0
            self.tree = tree
        else:
            with jax.ensure_compile_time_eval():
                ns = jnp.array([jnp.shape(x)[0] for x in jax.tree_leaves(tree)], dtype=idx_dtype)
                n = ns[0]
                assert jnp.all(ns == n)
            self.n = n
            self.tree = tree

    def __len__(self):
        return self.n

    def __getitem__(self, idx : ArrayLike) -> T:
        idx = jnp.array(idx, dtype=idx_dtype)
        assert idx.ndim == 0
        return jax.tree.map(
            lambda x: x[idx],
            self.tree
        )

    @property
    def structure(self):
        return jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape[1:], x.dtype),
            self.tree
        )

    def slice(self, off : ArrayLike, length : ArrayLike) -> T:
        # the length must be a scalar
        length = np.array(min(len(self), length)).item()
        return PyTreeData(jax.tree.map(
            lambda x: jax.lax.dynamic_slice(x,
                    jnp.broadcast_to(jnp.array(off, dtype=idx_dtype), (x.ndim,)),
                    (length,) + x.shape[1:]),
            self.tree
        ))
    
    def as_pytree(self) -> T:
        return self.tree
    
    def append(self, data: Data[T]) -> "PyTreeData[T]":
        tree = data.as_pytree()
        if tree is None: return self
        if self.tree is None: return PyTreeData(tree)
        tree = jax.tree.map(lambda x, y: jnp.concatenate((x,y), axis=0), self.tree, tree)
        return PyTreeData(tree)

jax.tree_util.register_pytree_node(
    PyTreeData,
    lambda d: ((d.tree,), None),
    lambda n, c: PyTreeData(c[0])
)

@dataclass
class IndexedDataStream(DataStream[T]):
    data: Data[T]
    offset: jax.Array
    max_offset: int
    batch_shape: Sequence[int]

    shuffle_key: jax.Array | None
    indices: jax.Array | None
    resample : bool

    @staticmethod
    def create(data, max_offset, batch_shape,
               shuffle_key=None, resample=False, ):
        indices_per_batch = math.prod(batch_shape)
        if indices_per_batch > max_offset: 
            # reduce batch_shape to fit at least one batch
            batch_rem = math.prod(batch_shape[1:])
            leading_axis = max_offset // batch_rem
            if leading_axis > 0:
                batch_shape = (leading_axis,) + tuple(batch_shape[1:])
                indices_per_batch = math.prod(batch_shape)

        batches = max_offset // indices_per_batch
        max_offset = batches * indices_per_batch
        if shuffle_key is not None and not resample:
            shuffle_key, r = jax.random.split(shuffle_key)
            indices = jax.random.permutation(r, max_offset)
        else: indices = None
        return IndexedDataStream(
            data=data,
            offset=jnp.zeros((), dtype=idx_dtype),
            max_offset=max_offset,
            batch_shape=batch_shape,
            shuffle_key=shuffle_key,
            indices=indices,
            resample=resample,
        )

    @F.jit
    def __len__(self):
        batch_size = math.prod(self.batch_shape)
        return (self.max_offset - self.offset) // batch_size

    @F.jit
    def has_next(self):
        return self.offset < self.max_offset

    @F.jit
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
    
    @F.jit
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
    max_offset: int
    batch_shape: Sequence[int] | None = None
    shuffle_key: jax.Array | None = None
    resample : bool = False

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