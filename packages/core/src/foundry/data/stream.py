from foundry.core.dataclasses import dataclass, field
from typing import (
    TypeVar, Generic, Callable, Generator
)
from contextlib import contextmanager

import jax

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
    
    # Optional to override
    def map(self, fn: Callable[[T], V]) -> "DataStream[V]":
        return MappedStream(self, fn)

@dataclass
class MappedStream(DataStream[T]):
    stream: DataStream[V]
    fn: Callable[[V], T]

    def __len__(self):
        return len(self.stream)

    def has_next(self):
        return self.stream.has_next()

    def next(self):
        stream, batch = self.stream.next()
        stream = MappedStream(stream, self.fn)
        batch = jax.vmap(self.fn)(batch)
        return batch

    def reset(self):
        return MappedStream(self.stream.reset(), self.fn)


class StreamBuilder(Generic[T]):
    def batch(self, batch_size: int) -> "StreamBuilder[T]":
        raise NotImplementedError()
    
    def shuffle(self, rng_key: jax.Array, resample=False) -> "StreamBuilder[T]":
        raise NotImplementedError()
    
    def map(self, fn: Callable[[T], V]) -> "StreamBuilder[V]":
        return MappedStreamBuilder(self, fn)
    
    @contextmanager
    def build(self) -> Generator[DataStream[T], None, None]:
        raise NotImplementedError()

@dataclass
class MappedStreamBuilder(StreamBuilder[T]):
    builder: StreamBuilder[V]
    fn: Callable[[V], T]

    def batch(self, batch_size: int) -> "MappedStreamBuilder[T]":
        return MappedStreamBuilder(self.builder.batch(batch_size), self.fn)
    
    def shuffle(self, rng_key : jax.Array, resample=False) -> "MappedStreamBuilder[T]":
        return MappedStreamBuilder(
            self.builder.shuffle(rng_key, resample), self.fn
        )

    @contextmanager
    def build(self) -> Generator[DataStream[T], None, None]:
        with self.builder.build() as stream:
            yield MappedStream(stream, self.fn)
