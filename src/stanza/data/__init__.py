import abc
import itertools
import jax
import jax.numpy as jnp
import multiprocessing.pool as mp_pool

from functools import partial

from typing import TypeVar, Generic, Iterator, Generator, Callable, Optional

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

    def slice(self, off : int, length : int) -> T:
        off = off or 0
        length = length or len(self) - off
        idxs = jnp.arange(off, off+length)
        return jax.vmap(lambda i: self[i])(idxs)
    
    def map(self, fn : Callable[[T], V]) -> "Mapped[V]":
        return Mapped(self, fn)

    def cache(self) -> "Data[T]":
        return PyTreeData(self.slice(0, len(self)))

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
        return jax.tree_map(
            lambda x: x[idx],
            self.tree
        )
    
    def slice(self, off : int, length : int) -> T:
        return jax.tree_map(
            lambda x: x[off:off+length],
            self.tree
        )
    
    @staticmethod
    def from_data(data : Data[T]) -> "PyTreeData[T]":
        return PyTreeData(data.slice(0, len(data)))

class IOData(Data[T], abc.ABC):
    def __init__(self):
        ex = self._fetch(0)
        self._ex = jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), ex)

    @abc.abstractmethod
    def _fetch(self, idx: jax.Array) -> T: ...

    def __getitem__(self, idx : jax.Array) -> T:
        data = jax.pure_callback(self._fetch, self._ex, idx)
        return data

@partial(jax.jit,static_argnums=(1,))
def _shuffle(rng_key, len):
    return jax.random.permutation(rng_key, jnp.arange(len))

class DataLoader(Generic[T]):
    def __init__(self, data: Data[T], *, batch_size : int =1,
                shuffle : bool =False, rng_key : jax.Array = None, drop_jagged : bool =False,
                num_workers : int = 1, chunksize : int =16):
        if shuffle and rng_key is None:
            raise ValueError("Must provide rng_key if shuffle=True")
        self.rng_key = rng_key
        self.data = data
        self.batch_size = batch_size
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.drop_jagged = drop_jagged
        self.pool = mp_pool.ThreadPool(max(num_workers,1))
    
        def _get_batch(indices):
            batch = jax.vmap(lambda i: data[i])(indices)
            return jax.tree_map(
                lambda x: jax.device_put(x), batch
            )
        self._get_batch = jax.jit(_get_batch)
    
    def cycle(self) -> Generator[T, None, None]:
        while True:
            for data in self:
                yield data

    def __iter__(self) -> Iterator[T]:
        if self.shuffle:
            rng_key, subkey = jax.random.split(self.rng_key)
            indices = _shuffle(subkey, len(self.data))
            self.rng_key = rng_key
        else:
            indices = jnp.arange(len(self.data))
        if len(indices) % self.batch_size != 0:
            final_batch_len = len(indices) % self.batch_size
            last_batch = indices[-final_batch_len:]
            indices = indices[:-final_batch_len]
        else:
            last_batch = None
        indices = jnp.reshape(indices, (-1, self.batch_size))
        batch_indices = iter(indices)
        if last_batch is not None and \
                (not self.drop_jagged or indices.shape[0] == 0):
            batch_indices = itertools.chain(batch_indices, [last_batch])
        return self.pool.imap(
            self._get_batch,
            batch_indices, chunksize=self.chunksize
        )
   
    def __len__(self) -> int:
        return (
            len(self.data) // self.batch_size if self.drop_jagged else 
            (len(self.data) + self.batch_size - 1) // self.batch_size
        )