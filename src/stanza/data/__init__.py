import abc
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

    @staticmethod
    def from_pytree(tree: T) -> "PyTreeData[T]":
        return PyTreeData(tree)

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
        super().__init__()
        ns = jnp.array([x.shape[0] for x in jax.tree_leaves(tree)])
        n = ns[0]
        assert jnp.all(ns == n)
        self.n = n
        self.tree = tree

    def __len__(self):
        return self.n

    def __getitem__(self, idx : int) -> T:
        return jax.tree_map(
            lambda x: x[idx],
            self.tree
        )
    
    def slice(self, off : int, length : int) -> T:
        return jax.tree_map(
            lambda x: x[off:off+length],
            self.tree
        )

def _get_batch(dataset, indices):
    batch = jax.vmap(lambda i: dataset[i])(indices)
    return jax.tree_map(
        lambda x: jax.device_put(x), batch
    )

@partial(jax.jit,static_argnums=(1,))
def _shuffle(rng_key, len):
    return jax.random.permutation(rng_key, jnp.arange(len))

class DataLoader(Generic[T]):
    def __init__(self, dataset : Data[T], *, batch_size : int =1,
                shuffle : bool =False, rng_key : jax.Array = None, drop_jagged : bool =False,
                num_workers : int = 1, chunksize : int =16):
        if shuffle and rng_key is None:
            raise ValueError("Must provide rng_key if shuffle=True")
        self.rng_key = rng_key
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.pool = mp_pool.ThreadPool(max(num_workers,1))
    
        def _get_batch(indices):
            batch = jax.vmap(lambda i: dataset[i])(indices)
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
            indices = _shuffle(subkey, len(self.dataset))
            self.rng_key = rng_key
        else:
            indices = jnp.arange(len(self.dataset))
        if len(indices) % self.batch_size != 0:
            final_batch_len = len(indices) % self.batch_size
            indices = indices[:-final_batch_len]
        indices = jnp.reshape(indices, (-1, self.batch_size))
        batch_indices = iter(indices)
        return self.pool.imap(
            self._get_batch,
            batch_indices, chunksize=self.chunksize
        )
   
    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size