import abc
import jax
import jax.numpy as jnp
import multiprocessing.pool as mp_pool

from functools import partial

from typing import TypeVar, Generic, Iterator, Generator
T = TypeVar('T')

class Data(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def __len__(self) -> int:
        ...

    @abc.abstractmethod
    def __getitem__(self, idx : int) -> T:
        ...

    def get_batch(self, batch_indices: jax.Array) -> T:
        batch = [self[i] for i in batch_indices]
        combined_batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *batch)
        return combined_batch


def slice_dataset(dataset: Data[T], start : int, end : int) -> T:
    if start is None: start = 0
    if end is None: end = len(dataset)
    if start < 0: start = len(dataset) + start
    if end < 0: end = len(dataset) + end

    data = []
    for i in range(start, end):
        data.append(dataset[i])
    combined_data = jax.tree_map(lambda *xs: jnp.stack(xs), *data)
    return combined_data


def to_pytree(dataset : Data[T]) -> T:
    samples = []
    for i in range(len(dataset)):
        samples.append(dataset[i])
    tree = jax.tree_map(lambda *xs: jnp.stack(xs), *samples)
    return tree
# A pytorch dataset from a jax pytree
class PyTreeData(Data):
    def __init__(self, tree):
        super().__init__()
        ns = jnp.array([x.shape[0] for x in jax.tree_leaves(tree)])
        n = ns[0]
        assert jnp.all(ns == n)
        self.n = n
        self.tree = tree

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return jax.tree_map(
            lambda x: x[idx],
            self.tree
        )

    def get_batch(self, batch_indices):
        return jax.vmap(self.__getitem__)(batch_indices)

def _get_batch(dataset, indices):
    batch = dataset.get_batch(indices)
    return jax.tree_map(
        lambda x: jax.device_put(x), batch
    )

@partial(jax.jit,static_argnums=(1,))
def _shuffle(rng_key, len):
    return jax.random.permutation(rng_key, jnp.arange(len))

class DataLoader(Generic[T]):
    def __init__(self, dataset : Data[T], *, batch_size=1,
                shuffle=False, rng_key=None, drop_jagged=False,
                batch_sampler=None, num_workers=1, chunksize=16):
        if shuffle and rng_key is None:
            raise ValueError("Must provide rng_key if shuffle=True")
        self.rng_key = rng_key
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.pool = mp_pool.ThreadPool(max(num_workers,1))
    
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
            partial(_get_batch, self.dataset),
            batch_indices, chunksize=self.chunksize
        )
    
    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size

def cycle(loader : DataLoader[T]) -> Generator[T, None, None]:
    while True:
        for data in loader:
            yield data