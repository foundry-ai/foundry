from typing import Generic, TypeVar
from .core import Data

T = TypeVar("T")

class DataLoader(Generic[T]):
    def __init__(self, data: Data[T], *, 
                batch_size : int = 1,
                shuffle : bool = False,
                rng_key : jax.Array = None,
                drop_jagged : bool = False,
                num_workers : int = 1, chunksize : int = 16):
        if shuffle and rng_key is None:
            raise ValueError("Must provide rng_key if shuffle=True or transforms are provided")
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

        if self.batch_size is None:
            def loader():
                data = self._get_batch(indices)
                yield data
            return loader()
        else:
            if len(indices) % self.batch_size != 0:
                final_batch_len = len(indices) % self.batch_size
                last_batch = indices[-final_batch_len:]
                indices = indices[:-final_batch_len]
            else:
                last_batch = None
            indices = jnp.reshape(indices, (-1, self.batch_size))
            if last_batch is not None and \
                    (not self.drop_jagged or indices.shape[0] == 0):
                indices = iter(indices)
                indices = itertools.chain(indices, [last_batch])
            else:
                indices = iter(indices)
            return self.pool.imap(
                self._get_batch,
                indices, chunksize=self.chunksize
            )
   
    def __len__(self) -> int:
        return (
            len(self.data) // self.batch_size if self.drop_jagged else 
            (len(self.data) + self.batch_size - 1) // self.batch_size
        )