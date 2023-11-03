from stanza.dataclasses import dataclass, field
from stanza.data import Data

import jax
import jax.numpy as jnp

from PIL import Image

class Storage:
    @property
    def length(self):
        pass

    def get(self, idx):
        pass

    def get_batch(self, idx):
        data = []
        for i in idx:
            data.append(self.get(i))
        return jax.tree_map(
            lambda *x: jnp.stack(x, axis=0), 
            *data
        )

class FolderImageStorage(Storage):
    def __init__(self, folder_path, files=None,
                 start=None, end=None):
        self.folder_path = folder_path
        self.files = files or list(folder_path.iterdir())[start:end]
    
    @property
    def length(self):
        return len(self.files)
    
    def get(self, idx):
        p = self.folder_path / self.files[idx]
        img = Image.open(p)
        img = jnp.asarray(img)
        # reorder the BGR channels to RGB
        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]
        img = jnp.stack((r,g,b), axis=-1)
        return img

@dataclass
class StoredData(Data):
    storage: Storage = field(jax_static=True)
    shuffle_indices: jnp.ndarray = None

    @staticmethod
    def _length(self):
        return jnp.array(self.storage.length, dtype=jnp.int32)
    
    @staticmethod
    def _get(self, idx):
        if idx.ndim == 0:
            return self.storage.get(idx)
        elif idx.ndim == 1:
            return self.storage.get_batch(idx)
        else:
            raise ValueError("Invalid index shape")

    @property
    def start(self):
        return 0

    def next(self, it):
        return it + 1
    
    def advance(self, it, n):
        return it + n

    @jax.jit
    def get(self, idx):
        with jax.ensure_compile_time_eval():
            sample = self.storage.get(0)
        if self.shuffle_indices is not None:
            idx = self.shuffle_indices[idx]
        return jax.pure_callback(
            self._get, sample,
            self, idx, vectorized=True
        )

    def remaining(self, idx):
        return self.length - idx

    @property
    def length(self):
        if self.shuffle_indices is not None:
            return self.shuffle_indices.shape[0]
        return jax.pure_callback(
            self._length,
            jnp.zeros((), dtype=jnp.int32), self
        )
    
    def is_end(self, idx):
        return idx == self.length

    def shuffle(self, rng_key):
        with jax.ensure_compile_time_eval():
            N = self._length(self)
        indices = jax.random.permutation(rng_key, N)
        return StoredData(
            storage=self.storage,
            shuffle_indices=indices
        )